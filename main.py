import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
import colorama
from colorama import Fore, Style
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/kaye_L7_30fps/2")
    parser.add_argument("--save-as", default="kaye_L7_30fps_2")
    parser.add_argument("--calib", default="calib/FOV110_RES600.yaml")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--no-viz", default=True)
    parser.add_argument("--sample-rate", type=int, default=2, help="Dataset sampling rate (1 = use every frame, 2 = use every other frame, etc.)")
    parser.add_argument("--start-frame", type=int, default=3500, help="Frame index to start processing from")
    parser.add_argument("--end-frame", type=int, default=7500, help="Frame index to end processing (inclusive, default is the last frame)")
    return parser.parse_args()

def main():
    """
    Main entry point for MAST3R SLAM system.
    
    Handles initialization, processing, and cleanup for the SLAM system.
    """
    try:
        # Initialize environment
        device, datetime_now = setup_environment()
        save_frames = False
        
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        load_config(args.config)
        print(f"{Fore.GREEN}[CONFIG] Dataset: {args.dataset}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[CONFIG] Configuration loaded: {args.config}{Style.RESET_ALL}")
        
        # Set up multiprocessing
        manager = mp.Manager()
        main2viz, viz2main = setup_multiprocessing(manager, args.no_viz)
        
        # Load and configure dataset
        sample_rate = args.sample_rate if args.sample_rate is not None else config["dataset"]["subsample"]
        dataset, start_frame, end_frame = load_and_configure_dataset(
            args.dataset, 
            sample_rate,
            args.start_frame,
            args.end_frame
        )
        
        # Get image dimensions
        h, w = dataset.get_img_shape()[0]
        
        # Load calibration if provided
        if args.calib:
            dataset = load_calibration(args.calib, dataset, config)
        
        # Set up shared memory
        keyframes, states = setup_shared_memory(manager, h, w)
        
        # Start visualization if enabled
        viz = None
        if not args.no_viz:
            viz = start_visualization(config, states, keyframes, main2viz, viz2main)
        
        # Load MAST3R model
        model = load_model(device)
        
        # Set up calibration
        K = setup_calibration(dataset, config, device, keyframes)
        
        # Prepare saving environment
        save_dir, seq_name = prepare_saving_environment(args, dataset)
        
        # Initialize tracker
        tracker = FrameTracker(model, keyframes, device)
        
        # Start backend process
        backend = start_backend_process(config, model, states, keyframes, K)
        
        # Process frames
        frames, frames_processed, last_processed_frame, last_msg = process_frames(
            config, model, dataset, start_frame, end_frame, sample_rate, 
            tracker, keyframes, states, device, viz2main, save_frames
        )
        
        # Save results
        save_results(
            save_dir, seq_name, dataset, keyframes, last_msg, 
            frames, start_frame, last_processed_frame, save_frames, datetime_now
        )
        
        # Clean up
        cleanup(backend, viz, args.no_viz, frames_processed, start_frame)
        
    except Exception as e:
        print(f"{Fore.RED}\n[ERROR] An unexpected error occurred: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    return 0


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print(f"{Fore.YELLOW}\n[RELOC] Relocalization attempt against KF {n_kf - 1} and {kf_idx}{Style.RESET_ALL}")
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print(f"{Fore.GREEN}\n[RELOC] ✓ Successful relocalization!{Style.RESET_ALL}")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print(f"{Fore.RED}\n[RELOC] ✗ Failed to relocalize{Style.RESET_ALL}")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    print(f"{Fore.BLUE}\n[BACKEND] Backend process started{Style.RESET_ALL}")
    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print(f"{Fore.CYAN}\n[BACKEND] Database retrieval for KF {idx}: {lc_inds}{Style.RESET_ALL}")

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
    
    print(f"{Fore.BLUE}\n[BACKEND] Backend process terminated{Style.RESET_ALL}")


def print_banner():
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                   MAST3R SLAM PROCESSING                     ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """
    print(banner)

def setup_environment():
    """Initialize environment settings."""
    colorama.init()
    print_banner()
    
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    return "cuda:0", str(datetime.datetime.now()).replace(" ", "_")

def setup_multiprocessing(manager, no_viz):
    """Set up multiprocessing queues."""
    main2viz = new_queue(manager, no_viz)
    viz2main = new_queue(manager, no_viz)
    return main2viz, viz2main

def load_and_configure_dataset(dataset_path, sample_rate, start_frame, end_frame):
    """Load dataset and configure sampling parameters."""
    print(f"{Fore.YELLOW}[DATASET] Loading dataset...{Style.RESET_ALL}", end="")
    sys.stdout.flush()
    dataset = load_dataset(dataset_path)
    
    print(f"\r{Fore.GREEN}[DATASET] Dataset loaded: {len(dataset)} frames{Style.RESET_ALL}")
    print(f"{Fore.GREEN}[CONFIG] Using sampling rate: {sample_rate} (1/{sample_rate} frames){Style.RESET_ALL}")
    print(f"{Fore.GREEN}[DATASET] Total frames available: {len(dataset)} frames{Style.RESET_ALL}")
    
    # Validate start frame
    start_frame = validate_start_frame(start_frame, len(dataset))
    
    # Validate end frame
    end_frame = validate_end_frame(end_frame, start_frame, len(dataset))
    
    return dataset, start_frame, end_frame

def validate_start_frame(start_frame, dataset_length):
    """Validate and adjust start frame if needed."""
    if start_frame < 0:
        print(f"{Fore.RED}[CONFIG] Invalid start frame: {start_frame}, using 0 instead{Style.RESET_ALL}")
        return 0
    elif start_frame >= dataset_length:
        print(f"{Fore.RED}[CONFIG] Start frame {start_frame} exceeds dataset length {dataset_length}, using 0 instead{Style.RESET_ALL}")
        return 0
    else:
        print(f"{Fore.GREEN}[CONFIG] Starting from frame: {start_frame}{Style.RESET_ALL}")
        return start_frame

def validate_end_frame(end_frame, start_frame, dataset_length):
    """Validate and adjust end frame if needed."""
    if end_frame is None:
        end_frame = dataset_length - 1
        print(f"{Fore.GREEN}[CONFIG] Processing until the last frame: {end_frame}{Style.RESET_ALL}")
    elif end_frame < start_frame:
        print(f"{Fore.RED}[CONFIG] End frame {end_frame} is before start frame {start_frame}, using last frame instead{Style.RESET_ALL}")
        end_frame = dataset_length - 1
    elif end_frame >= dataset_length:
        print(f"{Fore.RED}[CONFIG] End frame {end_frame} exceeds dataset length {dataset_length}, using last frame instead{Style.RESET_ALL}")
        end_frame = dataset_length - 1
    else:
        print(f"{Fore.GREEN}[CONFIG] Processing until frame: {end_frame}{Style.RESET_ALL}")
    return end_frame

def load_calibration(calib_path, dataset, config):
    """Load camera calibration if provided."""
    if not calib_path:
        return None
    
    print(f"{Fore.YELLOW}\n[CALIB] Loading calibration from {calib_path}...{Style.RESET_ALL}", end="")
    sys.stdout.flush()
    with open(calib_path, "r") as f:
        intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
    config["use_calib"] = True
    dataset.use_calibration = True
    dataset.camera_intrinsics = Intrinsics.from_calib(
        dataset.img_size,
        intrinsics["width"],
        intrinsics["height"],
        intrinsics["calibration"],
    )
    print(f"\r{Fore.GREEN}[CALIB] Calibration loaded successfully{Style.RESET_ALL}")
    return dataset

def setup_shared_memory(manager, h, w):
    """Set up shared memory for keyframes and states."""
    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)
    return keyframes, states

def start_visualization(config, states, keyframes, main2viz, viz2main):
    """Start visualization process if enabled."""
    print(f"{Fore.BLUE}\n[VIZ] Starting visualization process{Style.RESET_ALL}")
    viz = mp.Process(
        target=run_visualization,
        args=(config, states, keyframes, main2viz, viz2main),
    )
    viz.start()
    return viz

def load_model(device):
    """Load MAST3R model."""
    print(f"{Fore.YELLOW}\n[MODEL] Loading MAST3R model...{Style.RESET_ALL}", end="")
    sys.stdout.flush()
    model = load_mast3r(device=device)
    model.share_memory()
    print(f"\r{Fore.GREEN}\n[MODEL] MAST3R model loaded successfully{Style.RESET_ALL}")
    return model

def setup_calibration(dataset, config, device, keyframes):
    """Set up calibration if available."""
    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print(f"{Fore.RED}\n[ERROR] No calibration provided for this dataset!{Style.RESET_ALL}")
        sys.exit(0)
    
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)
    
    return K

def prepare_saving_environment(args, dataset):
    """Prepare saving directory and remove old files."""
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()
    return save_dir, seq_name

def start_backend_process(config, model, states, keyframes, K):
    """Start backend processing."""
    print(f"{Fore.BLUE}[BACKEND] Starting backend process{Style.RESET_ALL}")
    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()
    return backend

def handle_init_mode(model, frame, keyframes, states, frames_processed, pbar, i):
    """Handle INIT mode processing."""
    X_init, C_init = mast3r_inference_mono(model, frame)
    frame.update_pointmap(X_init, C_init)
    keyframes.append(frame)
    states.queue_global_optimization(len(keyframes) - 1)
    states.set_mode(Mode.TRACKING)
    states.set_frame(frame)
    frames_processed += 1
    pbar.update(1)
    return frames_processed, i + 1, True  # Return True to continue

def handle_tracking_mode(tracker, frame, states):
    """Handle TRACKING mode processing."""
    add_new_kf, match_info, try_reloc = tracker.track(frame)
    if try_reloc:
        states.set_mode(Mode.RELOC)
    states.set_frame(frame)
    return add_new_kf

def handle_reloc_mode(model, frame, states, config):
    """Handle RELOC mode processing."""
    X, C = mast3r_inference_mono(model, frame)
    frame.update_pointmap(X, C)
    states.set_frame(frame)
    states.queue_reloc()
    # In single threaded mode, make sure relocalization happen for every frame
    while config["single_thread"]:
        with states.lock:
            if states.reloc_sem.value == 0:
                break
        time.sleep(0.01)

def handle_keyframe_addition(frame, keyframes, states, config):
    """Handle adding a new keyframe."""
    keyframes.append(frame)
    states.queue_global_optimization(len(keyframes) - 1)
    # In single threaded mode, wait for the backend to finish
    while config["single_thread"]:
        with states.lock:
            if len(states.global_optimizer_tasks) == 0:
                break
        time.sleep(0.01)

def process_frames(config, model, dataset, start_frame, end_frame, sample_rate, tracker, keyframes, states, device, viz2main, save_frames=False):
    """Main processing loop for frames."""
    i = start_frame
    fps_timer = time.time()
    frames = []
    
    # Calculate total frames to process accounting for sampling rate
    total_frames = ((end_frame - start_frame + 1) + sample_rate - 1) // sample_rate  # Ceiling division
    print(f"{Fore.GREEN}\n[CONFIG] Processing approximately {total_frames} frames from {start_frame} to {end_frame} with sampling rate {sample_rate}{Style.RESET_ALL}")
    
    # Create progress bar for frame processing
    pbar = tqdm.tqdm(
        total=total_frames,
        desc=f"{Fore.CYAN}[PROCESSING]{Style.RESET_ALL}",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=100
    )

    print(f"{Fore.GREEN}\n[MAIN] Starting frame processing loop from frame {start_frame} to {end_frame}{Style.RESET_ALL}")
    
    # Track actual frames processed
    frames_processed = 0
    last_msg = WindowMsg()
    
    try:
        while True:
            mode = states.get_mode()
            msg = try_get_msg(viz2main)
            last_msg = msg if msg is not None else last_msg
            
            # Check termination conditions
            if last_msg.is_terminated or i > end_frame:
                states.set_mode(Mode.TERMINATED)
                break

            # Handle pause state
            if last_msg.is_paused and not last_msg.next:
                states.pause()
                time.sleep(0.01)
                continue
            
            if not last_msg.is_paused:
                states.unpause()

            # Only process frames that match the sampling pattern
            if (i - start_frame) % sample_rate == 0:
                timestamp, img = dataset[i]
                if save_frames:
                    frames.append(img)

                # Get frame's last camera pose
                T_WC = (
                    lietorch.Sim3.Identity(1, device=device)
                    if i == start_frame  # Initialize at start_frame
                    else states.get_frame().T_WC
                )
                frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

                # Handle different modes
                if mode == Mode.INIT:
                    frames_processed, i, continue_loop = handle_init_mode(
                        model, frame, keyframes, states, frames_processed, pbar, i
                    )
                    if continue_loop:
                        continue

                elif mode == Mode.TRACKING:
                    add_new_kf = handle_tracking_mode(tracker, frame, states)
                    if add_new_kf:
                        handle_keyframe_addition(frame, keyframes, states, config)

                elif mode == Mode.RELOC:
                    handle_reloc_mode(model, frame, states, config)

                else:
                    raise Exception("Invalid mode")
                
                # Update progress bar with the current FPS
                if frames_processed % 10 == 0:
                    current_fps = frames_processed / (time.time() - fps_timer)
                    pbar.set_postfix({"FPS": f"{current_fps:.2f}"})
                    
                frames_processed += 1
                pbar.update(1)
            
            # Always increment the frame counter, whether we processed this frame or not
            i += 1
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}\n[MAIN] Process interrupted by user{Style.RESET_ALL}")
        states.set_mode(Mode.TERMINATED)
    finally:
        pbar.close()
    
    return frames, frames_processed, i - 1, last_msg

def save_results(save_dir, seq_name, dataset, keyframes, last_msg, frames, start_frame, last_processed_frame, save_frames, datetime_now):
    """Save trajectory and reconstruction results."""
    print(f"{Fore.GREEN}\n[SAVING] Saving trajectory and reconstruction...{Style.RESET_ALL}")
    eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
    eval.save_reconstruction(
        save_dir,
        f"{seq_name}.ply",
        keyframes,
        last_msg.C_conf_threshold,
    )
    
    # Save trajectory with camera positions
    print(f"{Fore.GREEN}\n[SAVING] Saving trajectory with camera positions...{Style.RESET_ALL}")
    eval.save_reconstruction_with_camera_trajectory(
        save_dir,
        f"{seq_name}_with_cameras.ply",
        keyframes,
        dataset,
        last_msg.C_conf_threshold,
        start_frame=start_frame,
        end_frame=last_processed_frame,
        show_frustums=False
    )    
    
    eval.save_keyframes(
        save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
    )
    
    save_frames_to_disk(frames, save_frames, datetime_now, start_frame)

def save_frames_to_disk(frames, save_frames, datetime_now, start_frame):
    """Save processed frames to disk if enabled."""
    if save_frames and frames:
        print(f"{Fore.BLUE}\n[SAVING] Saving {len(frames)} frames...{Style.RESET_ALL}")
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(
            enumerate(frames), 
            total=len(frames),
            desc=f"{Fore.CYAN}\n[FRAMES]{Style.RESET_ALL}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        ):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i + start_frame}.png", frame)

def cleanup(backend, viz, no_viz, frames_processed, start_frame):
    """Clean up processes and print final statistics."""
    print(f"{Fore.GREEN}\n[MAIN] Processing completed successfully{Style.RESET_ALL}")
    print(f"{Fore.GREEN}[STATS] Processed {frames_processed} frames starting from frame {start_frame}{Style.RESET_ALL}")
    backend.join()
    if not no_viz:
        viz.join()
    
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                 MAST3R SLAM PROCESS COMPLETE                 ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """)
    
    # Reset colorama settings
    colorama.deinit()

if __name__ == "__main__":
    main()