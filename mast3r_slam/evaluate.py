import pathlib
from typing import Optional, List
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)

    print(f"Saved reconstruction to {savedir / filename}")

def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


# def save_ply(filename, points, colors):
#     colors = colors.astype(np.uint8)
#     # Combine XYZ and RGB into a structured array
#     pcd = np.empty(
#         len(points),
#         dtype=[
#             ("x", "f4"),
#             ("y", "f4"),
#             ("z", "f4"),
#             ("red", "u1"),
#             ("green", "u1"),
#             ("blue", "u1"),
#         ],
#     )
#     pcd["x"], pcd["y"], pcd["z"] = points.T
#     pcd["red"], pcd["green"], pcd["blue"] = colors.T
#     vertex_element = PlyElement.describe(pcd, "vertex")
#     ply_data = PlyData([vertex_element], text=False)
#     ply_data.write(filename)


def save_ply(filename, points, colors, target_size_mb=200):
    """
    Save point cloud as PLY file, downsampling if necessary to meet target file size.
    
    Args:
        filename: Output file path
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        target_size_mb: Target file size in MB
    """
    import io
    import numpy as np
    from plyfile import PlyData, PlyElement
    
    colors = colors.astype(np.uint8)
    max_size_mb = target_size_mb + 5
    
    # Function to create and measure a PLY file with given sampling rate
    def try_sampling_rate(rate):
        # Sample points
        n_points = len(points)
        n_sample = max(int(n_points * rate), 1)
        indices = np.random.choice(n_points, size=n_sample, replace=False)
        
        # Create PLY data
        pcd = np.empty(n_sample, dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")
        ])
        
        pcd["x"], pcd["y"], pcd["z"] = points[indices].T
        pcd["red"], pcd["green"], pcd["blue"] = colors[indices].T
        
        ply_data = PlyData([PlyElement.describe(pcd, "vertex")], text=False)
        
        # Check size
        buffer = io.BytesIO()
        ply_data.write(buffer)
        size_mb = buffer.tell() / (1024 * 1024)
        
        return ply_data, size_mb
    
    # Try with full dataset first
    ply_data, size_mb = try_sampling_rate(1.0)
    
    # If already small enough, save and return
    if size_mb <= max_size_mb:
        ply_data.write(filename)
        return
    
    # Simple iterative approach to find suitable sampling rate
    rates = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
    
    for rate in rates:
        ply_data, size_mb = try_sampling_rate(rate)
        if size_mb <= max_size_mb:
            break
    
    # Save final result
    ply_data.write(filename)


def extract_keyframe_data(keyframes):
    """
    Extract and sort pose data from keyframes
    
    Args:
        keyframes: SharedKeyframes object with stored keyframes
    
    Returns:
        tuple: (sorted_frame_ids, sorted_poses)
    """
    kf_poses = []  # [x, y, z, qx, qy, qz, qw]
    kf_ids = []
    
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        kf_id = keyframe.frame_id
        kf_ids.append(kf_id)
        
        # Extract pose (position and orientation)
        T_WC = as_SE3(keyframe.T_WC)
        pose = T_WC.data.numpy().reshape(-1)  # [x, y, z, qx, qy, qz, qw]
        kf_poses.append(pose)
    
    # Sort keyframes by frame_id
    kf_data = sorted(zip(kf_ids, kf_poses))
    kf_ids = [data[0] for data in kf_data]
    kf_poses = [data[1] for data in kf_data]
    
    return kf_ids, kf_poses


def find_interpolation_params(frame_id, kf_ids):
    """
    Find the appropriate keyframes for interpolation and the interpolation parameter
    
    Args:
        frame_id: Current frame ID
        kf_ids: Sorted list of keyframe IDs
    
    Returns:
        tuple: (before_idx, after_idx, t_rel) or (exact_idx, None, None) for exact matches
              Returns (-1, 0, 0) if before first keyframe
              Returns (last_idx, None, None) if after last keyframe
    """
    # If this is a keyframe, use its exact pose
    if frame_id in kf_ids:
        idx = kf_ids.index(frame_id)
        return (idx, None, None)
    
    # Before first keyframe
    if frame_id < kf_ids[0]:
        return (-1, 0, 0)
        
    # After last keyframe
    if frame_id > kf_ids[-1]:
        return (len(kf_ids) - 1, None, None)
    
    # Find keyframes before and after current frame
    for i in range(len(kf_ids) - 1):
        if kf_ids[i] <= frame_id <= kf_ids[i+1]:
            # Calculate interpolation parameter
            range_frames = kf_ids[i+1] - kf_ids[i]
            if range_frames == 0:  # Avoid division by zero
                t_rel = 0
            else:
                t_rel = (frame_id - kf_ids[i]) / range_frames
            
            return (i, i+1, t_rel)
    
    # This should never happen if input is valid
    raise ValueError(f"Could not find interpolation parameters for frame {frame_id}")


def estimate_keyframe_velocities(kf_ids, kf_poses):
    """
    Estimate velocities at each keyframe based on neighboring keyframes
    
    Args:
        kf_ids: Sorted list of keyframe IDs
        kf_poses: Sorted list of keyframe poses [x, y, z, qx, qy, qz, qw]
    
    Returns:
        numpy.array: Array of velocity vectors at each keyframe
    """
    n_keyframes = len(kf_ids)
    velocities = []
    
    for i in range(n_keyframes):
        if i == 0:
            # First keyframe: use forward difference
            if n_keyframes > 1:
                frame_diff = kf_ids[1] - kf_ids[0]
                if frame_diff == 0:  # Avoid division by zero
                    vel = np.zeros(3)
                else:
                    vel = (kf_poses[1][:3] - kf_poses[0][:3]) / frame_diff
            else:
                vel = np.zeros(3)
        elif i == n_keyframes - 1:
            # Last keyframe: use backward difference
            frame_diff = kf_ids[i] - kf_ids[i-1]
            if frame_diff == 0:  # Avoid division by zero
                vel = np.zeros(3)
            else:
                vel = (kf_poses[i][:3] - kf_poses[i-1][:3]) / frame_diff
        else:
            # Middle keyframes: use central difference
            frame_diff_next = kf_ids[i+1] - kf_ids[i]
            frame_diff_prev = kf_ids[i] - kf_ids[i-1]
            
            if frame_diff_next == 0 or frame_diff_prev == 0:  # Avoid division by zero
                vel = np.zeros(3)
            else:
                # Calculate weighted central difference
                vel_next = (kf_poses[i+1][:3] - kf_poses[i][:3]) / frame_diff_next
                vel_prev = (kf_poses[i][:3] - kf_poses[i-1][:3]) / frame_diff_prev
                
                # Weight the velocities by the inverse of the frame distance
                weight_next = 1 / frame_diff_next
                weight_prev = 1 / frame_diff_prev
                total_weight = weight_next + weight_prev
                
                vel = (weight_next * vel_next + weight_prev * vel_prev) / total_weight
        
        velocities.append(vel)
    
    return np.array(velocities)


def interpolate_position_with_momentum(pos_before, pos_after, vel_before, vel_after, t):
    """
    Hermite spline interpolation between two 3D positions with momentum
    
    Args:
        pos_before: Starting position [x, y, z]
        pos_after: Ending position [x, y, z]
        vel_before: Velocity at starting position [vx, vy, vz]
        vel_after: Velocity at ending position [vx, vy, vz]
        t: Interpolation parameter [0, 1]
    
    Returns:
        numpy.array: Interpolated position
    """
    # Hermite basis functions
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    # Scale velocities by the frame difference (assuming unit frame difference)
    # For actual frame differences, multiply these by the frame count between keyframes
    
    # Compute interpolated position using Hermite spline
    interp_pos = (h00 * pos_before + 
                  h10 * vel_before + 
                  h01 * pos_after + 
                  h11 * vel_after)
    
    return interp_pos


def interpolate_quaternion(quat_before, quat_after, t):
    """
    Spherical Linear Interpolation (SLERP) between two quaternions
    
    Args:
        quat_before: Starting quaternion [qx, qy, qz, qw]
        quat_after: Ending quaternion [qx, qy, qz, qw]
        t: Interpolation parameter [0, 1]
    
    Returns:
        numpy.array: Interpolated and normalized quaternion
    """
    # Ensure quaternion dot product is positive (shortest path)
    if np.dot(quat_before, quat_after) < 0:
        quat_after = -quat_after
    
    # Simple SLERP implementation
    dot = np.clip(np.dot(quat_before, quat_after), -1, 1)
    theta = np.arccos(dot)
    
    if abs(theta) < 1e-6:
        # If quaternions are very close, use linear interpolation
        interp_quat = quat_before + t * (quat_after - quat_before)
    else:
        # SLERP formula
        sin_theta = np.sin(theta)
        interp_quat = (np.sin((1-t)*theta) / sin_theta) * quat_before + (np.sin(t*theta) / sin_theta) * quat_after
    
    # Normalize quaternion
    return interp_quat / np.linalg.norm(interp_quat)


def interpolate_camera_poses(keyframes, timestamps, all_frame_ids):
    """
    Interpolate camera poses for all frames based on keyframe poses
    
    Args:
        keyframes: SharedKeyframes object with stored keyframes
        timestamps: Timestamps for all frames
        all_frame_ids: List of all frame IDs to interpolate
    
    Returns:
        positions: Array of camera positions for all frames
        quaternions: Array of camera orientations as quaternions
    """
    # Extract and sort keyframe data
    kf_ids, kf_poses = extract_keyframe_data(keyframes)
    
    # Calculate velocities for momentum-based interpolation
    kf_velocities = estimate_keyframe_velocities(kf_ids, kf_poses)
    
    # Initialize arrays for all frame positions and orientations
    all_positions = []
    all_quaternions = []
    
    # For each frame, interpolate the camera pose
    for frame_id in all_frame_ids:
        # Find appropriate interpolation keyframes and parameter
        interp_params = find_interpolation_params(frame_id, kf_ids)
        before_idx, after_idx, t_rel = interp_params
        
        if after_idx is None:
            # Exact keyframe match or before/after bounds
            pose = kf_poses[before_idx]
            all_positions.append(pose[:3])
            all_quaternions.append(pose[3:7])
        else:
            # Interpolate between keyframes
            pos_before = kf_poses[before_idx][:3]
            pos_after = kf_poses[after_idx][:3]
            vel_before = kf_velocities[before_idx]
            vel_after = kf_velocities[after_idx]
            
            # Scale velocities by the frame difference for proper momentum
            frame_diff = kf_ids[after_idx] - kf_ids[before_idx]
            
            # Use momentum-based interpolation for position
            interp_pos = interpolate_position_with_momentum(
                pos_before, 
                pos_after, 
                vel_before * frame_diff,  # Scale velocity by frame difference
                vel_after * frame_diff,   # Scale velocity by frame difference
                t_rel
            )
            
            # Use SLERP for quaternion interpolation (unchanged)
            quat_before = kf_poses[before_idx][3:7]
            quat_after = kf_poses[after_idx][3:7]
            interp_quat = interpolate_quaternion(quat_before, quat_after, t_rel)
            
            all_positions.append(interp_pos)
            all_quaternions.append(interp_quat)
    
    return np.array(all_positions), np.array(all_quaternions)


def add_camera_frustum_points(position, quaternion, scale=0.05, line_density=10):
    """
    Create a detailed camera frustum representation with densely sampled line points
    
    Args:
        position: Camera position [3]
        quaternion: Camera orientation as quaternion [x,y,z,w]
        scale: Size of the frustum
        line_density: Number of points to sample along each frustum edge
    
    Returns:
        points: Points for camera frustum lines
    """
    # Convert quaternion to rotation matrix
    qx, qy, qz, qw = quaternion
    
    # Compute rotation matrix from quaternion
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    
    # Define frustum vertices in camera coordinates
    frustum_vertices = np.array([
        [0, 0, 0],                 # Camera center
        [scale, scale, scale*2],   # Top-right
        [scale, -scale, scale*2],  # Bottom-right
        [-scale, -scale, scale*2], # Bottom-left
        [-scale, scale, scale*2],  # Top-left
    ])
    
    # Transform to world coordinates
    world_vertices = []
    for vertex in frustum_vertices:
        # Rotate and translate
        world_vertex = position + R @ vertex
        world_vertices.append(world_vertex)
    
    frustum_points = []
    
    # Define the edges of the frustum pyramid
    edges = [
        (0, 1),  # Center to top-right
        (0, 2),  # Center to bottom-right
        (0, 3),  # Center to bottom-left
        (0, 4),  # Center to top-left
        (1, 2),  # Top-right to bottom-right
        (2, 3),  # Bottom-right to bottom-left
        (3, 4),  # Bottom-left to top-left
        (4, 1),  # Top-left to top-right
    ]
    
    # Add densely sampled points along each edge
    for start_idx, end_idx in edges:
        start = world_vertices[start_idx]
        end = world_vertices[end_idx]
        
        # Sample points along the line
        for t in np.linspace(0, 1, line_density):
            point = start + t * (end - start)
            frustum_points.append(point)
    
    return np.array(frustum_points)


def save_reconstruction_with_camera_trajectory(
    savedir, 
    filename, 
    keyframes, 
    dataset, 
    c_conf_threshold, 
    max_frames=None,
    start_frame=0,
    end_frame=None,
    show_frustums=True
):
    """
    Save reconstruction with camera trajectory visualization
    
    Args:
        savedir: Directory to save the PLY file
        filename: Output filename
        keyframes: SharedKeyframes object
        dataset: Dataset object with timestamps
        c_conf_threshold: Confidence threshold for points
        max_frames: Maximum number of frames to process (legacy parameter)
        start_frame: First frame to include in the trajectory
        end_frame: Last frame to include in the trajectory (inclusive)
        show_frustums: Whether to show camera view frustums
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    
    # Extract point cloud from keyframes (same as original)
    pointclouds = []
    colors = []
    
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    
    # Determine frame range to process
    if end_frame is None:
        # If end_frame is not provided, use max_frames for backwards compatibility
        if max_frames is not None:
            end_frame = min(start_frame + max_frames - 1, len(dataset) - 1)
        else:
            end_frame = len(dataset) - 1
    
    # Validate frame range
    start_frame = max(0, min(start_frame, len(dataset) - 1))
    end_frame = max(start_frame, min(end_frame, len(dataset) - 1))
    
    print(f"Creating camera trajectory for frames {start_frame} to {end_frame}")
    
    # Create frame IDs for selected frame range
    all_frame_ids = list(range(start_frame, end_frame + 1))
    
    # Interpolate camera poses for all frames
    camera_positions, camera_quaternions = interpolate_camera_poses(
        keyframes, dataset.timestamps, all_frame_ids
    )
    
    # Add camera positions as simple individual points (not spheres)
    camera_trajectory_points = []
    camera_trajectory_colors = []
    
    # Create colored camera trajectory points
    for i, pos in enumerate(camera_positions):
        # Calculate color based on position in sequence (blue to red gradient)
        t = i / (len(camera_positions) - 1) if len(camera_positions) > 1 else 0
        color = np.array([int(255 * t), 0, int(255 * (1-t))], dtype=np.uint8)
        
        # Add just the central point (no sphere)
        camera_trajectory_points.append(pos)
        camera_trajectory_colors.append(color)
    
    # Add camera trajectory points to point cloud
    if camera_trajectory_points:
        camera_trajectory_points = np.array(camera_trajectory_points)
        camera_trajectory_colors = np.array(camera_trajectory_colors)
        pointclouds.append(camera_trajectory_points)
        colors.append(camera_trajectory_colors)
    
    # Add camera frustums if requested
    if show_frustums:
        frustum_points = []
        frustum_colors = []
        
        # Add camera frustums (every 30th frame to avoid clutter)
        frustum_interval = 30
        for i in range(0, len(camera_positions), frustum_interval):
            # Get camera frustum points with higher density of points (20 points per line)
            cam_frustum = add_camera_frustum_points(
                camera_positions[i], 
                camera_quaternions[i],
                scale=0.05,
                line_density=20  # Increase point density along frustum edges
            )
            
            # Color based on position in sequence (blue to red)
            t = i / (len(camera_positions) - 1) if len(camera_positions) > 1 else 0
            color = np.array([int(255 * t), 0, int(255 * (1-t))], dtype=np.uint8)
            
            # Add to frustum points and colors
            frustum_points.append(cam_frustum)
            frustum_colors.append(np.tile(color, (len(cam_frustum), 1)))
        
        # Add frustums to point cloud
        if frustum_points:
            frustum_points = np.vstack(frustum_points)
            frustum_colors = np.vstack(frustum_colors)
            pointclouds.append(frustum_points)
            colors.append(frustum_colors)
    
    # Combine all points and colors
    all_points = np.vstack(pointclouds)
    all_colors = np.vstack(colors)
    
    # Save as PLY
    save_ply(savedir / filename, all_points, all_colors)