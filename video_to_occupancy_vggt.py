import os
import sys
import cv2
import numpy as np
import torch
import trimesh
import tempfile

# Add the vggt folder to the Python path so its subpackages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import (
    closed_form_inverse_se3,
    unproject_depth_map_to_point_map,
)


def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = max_frames  # Fallback

    step = max(1, total_frames // max_frames)

    idx = 0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            count += 1
            if count >= max_frames:
                break
        idx += 1

    cap.release()
    return frames


def _save_frames_to_tmpdir(frames):
    """Save numpy frames to a temporary directory and return the list of file paths.
    VGGT's load_and_preprocess_images expects file paths, not raw arrays."""
    tmp_dir = tempfile.mkdtemp(prefix="vggt_frames_")
    paths = []
    for i, frame in enumerate(frames):
        from PIL import Image

        img = Image.fromarray(frame)
        p = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        img.save(p)
        paths.append(p)
    return paths, tmp_dir


def classify_points_by_height(points, return_masks=False):
    """Separate aligned 3D points into floor (free space) and wall (obstacle) sets.

    Strategy:
        1. Use RANSAC to robustly fit a plane to the ground points, handling camera tilt.
        2. Fallback to simple height percentiles if RANSAC fails or Open3D is unavailable.
        3. Floor points are within a thin band around the ground plane.
        4. Wall points start just above the floor and extend upwards.

    Returns:
        floor_points, wall_points  (both Nx3 arrays)
    """
    y_range = np.percentile(points[:, 1], 95) - np.percentile(points[:, 1], 5)

    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Downsample for faster plane fitting
        if len(points) > 50000:
            pcd_down = pcd.random_down_sample(50000 / len(points))
        else:
            pcd_down = pcd
            
        plane_model, _ = pcd_down.segment_plane(distance_threshold=max(0.02 * y_range, 0.05),
                                                ransac_n=3,
                                                num_iterations=1000)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        
        # Ensure normal points mostly "up" (Y ascending)
        if normal[1] < 0:
            normal = -normal
            d = -d
            
        # Check if the detected plane is actually horizontal-ish
        if normal[1] >= 0.5:
            distances = np.dot(points, normal) + d
            
            # Re-anchor the floor level so it captures the bottom surface tightly
            dist_5th = np.percentile(distances, 5)
            distances = distances - dist_5th
            
            floor_mask = (distances >= -0.2 * y_range) & (distances < 0.10 * y_range)
            wall_mask = (distances >= 0.10 * y_range) & (distances < 0.4 * y_range)
            
            if return_masks:
                return points[floor_mask], points[wall_mask], floor_mask, wall_mask
            return points[floor_mask], points[wall_mask]
            
    except ImportError:
        pass # open3d not available, fallback to pure numpy
        
    # Fallback: Estimate floor level using simple percentiles
    y = points[:, 1]
    floor_y = np.percentile(y, 5)
    
    floor_band_top = floor_y + 0.10 * y_range
    wall_band_top = floor_y + 0.4 * y_range
    
    floor_mask = (y >= floor_y) & (y < floor_band_top)
    wall_mask = (y >= floor_band_top) & (y < wall_band_top)
    
    if return_masks:
        return points[floor_mask], points[wall_mask], floor_mask, wall_mask
    return points[floor_mask], points[wall_mask]


def points_to_occupancy_grid(floor_points, wall_points, cam_points=None, grid_res=0.05):
    """Build a 2D occupancy grid from classified 3D points.

    Color convention (standard robotics):
        - White (255,255,255) = free / traversable  (floor-level points detected)
        - Dark red (0,0,140)  = obstacle / wall      (wall-level points detected)
        - Black (0,0,0)       = unknown / unobserved
    """
    # Combine all points to determine grid bounds
    all_pts = []
    if len(floor_points) > 0:
        all_pts.append(floor_points)
    if len(wall_points) > 0:
        all_pts.append(wall_points)

    if len(all_pts) == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 0, 0)

    all_pts = np.concatenate(all_pts, 0)
    x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
    z_min, z_max = np.min(all_pts[:, 2]), np.max(all_pts[:, 2])

    if cam_points is not None and len(cam_points) > 0:
        cx, cz = cam_points[:, 0], cam_points[:, 2]
        x_min, x_max = min(x_min, np.min(cx)), max(x_max, np.max(cx))
        z_min, z_max = min(z_min, np.min(cz)), max(z_max, np.max(cz))

    # Add a small margin
    margin = grid_res * 10
    x_min, x_max = x_min - margin, x_max + margin
    z_min, z_max = z_min - margin, z_max + margin

    width = int(np.ceil((x_max - x_min) / grid_res))
    height = int(np.ceil((z_max - z_min) / grid_res))

    if width <= 0 or height <= 0:
        return np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 0, 0)

    # 3-channel grid: black = unknown
    grid = np.zeros((height, width, 3), dtype=np.uint8)

    def _project_to_grid(pts):
        x, z = pts[:, 0], pts[:, 2]
        c = np.clip(np.floor((x - x_min) / grid_res).astype(int), 0, width - 1)
        r = np.clip(np.floor((z - z_min) / grid_res).astype(int), 0, height - 1)
        return r, c

    # 1) Paint floor cells white (free space)
    if len(floor_points) > 0:
        fr, fc = _project_to_grid(floor_points)
        grid[fr, fc] = [255, 255, 255]

    # 2) Paint wall cells dark red (obstacle) — overwrites floor where walls stand
    if len(wall_points) > 0:
        wr, wc = _project_to_grid(wall_points)
        grid[wr, wc] = [0, 0, 140]  # BGR: dark red

    return grid, (x_min, z_min, x_max, z_max)


def main(video_path, output_path, conf_percentile=40.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    print(f"Loading VGGT model on {device} (dtype={dtype})...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval()
    model = model.to(device)

    # --- Extract video frames and save to temp dir (VGGT expects file paths) ---
    print(f"Extracting frames from {video_path}")
    frames = extract_frames(video_path, max_frames=20)
    if len(frames) == 0:
        print("No frames extracted.")
        return

    print(f"Saving {len(frames)} frames to temp dir for VGGT preprocessing...")
    frame_paths, tmp_dir = _save_frames_to_tmpdir(frames)

    # --- Preprocess and run inference ---
    images = load_and_preprocess_images(frame_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Decode pose encoding → extrinsic (S,3,4) and intrinsic (S,3,3)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Move everything to numpy, remove batch dim
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    S = predictions["extrinsic"].shape[0]

    # --- Build 3D world points from depth maps ---
    print("Unprojecting depth maps to 3D world points...")
    depth_map = predictions["depth"]       # (S, H, W, 1)
    depth_conf = predictions["depth_conf"] # (S, H, W)
    extrinsics = predictions["extrinsic"]  # (S, 3, 4)
    intrinsics = predictions["intrinsic"]  # (S, 3, 3)

    world_points = unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics)  # (S, H, W, 3)

    # Confidence-based filtering threshold
    conf_thr = np.percentile(depth_conf, conf_percentile)

    # Flatten to Nx3 point cloud, applying confidence mask
    pts_all = []
    col_all = []

    # Colors: predictions["images"] is (S, 3, H, W) float [0,1]
    images_np = predictions["images"]  # (S, 3, H, W)
    images_u8 = (images_np.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # (S, H, W, 3)

    for i in range(S):
        d = depth_map[i].squeeze(-1)  # (H, W)
        conf = depth_conf[i]          # (H, W)

        valid = np.isfinite(d) & (d > 0) & (conf >= conf_thr)
        if not np.any(valid):
            continue

        wp = world_points[i]  # (H, W, 3)
        pts_all.append(wp[valid])
        col_all.append(images_u8[i][valid])

    if len(pts_all) == 0:
        print("No valid 3D points after filtering. Check thresholds or input video.")
        return

    points = np.concatenate(pts_all, 0)
    colors = np.concatenate(col_all, 0)

    print(f"Extracted {len(points)} points. Aligning to first camera...")

    # --- Align point cloud using the first camera (same logic as DA3 version) ---
    # Build 4x4 from the 3x4 w2c
    w2c0_34 = extrinsics[0]
    w2c0 = np.eye(4, dtype=np.float64)
    w2c0[:3, :4] = w2c0_34

    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0  # flip Y
    M[2, 2] = -1.0  # flip Z

    A_no_center = M @ w2c0

    center = np.median(points, axis=0) if len(points) > 0 else np.zeros(3)
    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center
    A = T_center @ A_no_center

    pts_homo = np.hstack([points, np.ones((len(points), 1))])
    pts_aligned = (A @ pts_homo.T).T[:, :3]

    # --- Camera positions ---
    print("Extracting camera positions...")
    # c2w = inverse of w2c (extrinsics)
    # closed_form_inverse_se3 expects (S,3,4) or (S,4,4), returns (S,4,4)
    c2w_all = closed_form_inverse_se3(extrinsics)  # (S, 4, 4)

    cam_centers_w = c2w_all[:, :3, 3]  # (S, 3)
    cam_homo = np.hstack([cam_centers_w, np.ones((S, 1))])
    cam_aligned = (A @ cam_homo.T).T[:, :3]

    # --- Classify points into floor and wall ---
    floor_pts, wall_pts = [], []
    if len(pts_aligned) > 0:
        print("Classifying 3D point cloud into floor and wall...")
        floor_pts, wall_pts, floor_mask, wall_mask = classify_points_by_height(pts_aligned, return_masks=True)
        print(f"  Floor points: {len(floor_pts)}, Wall points: {len(wall_pts)}")
        
        # Highlight obstacles (walls) with a differentiating color (e.g., Red)
        colors[wall_mask] = [255, 0, 0]

    # --- Save PLY ---
    ply_out = output_path.rsplit(".", 1)[0] + "_pointcloud.ply"
    print(f"Saving 3D point cloud to {ply_out}...")

    cam_pts = []
    cam_cols = []
    scene_scale = max(0.01, (np.max(pts_aligned[:, 0]) - np.min(pts_aligned[:, 0])) / 100.0)

    for i, c in enumerate(cam_aligned):
        sphere = trimesh.creation.icosphere(radius=scene_scale * 5.0, subdivisions=1)
        sphere.apply_translation(c)
        cam_pts.append(sphere.vertices)

        # Color gradient: First camera is Green, last goes towards Red
        ratio = i / max(1, len(cam_aligned) - 1)
        col = [int(255 * ratio), int(255 * (1 - ratio)), 0]
        cam_cols.append(np.tile(col, (len(sphere.vertices), 1)))

    cam_pts = np.vstack(cam_pts)
    cam_cols = np.vstack(cam_cols)

    all_pts = np.vstack([pts_aligned, cam_pts])
    all_cols = np.vstack([colors, cam_cols])

    pc = trimesh.points.PointCloud(vertices=all_pts, colors=all_cols)
    pc.export(ply_out)

    # --- Occupancy grid ---
    print("Converting 3D point cloud to 2D occupancy grid...")
    if len(pts_aligned) > 0:

        # Determine grid resolution
        span_x = np.max(pts_aligned[:, 0]) - np.min(pts_aligned[:, 0])
        grid_res = max(0.01, span_x / 200.0)  # Aiming for ~200 bins along width

        occupancy_grid, bounds = points_to_occupancy_grid(
            floor_pts,
            wall_pts,
            cam_aligned,
            grid_res=grid_res,
        )

        x_min, z_min, x_max, z_max = bounds
        print(f"Occupancy grid shape: {occupancy_grid.shape}")

        # Draw camera positions on the occupancy grid
        for i, c in enumerate(cam_aligned):
            cx, cz = c[0], c[2]
            c_col = int((cx - x_min) / grid_res)
            c_row = int((cz - z_min) / grid_res)

            if 0 <= c_col < occupancy_grid.shape[1] and 0 <= c_row < occupancy_grid.shape[0]:
                ratio = i / max(1, len(cam_aligned) - 1)
                # BGR format for OpenCV
                col = (0, int(255 * (1 - ratio)), int(255 * ratio))
                cv2.circle(
                    occupancy_grid,
                    (c_col, c_row),
                    radius=1,
                    color=col,
                    thickness=-1,
                )

                # Draw a line connecting consecutive cameras
                if i > 0:
                    prev_c = cam_aligned[i - 1]
                    p_col = int((prev_c[0] - x_min) / grid_res)
                    p_row = int((prev_c[2] - z_min) / grid_res)
                    if 0 <= p_col < occupancy_grid.shape[1] and 0 <= p_row < occupancy_grid.shape[0]:
                        cv2.line(
                            occupancy_grid,
                            (p_col, p_row),
                            (c_col, c_row),
                            (0, 0, 255),
                            max(1, int(1.0 / grid_res * 0.05)),
                        )

        # Flip vertically so Z-up maps to image top
        occupancy_grid = np.flipud(occupancy_grid)

        cv2.imwrite(output_path, occupancy_grid)
        print(f"Saved occupancy grid to {output_path}")
    else:
        print("Point cloud was empty. Check thresholds or input video.")

    # --- Cleanup temp frames ---
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--out", type=str, default="occupancy_grid_vggt.png", help="Output 2D grid image path")
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=40.0,
        help="Percentile of depth confidence below which points are filtered",
    )
    args = parser.parse_args()

    main(args.video, args.out, args.conf_percentile)
