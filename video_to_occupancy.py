import os
import cv2
import numpy as np
import torch
import trimesh

from depth_anything_3.api import DepthAnything3

def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        total_frames = max_frames # Fallback
        
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

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")

def _depths_to_world_points(depth, K, ext_w2c, images_u8, conf=None, conf_thr=1.05):
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)

    pts_all = []
    col_all = []

    for i in range(N):
        d = depth[i]
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))

        rays = K_inv @ pix[vidx].T
        Xc = rays * d_flat[vidx][None, :]
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)
        
        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)

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
            
            floor_mask = (distances >= -0.2 * y_range) & (distances < 0.05 * y_range)
            wall_mask = (distances >= 0.05 * y_range) & (distances < 0.4 * y_range)
            
            if return_masks:
                return points[floor_mask], points[wall_mask], floor_mask, wall_mask
            return points[floor_mask], points[wall_mask]
            
    except ImportError:
        pass # open3d not available, fallback to pure numpy
        
    # Fallback: Estimate floor level using simple percentiles
    y = points[:, 1]
    floor_y = np.percentile(y, 5)
    
    floor_band_top = floor_y + 0.05 * y_range
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

    # 2) Paint wall cells dark red (obstacle) - overwrites floor where walls stand
    if len(wall_points) > 0:
        wr, wc = _project_to_grid(wall_points)
        grid[wr, wc] = [0, 0, 140]  # BGR: dark red

    return grid, (x_min, z_min, x_max, z_max)

def main(video_path, output_path, model_name="da3-large-1.1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DA3 model: {model_name} on {device}")
    
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name.upper()}")
    model = model.to(device)
    
    print(f"Extracting frames from {video_path}")
    frames = extract_frames(video_path, max_frames=200)
    if len(frames) == 0:
        print("No frames extracted.")
        return
        
    print(f"Processing {len(frames)} frames through DA3...")
    prediction = model.inference(frames)
    
    print("Converting depth maps to 3D world points...")
    conf = prediction.conf
    conf_thr = np.percentile(conf, 40.0) if conf is not None else 1.05
    
    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    points, colors = _depths_to_world_points(
        prediction.depth, 
        prediction.intrinsics, 
        prediction.extrinsics,
        images_u8,
        conf,
        conf_thr
    )
    
    print(f"Extracted {len(points)} points. Aligning to first camera...")
    # Align the point cloud using the first camera
    w2c0 = _as_homogeneous44(prediction.extrinsics[0]).astype(np.float64)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0 # flip Y
    M[2, 2] = -1.0 # flip Z
    
    A_no_center = M @ w2c0
    
    if len(points) > 0:
        center = np.median(points, axis=0)
    else:
        center = np.zeros(3)
        
    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center
    A = T_center @ A_no_center
    
    ones = np.ones((len(points), 1))
    pts_homo = np.hstack([points, ones])
    pts_aligned = (A @ pts_homo.T).T[:, :3]
    
    print("Extracting camera positions...")
    cam_centers_w = []
    for ext in prediction.extrinsics:
        c2w = np.linalg.inv(_as_homogeneous44(ext))
        # Camera origin is [0,0,0,1] in camera coordinates
        Cw = (c2w @ np.array([0, 0, 0, 1.0]))[:3]
        cam_centers_w.append(Cw)
    
    cam_centers_w = np.array(cam_centers_w)
    cam_homo = np.hstack([cam_centers_w, np.ones((len(cam_centers_w), 1))])
    cam_aligned = (A @ cam_homo.T).T[:, :3]
    
    ply_out = output_path.rsplit('.', 1)[0] + "_pointcloud.ply"
    print(f"Saving 3D point cloud to {ply_out}...")
    
    # Optional: Highlight cameras in the point cloud by adding visible markers for them
    # We will add a small cluster of red points at each camera center
    cam_pts = []
    cam_cols = []
    scene_scale = max(0.01, (np.max(pts_aligned[:,0]) - np.min(pts_aligned[:,0])) / 100.0)
    
    for i, c in enumerate(cam_aligned):
        # Create a small sphere of points for visibility
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
    
    print("Converting 3D point cloud to 2D occupancy grid...")
    if len(pts_aligned) > 0:
        # Classify points into floor (free) and wall (obstacle) by height
        floor_pts, wall_pts, floor_mask, wall_mask = classify_points_by_height(pts_aligned, return_masks=True)
        print(f"  Floor points: {len(floor_pts)}, Wall points: {len(wall_pts)}")
        
        # Highlight obstacles (walls) with a differentiating color (e.g., Red)
        colors[wall_mask] = [255, 0, 0]
        
        # Determine grid resolution
        span_x = np.max(pts_aligned[:,0]) - np.min(pts_aligned[:,0])
        grid_res = max(0.01, span_x / 200.0) # Aiming for ~200 bins along width
        
        occupancy_grid, bounds = points_to_occupancy_grid(
            floor_pts, 
            wall_pts,
            cam_aligned,
            grid_res=grid_res
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
                cv2.circle(occupancy_grid, (c_col, c_row), radius=1, color=col, thickness=-1)
                
                # Draw a line connecting consecutive cameras
                if i > 0:
                    prev_c = cam_aligned[i-1]
                    p_col = int((prev_c[0] - x_min) / grid_res)
                    p_row = int((prev_c[2] - z_min) / grid_res)
                    if 0 <= p_col < occupancy_grid.shape[1] and 0 <= p_row < occupancy_grid.shape[0]:
                        cv2.line(occupancy_grid, (p_col, p_row), (c_col, c_row), (0, 0, 255), max(1, int(1.0 / grid_res * 0.05)))
        
        # The DA3 Z axis goes backward, to visualize as image Z going up requires flipping
        occupancy_grid = np.flipud(occupancy_grid) 
        
        cv2.imwrite(output_path, occupancy_grid)
        print(f"Saved occupancy grid to {output_path}")
    else:
        print("Point cloud was empty. Check thresholds or input video.")
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--out", type=str, default="occupancy_grid.png", help="Output 2D grid image path")
    parser.add_argument("--model", type=str, default="DA3-large-1.1", help="DA3 model name")
    args = parser.parse_args()
    
    main(args.video, args.out, args.model)
