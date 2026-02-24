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

def _depths_to_world_points_per_frame(depth, K, ext_w2c, images_u8, conf=None, conf_thr=1.05):
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
            pts_all.append(np.zeros((0, 3), dtype=np.float32))
            col_all.append(np.zeros((0, 3), dtype=np.uint8))
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

    return pts_all, col_all

def points_to_occupancy_grid_fixed_bounds(points, cam_points_up_to_i, bounds, grid_res=0.05, height_thresh=(-1.0, 1.0)):
    x_min, z_min, x_max, z_max, width, height = bounds
    
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(points) > 0:
        valid_mask = (points[:, 1] >= height_thresh[0]) & (points[:, 1] <= height_thresh[1])
        valid_points = points[valid_mask]
        
        if len(valid_points) > 0:
            x = valid_points[:, 0]
            z = valid_points[:, 2] 
            
            col_idx = np.floor((x - x_min) / grid_res).astype(int)
            row_idx = np.floor((z - z_min) / grid_res).astype(int)
            
            col_idx = np.clip(col_idx, 0, width - 1)
            row_idx = np.clip(row_idx, 0, height - 1)
            
            grid[row_idx, col_idx] = [255, 255, 255]
            
    # Draw cameras
    if cam_points_up_to_i is not None and len(cam_points_up_to_i) > 0:
        for i, c in enumerate(cam_points_up_to_i):
            cx, cz = c[0], c[2]
            c_col = int((cx - x_min) / grid_res)
            c_row = int((cz - z_min) / grid_res)
            
            if 0 <= c_col < width and 0 <= c_row < height:
                # We want the color gradient from Green (first cam) to Red (current/last cam)
                ratio = i / max(1, len(cam_points_up_to_i) - 1)
                col = (0, int(255 * (1 - ratio)), int(255 * ratio))
                cv2.circle(grid, (c_col, c_row), radius=max(2, int(3.0 / grid_res * 0.05)), color=col, thickness=-1)
                
                if i > 0:
                    prev_c = cam_points_up_to_i[i-1]
                    p_col = int((prev_c[0] - x_min) / grid_res)
                    p_row = int((prev_c[2] - z_min) / grid_res)
                    if 0 <= p_col < width and 0 <= p_row < height:
                        cv2.line(grid, (p_col, p_row), (c_col, c_row), (0, 0, 255), max(1, int(1.0 / grid_res * 0.05)))
                        
    # Flip to make Z go up (since Z is depth backwards to camera)
    grid = np.flipud(grid)
    return grid

def main(video_path, output_path, model_name="DA3-GIANT-1.1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DA3 model: {model_name} on {device}")
    
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name.upper()}")
    model = model.to(device)
    
    print(f"Extracting frames from {video_path}")
    frames = extract_frames(video_path, max_frames=40)
    if len(frames) == 0:
        print("No frames extracted.")
        return
        
    print(f"Processing {len(frames)} frames through DA3...")
    prediction = model.inference(frames)
    
    print("Converting depth maps to 3D world points...")
    conf = prediction.conf
    conf_thr = np.percentile(conf, 40.0) if conf is not None else 1.05
    images_u8 = prediction.processed_images
    
    pts_all_frames, col_all_frames = _depths_to_world_points_per_frame(
        prediction.depth, prediction.intrinsics, prediction.extrinsics,
        images_u8, conf, conf_thr
    )
    
    # Align the point cloud using the first camera
    w2c0 = _as_homogeneous44(prediction.extrinsics[0]).astype(np.float64)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0 # flip Y
    M[2, 2] = -1.0 # flip Z
    
    A_no_center = M @ w2c0
    
    # Calculate median center for all points together
    all_raw_pts = np.vstack([p for p in pts_all_frames if len(p) > 0])
    if len(all_raw_pts) > 0:
        center = np.median(all_raw_pts, axis=0)
    else:
        center = np.zeros(3)
        
    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center
    A = T_center @ A_no_center
    
    print("Aligning frames to global coordinate system...")
    pts_aligned_frames = []
    for pts in pts_all_frames:
        if len(pts) > 0:
            pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
            pts_aligned = (A @ pts_homo.T).T[:, :3]
            pts_aligned_frames.append(pts_aligned)
        else:
            pts_aligned_frames.append(np.zeros((0,3)))
            
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
    
    all_pts_concat = np.vstack([p for p in pts_aligned_frames if len(p) > 0])
    if len(all_pts_concat) == 0:
        print("Point cloud was empty. Check thresholds or input video.")
        return
        
    y = all_pts_concat[:, 1]
    y_median = np.median(y)
    y_std = np.std(y)
    h_thresh = (y_median - y_std, y_median + y_std * 0.5) 
    
    x_min = min(np.min(all_pts_concat[:,0]), np.min(cam_aligned[:,0]))
    x_max = max(np.max(all_pts_concat[:,0]), np.max(cam_aligned[:,0]))
    z_min = min(np.min(all_pts_concat[:,2]), np.min(cam_aligned[:,2]))
    z_max = max(np.max(all_pts_concat[:,2]), np.max(cam_aligned[:,2]))
    
    grid_res = max(0.01, (x_max - x_min) / 200.0)
    margin = grid_res * 10
    x_min, x_max = x_min - margin, x_max + margin
    z_min, z_max = z_min - margin, z_max + margin
    
    width = int(np.ceil((x_max - x_min) / grid_res))
    height = int(np.ceil((z_max - z_min) / grid_res))
    bounds = (x_min, z_min, x_max, z_max, width, height)
    
    print(f"Global bounds computed. Occupancy grid shape: ({height}, {width}, 3)")
    print(f"Creating animation video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5.0 # E.g., 5 frames per second for visualization
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    accum_pts = []
    for i in range(len(pts_aligned_frames)):
        if len(pts_aligned_frames[i]) > 0:
             accum_pts.append(pts_aligned_frames[i])
             
        curr_pts = np.vstack(accum_pts) if accum_pts else np.zeros((0,3))
        curr_cams = cam_aligned[:i+1]

        frame_img = points_to_occupancy_grid_fixed_bounds(curr_pts, curr_cams, bounds, grid_res, h_thresh)
        out.write(frame_img)
        
    out.release()
    print("Video generation complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--out", type=str, default="occupancy_grid_anim.mp4", help="Output 2D grid video path")
    parser.add_argument("--model", type=str, default="DA3-GIANT-1.1", help="DA3 model name")
    args = parser.parse_args()
    
    main(args.video, args.out, args.model)
