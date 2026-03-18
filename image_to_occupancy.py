import os
import sys
import glob
import cv2
import numpy as np
import torch
import trimesh

# Add the Depth-Anything-3 src folder to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything-3", "src"))

from depth_anything_3.api import DepthAnything3


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """Ensure a (3,4) or (4,4) extrinsic matrix is returned as (4,4)."""
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"Extrinsic must be (3,4) or (4,4), got {ext.shape}")


def load_images(image_paths):
    """Load a list of image file paths as RGB numpy arrays.

    Args:
        image_paths : list of file path strings (supports glob patterns)

    Returns:
        frames : list of (H, W, 3) uint8 numpy arrays in RGB order
    """
    # Expand any glob patterns
    expanded = []
    for p in image_paths:
        matches = sorted(glob.glob(p))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(p)  # keep as-is; will fail below with a clear error

    frames = []
    for p in expanded:
        img = cv2.imread(p)
        if img is None:
            print(f"  WARNING: could not read image '{p}', skipping.")
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(f"  Loaded {p}  ({frames[-1].shape[1]}x{frames[-1].shape[0]})")

    return frames


def _unproject_depth_to_world(depth, intrinsics, extrinsics, images_u8, conf=None, conf_thr=0.0):
    """Unproject depth maps (N,H,W) to 3D world points.

    Args:
        depth      : (N, H, W) float depth maps
        intrinsics : (N, 3, 3) camera intrinsic matrices
        extrinsics : (N, 3,4) or (N, 4,4) world-to-camera (w2c) extrinsics
        images_u8  : (N, H, W, 3) uint8 RGB images
        conf       : (N, H, W) confidence maps or None
        conf_thr   : confidence threshold for filtering

    Returns:
        points : (M, 3) float32 world points
        colors : (M, 3) uint8 RGB colors
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3).astype(np.float64)

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

        K_inv = np.linalg.inv(intrinsics[i].astype(np.float64))
        c2w = np.linalg.inv(_as_homogeneous44(extrinsics[i].astype(np.float64)))

        rays = K_inv @ pix[vidx].T          # (3, M)
        Xc = rays * d_flat[vidx][None, :]   # (3, M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])  # (4, M)
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)         # (M, 3)

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

        if len(points) > 50000:
            pcd_down = pcd.random_down_sample(50000 / len(points))
        else:
            pcd_down = pcd

        plane_model, _ = pcd_down.segment_plane(
            distance_threshold=max(0.02 * y_range, 0.05),
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])

        if normal[1] < 0:
            normal = -normal
            d = -d

        if normal[1] >= 0.5:
            distances = np.dot(points, normal) + d
            dist_5th = np.percentile(distances, 5)
            distances = distances - dist_5th

            floor_mask = (distances >= -0.2 * y_range) & (distances < 0.10 * y_range)
            wall_mask  = (distances >= 0.10 * y_range) & (distances < 0.4  * y_range)

            if return_masks:
                return points[floor_mask], points[wall_mask], floor_mask, wall_mask
            return points[floor_mask], points[wall_mask]

    except ImportError:
        pass  # open3d not available, fallback to pure numpy

    # Fallback: estimate floor level using simple percentiles
    y = points[:, 1]
    floor_y = np.percentile(y, 5)

    floor_band_top = floor_y + 0.10 * y_range
    wall_band_top  = floor_y + 0.4  * y_range

    floor_mask = (y >= floor_y)        & (y < floor_band_top)
    wall_mask  = (y >= floor_band_top) & (y < wall_band_top)

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

    margin = grid_res * 10
    x_min, x_max = x_min - margin, x_max + margin
    z_min, z_max = z_min - margin, z_max + margin

    width  = int(np.ceil((x_max - x_min) / grid_res))
    height = int(np.ceil((z_max - z_min) / grid_res))

    if width <= 0 or height <= 0:
        return np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 0, 0)

    grid = np.zeros((height, width, 3), dtype=np.uint8)

    def _project_to_grid(pts):
        x, z = pts[:, 0], pts[:, 2]
        c = np.clip(np.floor((x - x_min) / grid_res).astype(int), 0, width  - 1)
        r = np.clip(np.floor((z - z_min) / grid_res).astype(int), 0, height - 1)
        return r, c

    if len(floor_points) > 0:
        fr, fc = _project_to_grid(floor_points)
        grid[fr, fc] = [255, 255, 255]

    if len(wall_points) > 0:
        wr, wc = _project_to_grid(wall_points)
        grid[wr, wc] = [0, 0, 140]  # BGR: dark red

    return grid, (x_min, z_min, x_max, z_max)


def main(image_paths, output_path, model_name="DA3-LARGE-1.1", conf_percentile=40.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Depth Anything 3 model: {model_name} on {device}...")
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}")
    model.eval()
    model = model.to(device)

    # --- Load images ---
    print(f"Loading {len(image_paths)} image path(s)...")
    frames = load_images(image_paths)
    if len(frames) == 0:
        print("No images loaded. Check the provided paths.")
        return
    print(f"Loaded {len(frames)} image(s).")

    # --- Run DA3 inference ---
    print("Running DA3 inference...")
    prediction = model.inference(frames)

    depth      = prediction.depth            # (N, H, W)
    conf       = prediction.conf             # (N, H, W) or None
    extrinsics = prediction.extrinsics       # (N, 3,4) or (N, 4,4) w2c
    intrinsics = prediction.intrinsics       # (N, 3, 3)
    images_u8  = prediction.processed_images # (N, H, W, 3) uint8

    N = depth.shape[0]

    # Confidence-based filtering threshold
    conf_thr = np.percentile(conf, conf_percentile) if conf is not None else 0.0

    # --- Unproject depth maps to 3D world points ---
    print("Unprojecting depth maps to 3D world points...")
    points, colors = _unproject_depth_to_world(
        depth, intrinsics, extrinsics, images_u8, conf, conf_thr
    )

    if len(points) == 0:
        print("No valid 3D points after filtering. Check thresholds or input images.")
        return

    print(f"Extracted {len(points)} points. Aligning to first camera...")

    # --- Align point cloud to first camera frame ---
    w2c0 = _as_homogeneous44(extrinsics[0].astype(np.float64))

    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0  # flip Y
    M[2, 2] = -1.0  # flip Z

    A_no_center = M @ w2c0

    center   = np.median(points, axis=0)
    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center
    A = T_center @ A_no_center

    pts_homo    = np.hstack([points, np.ones((len(points), 1))])
    pts_aligned = (A @ pts_homo.T).T[:, :3]

    # --- Camera positions ---
    print("Extracting camera positions...")
    cam_centers_w = []
    for ext in extrinsics:
        c2w = np.linalg.inv(_as_homogeneous44(ext.astype(np.float64)))
        cam_centers_w.append((c2w @ np.array([0.0, 0.0, 0.0, 1.0]))[:3])

    cam_centers_w = np.array(cam_centers_w)  # (N, 3)
    cam_homo    = np.hstack([cam_centers_w, np.ones((N, 1))])
    cam_aligned = (A @ cam_homo.T).T[:, :3]

    # --- Classify points into floor and wall ---
    floor_pts, wall_pts = [], []
    if len(pts_aligned) > 0:
        print("Classifying 3D point cloud into floor and wall...")
        floor_pts, wall_pts, floor_mask, wall_mask = classify_points_by_height(
            pts_aligned, return_masks=True
        )
        print(f"  Floor points: {len(floor_pts)}, Wall points: {len(wall_pts)}")
        colors[wall_mask] = [255, 0, 0]  # paint obstacle points red in PLY

    # --- Save PLY ---
    ply_out = output_path.rsplit(".", 1)[0] + "_pointcloud.ply"
    print(f"Saving 3D point cloud to {ply_out}...")

    cam_pts  = []
    cam_cols = []
    scene_scale = max(0.01, (np.max(pts_aligned[:, 0]) - np.min(pts_aligned[:, 0])) / 100.0)

    for i, c in enumerate(cam_aligned):
        sphere = trimesh.creation.icosphere(radius=scene_scale * 5.0, subdivisions=1)
        sphere.apply_translation(c)
        cam_pts.append(sphere.vertices)

        ratio = i / max(1, len(cam_aligned) - 1)
        col   = [int(255 * ratio), int(255 * (1 - ratio)), 0]  # green → red gradient
        cam_cols.append(np.tile(col, (len(sphere.vertices), 1)))

    cam_pts  = np.vstack(cam_pts)
    cam_cols = np.vstack(cam_cols)

    all_pts  = np.vstack([pts_aligned, cam_pts])
    all_cols = np.vstack([colors, cam_cols])

    pc = trimesh.points.PointCloud(vertices=all_pts, colors=all_cols)
    pc.export(ply_out)

    # --- Occupancy grid ---
    print("Converting 3D point cloud to 2D occupancy grid...")
    if len(pts_aligned) > 0:

        span_x   = np.max(pts_aligned[:, 0]) - np.min(pts_aligned[:, 0])
        grid_res = max(0.01, span_x / 200.0)

        occupancy_grid, bounds = points_to_occupancy_grid(
            floor_pts,
            wall_pts,
            cam_aligned,
            grid_res=grid_res,
        )

        x_min, z_min, x_max, z_max = bounds
        print(f"Occupancy grid shape: {occupancy_grid.shape}")

        # Draw camera positions and trajectory on the grid
        for i, c in enumerate(cam_aligned):
            cx, cz = c[0], c[2]
            c_col  = int((cx - x_min) / grid_res)
            c_row  = int((cz - z_min) / grid_res)

            if 0 <= c_col < occupancy_grid.shape[1] and 0 <= c_row < occupancy_grid.shape[0]:
                ratio = i / max(1, len(cam_aligned) - 1)
                col = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR
                cv2.circle(occupancy_grid, (c_col, c_row), radius=1, color=col, thickness=-1)

                if i > 0:
                    prev_c = cam_aligned[i - 1]
                    p_col  = int((prev_c[0] - x_min) / grid_res)
                    p_row  = int((prev_c[2] - z_min) / grid_res)
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
        print("Point cloud was empty. Check thresholds or input images.")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a 2D occupancy map from one or more images using Depth Anything 3."
    )
    parser.add_argument(
        "images", nargs="+",
        help="One or more image file paths (or glob patterns, e.g. 'frames/*.png')"
    )
    parser.add_argument("--out",   type=str, default="occupancy_grid.png",
                        help="Output 2D occupancy grid image path")
    parser.add_argument("--model", type=str, default="DA3-LARGE-1.1",
                        help="Depth Anything 3 model name (e.g. DA3-LARGE-1.1, DA3-BASE-1.1)")
    parser.add_argument("--conf_percentile", type=float, default=40.0,
                        help="Percentile of depth confidence below which points are filtered")
    args = parser.parse_args()

    main(args.images, args.out, args.model, args.conf_percentile)
