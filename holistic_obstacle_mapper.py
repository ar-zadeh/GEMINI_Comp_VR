import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=ext.dtype)
        out[:3, :4] = ext
        return out
    raise ValueError(f"Invalid extrinsic shape: {ext.shape}, expected (3,4) or (4,4)")


def _uniform_indices(total: int, count: int) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=np.int32)
    count = max(1, min(count, total))
    return np.linspace(0, total - 1, num=count, dtype=np.int32)


def _iter_input_images(input_path: str, frame_stride: int = 1, max_frames: int | None = None, uniform_frames: int | None = None) -> list[np.ndarray]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    frames: list[np.ndarray] = []
    if path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        image_files = sorted([p for p in path.iterdir() if p.suffix.lower() in exts])
        if len(image_files) == 0:
            raise ValueError(f"No images found in directory: {input_path}")

        if uniform_frames is not None and uniform_frames > 0:
            chosen = _uniform_indices(len(image_files), uniform_frames)
            for idx in chosen:
                img_bgr = cv2.imread(str(image_files[int(idx)]), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if len(frames) == 0:
                raise ValueError(f"Could not read sampled images from: {input_path}")
            return frames

        for idx, image_file in enumerate(image_files):
            if idx % max(1, frame_stride) != 0:
                continue
            img_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
        return frames

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    if uniform_frames is not None and uniform_frames > 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            chosen = _uniform_indices(total_frames, uniform_frames)
            for idx in chosen:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            cap.release()
            if len(frames) == 0:
                raise ValueError(f"No sampled frames decoded from input: {input_path}")
            return frames

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % max(1, frame_stride) == 0:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from input: {input_path}")
    return frames


def _camera_centers_from_extrinsics(extrinsics_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c2w_all = []
    centers = []
    for ext in extrinsics_w2c:
        ext44 = _as_homogeneous44(ext).astype(np.float64)
        c2w = np.linalg.inv(ext44)
        c2w_all.append(c2w.astype(np.float32))
        centers.append(c2w[:3, 3].astype(np.float32))
    return np.stack(c2w_all, axis=0), np.stack(centers, axis=0)


def _depth_to_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    conf: np.ndarray | None,
    conf_threshold: float,
    depth_min_m: float,
    depth_max_m: float,
    point_stride: int,
) -> np.ndarray:
    n, h, w = depth.shape
    ys = np.arange(0, h, max(1, point_stride), dtype=np.int32)
    xs = np.arange(0, w, max(1, point_stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pix = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), np.ones(grid_x.size)], axis=1)

    points_world = []
    for i in range(n):
        d = depth[i, ys[:, None], xs[None, :]].reshape(-1)
        valid = np.isfinite(d) & (d >= depth_min_m) & (d <= depth_max_m)
        if conf is not None:
            c = conf[i, ys[:, None], xs[None, :]].reshape(-1)
            valid &= np.isfinite(c) & (c >= conf_threshold)
        if not np.any(valid):
            continue

        p = pix[valid]
        d_valid = d[valid]

        k_inv = np.linalg.inv(intrinsics[i]).astype(np.float64)
        rays = (k_inv @ p.T)
        x_cam = rays * d_valid[None, :]

        c2w = np.linalg.inv(_as_homogeneous44(extrinsics_w2c[i]).astype(np.float64))
        x_cam_h = np.vstack([x_cam, np.ones((1, x_cam.shape[1]))])
        x_world = (c2w @ x_cam_h)[:3, :].T.astype(np.float32)
        points_world.append(x_world)

    if len(points_world) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points_world, axis=0)


def _depth_to_world_points_adaptive(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    conf: np.ndarray | None,
    conf_threshold: float,
    depth_min_m: float,
    depth_max_m: float,
    point_stride: int,
) -> np.ndarray:
    points_world = _depth_to_world_points(
        depth=depth,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        conf=conf,
        conf_threshold=conf_threshold,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        point_stride=point_stride,
    )

    if len(points_world) > 2000:
        return points_world

    relaxed_conf = 0.0 if conf is not None else conf_threshold
    relaxed_stride = max(1, point_stride // 2)
    points_relaxed = _depth_to_world_points(
        depth=depth,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        conf=conf,
        conf_threshold=relaxed_conf,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        point_stride=relaxed_stride,
    )
    return points_relaxed if len(points_relaxed) > 0 else points_world


def _select_vertical_axis_sign(points_xyz: np.ndarray, cam_xyz: np.ndarray, floor_percentile: float) -> tuple[float, float, float]:
    if len(points_xyz) == 0 or len(cam_xyz) == 0:
        return 1.0, 0.0, 1.0

    y_points = points_xyz[:, 1].astype(np.float64)
    y_cams = cam_xyz[:, 1].astype(np.float64)

    best = None
    for sign in (1.0, -1.0):
        yp = sign * y_points
        yc = sign * y_cams
        floor = np.percentile(yp, floor_percentile)
        cam_height = float(np.median(yc) - floor)
        score = cam_height
        if best is None or score > best[0]:
            best = (score, sign, floor, max(cam_height, 1e-4))

    _, sign, floor, cam_height = best
    return float(sign), float(floor), float(cam_height)


def _build_occupancy_grid(
    points_xyz: np.ndarray,
    cam_xyz: np.ndarray,
    resolution_m: float,
    floor_y: float,
    camera_height: float,
    sign_y: float,
    floor_clearance_ratio: float,
    obstacle_height_ratio: float,
    min_cell_points: int,
    path_radius_cells: int,
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
    if len(points_xyz) == 0:
        empty = np.full((64, 64), 0.5, dtype=np.float32)
        return empty, (0.0, 0.0, 1.0, 1.0), np.zeros((0, 2), dtype=np.float32)

    x = points_xyz[:, 0]
    z = points_xyz[:, 2]
    y = sign_y * points_xyz[:, 1]
    cam_x = cam_xyz[:, 0]
    cam_z = cam_xyz[:, 2]

    x_min = float(min(np.min(x), np.min(cam_x)))
    x_max = float(max(np.max(x), np.max(cam_x)))
    z_min = float(min(np.min(z), np.min(cam_z)))
    z_max = float(max(np.max(z), np.max(cam_z)))

    margin = max(resolution_m * 8.0, 0.1)
    x_min -= margin
    x_max += margin
    z_min -= margin
    z_max += margin

    width = max(8, int(np.ceil((x_max - x_min) / resolution_m)))
    height = max(8, int(np.ceil((z_max - z_min) / resolution_m)))

    floor_upper = floor_y + max(0.01, floor_clearance_ratio * camera_height)
    obstacle_upper = floor_y + max(0.05, obstacle_height_ratio * camera_height)

    floor_mask = y <= floor_upper
    obstacle_mask = (y > floor_upper) & (y <= obstacle_upper)

    floor_count = np.zeros((height, width), dtype=np.int32)
    obstacle_count = np.zeros((height, width), dtype=np.int32)

    def to_grid(px: np.ndarray, pz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = np.floor((px - x_min) / resolution_m).astype(np.int32)
        rows = np.floor((pz - z_min) / resolution_m).astype(np.int32)
        cols = np.clip(cols, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)
        return rows, cols

    if np.any(floor_mask):
        r, c = to_grid(x[floor_mask], z[floor_mask])
        np.add.at(floor_count, (r, c), 1)
    if np.any(obstacle_mask):
        r, c = to_grid(x[obstacle_mask], z[obstacle_mask])
        np.add.at(obstacle_count, (r, c), 1)

    occupancy = np.full((height, width), 0.5, dtype=np.float32)
    free_cells = floor_count >= max(1, min_cell_points)
    obstacle_cells = obstacle_count >= max(1, min_cell_points)
    occupancy[free_cells] = 0.0
    occupancy[obstacle_cells] = 1.0

    if np.any(occupancy != 0.5):
        known_mask = (occupancy != 0.5).astype(np.uint8)
        occ_mask = (occupancy == 1.0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        known_mask = cv2.morphologyEx(known_mask, cv2.MORPH_CLOSE, kernel)
        occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE, kernel)
        occupancy[known_mask > 0] = 0.0
        occupancy[occ_mask > 0] = 1.0

    cam_rows, cam_cols = to_grid(cam_x, cam_z)
    cam_grid = np.stack([cam_cols, cam_rows], axis=1).astype(np.float32)

    if len(cam_grid) >= 1 and path_radius_cells > 0:
        for cxy in cam_grid.astype(np.int32):
            cv2.circle(
                occupancy,
                (int(cxy[0]), int(cxy[1])),
                int(path_radius_cells),
                color=0.0,
                thickness=-1,
            )

    return occupancy, (x_min, z_min, x_max, z_max), cam_grid


def _render_occupancy_png(
    occupancy: np.ndarray,
    cam_grid: np.ndarray,
    output_png: str,
) -> None:
    h, w = occupancy.shape
    canvas = np.full((h, w, 3), 128, dtype=np.uint8)
    canvas[occupancy <= 0.01] = (255, 255, 255)
    canvas[occupancy >= 0.99] = (0, 0, 0)

    if len(cam_grid) >= 1:
        pts = cam_grid.astype(np.int32)
        for i in range(1, len(pts)):
            cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), (0, 170, 255), 2)
        for i, p in enumerate(pts):
            ratio = i / max(1, len(pts) - 1)
            color = (0, int(255 * (1 - ratio)), int(255 * ratio))
            cv2.circle(canvas, tuple(p), 3, color, -1)

        start = tuple(pts[0])
        end = tuple(pts[-1])
        cv2.circle(canvas, start, 5, (0, 255, 0), 2)
        cv2.circle(canvas, end, 5, (0, 0, 255), 2)

    canvas = np.flipud(canvas)
    cv2.imwrite(output_png, canvas)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = args.model.upper()
    model_repo = model_name if "/" in model_name else f"depth-anything/{model_name}"

    print(f"[INFO] Loading model: {model_repo} on {device}")
    model = DepthAnything3.from_pretrained(model_repo).to(device)

    print(f"[INFO] Reading frames from: {args.input}")
    frames = _iter_input_images(
        args.input,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        uniform_frames=args.uniform_frames,
    )
    print(f"[INFO] Using {len(frames)} frames")

    print("[INFO] Running DA3 inference")
    prediction = model.inference(
        image=frames,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
    )

    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("Model did not return camera parameters (extrinsics/intrinsics).")

    conf_threshold = (
        float(np.percentile(prediction.conf, args.depth_conf_percentile))
        if prediction.conf is not None
        else args.depth_conf_fallback
    )
    print(f"[INFO] Confidence threshold: {conf_threshold:.4f}")

    points_world = _depth_to_world_points_adaptive(
        depth=prediction.depth,
        intrinsics=prediction.intrinsics,
        extrinsics_w2c=prediction.extrinsics,
        conf=prediction.conf,
        conf_threshold=conf_threshold,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        point_stride=args.point_stride,
    )
    print(f"[INFO] Fused points: {len(points_world)}")

    poses_c2w, cam_xyz = _camera_centers_from_extrinsics(prediction.extrinsics)

    sign_y, floor_y, camera_height = _select_vertical_axis_sign(
        points_world, cam_xyz, floor_percentile=args.floor_percentile
    )
    print(
        f"[INFO] Vertical sign={sign_y:+.0f}, floor_y={floor_y:.4f}, camera_height={camera_height:.4f}"
    )

    occupancy, bounds, cam_grid = _build_occupancy_grid(
        points_xyz=points_world,
        cam_xyz=cam_xyz,
        resolution_m=args.map_resolution_m,
        floor_y=floor_y,
        camera_height=camera_height,
        sign_y=sign_y,
        floor_clearance_ratio=args.floor_clearance_ratio,
        obstacle_height_ratio=args.obstacle_height_ratio,
        min_cell_points=args.min_cell_points,
        path_radius_cells=args.path_radius_cells,
    )

    output_prefix = args.output_prefix
    npz_path = f"{output_prefix}.npz"
    png_path = f"{output_prefix}.png"

    if len(points_world) > args.max_points_in_npz:
        ids = np.random.default_rng(args.seed).choice(
            len(points_world), size=args.max_points_in_npz, replace=False
        )
        world_points_sample = points_world[ids]
    else:
        world_points_sample = points_world

    np.savez_compressed(
        npz_path,
        occupancy=occupancy,
        origin_xz=np.array([bounds[0], bounds[1]], dtype=np.float32),
        resolution_m=np.float32(args.map_resolution_m),
        trajectory_xyz=cam_xyz.astype(np.float32),
        poses_c2w=poses_c2w.astype(np.float32),
        world_points_sample=world_points_sample.astype(np.float32),
    )
    _render_occupancy_png(occupancy, cam_grid, png_path)

    print(f"[DONE] Saved map npz: {os.path.abspath(npz_path)}")
    print(f"[DONE] Saved map image: {os.path.abspath(png_path)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a 2D occupancy map from DepthAnything3 depth and estimated camera poses."
    )

    parser.add_argument("--input", type=str, required=True, help="Input video file or image directory")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="holistic_obstacle_map",
        help="Output prefix for .npz and .png",
    )

    parser.add_argument("--model", type=str, default="DA3-LARGE-1.1", help="DA3 model name or HF repo")
    parser.add_argument("--device", type=str, default="", help="Device override, e.g. cuda or cpu")
    parser.add_argument("--use_ray_pose", action="store_true", help="Enable ray-based pose estimation")
    parser.add_argument(
        "--ref_view_strategy",
        type=str,
        default="middle",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="Reference view selection strategy",
    )

    parser.add_argument(
        "--uniform_frames",
        type=int,
        default=40,
        help="Sample this many frames/images uniformly across input",
    )
    parser.add_argument("--frame_stride", type=int, default=1, help="Used when --uniform_frames <= 0")
    parser.add_argument("--max_frames", type=int, default=180, help="Maximum number of frames to process")
    parser.add_argument("--process_res", type=int, default=504, help="DA3 processing resolution")
    parser.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
        help="DA3 resize strategy",
    )

    parser.add_argument("--map_resolution_m", type=float, default=0.03, help="Occupancy cell size in meters")
    parser.add_argument("--depth_conf_percentile", type=float, default=20.0, help="Confidence percentile")
    parser.add_argument("--depth_conf_fallback", type=float, default=1.0, help="Fallback conf threshold")
    parser.add_argument("--depth_min_m", type=float, default=0.2, help="Minimum valid depth")
    parser.add_argument("--depth_max_m", type=float, default=20.0, help="Maximum valid depth")
    parser.add_argument("--point_stride", type=int, default=2, help="Pixel stride for point extraction")

    parser.add_argument("--floor_percentile", type=float, default=20.0, help="Percentile for floor level")
    parser.add_argument(
        "--floor_clearance_ratio",
        type=float,
        default=0.05,
        help="Floor-band thickness as ratio of estimated camera height",
    )
    parser.add_argument(
        "--obstacle_height_ratio",
        type=float,
        default=0.80,
        help="Upper obstacle cutoff as ratio of estimated camera height",
    )

    parser.add_argument("--min_cell_points", type=int, default=1, help="Min points per grid cell")
    parser.add_argument("--path_radius_cells", type=int, default=3, help="Free-space radius around trajectory")

    parser.add_argument(
        "--max_points_in_npz",
        type=int,
        default=300_000,
        help="Maximum sampled world points saved to NPZ",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for downsampling")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
