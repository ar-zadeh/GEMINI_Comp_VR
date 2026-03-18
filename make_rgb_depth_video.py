import argparse
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
    raise ValueError(f"Invalid extrinsic shape: {ext.shape}")


def sample_video_frames_uniform(video_path: str, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Failed to read frame count from: {video_path}")

    num_frames = max(1, min(num_frames, total))
    idxs = np.linspace(0, total - 1, num_frames, dtype=np.int32)

    frames_rgb = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames_rgb) == 0:
        raise RuntimeError("No frames decoded from input video.")
    return frames_rgb


def depth_to_colormap(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if depth_max <= depth_min:
        depth_min = float(np.min(d))
        depth_max = float(np.max(d))
        if depth_max <= depth_min:
            depth_max = depth_min + 1e-6

    d_norm = (d - depth_min) / (depth_max - depth_min)
    d_norm = np.clip(d_norm, 0.0, 1.0)
    d_u8 = (d_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)


def _estimate_floor_and_height(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    conf: np.ndarray | None,
    conf_percentile: float,
    floor_percentile: float,
    depth_min_m: float,
    depth_max_m: float,
    sample_stride: int,
) -> tuple[float, float, float, float]:
    conf_threshold = float(np.percentile(conf, conf_percentile)) if conf is not None else 0.0

    n, h, w = depth.shape
    ys = np.arange(0, h, max(1, sample_stride), dtype=np.int32)
    xs = np.arange(0, w, max(1, sample_stride), dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys)
    pix = np.stack([gx.reshape(-1), gy.reshape(-1), np.ones(gx.size)], axis=1)

    y_world_values = []
    cam_y_values = []
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

        c2w = np.linalg.inv(_as_homogeneous44(extrinsics[i]).astype(np.float64))
        x_world = (c2w[:3, :3] @ x_cam + c2w[:3, 3:4]).T
        y_world_values.append(x_world[:, 1])
        cam_y_values.append(float(c2w[1, 3]))

    if len(y_world_values) == 0 or len(cam_y_values) == 0:
        return 1.0, 0.0, 1.0, conf_threshold

    y_all = np.concatenate(y_world_values).astype(np.float64)
    cam_y = np.array(cam_y_values, dtype=np.float64)

    best = None
    for sign in (1.0, -1.0):
        ysigned = sign * y_all
        csigned = sign * cam_y
        floor = float(np.percentile(ysigned, floor_percentile))
        cam_h = float(np.median(csigned) - floor)
        if best is None or cam_h > best[0]:
            best = (cam_h, sign, floor)

    cam_height, sign_y, floor_y = best
    cam_height = max(cam_height, 1e-4)
    return float(sign_y), float(floor_y), float(cam_height), float(conf_threshold)


def _obstacle_mask_for_frame(
    depth_i: np.ndarray,
    intrinsics_i: np.ndarray,
    extrinsics_i: np.ndarray,
    conf_i: np.ndarray | None,
    conf_threshold: float,
    sign_y: float,
    floor_y: float,
    floor_clearance_ratio: float,
    obstacle_height_ratio: float,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    h, w = depth_i.shape
    ys = np.arange(h, dtype=np.int32)
    xs = np.arange(w, dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys)

    d = depth_i.reshape(-1)
    valid = np.isfinite(d) & (d >= depth_min_m) & (d <= depth_max_m)
    if conf_i is not None:
        c = conf_i.reshape(-1)
        valid &= np.isfinite(c) & (c >= conf_threshold)

    if not np.any(valid):
        return np.zeros((h, w), dtype=np.uint8)

    pix = np.stack([gx.reshape(-1), gy.reshape(-1), np.ones(h * w)], axis=1)
    p = pix[valid]
    d_valid = d[valid]

    k_inv = np.linalg.inv(intrinsics_i).astype(np.float64)
    rays = (k_inv @ p.T)
    x_cam = rays * d_valid[None, :]

    c2w = np.linalg.inv(_as_homogeneous44(extrinsics_i).astype(np.float64))
    x_world = (c2w[:3, :3] @ x_cam + c2w[:3, 3:4])
    y_world = sign_y * x_world[1, :]

    floor_upper = floor_y + max(0.01, floor_clearance_ratio)
    obstacle_upper = floor_y + max(0.05, obstacle_height_ratio)
    obs_valid = (y_world > floor_upper) & (y_world <= obstacle_upper)

    out = np.zeros(h * w, dtype=np.uint8)
    out[np.flatnonzero(valid)[obs_valid]] = 255
    return out.reshape(h, w)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = args.model.upper()
    model_repo = model_name if "/" in model_name else f"depth-anything/{model_name}"

    print(f"[INFO] Sampling {args.num_frames} frames from {args.input}")
    frames_rgb = sample_video_frames_uniform(args.input, args.num_frames)
    print(f"[INFO] Decoded {len(frames_rgb)} frames")

    print(f"[INFO] Loading model {model_repo} on {device}")
    model = DepthAnything3.from_pretrained(model_repo).to(device)

    print("[INFO] Running depth inference")
    pred = model.inference(
        image=frames_rgb,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy="middle",
    )

    rgb = pred.processed_images
    depth = pred.depth
    conf = pred.conf
    intrinsics = pred.intrinsics
    extrinsics = pred.extrinsics
    if rgb is None or depth is None:
        raise RuntimeError("Model output missing processed images or depth.")
    if intrinsics is None or extrinsics is None:
        raise RuntimeError("Model output missing camera intrinsics/extrinsics.")

    if args.depth_vmin is None or args.depth_vmax is None:
        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size == 0:
            dmin, dmax = 0.0, 1.0
        else:
            dmin = float(np.percentile(finite_depth, 2.0))
            dmax = float(np.percentile(finite_depth, 98.0))
    else:
        dmin, dmax = float(args.depth_vmin), float(args.depth_vmax)

    sign_y, floor_y, camera_height, conf_threshold = _estimate_floor_and_height(
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        conf=conf,
        conf_percentile=args.conf_percentile,
        floor_percentile=args.floor_percentile,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        sample_stride=args.floor_estimate_stride,
    )
    floor_clearance = args.floor_clearance_ratio * camera_height
    obstacle_height = args.obstacle_height_ratio * camera_height
    print(
        f"[INFO] Obstacle model: sign={sign_y:+.0f}, floor={floor_y:.4f}, cam_h={camera_height:.4f}, conf_thr={conf_threshold:.3f}"
    )

    h, w = rgb.shape[1], rgb.shape[2]
    out_w, out_h = w * 2, h
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    for i in range(len(rgb)):
        left = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
        right = depth_to_colormap(depth[i], dmin, dmax)

        if args.highlight_obstacles:
            mask = _obstacle_mask_for_frame(
                depth_i=depth[i],
                intrinsics_i=intrinsics[i],
                extrinsics_i=extrinsics[i],
                conf_i=conf[i] if conf is not None else None,
                conf_threshold=conf_threshold,
                sign_y=sign_y,
                floor_y=floor_y,
                floor_clearance_ratio=floor_clearance,
                obstacle_height_ratio=obstacle_height,
                depth_min_m=args.depth_min_m,
                depth_max_m=args.depth_max_m,
            )

            obstacle_tint = np.zeros_like(right)
            obstacle_tint[:, :, 2] = 255
            alpha = 0.45
            obs = mask > 0
            if np.any(obs):
                right_obs = right[obs].astype(np.float32)
                tint_obs = obstacle_tint[obs].astype(np.float32)
                right[obs] = ((1.0 - alpha) * right_obs + alpha * tint_obs).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(right, contours, -1, (0, 255, 255), 1)

        frame = np.concatenate([left, right], axis=1)
        cv2.putText(frame, "RGB", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, "Depth", (w + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if args.highlight_obstacles:
            cv2.putText(
                frame,
                "Obstacles",
                (w + 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (50, 220, 255),
                2,
            )
        writer.write(frame)

    writer.release()
    print(f"[DONE] Saved video: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create side-by-side RGB + depth video using DA3.")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="rgb_depth_side_by_side.mp4", help="Output mp4 path")
    parser.add_argument("--model", type=str, default="DA3-LARGE-1.1", help="DA3 model name or HF repo")
    parser.add_argument("--device", type=str, default="", help="Device override, e.g. cuda or cpu")
    parser.add_argument("--num_frames", type=int, default=40, help="Uniformly sampled frame count")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    parser.add_argument("--process_res", type=int, default=504, help="Inference resize resolution")
    parser.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
        help="Inference resize policy",
    )
    parser.add_argument("--use_ray_pose", action="store_true", help="Use ray-based pose head")
    parser.add_argument("--depth_vmin", type=float, default=None, help="Fixed depth min for colormap")
    parser.add_argument("--depth_vmax", type=float, default=None, help="Fixed depth max for colormap")
    parser.add_argument("--highlight_obstacles", action="store_true", help="Overlay occupancy-style obstacles on depth")
    parser.add_argument("--conf_percentile", type=float, default=20.0, help="Confidence percentile for valid depth")
    parser.add_argument("--floor_percentile", type=float, default=20.0, help="Percentile used to estimate floor")
    parser.add_argument("--depth_min_m", type=float, default=0.2, help="Minimum valid depth for obstacle mask")
    parser.add_argument("--depth_max_m", type=float, default=20.0, help="Maximum valid depth for obstacle mask")
    parser.add_argument(
        "--floor_clearance_ratio",
        type=float,
        default=0.05,
        help="Lower obstacle bound above floor, as fraction of camera height",
    )
    parser.add_argument(
        "--obstacle_height_ratio",
        type=float,
        default=0.80,
        help="Upper obstacle bound above floor, as fraction of camera height",
    )
    parser.add_argument("--floor_estimate_stride", type=int, default=8, help="Stride for floor estimation sampling")
    main(parser.parse_args())
