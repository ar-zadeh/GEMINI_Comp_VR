"""video_to_occupancy_live.py

Plays the input video side-by-side with the occupancy map being built frame
by frame.  After processing all frames with Depth-Anything-3 the script shows:

    [ Video frame (left) | Occupancy map so far (right) ]

Press Q or Esc during playback to quit early.
The composite video is also saved to disk.

Usage
-----
    python video_to_occupancy_live.py --video my_video.mp4 [--out out.mp4]
        [--model DA3-LARGE-1.1] [--fps 5] [--conf_percentile 40]
        [--grid_res 0] [--no_display]
"""

import os
import sys
import cv2
import numpy as np
import torch

# ── Depth-Anything-3 path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything-3", "src"))
from depth_anything_3.api import DepthAnything3


# ════════════════════════════════════════════════════════════════════════════
#  Helpers (adapted from video_to_occupancy.py)
# ════════════════════════════════════════════════════════════════════════════

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"Extrinsic must be (3,4) or (4,4), got {ext.shape}")


def extract_frames(video_path, max_frames=20):
    """Sample up to max_frames frames evenly, always including the last frame."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = max_frames

    sample_slots = max_frames - 1
    step = max(1, total_frames // max(1, sample_slots))

    idx, count, last_sampled_idx = 0, 0, -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0 and count < sample_slots:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_sampled_idx = idx
            count += 1
        idx += 1

    true_last = total_frames - 1
    if last_sampled_idx != true_last:
        cap.set(cv2.CAP_PROP_POS_FRAMES, true_last)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def _unproject_per_frame(depth, intrinsics, extrinsics, images_u8, conf=None, conf_thr=0.0):
    """Return a list of (pts, cols) arrays – one entry per frame.

    Each pts array has shape (M_i, 3) in world coordinates (before alignment).
    Frames with no valid points get empty arrays.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3).astype(np.float64)

    result_pts = []
    result_cols = []

    for i in range(N):
        d = depth[i]
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            result_pts.append(np.zeros((0, 3), dtype=np.float32))
            result_cols.append(np.zeros((0, 3), dtype=np.uint8))
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(intrinsics[i].astype(np.float64))
        c2w = np.linalg.inv(_as_homogeneous44(extrinsics[i].astype(np.float64)))

        rays = K_inv @ pix[vidx].T               # (3, M)
        Xc   = rays * d_flat[vidx][None, :]      # (3, M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw   = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M, 3)

        result_pts.append(Xw)
        result_cols.append(images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8))

    return result_pts, result_cols


def _classify_height(points):
    """Return (floor_mask, wall_mask) boolean arrays for a (N,3) point cloud.

    Uses RANSAC plane fitting (Open3D) when available, otherwise falls back to
    simple percentile thresholds.  The logic mirrors video_to_occupancy.py.
    """
    y_range = np.percentile(points[:, 1], 95) - np.percentile(points[:, 1], 5)
    y_range = max(y_range, 1e-3)

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd_down = pcd.random_down_sample(50000 / len(points)) if len(points) > 50000 else pcd
        plane_model, _ = pcd_down.segment_plane(
            distance_threshold=max(0.02 * y_range, 0.05),
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        if normal[1] < 0:
            normal, d = -normal, -d

        if normal[1] >= 0.5:
            distances = np.dot(points, normal) + d
            distances -= np.percentile(distances, 5)
            floor_mask = (distances >= -0.2 * y_range) & (distances < 0.10 * y_range)
            wall_mask  = (distances >= 0.10 * y_range) & (distances < 0.4  * y_range)
            return floor_mask, wall_mask

    except ImportError:
        pass

    # Fallback
    y = points[:, 1]
    floor_y       = np.percentile(y, 5)
    floor_band_top = floor_y + 0.10 * y_range
    wall_band_top  = floor_y + 0.40 * y_range
    floor_mask = (y >= floor_y)       & (y < floor_band_top)
    wall_mask  = (y >= floor_band_top) & (y < wall_band_top)
    return floor_mask, wall_mask


def _build_occupancy_grid(floor_pts_accum, wall_pts_accum, cam_accum,
                          bounds, grid_res):
    """Render a colour occupancy grid from accumulated floor / wall points.

    Colours:
        White  (255,255,255) = free space (floor)
        Dark-red (0,0,140)  = obstacle / wall
        Black  (0,0,0)      = unknown
    Camera path is drawn in a green→red gradient; path shown as red line.
    """
    x_min, z_min, x_max, z_max = bounds
    width  = int(np.ceil((x_max - x_min) / grid_res))
    height = int(np.ceil((z_max - z_min) / grid_res))
    width, height = max(width, 1), max(height, 1)

    grid = np.zeros((height, width, 3), dtype=np.uint8)

    def _proj(pts):
        c = np.clip(np.floor((pts[:, 0] - x_min) / grid_res).astype(int), 0, width  - 1)
        r = np.clip(np.floor((pts[:, 2] - z_min) / grid_res).astype(int), 0, height - 1)
        return r, c

    if len(floor_pts_accum) > 0:
        fr, fc = _proj(floor_pts_accum)
        grid[fr, fc] = [255, 255, 255]

    if len(wall_pts_accum) > 0:
        wr, wc = _proj(wall_pts_accum)
        grid[wr, wc] = [0, 0, 140]

    # Camera path
    for idx, cam in enumerate(cam_accum):
        cc = int((cam[0] - x_min) / grid_res)
        cr = int((cam[2] - z_min) / grid_res)
        if 0 <= cc < width and 0 <= cr < height:
            ratio = idx / max(1, len(cam_accum) - 1)
            col   = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR green→red
            cv2.circle(grid, (cc, cr), radius=2, color=col, thickness=-1)
            if idx > 0:
                pc  = cam_accum[idx - 1]
                pcc = int((pc[0] - x_min) / grid_res)
                pcr = int((pc[2] - z_min) / grid_res)
                if 0 <= pcc < width and 0 <= pcr < height:
                    cv2.line(grid, (pcc, pcr), (cc, cr), (0, 0, 200), 1)

    return np.flipud(grid)


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main(video_path, output_path, model_name="DA3-LARGE-1.1",
         fps=5.0, conf_percentile=40.0, grid_res_override=0.0,
         display=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"Loading {model_name} on {device} …")
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}")
    model.eval().to(device)

    # ── 2. Extract frames ────────────────────────────────────────────────────
    print(f"Extracting frames from {video_path} …")
    frames = extract_frames(video_path, max_frames=20)
    if not frames:
        print("No frames extracted. Exiting.")
        return
    N = len(frames)
    print(f"  {N} frames extracted.")

    # ── 3. DA3 inference ─────────────────────────────────────────────────────
    print("Running DA3 inference …")
    pred = model.inference(frames)

    depth      = pred.depth             # (N, H, W)
    conf       = pred.conf              # (N, H, W) or None
    extrinsics = pred.extrinsics        # (N, 4, 4) or (N, 3, 4) w2c
    intrinsics = pred.intrinsics        # (N, 3, 3)
    images_u8  = pred.processed_images  # (N, H, proc, 3) uint8

    conf_thr = float(np.percentile(conf, conf_percentile)) if conf is not None else 0.0

    # ── 4. Per-frame unprojection to world coords ─────────────────────────────
    print("Unprojecting depth maps …")
    pts_per_frame, _cols_per_frame = _unproject_per_frame(
        depth, intrinsics, extrinsics, images_u8, conf, conf_thr
    )

    # ── 5. Alignment transform (first-camera → centred Y-up frame) ───────────
    print("Computing alignment …")
    all_raw = np.vstack([p for p in pts_per_frame if len(p) > 0])
    if len(all_raw) == 0:
        print("No valid 3D points. Exiting.")
        return

    w2c0 = _as_homogeneous44(extrinsics[0].astype(np.float64))
    M    = np.diag([1.0, -1.0, -1.0, 1.0])   # flip Y and Z
    center    = np.median(all_raw, axis=0)
    T_center  = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center
    A = T_center @ (M @ w2c0)

    def _apply_A(pts_nx3):
        h = np.hstack([pts_nx3, np.ones((len(pts_nx3), 1))])
        return (A @ h.T).T[:, :3].astype(np.float32)

    pts_aligned = [_apply_A(p) if len(p) > 0 else p for p in pts_per_frame]

    # Camera positions
    cam_centers_world = []
    for ext in extrinsics:
        c2w = np.linalg.inv(_as_homogeneous44(ext.astype(np.float64)))
        cam_centers_world.append((c2w @ [0, 0, 0, 1])[:3])
    cam_centers_world = np.array(cam_centers_world)
    cam_aligned = _apply_A(cam_centers_world)  # (N, 3)

    # ── 6. Global height classification ──────────────────────────────────────
    print("Classifying floor vs. wall globally …")
    all_aligned = np.vstack([p for p in pts_aligned if len(p) > 0])
    floor_mask_global, wall_mask_global = _classify_height(all_aligned)

    # Split masks back to per-frame
    split_idx = np.cumsum([len(p) for p in pts_aligned])
    split_idx = np.insert(split_idx, 0, 0)

    floor_per_frame = []
    wall_per_frame  = []
    cursor = 0
    for p in pts_aligned:
        n = len(p)
        if n > 0:
            fm = floor_mask_global[cursor: cursor + n]
            wm = wall_mask_global [cursor: cursor + n]
            floor_per_frame.append(p[fm])
            wall_per_frame.append(p[wm])
            cursor += n
        else:
            floor_per_frame.append(np.zeros((0, 3), np.float32))
            wall_per_frame.append(np.zeros((0, 3), np.float32))

    print(f"  Floor points total: {sum(len(f) for f in floor_per_frame)}")
    print(f"  Wall  points total: {sum(len(w) for w in wall_per_frame)}")

    # ── 7. Fixed occupancy-grid bounds ────────────────────────────────────────
    margin = 0.5
    x_min = min(np.min(all_aligned[:, 0]), np.min(cam_aligned[:, 0])) - margin
    x_max = max(np.max(all_aligned[:, 0]), np.max(cam_aligned[:, 0])) + margin
    z_min = min(np.min(all_aligned[:, 2]), np.min(cam_aligned[:, 2])) - margin
    z_max = max(np.max(all_aligned[:, 2]), np.max(cam_aligned[:, 2])) + margin
    bounds = (x_min, z_min, x_max, z_max)

    span_x = x_max - x_min
    grid_res = grid_res_override if grid_res_override > 0 else max(0.01, span_x / 200.0)
    print(f"  Grid resolution: {grid_res:.4f} m/px")

    # ── 8. Side-by-side display dimensions ───────────────────────────────────
    # Target height for the composite window
    target_h = 480

    # Video frame size (use first processed frame from DA3)
    src_h, src_w = images_u8[0].shape[:2]
    vid_scale    = target_h / src_h
    vid_w        = int(src_w * vid_scale)
    vid_h        = target_h

    # Occupancy grid size
    occ_w = int(np.ceil((x_max - x_min) / grid_res))
    occ_h = int(np.ceil((z_max - z_min) / grid_res))
    occ_scale = target_h / max(occ_h, 1)
    occ_disp_w = int(occ_w * occ_scale)
    occ_disp_h = target_h

    composite_w = vid_w + occ_disp_w
    composite_h = target_h

    # ── 9. Video writer (H.264 via ffmpeg pipe) ──────────────────────────────
    import subprocess, shutil

    ffmpeg_bin = shutil.which("ffmpeg")
    ff_proc    = None
    writer     = None   # kept as sentinel; actual writing goes through ff_proc

    if ffmpeg_bin:
        ff_cmd = [
            ffmpeg_bin, "-y",
            "-f",  "rawvideo",
            "-vcodec", "rawvideo",
            "-s",  f"{composite_w}x{composite_h}",
            "-pix_fmt", "bgr24",
            "-r",  str(fps),
            "-i",  "pipe:0",
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf",    "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        ff_proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
        print(f"Video will be saved as H.264 MP4 to: {output_path}")
    else:
        # ffmpeg not found — fall back to OpenCV mp4v
        print("ffmpeg not found; falling back to OpenCV mp4v writer.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (composite_w, composite_h))
        if not writer.isOpened():
            print(f"Warning: Could not open VideoWriter for {output_path}")
            writer = None

    if display:
        cv2.namedWindow("DA3 Occupancy Map – Live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DA3 Occupancy Map – Live", composite_w, composite_h)

    # ── 10. Playback loop ────────────────────────────────────────────────────
    print(f"\nPlaying {N} frames …  (press Q / Esc to quit early)")

    accum_floor = []
    accum_wall  = []
    delay_ms    = max(1, int(1000.0 / fps))

    for i in range(N):
        # Accumulate points
        if len(floor_per_frame[i]) > 0:
            accum_floor.append(floor_per_frame[i])
        if len(wall_per_frame[i]) > 0:
            accum_wall.append(wall_per_frame[i])

        curr_floor = np.vstack(accum_floor) if accum_floor else np.zeros((0, 3), np.float32)
        curr_wall  = np.vstack(accum_wall)  if accum_wall  else np.zeros((0, 3), np.float32)
        curr_cams  = cam_aligned[: i + 1]

        # ── Left panel: original video frame (BGR) ────────────────────────
        frame_rgb = images_u8[i]                                 # (H, W, 3) uint8 RGB
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        left_panel = cv2.resize(frame_bgr, (vid_w, vid_h))

        # Frame index label
        cv2.putText(left_panel, f"Frame {i+1}/{N}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)

        # ── Right panel: occupancy map ────────────────────────────────────
        occ = _build_occupancy_grid(curr_floor, curr_wall, curr_cams, bounds, grid_res)
        right_panel = cv2.resize(occ, (occ_disp_w, occ_disp_h),
                                 interpolation=cv2.INTER_NEAREST)

        # Legend overlay
        legend_items = [
            ((255, 255, 255), "Free space"),
            ((0, 0, 140),     "Obstacle"),
            ((0, 200, 0),     "1st camera"),
            ((0, 0, 200),     "Last camera"),
        ]
        for li, (colour, label) in enumerate(legend_items):
            y0 = 22 + li * 22
            cv2.rectangle(right_panel, (8, y0 - 12), (22, y0 + 2), colour, -1)
            cv2.putText(right_panel, label, (26, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Composite ─────────────────────────────────────────────────────
        composite = np.hstack([left_panel, right_panel])

        # Separator line
        cv2.line(composite, (vid_w, 0), (vid_w, composite_h), (100, 100, 100), 2)

        # Title bar
        title = f"Depth-Anything-3  |  {model_name}  |  frame {i+1}/{N}"
        cv2.putText(composite, title, (vid_w + 8, composite_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        if ff_proc is not None:
            ff_proc.stdin.write(composite.tobytes())
        elif writer is not None:
            writer.write(composite)

        if display:
            cv2.imshow("DA3 Occupancy Map – Live", composite)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or Esc
                print("Playback interrupted by user.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if ff_proc is not None:
        ff_proc.stdin.close()
        ff_proc.wait()
        print(f"\nSaved composite video to: {output_path}")
    elif writer is not None:
        writer.release()
        print(f"\nSaved composite video to: {output_path}")

    if display:
        cv2.destroyAllWindows()

    print("Done.")


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Play video + live occupancy-map side by side using DA3."
    )
    parser.add_argument("--video", required=True,
                        help="Input video path.")
    parser.add_argument("--out", default="occupancy_live.mp4",
                        help="Output composite video path (default: occupancy_live.mp4).")
    parser.add_argument("--model", default="DA3-LARGE-1.1",
                        help="DA3 model name (e.g. DA3-LARGE-1.1, DA3-BASE-1.1).")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Playback / output FPS (default: 5).")
    parser.add_argument("--conf_percentile", type=float, default=40.0,
                        help="Confidence percentile threshold for depth filtering.")
    parser.add_argument("--grid_res", type=float, default=0.0,
                        help="Occupancy-grid resolution in metres.  0 = auto (default).")
    parser.add_argument("--no_display", action="store_true",
                        help="Skip the live OpenCV window (only write the output video).")
    args = parser.parse_args()

    main(
        video_path       = args.video,
        output_path      = args.out,
        model_name       = args.model,
        fps              = args.fps,
        conf_percentile  = args.conf_percentile,
        grid_res_override= args.grid_res,
        display          = not args.no_display,
    )
