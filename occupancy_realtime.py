import os
import sys
import cv2
import numpy as np
import torch
from collections import deque

# Add DA3 source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything-3", "src"))
from depth_anything_3.api import DepthAnything3


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=ext.dtype)
        out[:3, :4] = ext
        return out
    raise ValueError(f"Extrinsic shape must be (3,4) or (4,4), got {ext.shape}")


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Estimate Sim(3) mapping src -> dst using Umeyama.

    src, dst: (N,3), N>=3
    Returns (s, R, t) such that dst ≈ s*R@src + t
    """
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for Sim(3) estimation")

    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    var_src = np.mean(np.sum(src_c**2, axis=1))
    scale = np.trace(np.diag(D) @ S) / max(var_src, 1e-12)
    t = mu_dst - scale * (R @ mu_src)
    return float(scale), R, t


def _unproject_frame(depth, K, ext_w2c, image_u8, conf=None, conf_thr=0.0, max_pts=20000):
    """Unproject one frame into local world coordinates."""
    H, W = depth.shape

    valid = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        valid &= conf >= conf_thr

    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    vidx = np.flatnonzero(valid.reshape(-1))
    if len(vidx) > max_pts:
        pick = np.random.choice(len(vidx), size=max_pts, replace=False)
        vidx = vidx[pick]

    us = (vidx % W).astype(np.float64)
    vs = (vidx // W).astype(np.float64)
    pix = np.stack([us, vs, np.ones_like(us)], axis=1)

    d = depth.reshape(-1)[vidx].astype(np.float64)
    K_inv = np.linalg.inv(K.astype(np.float64))
    c2w = np.linalg.inv(_as_homogeneous44(ext_w2c.astype(np.float64)))

    rays = (K_inv @ pix.T)                       # (3, M)
    Xc = rays * d[None, :]                       # (3, M)
    Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
    Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)

    cols = image_u8.reshape(-1, 3)[vidx].astype(np.uint8)
    return Xw, cols


def _classify_floor_wall_simple(points):
    """Fast per-frame height band classifier (fallback-style)."""
    if len(points) == 0:
        return points, points

    y = points[:, 1]
    y_lo = np.percentile(y, 5)
    y_hi = np.percentile(y, 95)
    y_range = max(y_hi - y_lo, 1e-3)

    floor_top = y_lo + 0.10 * y_range
    wall_top = y_lo + 0.40 * y_range

    floor = points[(y >= y_lo) & (y < floor_top)]
    wall = points[(y >= floor_top) & (y < wall_top)]
    return floor, wall


def _apply_sim3(points, s, R, t):
    if len(points) == 0:
        return points
    return (s * (R @ points.T).T + t).astype(np.float32)


def _points_to_cells(points, grid_res):
    if len(points) == 0:
        return np.empty((0, 2), np.int32)
    ix = np.floor(points[:, 0] / grid_res).astype(np.int32)
    iz = np.floor(points[:, 2] / grid_res).astype(np.int32)
    return np.stack([ix, iz], axis=1)


def _render_grid(floor_cells, wall_cells, cam_cells, cell_size=3):
    if len(floor_cells) == 0 and len(wall_cells) == 0 and len(cam_cells) == 0:
        return np.zeros((240, 320, 3), np.uint8)

    all_cells = []
    if len(floor_cells) > 0:
        all_cells.append(floor_cells)
    if len(wall_cells) > 0:
        all_cells.append(wall_cells)
    if len(cam_cells) > 0:
        all_cells.append(cam_cells)

    all_cells = np.vstack(all_cells)
    x_min, z_min = np.min(all_cells[:, 0]), np.min(all_cells[:, 1])
    x_max, z_max = np.max(all_cells[:, 0]), np.max(all_cells[:, 1])

    w = int((x_max - x_min + 1) * cell_size)
    h = int((z_max - z_min + 1) * cell_size)
    w = max(w, 20)
    h = max(h, 20)

    grid = np.zeros((h, w, 3), np.uint8)

    if len(floor_cells) > 0:
        xf = (floor_cells[:, 0] - x_min) * cell_size
        zf = (floor_cells[:, 1] - z_min) * cell_size
        for xx, zz in zip(xf, zf):
            y0 = h - 1 - zz - (cell_size - 1)
            y1 = h - zz
            x0 = xx
            x1 = xx + cell_size
            grid[y0:y1, x0:x1] = (255, 255, 255)

    if len(wall_cells) > 0:
        xw = (wall_cells[:, 0] - x_min) * cell_size
        zw = (wall_cells[:, 1] - z_min) * cell_size
        for xx, zz in zip(xw, zw):
            y0 = h - 1 - zz - (cell_size - 1)
            y1 = h - zz
            x0 = xx
            x1 = xx + cell_size
            grid[y0:y1, x0:x1] = (0, 0, 140)

    if len(cam_cells) > 0:
        pts = []
        for i, (cx, cz) in enumerate(cam_cells):
            px = int((cx - x_min) * cell_size + cell_size // 2)
            py = int(h - 1 - (cz - z_min) * cell_size - cell_size // 2)
            pts.append((px, py))
            ratio = i / max(1, len(cam_cells) - 1)
            col = (0, int(255 * (1 - ratio)), int(255 * ratio))
            cv2.circle(grid, (px, py), radius=max(2, cell_size), color=col, thickness=-1)
        for i in range(1, len(pts)):
            cv2.line(grid, pts[i - 1], pts[i], (0, 0, 220), 1)

    return grid


def _show_side_by_side(frame_bgr, occ_bgr, target_h=520, right_w=720):
    fh, fw = frame_bgr.shape[:2]
    oh, ow = occ_bgr.shape[:2]

    sf = target_h / max(1, fh)
    so = target_h / max(1, oh)

    left = cv2.resize(frame_bgr, (int(fw * sf), target_h), interpolation=cv2.INTER_LINEAR)
    right_auto_w = int(ow * so)
    if right_w <= 0:
        right_w = right_auto_w
    right = cv2.resize(occ_bgr, (int(right_w), target_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.hstack([left, right])
    cv2.line(canvas, (left.shape[1], 0), (left.shape[1], target_h), (80, 80, 80), 2)
    return canvas


def main(source, out, model_name="DA3-LARGE-1.1", chunk_size=16, overlap=8,
         stream_fps=10.0, conf_percentile=40.0, grid_res=0.05,
         max_points_per_frame=12000, no_display=False):

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {model_name} on {device}...")
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}")
    model.eval().to(device)

    # Source: webcam index or video file
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    ffmpeg_proc = None
    writer = None

    frame_buffer = []
    global_frame_idx = 0
    processed_until = 0

    # Global sparse map storage (cells)
    floor_set = set()
    wall_set = set()
    cam_path_cells = []

    # Chunk alignment state
    prev_overlap_global_cam = None  # (overlap,3) global coords from last chunk

    # Output writer will be initialized on first rendered frame
    output_initialized = False

    print("Starting stream. Press Q or ESC to stop.")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)
        global_frame_idx += 1

        should_process = len(frame_buffer) >= chunk_size
        if not should_process:
            if not no_display:
                preview = _show_side_by_side(
                    frame_bgr,
                    np.zeros((240, 320, 3), np.uint8),
                    target_h=520,
                    right_w=720,
                )
                cv2.putText(preview, "Warming up buffer...", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Realtime Occupancy", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
            continue

        # Process one chunk (with overlap sliding window)
        chunk = frame_buffer[:chunk_size]
        pred = model.inference(chunk)

        depth = pred.depth
        conf = pred.conf
        ext = pred.extrinsics
        K = pred.intrinsics
        imgs_u8 = pred.processed_images

        conf_thr = float(np.percentile(conf, conf_percentile)) if conf is not None else 0.0

        # Local chunk transform: orient with first camera (same idea as existing code)
        w2c0 = _as_homogeneous44(ext[0].astype(np.float64))
        M = np.eye(4, dtype=np.float64)
        M[1, 1] = -1.0
        M[2, 2] = -1.0
        A_local = M @ w2c0

        # Camera centers in local chunk coords
        cams_local = []
        for e in ext:
            c2w = np.linalg.inv(_as_homogeneous44(e.astype(np.float64)))
            cw = (c2w @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
            cw_h = np.hstack([cw, 1.0])
            cl = (A_local @ cw_h)[:3].astype(np.float32)
            cams_local.append(cl)
        cams_local = np.array(cams_local)

        # Estimate chunk->global Sim3 using overlap camera centers
        if prev_overlap_global_cam is None:
            s_g, R_g, t_g = 1.0, np.eye(3), np.zeros(3)
        else:
            curr_overlap_local = cams_local[:overlap]
            if len(curr_overlap_local) >= 3 and len(prev_overlap_global_cam) >= 3:
                s_g, R_g, t_g = _umeyama_sim3(curr_overlap_local, prev_overlap_global_cam)
            else:
                s_g, R_g, t_g = 1.0, np.eye(3), np.zeros(3)

        # For first chunk use all frames. For subsequent chunks, only add new frames.
        start_local = 0 if processed_until == 0 else overlap

        for i in range(start_local, chunk_size):
            pts_local, _ = _unproject_frame(
                depth[i], K[i], ext[i], imgs_u8[i],
                conf=conf[i] if conf is not None else None,
                conf_thr=conf_thr,
                max_pts=max_points_per_frame,
            )
            if len(pts_local) == 0:
                continue

            # orient with local chunk frame
            pts_h = np.hstack([pts_local, np.ones((len(pts_local), 1), dtype=np.float32)])
            pts_local_aligned = (A_local @ pts_h.T).T[:, :3].astype(np.float32)

            # map to global
            pts_global = _apply_sim3(pts_local_aligned, s_g, R_g, t_g)

            floor_pts, wall_pts = _classify_floor_wall_simple(pts_global)

            f_cells = _points_to_cells(floor_pts, grid_res)
            w_cells = _points_to_cells(wall_pts, grid_res)

            for c in f_cells:
                floor_set.add((int(c[0]), int(c[1])))
            for c in w_cells:
                wall_set.add((int(c[0]), int(c[1])))

            # Obstacle overrides floor
            for c in w_cells:
                floor_set.discard((int(c[0]), int(c[1])))

            # Camera path
            cam_global = _apply_sim3(cams_local[i:i+1], s_g, R_g, t_g)[0]
            cam_cell = _points_to_cells(cam_global.reshape(1, 3), grid_res)[0]
            cam_path_cells.append((int(cam_cell[0]), int(cam_cell[1])))

            # Render + show latest output after each new frame
            floor_arr = np.array(list(floor_set), dtype=np.int32) if floor_set else np.zeros((0, 2), np.int32)
            wall_arr = np.array(list(wall_set), dtype=np.int32) if wall_set else np.zeros((0, 2), np.int32)
            cam_arr = np.array(cam_path_cells, dtype=np.int32) if cam_path_cells else np.zeros((0, 2), np.int32)

            occ = _render_grid(floor_arr, wall_arr, cam_arr, cell_size=3)
            left = cv2.cvtColor(imgs_u8[i], cv2.COLOR_RGB2BGR)
            panel = _show_side_by_side(left, occ, target_h=520, right_w=720)

            cv2.putText(panel, f"Global frames processed: {processed_until + (i - start_local) + 1}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2, cv2.LINE_AA)

            if not output_initialized:
                h, w = panel.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out, fourcc, stream_fps, (w, h))
                output_initialized = True

            if writer is not None:
                writer.write(panel)

            if not no_display:
                cv2.imshow("Realtime Occupancy", panel)
                key = cv2.waitKey(max(1, int(1000.0 / stream_fps))) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    cap.release()
                    if writer is not None:
                        writer.release()
                    cv2.destroyAllWindows()
                    print(f"Saved output to: {out}")
                    return

        # Store global camera coords of current chunk overlap tail for next chunk alignment
        tail_local = cams_local[chunk_size - overlap:]
        prev_overlap_global_cam = _apply_sim3(tail_local, s_g, R_g, t_g)

        # Slide window by stride = chunk_size - overlap
        stride = chunk_size - overlap
        frame_buffer = frame_buffer[stride:]
        processed_until += stride

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Saved output to: {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Realtime occupancy mapping with DA3 chunk streaming.")
    parser.add_argument("--source", type=str, default="0", help="Video source: webcam index (e.g. 0) or file path.")
    parser.add_argument("--out", type=str, default="occupancy_realtime.mp4", help="Output video path (.mp4).")
    parser.add_argument("--model", type=str, default="DA3-LARGE-1.1", help="DA3 model name.")
    parser.add_argument("--chunk_size", type=int, default=16, help="Frames per inference chunk.")
    parser.add_argument("--overlap", type=int, default=8, help="Overlap frames between consecutive chunks.")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback/output FPS.")
    parser.add_argument("--conf_percentile", type=float, default=40.0, help="Confidence percentile threshold.")
    parser.add_argument("--grid_res", type=float, default=0.05, help="Occupancy grid resolution (m/cell).")
    parser.add_argument("--max_points_per_frame", type=int, default=12000, help="Point downsample cap per frame.")
    parser.add_argument("--no_display", action="store_true", help="Disable GUI display; only save video.")

    args = parser.parse_args()

    main(
        source=args.source,
        out=args.out,
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        stream_fps=args.fps,
        conf_percentile=args.conf_percentile,
        grid_res=args.grid_res,
        max_points_per_frame=args.max_points_per_frame,
        no_display=args.no_display,
    )
