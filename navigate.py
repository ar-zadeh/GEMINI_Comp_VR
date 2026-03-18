"""
navigate.py  –  Interactive path planner on the occupancy grid.

Workflow
--------
1. Load the companion _navdata.npz produced by video_to_occupancy.py.
2. Display the last video frame in a window.
3. The user draws a bounding box (click + drag) around the destination.
4. Press ENTER or SPACE to confirm, ESC to redraw.
5. The bounding-box centre is back-projected through depth to 3D world space,
   then mapped to a grid cell.
6. A* finds a path from the last camera position to that cell.
7. Both windows update: the frame shows the selected box, the occupancy grid
   shows the planned path (cyan line, green = start, red = goal).

Usage
-----
    python navigate.py --data occupancy_grid_navdata.npz
"""

import argparse
import heapq
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helper: SE3 inverse (same logic as main pipeline)
# ---------------------------------------------------------------------------
def _as_h44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext.astype(np.float64)
    H = np.eye(4, dtype=np.float64)
    H[:3, :4] = ext
    return H


# ---------------------------------------------------------------------------
# A* on the occupancy grid
# ---------------------------------------------------------------------------
def _astar(grid_passable: np.ndarray, start: tuple, goal: tuple):
    """8-connected A* on a boolean passable grid.

    Parameters
    ----------
    grid_passable : (H, W) bool  – True where the robot can move
    start, goal   : (row, col)

    Returns
    -------
    path : list of (row, col) from start to goal, inclusive; or [] if none found.
    """
    H, W = grid_passable.shape

    def h(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    open_heap = []
    heapq.heappush(open_heap, (h(start, goal), 0.0, start))
    came_from = {start: None}
    g_score   = {start: 0.0}

    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    diag_cost = 2 ** 0.5

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        if g > g_score.get(current, float("inf")) + 1e-9:
            continue

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not grid_passable[nr, nc]:
                continue
            step = diag_cost if (dr != 0 and dc != 0) else 1.0
            ng = g + step
            if ng < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = ng
                came_from[(nr, nc)] = current
                heapq.heappush(open_heap, (ng + h((nr, nc), goal), ng, (nr, nc)))

    return []   # no path found


# ---------------------------------------------------------------------------
# Back-project a pixel on the last frame → aligned world point → grid cell
# ---------------------------------------------------------------------------
def pixel_to_grid_cell(u_orig, v_orig, last_frame_orig, last_depth,
                        last_intrinsic, last_extrinsic, alignment_matrix,
                        bounds, grid_res, grid_h):
    """Convert a pixel coordinate on the original last frame to a grid (row, col).

    Returns (row, col) in the *flipped* grid coordinate system,
    or None if the depth is invalid.
    """
    H_orig, W_orig = last_frame_orig.shape[:2]
    H_proc, W_proc = last_depth.shape

    # Scale pixel to processed-image space (DA3 resizes inputs)
    u_proc = u_orig * W_proc / W_orig
    v_proc = v_orig * H_proc / H_orig

    # Bilinear sample depth
    u0, v0 = int(np.clip(u_proc, 0, W_proc - 1)), int(np.clip(v_proc, 0, H_proc - 1))
    d = float(last_depth[v0, u0])

    if not np.isfinite(d) or d <= 0:
        return None

    # Back-project to camera space
    K_inv = np.linalg.inv(last_intrinsic.astype(np.float64))
    xc = K_inv @ np.array([u_proc, v_proc, 1.0]) * d

    # Camera → world (original world space before alignment)
    c2w = np.linalg.inv(_as_h44(last_extrinsic))
    xw  = (c2w @ np.array([xc[0], xc[1], xc[2], 1.0]))[:3]

    # Apply alignment transform
    A   = alignment_matrix.astype(np.float64)
    xa  = (A @ np.array([xw[0], xw[1], xw[2], 1.0]))[:3]

    x_min, z_min = bounds[0], bounds[1]

    col = int((xa[0] - x_min) / grid_res)
    row_unflipped = int((xa[2] - z_min) / grid_res)
    row = grid_h - 1 - row_unflipped   # undo flipud

    return row, col


# ---------------------------------------------------------------------------
# Interactive bounding-box drawer
# ---------------------------------------------------------------------------
class BoxDrawer:
    def __init__(self, image):
        self.orig   = image.copy()
        self.canvas = image.copy()
        self.box    = None          # (x1,y1,x2,y2) in image pixels
        self._start = None
        self._drawing = False

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start   = (x, y)
            self._drawing = True
            self.box      = None

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self.canvas = self.orig.copy()
            cv2.rectangle(self.canvas, self._start, (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            x1, y1 = self._start
            x2, y2 = x, y
            # Normalise so x1<x2, y1<y2
            self.box = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            self.canvas = self.orig.copy()
            cv2.rectangle(self.canvas, (self.box[0], self.box[1]),
                          (self.box[2], self.box[3]), (0, 255, 0), 2)
            # Draw centre cross
            cx = (self.box[0] + self.box[2]) // 2
            cy = (self.box[1] + self.box[3]) // 2
            cv2.drawMarker(self.canvas, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)

    def center(self):
        if self.box is None:
            return None
        return ((self.box[0] + self.box[2]) // 2,
                (self.box[1] + self.box[3]) // 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interactive path planner on the occupancy grid."
    )
    parser.add_argument("--data", required=True,
                        help="Path to the _navdata.npz file produced by video_to_occupancy.py")
    parser.add_argument("--dilate", type=int, default=1,
                        help="Obstacle dilation radius in grid cells (default: 1)")
    args = parser.parse_args()

    # ---- Load data --------------------------------------------------------
    print(f"Loading navigation data from {args.data} …")
    data = np.load(args.data, allow_pickle=False)

    occupancy_grid   = data["occupancy_grid"]          # (H, W, 3) uint8 annotated
    # Use clean grid (no camera trail) for pathfinding if available
    if "occupancy_grid_clean" in data:
        occupancy_grid_nav = data["occupancy_grid_clean"]
    else:
        occupancy_grid_nav = occupancy_grid             # backward compat
    bounds           = data["bounds"]                  # [x_min, z_min, x_max, z_max]
    grid_res         = float(data["grid_res"])
    cam_aligned      = data["cam_aligned"]             # (N, 3)
    alignment_matrix = data["alignment_matrix"]        # (4, 4)
    last_depth       = data["last_depth"]              # (H_proc, W_proc)
    last_intrinsic   = data["last_intrinsic"]          # (3, 3)
    last_extrinsic   = data["last_extrinsic"]          # (4, 4)
    last_frame_orig  = data["last_frame_orig"]         # (H_orig, W_orig, 3) RGB

    grid_h, grid_w = occupancy_grid.shape[:2]
    x_min, z_min   = bounds[0], bounds[1]

    # Convert last frame RGB → BGR for OpenCV display
    frame_bgr = cv2.cvtColor(last_frame_orig, cv2.COLOR_RGB2BGR)

    # ---- Build passable mask (from clean grid, so camera trail is not an obstacle) ----
    # White pixels (>=240 on all channels) = free space
    free_mask = np.all(occupancy_grid_nav >= 240, axis=2)

    if args.dilate > 0:
        kernel    = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*args.dilate+1, 2*args.dilate+1))
        # Dilate obstacles (non-free) to add a safety margin
        obstacle_mask = (~free_mask).astype(np.uint8)
        obstacle_dilated = cv2.dilate(obstacle_mask, kernel, iterations=1)
        passable = (obstacle_dilated == 0)
    else:
        passable = free_mask

    # ---- Start cell (last camera position) --------------------------------
    last_cam = cam_aligned[-1]
    sc       = int((last_cam[0] - x_min) / grid_res)
    sr_unflipped = int((last_cam[2] - z_min) / grid_res)
    sr       = grid_h - 1 - sr_unflipped

    sc = int(np.clip(sc, 0, grid_w - 1))
    sr = int(np.clip(sr, 0, grid_h - 1))

    # If the start cell is not passable, find nearest passable cell
    if not passable[sr, sc]:
        ys, xs = np.where(passable)
        if len(ys) > 0:
            dists = (ys - sr)**2 + (xs - sc)**2
            idx   = np.argmin(dists)
            sr, sc = int(ys[idx]), int(xs[idx])

    # ---- Interactive selection --------------------------------------------
    drawer = BoxDrawer(frame_bgr)
    WIN_FRAME = "Last Frame - Draw box, then press ENTER"
    WIN_GRID  = "Occupancy Grid - Planned Path"

    cv2.namedWindow(WIN_FRAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WIN_FRAME, drawer.canvas)   # must show before setMouseCallback
    cv2.waitKey(1)                         # pump Qt event loop so window is live
    cv2.setMouseCallback(WIN_FRAME, drawer.mouse_cb)

    print("\n  Draw a bounding box around your destination on the video frame.")
    print("  Press ENTER or SPACE to confirm  |  ESC to redraw  |  Q to quit.\n")

    grid_display = None

    while True:
        cv2.imshow(WIN_FRAME, drawer.canvas)
        if grid_display is not None:
            cv2.imshow(WIN_GRID, grid_display)

        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'),):
            break   # Q → quit any time

        if key == 27:                    # ESC → redraw box
            drawer.box    = None
            drawer.canvas = drawer.orig.copy()
            grid_display  = None
            print("  Box cleared – draw again.")
            continue

        if key in (13, 32):              # ENTER or SPACE → plan path
            center = drawer.center()
            if center is None:
                print("  No box drawn yet.")
                continue

            u_orig, v_orig = center
            print(f"  Selected pixel ({u_orig}, {v_orig}) on last frame …")

            cell = pixel_to_grid_cell(
                u_orig, v_orig,
                last_frame_orig, last_depth,
                last_intrinsic, last_extrinsic,
                alignment_matrix, bounds, grid_res, grid_h
            )

            if cell is None:
                print("  Depth invalid at that pixel – try again.")
                continue

            gr, gc = cell
            gr = int(np.clip(gr, 0, grid_h - 1))
            gc = int(np.clip(gc, 0, grid_w - 1))
            print(f"  Goal grid cell: row={gr}, col={gc}")

            # Snap goal to nearest passable cell if needed
            if not passable[gr, gc]:
                ys, xs = np.where(passable)
                if len(ys) > 0:
                    dists  = (ys - gr)**2 + (xs - gc)**2
                    idx    = np.argmin(dists)
                    gr, gc = int(ys[idx]), int(xs[idx])
                    print(f"  Goal snapped to nearest free cell: ({gr}, {gc})")

            # ---- A* (retry with no dilation if first attempt fails) -------
            print("  Running A* ...")
            path = _astar(passable, (sr, sc), (gr, gc))

            if not path and args.dilate > 0:
                print("  No path with dilation, retrying without dilation ...")
                path = _astar(free_mask, (sr, sc), (gr, gc))
                if path:
                    print(f"  Path found (no dilation): {len(path)} cells.")

            if not path:
                print("  No path found - the goal may be unreachable from the start.")
            else:
                print(f"  Path found: {len(path)} cells.")

            # ---- Draw result on grid copy ---------------------------------
            # occupancy_grid is BGR (built with OpenCV colour operations)
            grid_display = occupancy_grid.copy()

            # Draw path – cyan in BGR = [255, 255, 0]
            if path:
                for pr, pc in path:
                    grid_display[pr, pc] = [255, 255, 0]

            # Draw start (green circle)
            cv2.circle(grid_display, (sc, sr), 5, (0, 255, 0), -1)
            # Draw goal (red circle)
            cv2.circle(grid_display, (gc, gr), 5, (0, 0, 255), -1)

            # Draw bounding box projection label on the frame
            frame_annotated = drawer.canvas.copy()
            cv2.putText(frame_annotated,
                        f"Goal cell ({gr},{gc})",
                        (drawer.box[0], max(0, drawer.box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            drawer.canvas = frame_annotated

            # Scale up for better visibility if grid is small
            scale = max(1, 600 // max(grid_h, grid_w))
            if scale > 1:
                grid_display = cv2.resize(
                    grid_display,
                    (grid_w * scale, grid_h * scale),
                    interpolation=cv2.INTER_NEAREST
                )

            print("  Press ENTER/SPACE to select a new destination, Q or ESC to quit.")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
