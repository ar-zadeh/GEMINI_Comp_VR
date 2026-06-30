"""
Occupancy exploration for the Qwen VR agent.

This module keeps the polygon / mesh path out of the loop: it captures headset
views, estimates depth, avoids close objects directly from the current depth
view, delays 2D occupancy mapping until enough movement has happened, and
finally runs A* back to the start when a map exists.
"""

import base64
import heapq
import io
import json
import math
import multiprocessing as mproc
import os
import queue
import random
import re
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


UNKNOWN = 0
FREE = 1
OBSTACLE = 2


def _run_live_map_window(frame_queue, stop_event, window_name: str, max_size: int) -> None:
    latest = None
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except Exception as e:
        print(f"[Explore Map] Could not open live map window: {e}")
        return

    try:
        while not stop_event.is_set():
            try:
                while True:
                    latest = frame_queue.get_nowait()
            except queue.Empty:
                pass

            if latest is None:
                latest = np.zeros((320, 320, 3), dtype=np.uint8)

            frame = latest.copy()
            h, w = frame.shape[:2]
            scale = min(1.0, float(max_size) / max(h, w, 1))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_NEAREST,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                stop_event.set()
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


@dataclass
class OccupancyExploreConfig:
    map_size_m: float = 20.0
    grid_res_m: float = 0.10
    station_spacing_m: float = 1.25
    capture_yaw_offsets: Tuple[float, ...] = (0.0,)
    fov_degrees: float = 120.0
    max_stations: int = 120
    max_move_m: float = 1.0
    depth_baseline_move_m: float = 0.0
    obstacle_min_height_ratio: float = 0.30
    obstacle_max_height_ratio: float = 0.90
    treat_above_obstacle_band_as_blocking: bool = True
    obstacle_min_distance_m: float = 0.45
    require_known_free_forward_path: bool = True
    close_unknown_radius_m: float = 1.25
    min_depth_m: float = 0.10
    max_depth_m: float = 7.0
    depth_stride: int = 3
    conf_percentile: float = 40.0
    obstacle_inflation_m: float = 0.25
    settle_seconds: float = 0.20
    output_dir_name: str = "occupancy_explore"
    depth_engine: str = "vggt"
    model_name: str = "facebook/VGGT-1B"
    foundationstereo_repo: str = ""
    foundationstereo_checkpoint: str = ""
    foundationstereo_scale: float = 1.0
    foundationstereo_valid_iters: int = 8
    foundationstereo_max_disp: int = 192
    foundationstereo_min_disparity_px: float = 0.25
    foundationstereo_remove_invisible: bool = True
    foundationstereo_hierarchical: bool = False
    show_live_map: bool = True
    live_map_max_size_px: int = 800
    max_controller_follow_offset_m: float = 0.90
    split_stereo_capture: bool = True
    stereo_eye_separation_m: float = 0.064
    min_moves_before_mapping: int = 0
    forward_depth_safety_margin_m: float = 0.35
    forward_depth_corridor_width_ratio: float = 0.45
    forward_depth_vertical_min_ratio: float = 0.25
    forward_depth_vertical_max_ratio: float = 0.95
    forward_depth_min_valid_pixels: int = 25
    forward_depth_min_close_fraction: float = 0.05
    forward_depth_relative_close_ratio: float = 0.70
    forward_depth_unreliable_scale_near_wall_m: float = 1.5
    forward_depth_use_height_filter: bool = True
    forward_depth_obstacle_dilate_px: int = 5
    rotate_probability: float = 0.0
    forward_move_m: float = 1.0
    rotate_step_degrees: float = 20.0
    rotate_to_frontier: bool = True
    frontier_heading_lookahead_m: float = 2.0
    frontier_unknown_radius_m: float = 1.25
    frontier_novelty_weight: float = 0.08
    frontier_visited_penalty: float = 25.0
    max_frontier_candidates: int = 96
    avoid_visited_forward: bool = True
    visited_revisit_unknown_radius_m: float = 1.25
    max_rays_per_observation: int = 8000
    debug_output: str = "summary"
    export_ply: bool = True
    save_ply_each_move: bool = True
    max_ply_points: int = 300000


@dataclass
class Pose2D:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float


@dataclass
class Observation:
    pose: Pose2D
    yaw_offset: float
    image_rgb: np.ndarray
    station_idx: int = 0
    frame_label: str = ""
    depth: Optional[np.ndarray] = None
    intrinsic: Optional[np.ndarray] = None
    extrinsic: Optional[np.ndarray] = None
    conf: Optional[np.ndarray] = None
    depth_source: str = ""


class _ExploreStopped(Exception):
    pass


class OccupancyGrid:
    def __init__(self, center_x: float, center_z: float, config: OccupancyExploreConfig):
        self.config = config
        self.width = max(8, int(math.ceil(config.map_size_m / config.grid_res_m)))
        self.height = self.width
        half = 0.5 * self.width * config.grid_res_m
        self.x_min = center_x - half
        self.z_min = center_z - half
        self.state = np.zeros((self.height, self.width), dtype=np.uint8)
        self.visited = np.zeros((self.height, self.width), dtype=np.uint8)
        self.forced_free = np.zeros((self.height, self.width), dtype=np.uint8)

    def world_to_cell(self, x: float, z: float) -> Optional[Tuple[int, int]]:
        c = int(math.floor((x - self.x_min) / self.config.grid_res_m))
        r = int(math.floor((z - self.z_min) / self.config.grid_res_m))
        if 0 <= r < self.height and 0 <= c < self.width:
            return r, c
        return None

    def cell_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        r, c = cell
        x = self.x_min + (c + 0.5) * self.config.grid_res_m
        z = self.z_min + (r + 0.5) * self.config.grid_res_m
        return x, z

    def mark_visited(self, x: float, z: float) -> None:
        cell = self.world_to_cell(x, z)
        if cell:
            self.visited[cell] = 1
            if self.state[cell] != OBSTACLE:
                self.state[cell] = FREE

    def _world_to_render_pixel(self, x: float, z: float) -> Optional[Tuple[int, int]]:
        cell = self.world_to_cell(x, z)
        if cell is None:
            return None
        r, c = cell
        return c, self.height - 1 - r

    def render(
        self,
        path_cells: Optional[Sequence[Tuple[int, int]]] = None,
        camera_pose: Optional[Pose2D] = None,
    ) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[self.state == FREE] = (255, 255, 255)
        img[self.state == OBSTACLE] = (0, 0, 140)
        img[self.visited > 0] = (0, 180, 0)
        if path_cells:
            for r, c in path_cells:
                if 0 <= r < self.height and 0 <= c < self.width:
                    img[r, c] = (0, 180, 255)
        img = np.flipud(img).copy()
        if camera_pose is not None:
            self._draw_camera_pose(img, camera_pose)
        return img

    def _draw_camera_pose(self, img: np.ndarray, pose: Pose2D) -> None:
        start = self._world_to_render_pixel(pose.x, pose.z)
        if start is None:
            return

        arrow_m = max(0.55, 4.0 * self.config.grid_res_m)
        yaw_rad = math.radians(-pose.yaw)
        end_x = pose.x + arrow_m * math.sin(yaw_rad)
        end_z = pose.z - arrow_m * math.cos(yaw_rad)
        end = self._world_to_render_pixel(end_x, end_z)
        if end is None:
            end = (
                int(round(start[0] + arrow_m * math.sin(yaw_rad) / self.config.grid_res_m)),
                int(round(start[1] + arrow_m * math.cos(yaw_rad) / self.config.grid_res_m)),
            )
            end = (
                max(0, min(self.width - 1, end[0])),
                max(0, min(self.height - 1, end[1])),
            )

        radius = max(3, int(round(0.18 / self.config.grid_res_m)))
        cv2.circle(img, start, radius + 2, (0, 0, 0), thickness=-1)
        cv2.circle(img, start, radius, (255, 0, 255), thickness=-1)
        cv2.arrowedLine(
            img,
            start,
            end,
            (0, 255, 255),
            thickness=max(2, radius // 2),
            tipLength=0.35,
        )
        cv2.circle(img, start, max(1, radius // 3), (255, 255, 255), thickness=-1)

    def save(
        self,
        path: Path,
        path_cells: Optional[Sequence[Tuple[int, int]]] = None,
        camera_pose: Optional[Pose2D] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.render(path_cells=path_cells, camera_pose=camera_pose))

    def close_unknown_count(self, center: Tuple[int, int], radius_m: float) -> int:
        radius = max(1, int(math.ceil(radius_m / self.config.grid_res_m)))
        cr, cc = center
        count = 0
        for r in range(max(0, cr - radius), min(self.height, cr + radius + 1)):
            for c in range(max(0, cc - radius), min(self.width, cc + radius + 1)):
                if (r - cr) ** 2 + (c - cc) ** 2 <= radius ** 2 and self.state[r, c] == UNKNOWN:
                    count += 1
        return count

    def has_unknown_near(self, center: Tuple[int, int], radius_m: float) -> bool:
        return self.close_unknown_count(center, radius_m) > 0

    def visited_fraction_near(self, center: Tuple[int, int], radius_m: float) -> float:
        radius = max(1, int(math.ceil(radius_m / self.config.grid_res_m)))
        cr, cc = center
        total = 0
        visited = 0
        for r in range(max(0, cr - radius), min(self.height, cr + radius + 1)):
            for c in range(max(0, cc - radius), min(self.width, cc + radius + 1)):
                if (r - cr) ** 2 + (c - cc) ** 2 <= radius ** 2:
                    total += 1
                    visited += int(self.visited[r, c] > 0)
        return float(visited / max(total, 1))

    def fill_close_unknowns(self, center: Tuple[int, int], radius_m: float) -> int:
        radius = max(1, int(math.ceil(radius_m / self.config.grid_res_m)))
        cr, cc = center
        changed = 0
        for r in range(max(0, cr - radius), min(self.height, cr + radius + 1)):
            for c in range(max(0, cc - radius), min(self.width, cc + radius + 1)):
                if (r - cr) ** 2 + (c - cc) ** 2 <= radius ** 2 and self.state[r, c] == UNKNOWN:
                    self.state[r, c] = FREE
                    self.forced_free[r, c] = 1
                    changed += 1
        return changed

    def fill_enclosed_unknowns(self) -> int:
        unknown = self.state == UNKNOWN
        reachable = np.zeros_like(unknown, dtype=np.uint8)
        queue: List[Tuple[int, int]] = []
        for c in range(self.width):
            if unknown[0, c]:
                queue.append((0, c))
            if unknown[self.height - 1, c]:
                queue.append((self.height - 1, c))
        for r in range(self.height):
            if unknown[r, 0]:
                queue.append((r, 0))
            if unknown[r, self.width - 1]:
                queue.append((r, self.width - 1))

        while queue:
            r, c = queue.pop()
            if reachable[r, c] or not unknown[r, c]:
                continue
            reachable[r, c] = 1
            for nr, nc in _neighbors4((r, c)):
                if 0 <= nr < self.height and 0 <= nc < self.width and not reachable[nr, nc]:
                    queue.append((nr, nc))

        enclosed = unknown & (reachable == 0)
        changed = int(np.count_nonzero(enclosed))
        self.state[enclosed] = FREE
        return changed


def _neighbors4(cell: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    r, c = cell
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _parse_pose(text: str) -> Pose2D:
    pos_match = re.search(r"Position:\s*\[([^\]]+)\]", text)
    rot_match = re.search(r"Rotation:\s*\[([^\]]+)\]", text)
    if not pos_match or not rot_match:
        raise ValueError(f"Could not parse pose from: {text}")

    def _numbers(group: str) -> List[float]:
        clean = group.replace("np.float64(", "").replace(")", "")
        return [float(x.strip()) for x in clean.split(",")]

    pos = _numbers(pos_match.group(1))
    rot = _numbers(rot_match.group(1))
    if len(pos) != 3 or len(rot) != 3:
        raise ValueError(f"Unexpected pose shape from: {text}")
    return Pose2D(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])


def _copy_pose(pose: Pose2D) -> Pose2D:
    return Pose2D(pose.x, pose.y, pose.z, pose.pitch, pose.yaw, pose.roll)


def _pose_with_right_offset(pose: Pose2D, right_m: float, effective_yaw: float) -> Pose2D:
    yaw_rad = math.radians(-effective_yaw)
    return Pose2D(
        pose.x + right_m * math.cos(yaw_rad),
        pose.y,
        pose.z + right_m * math.sin(yaw_rad),
        pose.pitch,
        pose.yaw,
        pose.roll,
    )


def _bresenham(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    cells = []
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def _inflate_obstacles(state: np.ndarray, radius_cells: int) -> np.ndarray:
    blocked = state == OBSTACLE
    if radius_cells <= 0 or not blocked.any():
        return blocked
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius_cells * 2 + 1, radius_cells * 2 + 1),
    )
    return cv2.dilate(blocked.astype(np.uint8), kernel) > 0


def _normalize_to_u8(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        h = np.eye(4, dtype=ext.dtype)
        h[:3, :4] = ext
        return h
    raise ValueError(f"Extrinsic must be (3,4) or (4,4), got {ext.shape}")


def _fit_similarity_2d(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fit dst ~= scale * src @ rotation.T + translation."""
    if len(src) == 0 or len(dst) == 0:
        return 1.0, np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    if len(src) == 1 or np.linalg.matrix_rank(src - np.mean(src, axis=0)) < 1:
        return 1.0, np.eye(2, dtype=np.float64), dst[0].astype(np.float64) - src[0].astype(np.float64)

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / len(src)
    u, singular_values, vt = np.linalg.svd(cov)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    var_src = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    scale = float(np.sum(singular_values * np.diag(correction)) / max(var_src, 1e-9))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    translation = dst_mean - scale * (src_mean @ rotation.T)
    return scale, rotation, translation


def _astar(
    grid: OccupancyGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_unknown_goal: bool = False,
) -> Optional[List[Tuple[int, int]]]:
    inflate = int(math.ceil(grid.config.obstacle_inflation_m / grid.config.grid_res_m))
    blocked = _inflate_obstacles(grid.state, inflate)

    def passable(cell: Tuple[int, int]) -> bool:
        r, c = cell
        if not (0 <= r < grid.height and 0 <= c < grid.width):
            return False
        if blocked[r, c] and cell not in {start, goal}:
            return False
        if grid.state[r, c] == FREE:
            return True
        return allow_unknown_goal and cell == goal

    if not passable(start) or not passable(goal):
        return None

    frontier: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in _neighbors4(current):
            if not passable(nxt):
                continue
            new_cost = cost_so_far[current] + 1.0
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return None

    path = []
    cur: Optional[Tuple[int, int]] = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


class OccupancyExplorer:
    _shared_model = None
    _shared_device = None
    _shared_dtype = None
    _shared_model_name = None
    _shared_depth_engine = None
    _shared_foundationstereo_repo = None
    _shared_foundationstereo_checkpoint = None

    def __init__(self, executor, log_dir: Path, config: Optional[OccupancyExploreConfig] = None):
        self.executor = executor
        self.log_dir = log_dir
        self.config = config or OccupancyExploreConfig()
        self.observations: List[Observation] = []
        self._live_map_queue = None
        self._live_map_stop = None
        self._live_map_process = None
        self._controller_offsets: Dict[str, Dict[str, float]] = {}
        self._stop_event = None
        self._ply_point_chunks: List[np.ndarray] = []
        self._ply_color_chunks: List[np.ndarray] = []
        self._ply_point_count = 0

    def explore(self, stop_event=None) -> str:
        run_dir = self._new_run_dir()
        self._stop_event = stop_event
        start_pose = self._get_pose()
        grid = OccupancyGrid(start_pose.x, start_pose.z, self.config)
        start_cell = grid.world_to_cell(start_pose.x, start_pose.z)
        if start_cell is None:
            return "Explore failed: initial pose is outside the map."

        grid.mark_visited(start_pose.x, start_pose.z)
        path_history: List[Tuple[int, int]] = [start_cell]
        status: Dict[str, object] = {
            "start_pose": asdict(start_pose),
            "config": asdict(self.config),
            "stations": [],
        }
        movement_count = 0
        mapping_started = int(self.config.min_moves_before_mapping) <= 0

        print(f"[Explore] Output directory: {run_dir}")
        self._capture_controller_offsets()
        self._apply_controller_offsets()
        if mapping_started:
            self._start_live_map()
            self._publish_live_map(grid, path_history, start_pose)
        else:
            print(
                "[Explore] Delaying 2D occupancy map until "
                f"{int(self.config.min_moves_before_mapping)} movement actions. "
                f"Forward safety uses current {self._depth_engine_label()} depth only."
            )
        try:
            for station_idx in range(self.config.max_stations):
                self._raise_if_stopped()

                pose = self._get_pose()
                current_cell = grid.world_to_cell(pose.x, pose.z)
                if current_cell is None:
                    status["error"] = "Current pose moved outside the occupancy map."
                    break

                grid.mark_visited(pose.x, pose.z)
                capture_count = len(self.config.capture_yaw_offsets)
                print(
                    f"[Explore] Station {station_idx + 1}: capture {capture_count} view"
                    f"{'' if capture_count == 1 else 's'} at x={pose.x:.2f}, z={pose.z:.2f}"
                )
                station_timings: Dict[str, float] = {}

                t0 = time.perf_counter()
                new_obs = self._capture_station(run_dir, station_idx, pose)
                station_timings["capture_seconds"] = time.perf_counter() - t0
                self._raise_if_stopped()

                pose = self._get_pose()
                current_cell = grid.world_to_cell(pose.x, pose.z)
                if current_cell is None:
                    status["error"] = "Depth-baseline move left the occupancy map."
                    break
                grid.mark_visited(pose.x, pose.z)
                if not path_history or path_history[-1] != current_cell:
                    path_history.append(current_cell)
                self.observations.extend(new_obs)

                t0 = time.perf_counter()
                self._infer_observations(new_obs, run_dir, force=False)
                station_timings["depth_inference_seconds"] = time.perf_counter() - t0
                self._raise_if_stopped()

                integration_stats: Dict[str, object] = {
                    "skipped": not mapping_started,
                    "reason": (
                        f"2D occupancy mapping delayed until {int(self.config.min_moves_before_mapping)} moves"
                        if not mapping_started
                        else ""
                    ),
                    "movement_count": movement_count,
                }
                unknown_count = 0
                close_filled = 0
                if mapping_started:
                    t0 = time.perf_counter()
                    self._reset_ply_accumulator()
                    integration_stats = self._rebuild_grid(grid, run_dir, debug_observations=new_obs)
                    station_timings["grid_update_seconds"] = time.perf_counter() - t0
                    self._raise_if_stopped()

                    unknown_count = grid.close_unknown_count(current_cell, self.config.close_unknown_radius_m)
                    if unknown_count > 0 and self.config.obstacle_max_height_ratio >= 1.0:
                        close_filled = grid.fill_close_unknowns(current_cell, self.config.close_unknown_radius_m)
                        unknown_count = grid.close_unknown_count(current_cell, self.config.close_unknown_radius_m)
                        print(f"[Explore] Filled {close_filled} close-range unknown cells as free after max height band.")
                else:
                    station_timings["grid_update_seconds"] = 0.0

                t0 = time.perf_counter()
                action_result = self._take_stochastic_explore_step(
                    grid,
                    pose,
                    latest_observations=new_obs,
                    allow_grid_blocking=mapping_started,
                    run_dir=run_dir,
                    station_idx=station_idx,
                    movement_count=movement_count,
                )
                map_pose = self._get_pose()
                station_timings["action_seconds"] = time.perf_counter() - t0
                if action_result.get("executed") in {"rotate", "forward"}:
                    movement_count += 1
                action_cell = action_result.get("cell")
                if isinstance(action_cell, tuple) and action_cell != current_cell:
                    path_history.append(action_cell)

                step_ply_path = None
                station_timings["step_ply_seconds"] = 0.0
                if action_result.get("executed") in {"rotate", "forward"} and self.config.save_ply_each_move:
                    t0 = time.perf_counter()
                    executed = str(action_result.get("executed", "move"))
                    step_ply_path = self._save_ply(
                        run_dir / "ply_by_move" / f"move_{movement_count:03d}_{executed}.ply"
                    )
                    station_timings["step_ply_seconds"] = time.perf_counter() - t0
                    if step_ply_path:
                        action_result["step_ply"] = str(step_ply_path)

                if not mapping_started and movement_count >= int(self.config.min_moves_before_mapping):
                    mapping_started = True
                    print(
                        f"[Explore] Movement count reached {movement_count}; "
                        "starting 2D occupancy map generation."
                    )
                    self._start_live_map()

                    t0 = time.perf_counter()
                    self._reset_ply_accumulator()
                    integration_stats = self._rebuild_grid(grid, run_dir, debug_observations=new_obs)
                    station_timings["grid_update_seconds"] += time.perf_counter() - t0
                    self._raise_if_stopped()
                    unknown_count = grid.close_unknown_count(current_cell, self.config.close_unknown_radius_m)

                status["stations"].append({
                    "index": station_idx + 1,
                    "pose": asdict(pose),
                    "movement_count": movement_count,
                    "mapping_started": mapping_started,
                    "close_unknown_cells": unknown_count,
                    "close_unknown_cells_filled": close_filled,
                    "action": action_result,
                    "integration": integration_stats,
                    "timings": station_timings,
                })

                t0 = time.perf_counter()
                if mapping_started:
                    grid.save(run_dir / "occupancy_latest.png", path_history, camera_pose=map_pose)
                    self._publish_live_map(grid, path_history, map_pose)
                station_timings["save_and_publish_seconds"] = time.perf_counter() - t0
                print(
                    "[Explore] Station timing: "
                    f"capture={station_timings['capture_seconds']:.2f}s, "
                    f"infer={station_timings['depth_inference_seconds']:.2f}s, "
                    f"grid={station_timings['grid_update_seconds']:.2f}s, "
                    f"action={station_timings['action_seconds']:.2f}s"
                )

            self._raise_if_stopped()
            filled = 0
            final_pose = self._get_pose()
            final_cell = grid.world_to_cell(final_pose.x, final_pose.z)
            return_path = None
            if mapping_started:
                self._reset_ply_accumulator()
                final_integration = self._rebuild_grid(grid, run_dir, debug_observations=None)
                status["final_integration"] = final_integration
                filled = grid.fill_enclosed_unknowns()
                return_path = _astar(grid, final_cell, start_cell) if final_cell else None
            if return_path:
                print(f"[Explore] Returning to start with A* path of {len(return_path)} cells.")
                self._walk_cells(grid, self._sample_path_by_distance(grid, return_path, self.config.max_move_m))
                self.executor.call(
                    "rotate_device",
                    device="headset",
                    pitch=start_pose.pitch,
                    yaw=start_pose.yaw,
                    roll=start_pose.roll,
                )
                self._apply_controller_offsets()
            else:
                if mapping_started:
                    print("[Explore] Could not find A* return path to the start.")
                else:
                    print("[Explore] Mapping threshold was not reached; skipping A* return path.")

            map_pose = self._get_pose()
            final_map_path = None
            latest_map_path = None
            if mapping_started:
                final_map_path = run_dir / "occupancy_final.png"
                latest_map_path = run_dir / "occupancy_latest.png"
                grid.save(final_map_path, return_path, camera_pose=map_pose)
                self._publish_live_map(grid, return_path, map_pose)
            ply_path = self._save_ply(run_dir / "occupancy_points_final.ply")
            time.sleep(2.0)
            status.update({
                "final_pose_before_return": asdict(final_pose),
                "filled_enclosed_unknown_cells": filled,
                "movement_count": movement_count,
                "mapping_started": mapping_started,
                "min_moves_before_mapping": int(self.config.min_moves_before_mapping),
                "obstacle_min_height_ratio": self.config.obstacle_min_height_ratio,
                "obstacle_max_height_ratio": self.config.obstacle_max_height_ratio,
                "model_debug_dir": str(run_dir / "model_debug"),
                "returned_to_start": bool(return_path),
                "final_map": str(final_map_path) if final_map_path else None,
                "latest_map": str(latest_map_path) if latest_map_path else None,
                "point_cloud_ply": str(ply_path) if ply_path else None,
            })
            with open(run_dir / "explore_summary.json", "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)

            return (
                f"Explore complete. Movement actions: {movement_count}. "
                f"Final map: {final_map_path if final_map_path else 'not generated before movement threshold'}. "
                f"Height band: {self.config.obstacle_min_height_ratio:.2f}-"
                f"{self.config.obstacle_max_height_ratio:.2f}. "
                f"Interior unknown cells filled: {filled}. "
                f"Returned to start with A*: {bool(return_path)}. "
                f"PLY: {ply_path if ply_path else 'not generated'}."
            )
        except _ExploreStopped:
            stopped_pose = self._get_pose()
            stopped_map = None
            if mapping_started:
                stopped_map = run_dir / "occupancy_stopped.png"
                grid.save(stopped_map, path_history, camera_pose=stopped_pose)
                self._publish_live_map(grid, path_history, stopped_pose)
            ply_path = self._save_ply(run_dir / "occupancy_points_stopped.ply")
            status.update({
                "stopped": True,
                "stop_reason": "user requested stop",
                "stopped_pose": asdict(stopped_pose),
                "movement_count": movement_count,
                "mapping_started": mapping_started,
                "min_moves_before_mapping": int(self.config.min_moves_before_mapping),
                "latest_map": str(stopped_map) if stopped_map else None,
                "model_debug_dir": str(run_dir / "model_debug"),
                "point_cloud_ply": str(ply_path) if ply_path else None,
            })
            with open(run_dir / "explore_summary.json", "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
            print("[Explore] Stopped by user request.")
            return (
                f"Explore stopped. Movement actions: {movement_count}. "
                f"Latest map: {stopped_map if stopped_map else 'not generated before movement threshold'}. "
                f"PLY: {ply_path if ply_path else 'not generated'}."
            )
        finally:
            self._stop_live_map()
            self._stop_event = None

    def _stop_requested(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _raise_if_stopped(self) -> None:
        if self._stop_requested():
            raise _ExploreStopped()

    def _reset_ply_accumulator(self) -> None:
        self._ply_point_chunks = []
        self._ply_color_chunks = []
        self._ply_point_count = 0

    def _start_live_map(self) -> None:
        if not self.config.show_live_map:
            return
        if self._live_map_process is not None and self._live_map_process.is_alive():
            return
        try:
            ctx = mproc.get_context("spawn")
            self._live_map_queue = ctx.Queue(maxsize=2)
            self._live_map_stop = ctx.Event()
            self._live_map_process = ctx.Process(
                target=_run_live_map_window,
                args=(
                    self._live_map_queue,
                    self._live_map_stop,
                    "Qwen Occupancy Explorer",
                    int(self.config.live_map_max_size_px),
                ),
                daemon=False,
            )
            self._live_map_process.start()
            print(f"[Explore Map] Live occupancy map window started pid={self._live_map_process.pid}")
        except Exception as e:
            print(f"[Explore Map] Failed to start live map window: {e}")
            self._live_map_queue = None
            self._live_map_stop = None
            self._live_map_process = None

    def _publish_live_map(
        self,
        grid: OccupancyGrid,
        path_cells: Optional[Sequence[Tuple[int, int]]] = None,
        camera_pose: Optional[Pose2D] = None,
    ) -> None:
        if self._live_map_queue is None:
            return
        frame = grid.render(path_cells=path_cells, camera_pose=camera_pose)
        try:
            self._live_map_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._live_map_queue.get_nowait()
            except Exception:
                pass
            try:
                self._live_map_queue.put_nowait(frame)
            except Exception:
                pass
        except Exception:
            pass

    def _stop_live_map(self) -> None:
        if self._live_map_stop is not None:
            self._live_map_stop.set()
        if self._live_map_process is not None:
            self._live_map_process.join(timeout=1.5)
            if self._live_map_process.is_alive():
                self._live_map_process.terminate()
                self._live_map_process.join(timeout=0.5)
            print("[Explore Map] Live occupancy map window stopped.")
        if self._live_map_queue is not None:
            try:
                self._live_map_queue.close()
                self._live_map_queue.join_thread()
            except Exception:
                pass
        self._live_map_queue = None
        self._live_map_stop = None
        self._live_map_process = None

    def _capture_controller_offsets(self) -> None:
        """Snapshot controller offsets relative to the headset so they follow exploration moves."""
        mcp = getattr(self.executor, "module", None)
        if mcp is None or not hasattr(mcp, "current_poses") or not hasattr(mcp, "state_lock"):
            return

        try:
            with mcp.state_lock:
                headset = mcp.current_poses["headset"]
                headset_pos = headset["pos"]
                headset_rot = headset["rot"]
                headset_yaw = math.radians(-float(headset_rot[1]))
                offsets = {}

                for ctrl_name in ("controller1", "controller2"):
                    if ctrl_name not in mcp.current_poses:
                        continue
                    ctrl = mcp.current_poses[ctrl_name]
                    ctrl_pos = ctrl["pos"]
                    ctrl_rot = ctrl["rot"]

                    wx = float(ctrl_pos[0]) - float(headset_pos[0])
                    wy = float(ctrl_pos[1]) - float(headset_pos[1])
                    wz = float(ctrl_pos[2]) - float(headset_pos[2])
                    if math.sqrt(wx * wx + wy * wy + wz * wz) > self.config.max_controller_follow_offset_m:
                        self._controller_offsets = self._default_controller_offsets()
                        print("[Explore] Controller offset was too large; using natural follow offsets.")
                        return

                    offsets[ctrl_name] = {
                        "forward": wx * math.sin(headset_yaw) - wz * math.cos(headset_yaw),
                        "right": wx * math.cos(headset_yaw) + wz * math.sin(headset_yaw),
                        "up": wy,
                        "pitch": float(ctrl_rot[0]) - float(headset_rot[0]),
                        "yaw": float(ctrl_rot[1]) - float(headset_rot[1]),
                        "roll": float(ctrl_rot[2]) - float(headset_rot[2]),
                    }

            self._controller_offsets = offsets
            if offsets:
                print("[Explore] Controller offsets captured; controllers will follow headset exploration.")
        except Exception as e:
            print(f"[Explore] Could not capture controller offsets: {e}")

    @staticmethod
    def _default_controller_offsets() -> Dict[str, Dict[str, float]]:
        return {
            "controller1": {
                "forward": 0.3,
                "right": -0.3,
                "up": -0.5,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
            },
            "controller2": {
                "forward": 0.3,
                "right": 0.3,
                "up": -0.5,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
            },
        }

    def _apply_controller_offsets(self) -> None:
        """Move controllers with the headset using the saved local offsets."""
        if not self._controller_offsets:
            return
        mcp = getattr(self.executor, "module", None)
        if mcp is None or not hasattr(mcp, "current_poses") or not hasattr(mcp, "state_lock"):
            return

        try:
            with mcp.state_lock:
                headset = mcp.current_poses["headset"]
                headset_pos = headset["pos"]
                headset_rot = headset["rot"]
                headset_yaw = math.radians(-float(headset_rot[1]))

                for ctrl_name, offsets in self._controller_offsets.items():
                    if ctrl_name not in mcp.current_poses:
                        continue
                    fwd = offsets["forward"]
                    right = offsets["right"]
                    up = offsets["up"]

                    world_x = float(headset_pos[0]) + fwd * math.sin(headset_yaw) + right * math.cos(headset_yaw)
                    world_y = float(headset_pos[1]) + up
                    world_z = float(headset_pos[2]) + fwd * (-math.cos(headset_yaw)) + right * math.sin(headset_yaw)

                    mcp.current_poses[ctrl_name]["pos"] = [world_x, world_y, world_z]
                    mcp.current_poses[ctrl_name]["rot"] = [
                        float(headset_rot[0]) + offsets["pitch"],
                        float(headset_rot[1]) + offsets["yaw"],
                        float(headset_rot[2]) + offsets["roll"],
                    ]
            if hasattr(mcp, "broadcast_state"):
                mcp.broadcast_state()
        except Exception as e:
            print(f"[Explore] Could not apply controller offsets: {e}")

    def _new_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / self.config.output_dir_name / stamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_pose(self) -> Pose2D:
        return _parse_pose(self.executor.call("get_current_pose", device="headset"))

    def _capture_station_views(
        self,
        run_dir: Path,
        station_idx: int,
        pose: Pose2D,
        label_suffix: str,
    ) -> List[Observation]:
        observations = []
        for offset in self.config.capture_yaw_offsets:
            self._raise_if_stopped()
            frame_label = f"station_{station_idx:03d}_{label_suffix}_yaw_{int(offset):+04d}"
            yaw = pose.yaw + offset
            self.executor.call(
                "rotate_device",
                device="headset",
                pitch=pose.pitch,
                yaw=yaw,
                roll=pose.roll,
            )
            self._apply_controller_offsets()
            time.sleep(self.config.settle_seconds)
            self._raise_if_stopped()
            img_rgb = self._capture_rgb_image()
            self._raise_if_stopped()
            Image.fromarray(img_rgb).save(run_dir / f"{frame_label}.png")
            observations.extend(
                self._observations_from_capture(
                    img_rgb=img_rgb,
                    pose=pose,
                    yaw_offset=offset,
                    station_idx=station_idx,
                    frame_label=frame_label,
                )
            )

        self.executor.call(
            "rotate_device",
            device="headset",
            pitch=pose.pitch,
            yaw=pose.yaw,
            roll=pose.roll,
        )
        self._apply_controller_offsets()
        return observations

    def _observations_from_capture(
        self,
        img_rgb: np.ndarray,
        pose: Pose2D,
        yaw_offset: float,
        station_idx: int,
        frame_label: str,
    ) -> List[Observation]:
        images = self._split_stereo_capture(img_rgb)
        if len(images) == 1:
            return [
                Observation(
                    pose=_copy_pose(pose),
                    yaw_offset=yaw_offset,
                    image_rgb=images[0][1],
                    station_idx=station_idx,
                    frame_label=frame_label,
                )
            ]

        eye_half = 0.5 * max(0.0, float(self.config.stereo_eye_separation_m))
        eye_offsets = {"left": -eye_half, "right": eye_half}
        effective_yaw = pose.yaw + yaw_offset
        return [
            Observation(
                pose=_pose_with_right_offset(pose, eye_offsets.get(label, 0.0), effective_yaw),
                yaw_offset=yaw_offset,
                image_rgb=image,
                station_idx=station_idx,
                frame_label=f"{frame_label}_{label}",
            )
            for label, image in images
        ]

    def _split_stereo_capture(self, img_rgb: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        if not self.config.split_stereo_capture:
            return [("mono", img_rgb)]
        h, w = img_rgb.shape[:2]
        if w < 2 or w / max(h, 1) < 1.5:
            return [("mono", img_rgb)]
        mid = w // 2
        if mid <= 0 or w - mid <= 0:
            return [("mono", img_rgb)]
        return [
            ("left", img_rgb[:, :mid].copy()),
            ("right", img_rgb[:, mid:].copy()),
        ]

    def _capture_station(self, run_dir: Path, station_idx: int, pose: Pose2D) -> List[Observation]:
        observations = self._capture_station_views(run_dir, station_idx, pose, "base")
        self._raise_if_stopped()
        moved_pose = self._move_for_depth_baseline(pose)
        if moved_pose is None:
            return observations

        print(
            f"[Explore] Depth baseline moved to x={moved_pose.x:.2f}, "
            f"z={moved_pose.z:.2f}; capture {len(self.config.capture_yaw_offsets)} more view"
            f"{'' if len(self.config.capture_yaw_offsets) == 1 else 's'}."
        )
        observations.extend(self._capture_station_views(run_dir, station_idx, moved_pose, "baseline"))
        return observations

    def _move_for_depth_baseline(self, pose: Pose2D) -> Optional[Pose2D]:
        distance = max(0.0, float(self.config.depth_baseline_move_m))
        if distance <= 0.0:
            return None

        yaw_rad = math.radians(-pose.yaw)
        x = pose.x + distance * math.sin(yaw_rad)
        z = pose.z - distance * math.cos(yaw_rad)
        self._raise_if_stopped()
        self.executor.call("walk_path", x=x, z=z, steps=5)
        self._apply_controller_offsets()
        time.sleep(self.config.settle_seconds)
        self._raise_if_stopped()
        return self._get_pose()

    def _take_stochastic_explore_step(
        self,
        grid: OccupancyGrid,
        pose: Pose2D,
        latest_observations: Sequence[Observation],
        allow_grid_blocking: bool,
        run_dir: Optional[Path] = None,
        station_idx: int = 0,
        movement_count: int = 0,
    ) -> Dict[str, object]:
        self._raise_if_stopped()
        rotate_probability = min(1.0, max(0.0, float(self.config.rotate_probability)))
        roll = random.random()
        target_x, target_z = self._forward_target(pose, self.config.forward_move_m)
        depth_blocked, depth_reason, depth_distance, depth_stats = self._forward_depth_blocked(
            latest_observations,
            self.config.forward_move_m,
            run_dir=run_dir,
            station_idx=station_idx,
            movement_count=movement_count,
        )
        if depth_blocked:
            self._rotate_by_degrees(pose, self.config.rotate_step_degrees, f"blocked forward: {depth_reason}")
            return {
                "selected": "forward",
                "executed": "rotate",
                "blocked": True,
                "block_source": "depth",
                "block_reason": depth_reason,
                "blocked_distance_m": depth_distance,
                "depth_safety": depth_stats,
                "random_roll": roll,
                "rotate_probability": rotate_probability,
                "degrees": float(self.config.rotate_step_degrees),
                "target": {"x": target_x, "z": target_z},
            }

        if allow_grid_blocking and self.config.rotate_to_frontier:
            current_cell = grid.world_to_cell(pose.x, pose.z)
            if current_cell is not None:
                frontier_path = self._path_to_frontier(grid, current_cell)
                if frontier_path:
                    lookahead = self._lookahead_cell(
                        grid,
                        frontier_path,
                        self.config.frontier_heading_lookahead_m,
                    )
                    rotated = self._rotate_toward_cell(grid, lookahead, "nearest frontier")
                    if rotated:
                        return {
                            "selected": "frontier_heading",
                            "executed": "rotate",
                            "blocked": False,
                            "frontier_path_cells": len(frontier_path),
                            "frontier_lookahead_cell": lookahead,
                            "depth_safety": depth_stats,
                            "random_roll": roll,
                            "rotate_probability": rotate_probability,
                        }

        if roll < rotate_probability:
            self._rotate_by_degrees(pose, self.config.rotate_step_degrees, "scheduled exploration rotation")
            return {
                "selected": "rotate",
                "executed": "rotate",
                "random_roll": roll,
                "rotate_probability": rotate_probability,
                "degrees": float(self.config.rotate_step_degrees),
                "depth_safety": depth_stats,
            }

        blocked_cell = None
        blocked_distance = None
        if allow_grid_blocking:
            blocked, reason, blocked_cell, blocked_distance = self._forward_move_blocked(
                grid, pose, target_x, target_z
            )
        else:
            blocked, reason = False, ""

        if blocked:
            self._rotate_by_degrees(pose, self.config.rotate_step_degrees, f"blocked forward: {reason}")
            return {
                "selected": "forward",
                "executed": "rotate",
                "blocked": True,
                "block_source": "occupancy_grid",
                "block_reason": reason,
                "blocked_cell": blocked_cell,
                "blocked_distance_m": blocked_distance,
                "depth_safety": depth_stats,
                "random_roll": roll,
                "rotate_probability": rotate_probability,
                "degrees": float(self.config.rotate_step_degrees),
                "target": {"x": target_x, "z": target_z},
            }

        print(f"[Explore] Moving forward {self.config.forward_move_m:.2f} m.")
        self._raise_if_stopped()
        self.executor.call("walk_path", x=target_x, z=target_z, steps=10)
        self._apply_controller_offsets()
        time.sleep(self.config.settle_seconds)
        self._raise_if_stopped()
        new_pose = self._get_pose()
        new_cell = grid.world_to_cell(new_pose.x, new_pose.z)
        grid.mark_visited(new_pose.x, new_pose.z)
        return {
            "selected": "forward",
            "executed": "forward",
            "blocked": False,
            "random_roll": roll,
            "rotate_probability": rotate_probability,
            "distance_m": float(self.config.forward_move_m),
            "depth_safety": depth_stats,
            "used_grid_blocking": bool(allow_grid_blocking),
            "target": {"x": target_x, "z": target_z},
            "pose_after": asdict(new_pose),
            "cell": new_cell,
        }

    def _forward_depth_blocked(
        self,
        observations: Sequence[Observation],
        step_m: float,
        run_dir: Optional[Path] = None,
        station_idx: int = 0,
        movement_count: int = 0,
    ) -> Tuple[bool, str, Optional[float], Dict[str, object]]:
        usable = [
            obs for obs in observations
            if abs(float(obs.yaw_offset)) < 1e-6 and obs.depth is not None and obs.conf is not None
        ]
        if not usable:
            usable = [
                obs for obs in observations
                if abs(float(obs.yaw_offset)) < 1e-6 and obs.depth is not None
            ]
        if not usable:
            return True, "no current depth for forward view", None, {
                "usable_observations": 0,
                "decision": "blocked",
                "heatmap_paths": [],
            }

        threshold_m = max(0.0, float(step_m)) + max(0.0, float(self.config.forward_depth_safety_margin_m))
        min_valid_pixels = max(1, int(self.config.forward_depth_min_valid_pixels))
        min_close_fraction = max(0.0, min(1.0, float(self.config.forward_depth_min_close_fraction)))
        relative_close_ratio = max(0.05, min(1.0, float(self.config.forward_depth_relative_close_ratio)))
        unreliable_scale_near_wall_m = max(0.0, float(self.config.forward_depth_unreliable_scale_near_wall_m))
        obstacle_pixel_masks = self._forward_obstacle_pixel_masks(usable)
        height_filter_used = obstacle_pixel_masks is not None
        best_distance: Optional[float] = None
        per_obs = []
        heatmap_paths: List[str] = []

        for obs in usable:
            depth = obs.depth
            if depth is None:
                continue
            h, w = depth.shape
            x_margin = 0.5 * max(0.0, min(1.0, float(self.config.forward_depth_corridor_width_ratio)))
            x0 = int(max(0, math.floor((0.5 - x_margin) * w)))
            x1 = int(min(w, math.ceil((0.5 + x_margin) * w)))
            y0 = int(max(0, math.floor(max(0.0, min(1.0, self.config.forward_depth_vertical_min_ratio)) * h)))
            y1 = int(min(h, math.ceil(max(0.0, min(1.0, self.config.forward_depth_vertical_max_ratio)) * h)))
            if x1 <= x0 or y1 <= y0:
                continue

            roi_depth = depth[y0:y1, x0:x1].astype(np.float64)
            depth_valid = np.isfinite(roi_depth) & (roi_depth >= self.config.min_depth_m) & (roi_depth <= self.config.max_depth_m)
            valid = depth_valid.copy()
            if obs.conf is not None:
                conf_roi = obs.conf[y0:y1, x0:x1]
                finite_conf = conf_roi[np.isfinite(conf_roi)]
                if finite_conf.size:
                    conf_thr = float(np.percentile(finite_conf, self.config.conf_percentile))
                    valid &= conf_roi >= conf_thr
            raw_valid_count = int(np.count_nonzero(valid))

            obstacle_roi = None
            if obstacle_pixel_masks is not None:
                obstacle_mask = obstacle_pixel_masks.get(id(obs))
                if obstacle_mask is not None:
                    obstacle_roi = obstacle_mask[y0:y1, x0:x1]
                    valid &= obstacle_roi

            valid_depths = roi_depth[valid]
            valid_count = int(valid_depths.size)
            close_values = valid_depths <= threshold_m if valid_count else np.zeros((0,), dtype=bool)
            close_count = int(np.count_nonzero(close_values))
            close_fraction = float(close_count / max(valid_count, 1))
            close_roi = np.zeros(valid.shape, dtype=bool)
            if valid_count:
                close_roi[valid] = close_values
            nearest = float(np.percentile(valid_depths, 5)) if valid_count else None
            if nearest is not None and (best_distance is None or nearest < best_distance):
                best_distance = nearest

            full_valid = np.isfinite(depth) & (depth >= self.config.min_depth_m) & (depth <= self.config.max_depth_m)
            if obs.conf is not None:
                full_conf = obs.conf
                finite_conf = full_conf[np.isfinite(full_conf)]
                if finite_conf.size:
                    full_conf_thr = float(np.percentile(finite_conf, self.config.conf_percentile))
                    full_valid &= full_conf >= full_conf_thr
            full_depths = depth[full_valid].astype(np.float64)
            scene_median = float(np.percentile(full_depths, 50)) if full_depths.size else None
            scene_p90 = float(np.percentile(full_depths, 90)) if full_depths.size else None
            metric_scale_looks_reliable = scene_p90 is not None and scene_p90 > threshold_m * 1.25
            relatively_close = (
                nearest is not None
                and scene_median is not None
                and nearest <= scene_median * relative_close_ratio
            )
            close_obstacle_present = (
                close_count >= min_valid_pixels
                and close_fraction >= min_close_fraction
            )
            scale_confirmed_close = metric_scale_looks_reliable or relatively_close
            near_wall_with_unreliable_scale = (
                height_filter_used
                and not scale_confirmed_close
                and nearest is not None
                and nearest <= unreliable_scale_near_wall_m
            )
            blocks_forward = close_obstacle_present and (
                scale_confirmed_close or near_wall_with_unreliable_scale
            )
            per_obs.append({
                "frame_label": obs.frame_label,
                "height_filter_used": height_filter_used,
                "raw_valid_pixels": raw_valid_count,
                "obstacle_height_pixels": int(np.count_nonzero(obstacle_roi)) if obstacle_roi is not None else None,
                "valid_pixels": valid_count,
                "close_pixels": close_count,
                "close_fraction": close_fraction,
                "nearest_depth_p05_m": nearest,
                "scene_depth_median_m": scene_median,
                "scene_depth_p90_m": scene_p90,
                "metric_scale_looks_reliable": metric_scale_looks_reliable,
                "relatively_close": relatively_close,
                "close_obstacle_present": close_obstacle_present,
                "scale_confirmed_close": scale_confirmed_close,
                "near_wall_with_unreliable_scale": near_wall_with_unreliable_scale,
                "unreliable_scale_near_wall_m": unreliable_scale_near_wall_m,
                "blocks_forward": blocks_forward,
            })
            heatmap_path = self._save_forward_depth_heatmap(
                run_dir=run_dir,
                obs=obs,
                station_idx=station_idx,
                movement_count=movement_count,
                roi_bounds=(x0, y0, x1, y1),
                valid_roi=valid,
                close_roi=close_roi,
                threshold_m=threshold_m,
                stats=per_obs[-1],
            )
            if heatmap_path:
                per_obs[-1]["heatmap_path"] = heatmap_path
                heatmap_paths.append(heatmap_path)

        total_valid = sum(int(item["valid_pixels"]) for item in per_obs)
        total_close = sum(int(item["close_pixels"]) for item in per_obs)
        blocking_observations = [item for item in per_obs if item.get("blocks_forward")]
        stats: Dict[str, object] = {
            "usable_observations": len(usable),
            "valid_pixels": total_valid,
            "close_pixels": total_close,
            "blocking_observations": len(blocking_observations),
            "clearance_threshold_m": threshold_m,
            "min_close_fraction": min_close_fraction,
            "relative_close_ratio": relative_close_ratio,
            "unreliable_scale_near_wall_m": unreliable_scale_near_wall_m,
            "height_filter_used": height_filter_used,
            "nearest_depth_p05_m": best_distance,
            "heatmap_paths": heatmap_paths,
            "per_observation": per_obs,
        }
        if total_valid < min_valid_pixels:
            if height_filter_used:
                stats["decision"] = "clear"
                stats["clear_reason"] = "no obstacle-height points in forward corridor"
                return False, "", best_distance, stats
            stats["decision"] = "blocked"
            return True, "not enough valid depth pixels in forward corridor", best_distance, stats
        if blocking_observations:
            stats["decision"] = "blocked"
            return True, "depth object inside forward safety corridor", best_distance, stats
        stats["decision"] = "clear"
        return False, "", best_distance, stats

    def _forward_obstacle_pixel_masks(
        self,
        observations: Sequence[Observation],
    ) -> Optional[Dict[int, np.ndarray]]:
        if not self.config.forward_depth_use_height_filter:
            return None
        valid_observations = [
            obs for obs in observations
            if obs.depth is not None and obs.intrinsic is not None and obs.extrinsic is not None
        ]
        if not valid_observations:
            return None
        try:
            conf_values = [obs.conf for obs in valid_observations if obs.conf is not None]
            conf_thr = (
                float(np.percentile(np.concatenate([c.reshape(-1) for c in conf_values]), self.config.conf_percentile))
                if conf_values
                else 0.0
            )
            recon = self._reconstruct_model_aligned_points(valid_observations, conf_thr)
            if recon is None:
                return None
            points_per_obs, cam_aligned, sample_data = recon
            usable_points = [points for points in points_per_obs if len(points) > 0]
            if not usable_points:
                return None
            all_points = np.vstack(usable_points)
            _, obstacle_mask_global, _ = self._classify_height_like_live(all_points, cam_aligned)
            split_idx = np.cumsum([len(p) for p in points_per_obs])
            split_idx = np.insert(split_idx, 0, 0)

            masks: Dict[int, np.ndarray] = {}
            dilate_px = max(0, int(self.config.forward_depth_obstacle_dilate_px))
            kernel = (
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
                if dilate_px > 0
                else None
            )
            for idx, obs in enumerate(valid_observations):
                depth_shape = tuple(int(v) for v in sample_data[idx]["depth_shape"])
                mask = np.zeros(depth_shape, dtype=bool)
                points = points_per_obs[idx]
                if len(points) > 0:
                    start = split_idx[idx]
                    end = split_idx[idx + 1]
                    obstacle_sample = obstacle_mask_global[start:end]
                    valid_sample = sample_data[idx]["valid"]
                    xs = sample_data[idx]["xs"][valid_sample]
                    ys = sample_data[idx]["ys"][valid_sample]
                    if len(xs) == len(obstacle_sample):
                        mask[ys[obstacle_sample], xs[obstacle_sample]] = True
                if kernel is not None and mask.any():
                    mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
                masks[id(obs)] = mask
            return masks
        except Exception as e:
            print(f"[Explore] Forward height filter unavailable; falling back to raw depth safety: {e}")
            return None

    def _save_forward_depth_heatmap(
        self,
        run_dir: Optional[Path],
        obs: Observation,
        station_idx: int,
        movement_count: int,
        roi_bounds: Tuple[int, int, int, int],
        valid_roi: np.ndarray,
        close_roi: np.ndarray,
        threshold_m: float,
        stats: Dict[str, object],
    ) -> Optional[str]:
        if run_dir is None or obs.depth is None:
            return None
        try:
            debug_dir = run_dir / "depth_safety"
            debug_dir.mkdir(parents=True, exist_ok=True)
            label = obs.frame_label or f"station_{station_idx:03d}"
            safe_label = re.sub(r"[^a-zA-Z0-9_.+-]+", "_", label)
            prefix = f"station_{station_idx:03d}_move_{movement_count:03d}_{safe_label}"

            depth_u8 = _normalize_to_u8(obs.depth)
            heat = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            overlay = cv2.cvtColor(obs.image_rgb, cv2.COLOR_RGB2BGR)
            if overlay.shape[:2] != heat.shape[:2]:
                overlay = cv2.resize(overlay, (heat.shape[1], heat.shape[0]), interpolation=cv2.INTER_AREA)
            blended = cv2.addWeighted(overlay, 0.45, heat, 0.55, 0)

            x0, y0, x1, y1 = roi_bounds
            cv2.rectangle(blended, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 255), 2)

            roi_overlay = blended[y0:y1, x0:x1]
            if roi_overlay.size and valid_roi.shape == close_roi.shape:
                close_map = np.zeros(valid_roi.shape, dtype=np.uint8)
                close_map[close_roi] = 255
                if close_map.any():
                    red = np.zeros_like(roi_overlay)
                    red[:, :] = (0, 0, 255)
                    mask = close_map > 0
                    roi_overlay[mask] = cv2.addWeighted(roi_overlay[mask], 0.25, red[mask], 0.75, 0)
                valid_edge = cv2.Canny((valid_roi.astype(np.uint8) * 255), 50, 150)
                roi_overlay[valid_edge > 0] = (0, 255, 0)

            text_lines = [
                f"decision: {'BLOCK' if stats.get('blocks_forward') else 'CLEAR'}",
                f"height_filter: {stats.get('height_filter_used')}",
                f"obstacle_px: {stats.get('obstacle_height_pixels')}",
                f"nearest_p05: {stats.get('nearest_depth_p05_m')}",
                f"close: {stats.get('close_pixels')}/{stats.get('valid_pixels')} ({stats.get('close_fraction'):.3f})",
                f"threshold: {threshold_m:.3f}",
                f"rel_close: {stats.get('relatively_close')}",
                f"scale_ok: {stats.get('metric_scale_looks_reliable')}",
            ]
            y_text = 22
            for line in text_lines:
                cv2.putText(blended, line, (12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(blended, line, (12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_text += 20

            heatmap_path = debug_dir / f"{prefix}_heatmap.png"
            stats_path = debug_dir / f"{prefix}_stats.json"
            cv2.imwrite(str(heatmap_path), blended)
            payload = {
                "station_idx": station_idx,
                "movement_count_before_action": movement_count,
                "frame_label": obs.frame_label,
                "roi_bounds_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "clearance_threshold_m": float(threshold_m),
                **stats,
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return str(heatmap_path)
        except Exception as e:
            print(f"[Explore] Could not save forward depth heatmap: {e}")
            return None

    def _forward_target(self, pose: Pose2D, distance_m: float) -> Tuple[float, float]:
        distance = max(0.0, float(distance_m))
        yaw_rad = math.radians(-pose.yaw)
        return (
            pose.x + distance * math.sin(yaw_rad),
            pose.z - distance * math.cos(yaw_rad),
        )

    def _forward_move_blocked(
        self,
        grid: OccupancyGrid,
        pose: Pose2D,
        target_x: float,
        target_z: float,
    ) -> Tuple[bool, str, Optional[Tuple[int, int]], Optional[float]]:
        origin = grid.world_to_cell(pose.x, pose.z)
        target = grid.world_to_cell(target_x, target_z)
        if origin is None:
            return True, "current pose outside occupancy map", None, None
        if target is None:
            return True, "forward target outside occupancy map", None, None

        inflate = int(math.ceil(grid.config.obstacle_inflation_m / grid.config.grid_res_m))
        blocked = _inflate_obstacles(grid.state, inflate)

        step_m = math.hypot(target_x - pose.x, target_z - pose.z)
        if step_m <= 1e-6:
            return True, "zero forward step", None, 0.0

        yaw_rad = math.radians(-pose.yaw)
        fwd_x = math.sin(yaw_rad)
        fwd_z = -math.cos(yaw_rad)
        right_x = math.cos(yaw_rad)
        right_z = math.sin(yaw_rad)
        corridor_half_width = max(grid.config.grid_res_m, grid.config.obstacle_inflation_m)
        nearest_block: Optional[Tuple[float, Tuple[int, int]]] = None
        nearest_unknown: Optional[Tuple[float, Tuple[int, int]]] = None

        min_x = min(pose.x, target_x) - corridor_half_width - grid.config.grid_res_m
        max_x = max(pose.x, target_x) + corridor_half_width + grid.config.grid_res_m
        min_z = min(pose.z, target_z) - corridor_half_width - grid.config.grid_res_m
        max_z = max(pose.z, target_z) + corridor_half_width + grid.config.grid_res_m

        min_cell = grid.world_to_cell(min_x, min_z)
        max_cell = grid.world_to_cell(max_x, max_z)
        if min_cell is not None and max_cell is not None:
            r0, c0 = min(min_cell[0], max_cell[0]), min(min_cell[1], max_cell[1])
            r1, c1 = max(min_cell[0], max_cell[0]), max(min_cell[1], max_cell[1])
        else:
            r0, c0 = 0, 0
            r1, c1 = grid.height - 1, grid.width - 1

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (
                    grid.config.require_known_free_forward_path
                    and grid.state[r, c] == UNKNOWN
                ):
                    cell_x, cell_z = grid.cell_to_world((r, c))
                    rel_x = cell_x - pose.x
                    rel_z = cell_z - pose.z
                    forward_dist = rel_x * fwd_x + rel_z * fwd_z
                    if forward_dist > 0.0 and forward_dist <= step_m:
                        lateral_dist = abs(rel_x * right_x + rel_z * right_z)
                        if lateral_dist <= corridor_half_width + 0.5 * grid.config.grid_res_m:
                            if nearest_unknown is None or forward_dist < nearest_unknown[0]:
                                nearest_unknown = (forward_dist, (r, c))

                if not blocked[r, c]:
                    continue
                cell_x, cell_z = grid.cell_to_world((r, c))
                rel_x = cell_x - pose.x
                rel_z = cell_z - pose.z
                forward_dist = rel_x * fwd_x + rel_z * fwd_z
                if forward_dist <= 0.0 or forward_dist > step_m:
                    continue
                lateral_dist = abs(rel_x * right_x + rel_z * right_z)
                if lateral_dist > corridor_half_width + 0.5 * grid.config.grid_res_m:
                    continue
                if nearest_block is None or forward_dist < nearest_block[0]:
                    nearest_block = (forward_dist, (r, c))

        if nearest_block is not None:
            dist, cell = nearest_block
            return True, "inflated obstacle within forward step", cell, float(dist)

        if grid.config.avoid_visited_forward and grid.visited[target] > 0:
            unknown_near_target = grid.has_unknown_near(target, grid.config.visited_revisit_unknown_radius_m)
            if not unknown_near_target:
                return True, "forward target already visited and has no nearby unknown frontier", target, float(step_m)

        if nearest_unknown is not None:
            dist, cell = nearest_unknown
            return True, "unknown cell within forward corridor", cell, float(dist)

        for cell in _bresenham(origin, target)[1:]:
            r, c = cell
            if blocked[r, c]:
                cell_x, cell_z = grid.cell_to_world(cell)
                return True, "inflated obstacle on forward ray", cell, float(
                    math.hypot(cell_x - pose.x, cell_z - pose.z)
                )
            if grid.config.require_known_free_forward_path and grid.state[r, c] == UNKNOWN:
                cell_x, cell_z = grid.cell_to_world(cell)
                return True, "unknown cell on forward ray", cell, float(
                    math.hypot(cell_x - pose.x, cell_z - pose.z)
                )
        return False, "", None, None

    def _rotate_by_degrees(self, pose: Pose2D, degrees: float, reason: str) -> None:
        degrees = float(degrees)
        if abs(degrees) < 1e-6:
            return
        self._raise_if_stopped()
        yaw = pose.yaw + degrees
        print(f"[Explore] Rotating {degrees:+.1f} deg ({reason}).")
        self.executor.call(
            "rotate_device",
            device="headset",
            pitch=pose.pitch,
            yaw=yaw,
            roll=pose.roll,
        )
        self._apply_controller_offsets()
        time.sleep(self.config.settle_seconds)
        self._raise_if_stopped()

    def _yaw_toward(self, pose: Pose2D, target_x: float, target_z: float) -> Optional[float]:
        dx = float(target_x) - pose.x
        dz = float(target_z) - pose.z
        if math.hypot(dx, dz) < max(self.config.grid_res_m, 1e-3):
            return None
        return -math.degrees(math.atan2(dx, -dz))

    def _rotate_toward_cell(self, grid: OccupancyGrid, cell: Tuple[int, int], reason: str) -> bool:
        pose = self._get_pose()
        target_x, target_z = grid.cell_to_world(cell)
        yaw = self._yaw_toward(pose, target_x, target_z)
        if yaw is None:
            return False
        yaw_delta = ((yaw - pose.yaw + 180.0) % 360.0) - 180.0
        if abs(yaw_delta) < 3.0:
            return False
        print(f"[Explore] Rotating {yaw_delta:+.1f} deg toward {reason}.")
        self.executor.call(
            "rotate_device",
            device="headset",
            pitch=pose.pitch,
            yaw=yaw,
            roll=pose.roll,
        )
        self._apply_controller_offsets()
        time.sleep(self.config.settle_seconds)
        return True

    def _lookahead_cell(
        self,
        grid: OccupancyGrid,
        path: Sequence[Tuple[int, int]],
        lookahead_m: float,
    ) -> Tuple[int, int]:
        if not path:
            raise ValueError("Cannot choose a lookahead cell from an empty path.")
        target_distance = max(0.0, float(lookahead_m))
        if target_distance <= 0.0 or len(path) == 1:
            return path[-1]
        accum = 0.0
        prev = path[0]
        for cell in path[1:]:
            accum += math.dist(grid.cell_to_world(prev), grid.cell_to_world(cell))
            if accum >= target_distance:
                return cell
            prev = cell
        return path[-1]

    def _capture_rgb_image(self) -> np.ndarray:
        res = self.executor.call("inspect_surroundings")
        if isinstance(res, str) and res.startswith("Error"):
            raise RuntimeError(res)
        data = json.loads(res).get("data")
        if not data:
            raise RuntimeError("inspect_surroundings returned no image data")
        raw = base64.b64decode(data)
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

    @staticmethod
    def _normalize_depth_engine(depth_engine: str) -> str:
        value = (depth_engine or "vggt").strip().lower().replace("-", "_")
        aliases = {
            "foundation": "fast_foundationstereo",
            "foundation_stereo": "fast_foundationstereo",
            "foundationstereo": "fast_foundationstereo",
            "fast_foundation_stereo": "fast_foundationstereo",
            "fast_foundationstereo": "fast_foundationstereo",
            "ffs": "fast_foundationstereo",
            "vggt": "vggt",
        }
        return aliases.get(value, value)

    def _depth_engine_label(self) -> str:
        engine = self._normalize_depth_engine(self.config.depth_engine)
        return "Fast-FoundationStereo" if engine == "fast_foundationstereo" else "VGGT"

    @staticmethod
    def _default_foundationstereo_repo() -> str:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            os.environ.get("OPENEYE_FOUNDATIONSTEREO_REPO", ""),
            os.environ.get("FOUNDATIONSTEREO_REPO", ""),
            str(root / "Fast-FoundationStereo"),
            str(root / "FoundationStereo"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).expanduser().exists():
                return str(Path(candidate).expanduser())
        return ""

    @staticmethod
    def _default_foundationstereo_checkpoint(repo: str) -> str:
        candidates = [
            os.environ.get("OPENEYE_FOUNDATIONSTEREO_CKPT", ""),
            os.environ.get("FOUNDATIONSTEREO_CKPT", ""),
        ]
        if repo:
            repo_path = Path(repo).expanduser()
            candidates.extend([
                str(repo_path / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"),
            ])
        for candidate in candidates:
            if candidate and Path(candidate).expanduser().exists():
                return str(Path(candidate).expanduser())
        return candidates[0] if candidates else ""

    @classmethod
    def preload_model(
        cls,
        model_name: str = "facebook/VGGT-1B",
        depth_engine: str = "vggt",
        foundationstereo_repo: str = "",
        foundationstereo_checkpoint: str = "",
        foundationstereo_valid_iters: int = 8,
        foundationstereo_max_disp: int = 192,
    ):
        engine = OccupancyExplorer._normalize_depth_engine(depth_engine)
        if engine == "fast_foundationstereo":
            repo = foundationstereo_repo or OccupancyExplorer._default_foundationstereo_repo()
            ckpt = foundationstereo_checkpoint or OccupancyExplorer._default_foundationstereo_checkpoint(repo)
            if (
                cls._shared_model is not None
                and cls._shared_depth_engine == engine
                and cls._shared_foundationstereo_repo == repo
                and cls._shared_foundationstereo_checkpoint == ckpt
            ):
                return cls._shared_model
            if not repo or not Path(repo).expanduser().exists():
                raise RuntimeError(
                    "Fast-FoundationStereo repo not found. Set "
                    "OPENEYE_FOUNDATIONSTEREO_REPO or pass foundationstereo_repo."
                )
            if not ckpt or not Path(ckpt).expanduser().exists():
                raise RuntimeError(
                    "Fast-FoundationStereo checkpoint not found. Set "
                    "OPENEYE_FOUNDATIONSTEREO_CKPT or pass foundationstereo_checkpoint."
                )
            import torch

            repo_path = Path(repo).expanduser().resolve()
            if str(repo_path) not in os.sys.path:
                os.sys.path.insert(0, str(repo_path))
            cls._shared_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._shared_dtype = torch.float16
            cls._shared_model = torch.load(str(Path(ckpt).expanduser()), map_location="cpu", weights_only=False)
            if hasattr(cls._shared_model, "args"):
                cls._shared_model.args.valid_iters = int(foundationstereo_valid_iters)
                cls._shared_model.args.max_disp = int(foundationstereo_max_disp)
            cls._shared_model.eval().to(cls._shared_device)
            cls._shared_model_name = Path(ckpt).name
            cls._shared_depth_engine = engine
            cls._shared_foundationstereo_repo = str(repo_path)
            cls._shared_foundationstereo_checkpoint = str(Path(ckpt).expanduser().resolve())
            return cls._shared_model

        if engine != "vggt":
            raise ValueError(f"Unsupported depth_engine: {depth_engine}")
        if (
            cls._shared_model is not None
            and cls._shared_depth_engine == engine
            and cls._shared_model_name == model_name
        ):
            return cls._shared_model
        import torch

        root = Path(__file__).resolve().parents[2]
        vggt_src = root / "vggt"
        if str(vggt_src) not in os.sys.path:
            os.sys.path.insert(0, str(vggt_src))
        from vggt.models.vggt import VGGT

        cls._shared_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls._shared_dtype = (
            torch.bfloat16
            if cls._shared_device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        hf_model_name = model_name if "/" in model_name else f"facebook/{model_name}"
        cls._shared_model = VGGT.from_pretrained(hf_model_name)
        cls._shared_model.eval().to(cls._shared_device)
        cls._shared_model_name = model_name
        cls._shared_depth_engine = engine
        cls._shared_foundationstereo_repo = None
        cls._shared_foundationstereo_checkpoint = None
        return cls._shared_model

    def _load_model(self):
        return self.preload_model(
            self.config.model_name,
            depth_engine=self.config.depth_engine,
            foundationstereo_repo=self.config.foundationstereo_repo,
            foundationstereo_checkpoint=self.config.foundationstereo_checkpoint,
            foundationstereo_valid_iters=self.config.foundationstereo_valid_iters,
            foundationstereo_max_disp=self.config.foundationstereo_max_disp,
        )

    def _update_grid_incremental(
        self,
        grid: OccupancyGrid,
        observations: Sequence[Observation],
        run_dir: Optional[Path] = None,
    ) -> Dict[str, object]:
        return self._classify_and_integrate_observations(
            grid=grid,
            run_dir=run_dir,
            integrate_observations=observations,
            debug_observations=observations,
            reset_grid=False,
        )

    def _rebuild_grid(
        self,
        grid: OccupancyGrid,
        run_dir: Optional[Path] = None,
        debug_observations: Optional[Sequence[Observation]] = None,
    ) -> Dict[str, object]:
        return self._classify_and_integrate_observations(
            grid=grid,
            run_dir=run_dir,
            integrate_observations=None,
            debug_observations=debug_observations,
            reset_grid=True,
        )

    def _classify_and_integrate_observations(
        self,
        grid: OccupancyGrid,
        run_dir: Optional[Path],
        integrate_observations: Optional[Sequence[Observation]],
        debug_observations: Optional[Sequence[Observation]],
        reset_grid: bool,
    ) -> Dict[str, object]:
        self._raise_if_stopped()
        if reset_grid:
            grid.state.fill(UNKNOWN)
            grid.state[grid.visited > 0] = FREE
            grid.state[grid.forced_free > 0] = FREE
        if not self.observations:
            return {"valid_observations": 0, "integrated_observations": 0}

        valid_observations = [
            obs for obs in self.observations
            if obs.depth is not None and obs.intrinsic is not None and obs.extrinsic is not None
        ]
        if not valid_observations:
            return {"valid_observations": 0, "integrated_observations": 0}
        source = valid_observations[0].depth_source or "vggt"

        if integrate_observations is None:
            integrate_ids = {id(obs) for obs in valid_observations}
        else:
            integrate_ids = {id(obs) for obs in integrate_observations}
        if debug_observations is None:
            debug_ids = integrate_ids
        else:
            debug_ids = {id(obs) for obs in debug_observations}

        conf_values = [obs.conf for obs in valid_observations if obs.conf is not None]
        conf_thr = (
            float(np.percentile(np.concatenate([c.reshape(-1) for c in conf_values]), self.config.conf_percentile))
            if conf_values
            else 0.0
        )

        recon = self._reconstruct_model_aligned_points(valid_observations, conf_thr)
        if recon is None:
            return {"valid_observations": len(valid_observations), "integrated_observations": 0}
        points_per_obs, cam_aligned, sample_data = recon

        all_points = np.vstack([p for p in points_per_obs if len(p) > 0])
        if len(all_points) == 0:
            return {"valid_observations": len(valid_observations), "integrated_observations": 0}

        floor_mask_global, obstacle_mask_global, class_stats = self._classify_height_like_live(
            all_points,
            cam_aligned,
        )
        split_idx = np.cumsum([len(p) for p in points_per_obs])
        split_idx = np.insert(split_idx, 0, 0)

        model_cam_xz = cam_aligned[:, [0, 2]].astype(np.float64)
        vr_cam_xz = np.array(
            [[obs.pose.x, obs.pose.z] for obs in valid_observations],
            dtype=np.float64,
        )
        scale, rotation, translation = _fit_similarity_2d(model_cam_xz, vr_cam_xz)
        fit_label = "Fast-FoundationStereo metric" if source.startswith("fast_foundationstereo") else "VGGT->VR"
        print(
            f"[Explore] {fit_label} map fit: scale={scale:.3f}, "
            f"translation=({translation[0]:.2f}, {translation[1]:.2f})"
        )

        stats: Dict[str, object] = {
            "valid_observations": len(valid_observations),
            "integrated_observations": 0,
            "input_relevant_points": 0,
            "unique_free_cells": 0,
            "unique_obstacle_cells": 0,
            "traced_free_rays": 0,
            "traced_obstacle_rays": 0,
            "max_rays_per_observation": int(self.config.max_rays_per_observation),
        }

        for idx, obs in enumerate(valid_observations):
            self._raise_if_stopped()
            pts = points_per_obs[idx]
            if len(pts) == 0:
                continue
            start = split_idx[idx]
            end = split_idx[idx + 1]
            floor_mask = floor_mask_global[start:end]
            obstacle_mask = obstacle_mask_global[start:end]
            world_xz = scale * (pts[:, [0, 2]].astype(np.float64) @ rotation.T) + translation

            if id(obs) in integrate_ids:
                world_points = np.column_stack((world_xz[:, 0], pts[:, 1], world_xz[:, 1])).astype(np.float32)
                obs_stats = self._integrate_classified_points(
                    grid=grid,
                    obs=obs,
                    world_x=world_xz[:, 0],
                    world_z=world_xz[:, 1],
                    free_mask=floor_mask,
                    obstacle_mask=obstacle_mask,
                )
                self._append_ply_points(world_points, floor_mask, obstacle_mask)
                stats["integrated_observations"] = int(stats["integrated_observations"]) + 1
                for key in (
                    "input_relevant_points",
                    "unique_free_cells",
                    "unique_obstacle_cells",
                    "traced_free_rays",
                    "traced_obstacle_rays",
                ):
                    stats[key] = int(stats[key]) + int(obs_stats.get(key, 0))

            if id(obs) in debug_ids and self.config.debug_output.lower() != "none":
                self._save_classification_debug(
                    debug_dir=(run_dir / "model_debug") if run_dir is not None else None,
                    obs=obs,
                    xs=sample_data[idx]["xs"],
                    ys=sample_data[idx]["ys"],
                    valid=sample_data[idx]["valid"],
                    free_mask=floor_mask,
                    obstacle_mask=obstacle_mask,
                    floor_y=class_stats["floor_y"],
                    camera_height=class_stats["camera_height"],
                    floor_top=class_stats["floor_top"],
                    obstacle_low=class_stats["obstacle_low"],
                    obstacle_high=class_stats["obstacle_high"],
                    conf_thr=conf_thr,
                    depth_shape=sample_data[idx]["depth_shape"],
                )
            self._raise_if_stopped()
        grid.state[grid.visited > 0] = FREE
        grid.state[grid.forced_free > 0] = FREE
        return stats

    def _infer_observations(
        self,
        observations: Sequence[Observation],
        run_dir: Optional[Path] = None,
        force: bool = False,
    ) -> None:
        self._raise_if_stopped()
        engine = self._normalize_depth_engine(self.config.depth_engine)
        if engine == "fast_foundationstereo":
            self._infer_foundationstereo_observations(observations, run_dir=run_dir, force=force)
            return
        if engine != "vggt":
            raise ValueError(f"Unsupported depth_engine: {self.config.depth_engine}")
        pending = [
            obs for obs in observations
            if force or obs.depth is None or obs.intrinsic is None or obs.extrinsic is None
        ]
        if not pending:
            return
        model = self._load_model()
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self._raise_if_stopped()
        with tempfile.TemporaryDirectory(prefix="openeye_vggt_frames_") as tmp_dir:
            frame_paths = []
            for idx, obs in enumerate(pending):
                path = Path(tmp_dir) / f"frame_{idx:04d}.png"
                Image.fromarray(obs.image_rgb).save(path)
                frame_paths.append(str(path))

            images = load_and_preprocess_images(frame_paths).to(self._shared_device)
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=self._shared_dtype)
                if self._shared_device is not None and self._shared_device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad():
                with autocast_ctx:
                    predictions = model(images)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                images.shape[-2:],
            )
            predictions["extrinsic"] = extrinsics
            predictions["intrinsic"] = intrinsics

            for key in list(predictions.keys()):
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].detach().cpu().numpy().squeeze(0)

        self._raise_if_stopped()
        depth = predictions["depth"]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        conf = predictions.get("depth_conf")
        if conf is not None and conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]

        processed_images = predictions.get("images")
        if processed_images is not None:
            processed_images = (processed_images.transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8)

        for idx, obs in enumerate(pending):
            obs.depth = depth[idx]
            obs.intrinsic = predictions["intrinsic"][idx]
            obs.extrinsic = predictions["extrinsic"][idx]
            obs.conf = conf[idx] if conf is not None else None
            if run_dir is not None and self.config.debug_output.lower() == "full":
                self._save_model_debug_images(
                    debug_dir=run_dir / "model_debug",
                    obs=obs,
                    processed_image=processed_images[idx] if processed_images is not None else None,
                )

    def _infer_foundationstereo_observations(
        self,
        observations: Sequence[Observation],
        run_dir: Optional[Path] = None,
        force: bool = False,
    ) -> None:
        pending = [
            obs for obs in observations
            if force or obs.depth is None or obs.intrinsic is None or obs.extrinsic is None
        ]
        if not pending:
            return

        pairs: Dict[str, Dict[str, Observation]] = {}
        for obs in pending:
            label = obs.frame_label
            if label.endswith("_left"):
                pairs.setdefault(label[:-5], {})["left"] = obs
            elif label.endswith("_right"):
                pairs.setdefault(label[:-6], {})["right"] = obs

        if not pairs:
            raise RuntimeError(
                "Fast-FoundationStereo needs split stereo captures. "
                "Keep split_stereo_capture=True and provide side-by-side stereo images."
            )

        model = self._load_model()
        for pair_label, pair in pairs.items():
            self._raise_if_stopped()
            left = pair.get("left")
            right = pair.get("right")
            if left is None or right is None:
                raise RuntimeError(f"Missing stereo mate for Fast-FoundationStereo frame '{pair_label}'.")

            depth, disparity, intrinsic = self._infer_foundationstereo_pair(
                model,
                left.image_rgb,
                right.image_rgb,
            )
            left.depth = depth
            left.intrinsic = intrinsic
            left.extrinsic = np.eye(4, dtype=np.float32)
            left.conf = None
            left.depth_source = "fast_foundationstereo"

            right.depth = None
            right.intrinsic = None
            right.extrinsic = None
            right.conf = None
            right.depth_source = "fast_foundationstereo_right_reference"

            if run_dir is not None and self.config.debug_output.lower() == "full":
                self._save_model_debug_images(
                    debug_dir=run_dir / "model_debug",
                    obs=left,
                    processed_image=None,
                )
                debug_dir = run_dir / "model_debug"
                label = left.frame_label or f"station_{left.station_idx:03d}_foundationstereo"
                np.save(debug_dir / f"{label}_disparity.npy", disparity)
                disp_u8 = _normalize_to_u8(disparity)
                disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)
                cv2.imwrite(str(debug_dir / f"{label}_disparity_colormap.png"), disp_color)

    def _infer_foundationstereo_pair(
        self,
        model,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        from core.utils.utils import InputPadder

        h0, w0 = left_rgb.shape[:2]
        if right_rgb.shape[:2] != (h0, w0):
            right_rgb = cv2.resize(right_rgb, (w0, h0), interpolation=cv2.INTER_AREA)

        scale = max(0.05, float(self.config.foundationstereo_scale))
        if abs(scale - 1.0) > 1e-6:
            left_in = cv2.resize(left_rgb, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right_in = cv2.resize(right_rgb, (left_in.shape[1], left_in.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            left_in = left_rgb
            right_in = right_rgb
        h, w = left_in.shape[:2]

        device = self._shared_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img0 = torch.as_tensor(left_in, device=device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_in, device=device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        use_cuda = device.type == "cuda"
        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=True, dtype=torch.float16)
            if use_cuda
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                if self.config.foundationstereo_hierarchical and hasattr(model, "run_hierachical"):
                    disp = model.run_hierachical(
                        img0,
                        img1,
                        iters=int(self.config.foundationstereo_valid_iters),
                        test_mode=True,
                        small_ratio=0.5,
                    )
                else:
                    disp = model.forward(
                        img0,
                        img1,
                        iters=int(self.config.foundationstereo_valid_iters),
                        test_mode=True,
                        optimize_build_volume="pytorch1",
                    )
        disp = padder.unpad(disp.float())
        disparity = disp.detach().cpu().numpy().reshape(h, w).astype(np.float32)
        disparity = np.clip(disparity, 0.0, None)

        if self.config.foundationstereo_remove_invisible:
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            disparity[(xx - disparity) < 0] = np.nan

        intrinsic_scaled = self._estimate_pinhole_intrinsic(h, w)
        min_disp = max(1e-6, float(self.config.foundationstereo_min_disparity_px))
        valid = np.isfinite(disparity) & (disparity >= min_disp)
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        depth[valid] = (
            float(intrinsic_scaled[0, 0])
            * max(1e-6, float(self.config.stereo_eye_separation_m))
            / disparity[valid]
        ).astype(np.float32)

        if (h, w) != (h0, w0):
            depth = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_NEAREST)
            disparity = cv2.resize(disparity, (w0, h0), interpolation=cv2.INTER_NEAREST)
        intrinsic = self._estimate_pinhole_intrinsic(h0, w0)
        return depth, disparity, intrinsic

    def _estimate_pinhole_intrinsic(self, h: int, w: int) -> np.ndarray:
        fov = max(1.0, min(179.0, float(self.config.fov_degrees)))
        fx = (0.5 * float(w)) / math.tan(0.5 * math.radians(fov))
        fy = fx
        cx = 0.5 * (float(w) - 1.0)
        cy = 0.5 * (float(h) - 1.0)
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _save_model_debug_images(
        self,
        debug_dir: Path,
        obs: Observation,
        processed_image: Optional[np.ndarray],
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        label = obs.frame_label or f"station_{obs.station_idx:03d}_yaw_{int(obs.yaw_offset):+04d}"

        Image.fromarray(obs.image_rgb).save(debug_dir / f"{label}_input_rgb.png")
        if processed_image is not None:
            proc = np.asarray(processed_image)
            if proc.dtype != np.uint8:
                proc = np.clip(proc, 0, 255).astype(np.uint8)
            Image.fromarray(proc).save(debug_dir / f"{label}_vggt_processed_rgb.png")

        if obs.depth is not None:
            depth_u8 = _normalize_to_u8(obs.depth)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            cv2.imwrite(str(debug_dir / f"{label}_depth_colormap.png"), depth_color)
            np.save(debug_dir / f"{label}_depth.npy", obs.depth)

        if obs.conf is not None:
            conf_u8 = _normalize_to_u8(obs.conf)
            conf_color = cv2.applyColorMap(conf_u8, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(str(debug_dir / f"{label}_confidence_colormap.png"), conf_color)
            np.save(debug_dir / f"{label}_confidence.npy", obs.conf)

        if obs.intrinsic is not None:
            np.save(debug_dir / f"{label}_intrinsics.npy", obs.intrinsic)

        if obs.extrinsic is not None:
            np.save(debug_dir / f"{label}_extrinsics.npy", obs.extrinsic)

    def _reconstruct_model_aligned_points(
        self,
        observations: Sequence[Observation],
        conf_thr: float,
    ) -> Optional[Tuple[List[np.ndarray], np.ndarray, List[Dict[str, np.ndarray]]]]:
        if observations and (observations[0].depth_source or "").startswith("fast_foundationstereo"):
            return self._reconstruct_vr_metric_points(observations, conf_thr)

        points_raw: List[np.ndarray] = []
        sample_data: List[Dict[str, np.ndarray]] = []

        for obs in observations:
            self._raise_if_stopped()
            assert obs.depth is not None
            assert obs.intrinsic is not None
            assert obs.extrinsic is not None

            depth = obs.depth
            h, w = depth.shape
            stride = max(1, int(self.config.depth_stride))
            ys, xs = np.mgrid[0:h:stride, 0:w:stride]
            z_cam = depth[ys, xs].astype(np.float64)
            valid = np.isfinite(z_cam) & (z_cam >= self.config.min_depth_m) & (z_cam <= self.config.max_depth_m)
            if obs.conf is not None:
                valid &= obs.conf[ys, xs] >= conf_thr

            sample_data.append({
                "xs": xs,
                "ys": ys,
                "valid": valid,
                "depth_shape": np.array(depth.shape, dtype=np.int32),
            })

            if not np.any(valid):
                points_raw.append(np.zeros((0, 3), dtype=np.float32))
                continue

            pix = np.stack(
                [
                    xs[valid].astype(np.float64),
                    ys[valid].astype(np.float64),
                    np.ones(int(np.count_nonzero(valid)), dtype=np.float64),
                ],
                axis=0,
            )
            k_inv = np.linalg.inv(obs.intrinsic.astype(np.float64))
            c2w = np.linalg.inv(_as_homogeneous44(obs.extrinsic.astype(np.float64)))
            rays = k_inv @ pix
            x_cam = rays * z_cam[valid][None, :]
            x_cam_h = np.vstack([x_cam, np.ones((1, x_cam.shape[1]))])
            points_raw.append((c2w @ x_cam_h)[:3].T.astype(np.float32))

        usable = [p for p in points_raw if len(p) > 0]
        if not usable:
            return None

        all_raw = np.vstack(usable)
        first_ext = _as_homogeneous44(observations[0].extrinsic.astype(np.float64))
        axis_flip = np.diag([1.0, -1.0, -1.0, 1.0])
        center = np.median(all_raw, axis=0)
        recenter = np.eye(4, dtype=np.float64)
        recenter[:3, 3] = -center
        align = recenter @ (axis_flip @ first_ext)

        def _apply_align(points: np.ndarray) -> np.ndarray:
            if len(points) == 0:
                return points
            hpts = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
            return (align @ hpts.T).T[:, :3].astype(np.float32)

        points_aligned = [_apply_align(points) for points in points_raw]
        cam_centers = []
        for obs in observations:
            self._raise_if_stopped()
            c2w = np.linalg.inv(_as_homogeneous44(obs.extrinsic.astype(np.float64)))
            cam_centers.append((c2w @ [0, 0, 0, 1])[:3])
        cam_aligned = _apply_align(np.asarray(cam_centers, dtype=np.float32))
        return points_aligned, cam_aligned, sample_data

    def _reconstruct_vr_metric_points(
        self,
        observations: Sequence[Observation],
        conf_thr: float,
    ) -> Optional[Tuple[List[np.ndarray], np.ndarray, List[Dict[str, np.ndarray]]]]:
        points_per_obs: List[np.ndarray] = []
        sample_data: List[Dict[str, np.ndarray]] = []
        cam_points: List[np.ndarray] = []

        for obs in observations:
            self._raise_if_stopped()
            assert obs.depth is not None
            assert obs.intrinsic is not None
            depth = obs.depth
            h, w = depth.shape
            stride = max(1, int(self.config.depth_stride))
            ys, xs = np.mgrid[0:h:stride, 0:w:stride]
            z_cam = depth[ys, xs].astype(np.float64)
            valid = np.isfinite(z_cam) & (z_cam >= self.config.min_depth_m) & (z_cam <= self.config.max_depth_m)
            if obs.conf is not None:
                valid &= obs.conf[ys, xs] >= conf_thr

            sample_data.append({
                "xs": xs,
                "ys": ys,
                "valid": valid,
                "depth_shape": np.array(depth.shape, dtype=np.int32),
            })

            cam_origin = np.array([obs.pose.x, obs.pose.y, obs.pose.z], dtype=np.float64)
            cam_points.append(cam_origin.astype(np.float32))
            if not np.any(valid):
                points_per_obs.append(np.zeros((0, 3), dtype=np.float32))
                continue

            k = obs.intrinsic.astype(np.float64)
            x_cam = (xs[valid].astype(np.float64) - k[0, 2]) * z_cam[valid] / k[0, 0]
            y_down = (ys[valid].astype(np.float64) - k[1, 2]) * z_cam[valid] / k[1, 1]
            forward = z_cam[valid]

            yaw_rad = math.radians(-(obs.pose.yaw + obs.yaw_offset))
            right_x = math.cos(yaw_rad)
            right_z = math.sin(yaw_rad)
            fwd_x = math.sin(yaw_rad)
            fwd_z = -math.cos(yaw_rad)
            world_x = cam_origin[0] + x_cam * right_x + forward * fwd_x
            world_y = cam_origin[1] - y_down
            world_z = cam_origin[2] + x_cam * right_z + forward * fwd_z
            points = np.column_stack((world_x, world_y, world_z)).astype(np.float32)
            points_per_obs.append(points)

        if not any(len(points) > 0 for points in points_per_obs):
            return None
        return points_per_obs, np.asarray(cam_points, dtype=np.float32), sample_data

    def _classify_height_like_live(
        self,
        points: np.ndarray,
        cam_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        y_range = np.percentile(points[:, 1], 95) - np.percentile(points[:, 1], 5)
        y_range = max(float(y_range), 1e-3)
        min_ratio = max(0.0, float(self.config.obstacle_min_height_ratio))
        max_ratio = max(min_ratio, float(self.config.obstacle_max_height_ratio))

        def _bounds(camera_height: float) -> Tuple[float, float, float, float]:
            if not np.isfinite(camera_height) or camera_height <= 1e-3:
                camera_height = y_range
            floor_top = 0.10 * camera_height
            obstacle_low = min_ratio * camera_height
            obstacle_high = max_ratio * camera_height
            return floor_top, obstacle_low, obstacle_high, camera_height

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
            normal = np.array([a, b, c], dtype=np.float64)
            if normal[1] < 0:
                normal, d = -normal, -d
            if normal[1] >= 0.5:
                distances = np.dot(points, normal) + d
                floor_offset = float(np.percentile(distances, 5))
                heights = distances - floor_offset
                cam_heights = np.dot(cam_points, normal) + d - floor_offset
                cam_heights = cam_heights[np.isfinite(cam_heights) & (cam_heights > 0)]
                camera_height = float(np.median(cam_heights)) if len(cam_heights) else np.nan
                floor_top, obstacle_low, obstacle_high, camera_height = _bounds(camera_height)
                print(
                    f"[Explore] Camera height estimate: {camera_height:.3f} m; "
                    f"obstacle band: {obstacle_low:.3f}-{obstacle_high:.3f} m above floor "
                    f"({min_ratio:.2f}-{max_ratio:.2f}x camera height)"
                )
                floor_mask = (heights >= -0.2 * floor_top) & (heights < floor_top)
                if self.config.treat_above_obstacle_band_as_blocking:
                    obstacle_mask = heights >= obstacle_low
                else:
                    obstacle_mask = (heights >= obstacle_low) & (heights <= obstacle_high)
                return floor_mask, obstacle_mask, {
                    "floor_y": 0.0,
                    "camera_height": camera_height,
                    "floor_top": floor_top,
                    "obstacle_low": obstacle_low,
                    "obstacle_high": obstacle_high,
                }
        except ImportError:
            pass

        y = points[:, 1]
        floor_y = float(np.percentile(y, 5))
        cam_heights = cam_points[:, 1] - floor_y
        cam_heights = cam_heights[np.isfinite(cam_heights) & (cam_heights > 0)]
        camera_height = float(np.median(cam_heights)) if len(cam_heights) else np.nan
        floor_top_rel, obstacle_low_rel, obstacle_high_rel, camera_height = _bounds(camera_height)
        print(
            f"[Explore] Camera height estimate: {camera_height:.3f} m; "
            f"obstacle band: {obstacle_low_rel:.3f}-{obstacle_high_rel:.3f} m above floor "
            f"({min_ratio:.2f}-{max_ratio:.2f}x camera height)"
        )
        heights = y - floor_y
        floor_mask = (heights >= -0.2 * floor_top_rel) & (heights < floor_top_rel)
        if self.config.treat_above_obstacle_band_as_blocking:
            obstacle_mask = heights >= obstacle_low_rel
        else:
            obstacle_mask = (heights >= obstacle_low_rel) & (heights <= obstacle_high_rel)
        return floor_mask, obstacle_mask, {
            "floor_y": floor_y,
            "camera_height": camera_height,
            "floor_top": floor_y + floor_top_rel,
            "obstacle_low": floor_y + obstacle_low_rel,
            "obstacle_high": floor_y + obstacle_high_rel,
        }

    def _integrate_classified_points(
        self,
        grid: OccupancyGrid,
        obs: Observation,
        world_x: np.ndarray,
        world_z: np.ndarray,
        free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
    ) -> Dict[str, int]:
        origin = grid.world_to_cell(obs.pose.x, obs.pose.z)
        if origin is None:
            return {
                "input_relevant_points": 0,
                "unique_free_cells": 0,
                "unique_obstacle_cells": 0,
                "traced_free_rays": 0,
                "traced_obstacle_rays": 0,
            }

        relevant = free_mask | obstacle_mask
        relevant_count = int(np.count_nonzero(relevant))
        if relevant_count == 0:
            return {
                "input_relevant_points": 0,
                "unique_free_cells": 0,
                "unique_obstacle_cells": 0,
                "traced_free_rays": 0,
                "traced_obstacle_rays": 0,
            }

        xs = world_x[relevant].astype(np.float64)
        zs = world_z[relevant].astype(np.float64)
        free = free_mask[relevant]
        obstacle = obstacle_mask[relevant].copy()
        distance_from_headset = np.hypot(xs - obs.pose.x, zs - obs.pose.z)
        obstacle &= distance_from_headset >= self.config.obstacle_min_distance_m
        free &= ~obstacle

        cols = np.floor((xs - grid.x_min) / grid.config.grid_res_m).astype(np.int32)
        rows = np.floor((zs - grid.z_min) / grid.config.grid_res_m).astype(np.int32)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        if not np.any(inside):
            return {
                "input_relevant_points": relevant_count,
                "unique_free_cells": 0,
                "unique_obstacle_cells": 0,
                "traced_free_rays": 0,
                "traced_obstacle_rays": 0,
            }

        keys = (rows[inside].astype(np.int64) * grid.width) + cols[inside].astype(np.int64)
        obstacle_keys = np.unique(keys[obstacle[inside]])
        free_keys = np.unique(keys[free[inside]])
        if obstacle_keys.size and free_keys.size:
            free_keys = np.setdiff1d(free_keys, obstacle_keys, assume_unique=True)

        max_rays = int(self.config.max_rays_per_observation)
        if max_rays > 0:
            if obstacle_keys.size >= max_rays:
                obstacle_keys = self._evenly_sample_keys(obstacle_keys, max_rays)
                free_keys = np.zeros((0,), dtype=np.int64)
            else:
                remaining = max_rays - obstacle_keys.size
                free_keys = self._evenly_sample_keys(free_keys, remaining)

        def _trace_keys(endpoint_keys: np.ndarray, mark_obstacle: bool) -> int:
            traced = 0
            for idx, key in enumerate(endpoint_keys):
                if idx % 2048 == 0:
                    self._raise_if_stopped()
                er = int(key // grid.width)
                ec = int(key % grid.width)
                ray = _bresenham(origin, (er, ec))
                if len(ray) > 1:
                    for r, c in ray[:-1]:
                        if grid.state[r, c] != OBSTACLE:
                            grid.state[r, c] = FREE
                if mark_obstacle:
                    grid.state[er, ec] = OBSTACLE
                elif grid.state[er, ec] != OBSTACLE:
                    grid.state[er, ec] = FREE
                traced += 1
            return traced

        traced_free = _trace_keys(free_keys, mark_obstacle=False)
        traced_obstacle = _trace_keys(obstacle_keys, mark_obstacle=True)
        return {
            "input_relevant_points": relevant_count,
            "unique_free_cells": int(free_keys.size),
            "unique_obstacle_cells": int(obstacle_keys.size),
            "traced_free_rays": traced_free,
            "traced_obstacle_rays": traced_obstacle,
        }

    @staticmethod
    def _evenly_sample_keys(keys: np.ndarray, limit: int) -> np.ndarray:
        if limit <= 0 or keys.size == 0:
            return np.zeros((0,), dtype=np.int64)
        if keys.size <= limit:
            return keys
        indices = np.linspace(0, keys.size - 1, num=limit, dtype=np.int64)
        return keys[indices]

    def _append_ply_points(
        self,
        points: np.ndarray,
        free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
    ) -> None:
        if not self.config.export_ply:
            return
        max_points = int(self.config.max_ply_points)
        if max_points <= 0 or self._ply_point_count >= max_points or len(points) == 0:
            return
        valid = np.isfinite(points).all(axis=1)
        if not np.any(valid):
            return
        points = points[valid]
        free = free_mask[valid]
        obstacle = obstacle_mask[valid]
        remaining = max_points - self._ply_point_count
        if len(points) > remaining:
            idx = np.linspace(0, len(points) - 1, num=remaining, dtype=np.int64)
            points = points[idx]
            free = free[idx]
            obstacle = obstacle[idx]
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:] = (220, 180, 40)
        colors[free] = (245, 245, 245)
        colors[obstacle] = (210, 35, 35)
        self._ply_point_chunks.append(points.astype(np.float32))
        self._ply_color_chunks.append(colors)
        self._ply_point_count += len(points)

    def _save_ply(self, path: Path) -> Optional[Path]:
        if not self.config.export_ply:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        points, colors = self._build_ply_arrays_from_observations()
        if points is None or colors is None:
            if self._ply_point_count <= 0:
                return None
            points = np.vstack(self._ply_point_chunks)
            colors = np.vstack(self._ply_color_chunks)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point, color in zip(points, colors):
                f.write(
                    f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        print(f"[Explore] Wrote PLY point cloud: {path}")
        return path

    def _build_ply_arrays_from_observations(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        valid_observations = [
            obs for obs in self.observations
            if obs.depth is not None and obs.intrinsic is not None and obs.extrinsic is not None
        ]
        if not valid_observations:
            return None, None

        saved_stop_event = self._stop_event
        self._stop_event = None
        try:
            conf_values = [obs.conf for obs in valid_observations if obs.conf is not None]
            conf_thr = (
                float(np.percentile(np.concatenate([c.reshape(-1) for c in conf_values]), self.config.conf_percentile))
                if conf_values
                else 0.0
            )
            recon = self._reconstruct_model_aligned_points(valid_observations, conf_thr)
            if recon is None:
                return None, None
            points_per_obs, cam_aligned, _ = recon
            all_points = np.vstack([p for p in points_per_obs if len(p) > 0])
            if len(all_points) == 0:
                return None, None
            floor_mask_global, obstacle_mask_global, _ = self._classify_height_like_live(all_points, cam_aligned)

            model_cam_xz = cam_aligned[:, [0, 2]].astype(np.float64)
            vr_cam_xz = np.array(
                [[obs.pose.x, obs.pose.z] for obs in valid_observations],
                dtype=np.float64,
            )
            scale, rotation, translation = _fit_similarity_2d(model_cam_xz, vr_cam_xz)
            split_idx = np.cumsum([len(p) for p in points_per_obs])
            split_idx = np.insert(split_idx, 0, 0)

            point_chunks = []
            color_chunks = []
            point_count = 0
            max_points = int(self.config.max_ply_points)
            if max_points <= 0:
                return None, None

            for idx, pts in enumerate(points_per_obs):
                if len(pts) == 0 or point_count >= max_points:
                    continue
                start = split_idx[idx]
                end = split_idx[idx + 1]
                floor_mask = floor_mask_global[start:end]
                obstacle_mask = obstacle_mask_global[start:end]
                world_xz = scale * (pts[:, [0, 2]].astype(np.float64) @ rotation.T) + translation
                world_points = np.column_stack((world_xz[:, 0], pts[:, 1], world_xz[:, 1])).astype(np.float32)
                colors = np.zeros((len(world_points), 3), dtype=np.uint8)
                colors[:] = (220, 180, 40)
                colors[floor_mask] = (245, 245, 245)
                colors[obstacle_mask] = (210, 35, 35)

                remaining = max_points - point_count
                if len(world_points) > remaining:
                    sample_idx = np.linspace(0, len(world_points) - 1, num=remaining, dtype=np.int64)
                    world_points = world_points[sample_idx]
                    colors = colors[sample_idx]
                point_chunks.append(world_points)
                color_chunks.append(colors)
                point_count += len(world_points)

            if not point_chunks:
                return None, None
            return np.vstack(point_chunks), np.vstack(color_chunks)
        except Exception as e:
            print(f"[Explore] Could not rebuild final PLY from observations; using incremental points: {e}")
            return None, None
        finally:
            self._stop_event = saved_stop_event

    def _save_classification_debug(
        self,
        debug_dir: Optional[Path],
        obs: Observation,
        xs: np.ndarray,
        ys: np.ndarray,
        valid: np.ndarray,
        free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        floor_y: float,
        camera_height: float,
        floor_top: float,
        obstacle_low: float,
        obstacle_high: float,
        conf_thr: float,
        depth_shape: Tuple[int, int],
    ) -> None:
        if debug_dir is None:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        label = obs.frame_label or f"station_{obs.station_idx:03d}_yaw_{int(obs.yaw_offset):+04d}"

        valid_y = ys[valid]
        valid_x = xs[valid]
        ignored_mask = ~(free_mask | obstacle_mask)

        if self.config.debug_output.lower() == "full":
            depth_h, depth_w = depth_shape
            overlay = cv2.resize(obs.image_rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
            colors = np.zeros_like(overlay)
            colors[valid_y[free_mask], valid_x[free_mask]] = (255, 255, 255)
            colors[valid_y[obstacle_mask], valid_x[obstacle_mask]] = (140, 0, 0)
            colors[valid_y[ignored_mask], valid_x[ignored_mask]] = (0, 180, 255)
            mask = np.any(colors > 0, axis=2)
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * colors[mask]).astype(np.uint8)
            Image.fromarray(overlay).save(debug_dir / f"{label}_classification_overlay.png")

        stats = {
            "station_idx": obs.station_idx,
            "yaw_offset": obs.yaw_offset,
            "pose": asdict(obs.pose),
            "valid_sample_count": int(np.count_nonzero(valid)),
            "free_sample_count": int(np.count_nonzero(free_mask)),
            "obstacle_sample_count": int(np.count_nonzero(obstacle_mask)),
            "ignored_valid_sample_count": int(np.count_nonzero(ignored_mask)),
            "floor_y": floor_y,
            "camera_height": camera_height,
            "floor_top": floor_top,
            "obstacle_low": obstacle_low,
            "obstacle_high": obstacle_high,
            "treat_above_obstacle_band_as_blocking": self.config.treat_above_obstacle_band_as_blocking,
            "obstacle_min_distance_m": self.config.obstacle_min_distance_m,
            "confidence_threshold": conf_thr,
            "depth_stride": self.config.depth_stride,
        }
        with open(debug_dir / f"{label}_classification_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def _path_to_frontier(self, grid: OccupancyGrid, start: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        unknown = (grid.state == UNKNOWN).astype(np.float32)
        radius = max(1, int(math.ceil(self.config.frontier_unknown_radius_m / self.config.grid_res_m)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)).astype(np.float32)
        unknown_density = cv2.filter2D(unknown, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)

        frontier_goals = []
        for r in range(1, grid.height - 1):
            for c in range(1, grid.width - 1):
                if grid.state[r, c] != FREE:
                    continue
                unknown_neighbors = any(
                    grid.state[nr, nc] == UNKNOWN
                    for nr, nc in _neighbors4((r, c))
                    if 0 <= nr < grid.height and 0 <= nc < grid.width
                )
                if unknown_neighbors:
                    dist = abs(start[0] - r) + abs(start[1] - c)
                    novelty = float(unknown_density[r, c])
                    visited_penalty = float(self.config.frontier_visited_penalty) if grid.visited[r, c] else 0.0
                    visited_penalty += 6.0 * grid.visited_fraction_near(
                        (r, c),
                        self.config.visited_revisit_unknown_radius_m,
                    )
                    score = dist + visited_penalty - self.config.frontier_novelty_weight * novelty
                    frontier_goals.append((score, dist, novelty, (r, c)))

        frontier_goals.sort(key=lambda item: (item[0], item[1], -item[2]))
        for _, _, _, goal in frontier_goals[: max(1, int(self.config.max_frontier_candidates))]:
            path = _astar(grid, start, goal)
            if path and len(path) > 1:
                return path
        return None

    def _sample_path_by_distance(
        self,
        grid: OccupancyGrid,
        path: Sequence[Tuple[int, int]],
        spacing_m: float,
    ) -> List[Tuple[int, int]]:
        if len(path) <= 1:
            return []
        sampled = []
        accum = 0.0
        prev = path[0]
        for cell in path[1:]:
            accum += math.dist(grid.cell_to_world(prev), grid.cell_to_world(cell))
            if accum >= spacing_m:
                sampled.append(cell)
                accum = 0.0
            prev = cell
        if not sampled:
            sampled.append(path[-1])
        return sampled

    def _walk_cells(self, grid: OccupancyGrid, cells: Sequence[Tuple[int, int]]) -> None:
        for cell in cells:
            self._raise_if_stopped()
            x, z = grid.cell_to_world(cell)
            self.executor.call("walk_path", x=x, z=z, steps=10)
            self._apply_controller_offsets()
            grid.mark_visited(x, z)
            time.sleep(self.config.settle_seconds)
            self._raise_if_stopped()
