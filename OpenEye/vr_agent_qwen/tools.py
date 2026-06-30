"""
vr_agent/tools.py
-----------------
_get_tools(): factory that returns all agent tool functions as a list of callables.

Tools are grouped into:
  - Vision / Grounding
  - Object Tracking & Visual Servoing
  - White Cane Accessibility
  - Controller Inputs
  - Movement & Orientation
  - Utility
"""

import io
import json
import math
import os
import re
import time
import base64
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from PIL import Image

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import base64

from .config import LOG_DIR
from .logger import get_logger
from .occupancy_explorer import OccupancyExploreConfig, OccupancyExplorer

# ── Module-level references (set by _get_tools) ───────────────────────────────
_executor = None
_grounder = None
_tracker = None
_white_cane = None
_describer = None
_agent_ref = None


def _get_tools(executor, grounder, tracker, white_cane, describer, agent_ref):
    """
    Initialise module-level references and return all tool functions as a list.
    Call once during QwenAgent.__init__.
    """
    global _executor, _grounder, _tracker, _white_cane, _describer, _agent_ref
    _executor = executor
    _grounder = grounder
    _tracker = tracker
    _white_cane = white_cane
    _describer = describer
    _agent_ref = agent_ref

    mask_api_url = os.getenv("SAM3_MASK_API_URL", "http://127.0.0.1:8010").strip().rstrip("/")
    segmentation_backend = os.getenv(
        "SEGMENTATION_BACKEND", "api" if mask_api_url else "sam"
    ).strip().lower()
    if segmentation_backend not in {"sam", "api"}:
        segmentation_backend = "sam"

    def _using_api_backend() -> bool:
        return segmentation_backend == "api"

    # ── Shared helper ─────────────────────────────────────────────────────────

    def _log_action(tool_name, **kwargs):
        get_logger().info(f"[TOOL] {tool_name}({kwargs})")
        print(f"Action: {tool_name} {kwargs}")

    def _should_use_mss_capture() -> bool:
        return bool(_agent_ref and getattr(_agent_ref, "use_mss_capture", False))

    def _capture_mss_frame():
        """Capture a frame from the desktop using MSS and return (PIL image, jpeg bytes)."""
        try:
            import mss
        except ImportError as e:
            raise RuntimeError(
                "mss is required for visual servoing. Install it with `pip install mss`."
            ) from e

        with mss.mss() as sct:
            monitors = sct.monitors
            if not monitors:
                raise RuntimeError("No monitor found for MSS capture.")

            def _encode_raw(raw_img):
                pil_img_local = Image.frombytes("RGB", raw_img.size, raw_img.rgb)
                out_local = io.BytesIO()
                pil_img_local.save(out_local, format="JPEG", quality=95)
                return pil_img_local, out_local.getvalue()

            # Try region first if one was detected at startup.
            if _agent_ref and hasattr(_agent_ref, "mss_capture_region") and _agent_ref.mss_capture_region:
                region = dict(_agent_ref.mss_capture_region)

                # Clamp region to the monitor bounds to avoid XGetImage out-of-range errors.
                mon = monitors[1] if len(monitors) > 1 else monitors[0]
                mon_left = int(mon["left"])
                mon_top = int(mon["top"])
                mon_right = mon_left + int(mon["width"])
                mon_bottom = mon_top + int(mon["height"])

                left = max(mon_left, int(region.get("left", mon_left)))
                top = max(mon_top, int(region.get("top", mon_top)))
                right = min(mon_right, left + int(region.get("width", 0)))
                bottom = min(mon_bottom, top + int(region.get("height", 0)))
                width = right - left
                height = bottom - top

                if width > 10 and height > 10:
                    safe_region = {"left": left, "top": top, "width": width, "height": height}
                    try:
                        raw = sct.grab(safe_region)
                        return _encode_raw(raw)
                    except Exception as e:
                        get_logger().warning(f"MSS region capture failed ({safe_region}): {e}")

            # MSS index 1 is the primary monitor; index 0 is the virtual full desktop.
            monitor_index = 1
            try:
                monitor_index = int(os.getenv("MSS_MONITOR_INDEX", "1"))
            except ValueError:
                monitor_index = 1

            if monitor_index < 0 or monitor_index >= len(monitors):
                monitor_index = 1 if len(monitors) > 1 else 0

            raw = sct.grab(monitors[monitor_index])
            return _encode_raw(raw)

    def _capture_driver_frame():
        """Fallback frame capture via VR driver inspect_surroundings."""
        res = _executor.call("inspect_surroundings")
        if isinstance(res, str) and res.startswith("Error"):
            raise RuntimeError(res)

        try:
            data = json.loads(res).get("data")
            if not data:
                raise RuntimeError("inspect_surroundings returned no image data")
            img_bytes = base64.b64decode(data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            out = io.BytesIO()
            pil_img.save(out, format="JPEG", quality=95)
            return pil_img, out.getvalue()
        except Exception as e:
            raise RuntimeError(f"Driver frame parse failed: {e}") from e

    def _decode_mask_png_base64(mask_png_base64: str):
        raw = base64.b64decode(mask_png_base64)
        np_buf = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(np_buf, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            return None
        return decoded > 127

    def _build_multipart_form_data(fields: Dict[str, str], file_name: str, file_bytes: bytes):
        boundary = f"----vragentmask{int(time.time() * 1000)}"
        body = bytearray()

        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
            )
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="image"; filename="{file_name}"\r\n'.encode("utf-8")
        )
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(file_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        return bytes(body), boundary

    def _segment_boxes_via_api(pil_img, boxes_xywh: List[List[float]], text: str = "") -> Dict[int, object]:
        if not mask_api_url:
            raise RuntimeError(
                "SEGMENTATION_BACKEND=api but SAM3_MASK_API_URL is not set."
            )
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for API mask decoding but is not available.")

        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("Failed to encode image for segmentation API request.")

        serializable_boxes = []
        for b in boxes_xywh:
            if len(b) != 4:
                continue
            x, y, bw, bh = [float(v) for v in b]
            serializable_boxes.append([x, y, max(1.0, bw), max(1.0, bh)])

        conf = os.getenv("SAM3_MASK_API_CONF", "0.35")
        timeout_s = float(os.getenv("SAM3_MASK_API_TIMEOUT_SEC", "8.0"))
        text_prompt = (text or "").strip()

        fields = {
            "boxes": json.dumps(serializable_boxes),
            "text": text_prompt,
            "conf": conf,
            "include_individual_masks": "true",
        }
        body, boundary = _build_multipart_form_data(fields, "frame.jpg", encoded.tobytes())

        req = urllib.request.Request(
            f"{mask_api_url}/segment",
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Mask API HTTP {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Mask API connection failed: {e}") from e

        data = json.loads(payload)
        masks_by_idx: Dict[int, object] = {}

        for item in data.get("masks", []):
            # New SAM3 API uses input_box_index; keep index as a backward-compatible fallback.
            idx = item.get("input_box_index")
            if idx is None:
                idx = item.get("index")
            mask_b64 = item.get("mask_png_base64")
            if idx is None or not mask_b64:
                continue
            mask = _decode_mask_png_base64(mask_b64)
            if mask is not None:
                masks_by_idx[int(idx)] = mask

        if not masks_by_idx and len(serializable_boxes) == 1 and data.get("combined_mask_png_base64"):
            mask = _decode_mask_png_base64(data["combined_mask_png_base64"])
            if mask is not None:
                masks_by_idx[0] = mask

        return masks_by_idx

    def _segment_boxes_with_backend(
        pil_img,
        boxes_by_key: Dict[str, List[float]],
        prompts_by_key: Dict[str, str],
        frame_w: int,
        frame_h: int,
    ) -> Dict[str, object]:
        if _using_api_backend():
            keys = list(boxes_by_key.keys())
            boxes = [boxes_by_key[k] for k in keys]
            # Always send text context together with geometric boxes for better SAM3 disambiguation.
            prompt_parts = []
            for k in keys:
                label = (prompts_by_key.get(k, "") or "").strip()
                prompt_parts.append(f"{k}:{label}" if label else k)
            prompt_text = "; ".join(prompt_parts)
            masks_by_idx = _segment_boxes_via_api(pil_img, boxes, prompt_text)
            return {key: masks_by_idx.get(i) for i, key in enumerate(keys)}

        if not _tracker or not _tracker.available:
            raise RuntimeError("SAM tracker is not available.")

        inference_state = _tracker.processor.set_image(pil_img)
        out: Dict[str, object] = {}

        for key, box in boxes_by_key.items():
            box_x, box_y, box_w, box_h = box
            box_input_xywh = _tracker.torch.tensor([box_x, box_y, box_w, box_h]).view(-1, 4)
            box_input_cxcywh = _tracker.box_xywh_to_cxcywh(box_input_xywh)
            norm_box_cxcywh = _tracker.normalize_bbox(
                box_input_cxcywh, frame_w, frame_h
            ).flatten().tolist()

            _tracker.processor.reset_all_prompts(inference_state)
            prompt = prompts_by_key.get(key)
            if prompt:
                inference_state = _tracker.processor.set_text_prompt(prompt, inference_state)
            inference_state = _tracker.processor.add_geometric_prompt(
                state=inference_state, box=norm_box_cxcywh, label=True
            )

            mask = None
            if "masks" in inference_state and inference_state["masks"] is not None:
                m = inference_state["masks"].detach().cpu().numpy() > 0.5
                if m.ndim == 4:
                    mask = m[0, 0]
                elif m.ndim == 3:
                    mask = m[0]

            out[key] = mask

        return out

    # =========================================================================
    # MOVEMENT & ORIENTATION
    # =========================================================================

    def start_bridge():
        """Start the VR bridge connection."""
        _log_action("start_bridge")
        return _executor.call("start_vr_bridge")

    def move_relative(device: str, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move a device relative to current position. dz: -ve is forward, +ve is back."""
        _log_action("move_relative", device=device, dx=dx, dy=dy, dz=dz)
        return _executor.call("move_relative", device=device, dx=dx, dy=dy, dz=dz)

    def move_absolute(device: str, x: float, y: float, z: float):
        """Move a device to absolute coordinates."""
        _log_action("move_absolute", device=device, x=x, y=y, z=z)
        return _executor.call("teleport", device=device, x=x, y=y, z=z)

    def teleport(device: str, x: float, y: float, z: float):
        """Teleport to exact coordinates."""
        _log_action("teleport", device=device, x=x, y=y, z=z)
        return _executor.call("teleport", device=device, x=x, y=y, z=z)

    def rotate_device(device: str, pitch: float, yaw: float, roll: float):
        """Rotate device (degrees)."""
        _log_action("rotate_device", device=device, pitch=pitch, yaw=yaw, roll=roll)
        return _executor.call("rotate_device", device=device, pitch=pitch, yaw=yaw, roll=roll)

    def get_current_pose(device: str = "headset"):
        """
        Get the current position and rotation of a device.
        device: "headset", "controller1", "controller2", or "all"
        """
        _log_action("get_current_pose", device=device)
        if device.lower() in ["all", "everything"]:
            results = []
            for d in ["headset", "controller1", "controller2"]:
                res = _executor.call("get_current_pose", device=d)
                results.append(f"{d}: {res}")
            return "\n".join(results)
        return _executor.call("get_current_pose", device=device)

    def reset_controller_positions():
        """Reset both controllers to natural positions relative to the headset."""
        _log_action("reset_controller_positions")
        return _executor.call("reset_controller_positions")

    def reset_controller_orientation():
        """Put controllers in front of user: Left pointing DOWN, Right pointing UP."""
        _log_action("reset_controller_orientation")
        if _agent_ref and hasattr(_agent_ref, 'keyboard_ctrl') and _agent_ref.keyboard_ctrl:
            _agent_ref.keyboard_ctrl.apply_reset_pose()
            return "Controllers reset (Left DOWN, Right UP) using keyboard controller."
        _executor.call("position_controller_relative_to_headset",
                       controller="controller1", forward=0.3, right=-0.2, up=-0.3)
        _executor.call("rotate_device", device="controller1", pitch=90, yaw=0, roll=0)
        _executor.call("position_controller_relative_to_headset",
                       controller="controller2", forward=0.3, right=0.2, up=-0.3)
        _executor.call("rotate_device", device="controller2", pitch=45, yaw=0, roll=0)
        return "Controllers reset (Left DOWN, Right UP) via direct commands."

    def position_controller_relative_to_headset(
        controller: str,
        forward: float = -0.3,
        right: float = 0.0,
        up: float = -0.5
    ):
        """Position a controller relative to the headset position."""
        _log_action("position_controller_relative_to_headset",
                    controller=controller, forward=forward, right=right, up=up)
        return _executor.call("position_controller_relative_to_headset",
                              controller=controller, forward=forward, right=right, up=up)

    def open_menu_sequence():
        """Set rigid positions for headset and controllers, then open menu."""
        _log_action("open_menu_sequence")
        move_absolute("headset", 0.0, 1.5, 0.0)
        rotate_device("headset", -20, 0, 0)
        move_absolute("controller1", -0.18, 1.2, -0.4)
        rotate_device("controller1", 90, 0, -15)
        move_absolute("controller2", 0.16491913500649544, 1.2251347749891743, -0.23297393336220185)
        rotate_device("controller2", 48.7, 38.9, 0.0)
        return click_button("controller1", "menu")

    # =========================================================================
    # CONTROLLER INPUTS
    # =========================================================================

    def press_button(controller: str, button: str):
        """Hold a button down until release_button is called."""
        _log_action("press_button", controller=controller, button=button)
        return _executor.call("press_button", controller=controller, button=button)

    def release_button(controller: str, button: str):
        """Release a previously pressed button."""
        _log_action("release_button", controller=controller, button=button)
        return _executor.call("release_button", controller=controller, button=button)

    def click_button(controller: str, button: str, duration: float = 0.1):
        """Quick press & release of a button."""
        _log_action("click_button", controller=controller, button=button, duration=duration)
        return _executor.call("click_button", controller=controller, button=button, duration=duration)

    def set_trigger(controller: str, value: float):
        """Set analog trigger value (0.0 released → 1.0 fully pressed)."""
        _log_action("set_trigger", controller=controller, value=value)
        return _executor.call("set_trigger", controller=controller, value=value)

    def set_joystick(controller: str, x: float, y: float):
        """Set joystick position. x/y range: -1.0 to 1.0."""
        _log_action("set_joystick", controller=controller, x=x, y=y)
        return _executor.call("set_joystick", controller=controller, x=x, y=y)

    def move_joystick_direction(controller: str, direction: str, magnitude: float = 1.0):
        """Move joystick in a cardinal direction: up/down/left/right/center/forward/backward."""
        _log_action("move_joystick_direction", controller=controller,
                    direction=direction, magnitude=magnitude)
        return _executor.call("move_joystick_direction", controller=controller,
                              direction=direction, magnitude=magnitude)

    def click_trackpad_direction(controller: str, direction: str, duration: float = 1.0):
        """Move joystick to direction then click trackpad."""
        _log_action("click_trackpad_direction", controller=controller,
                    direction=direction, duration=duration)
        move_joystick_direction(controller, direction, magnitude=1.0)
        time.sleep(0.1)
        return click_button(controller, "trackpad", duration=duration)

    def perform_grab(controller: str):
        """Grab object: press grip + trigger together."""
        _log_action("perform_grab", controller=controller)
        return _executor.call("perform_grab", controller=controller)

    def perform_release(controller: str):
        """Release grabbed object: release grip + trigger."""
        _log_action("perform_release", controller=controller)
        return _executor.call("perform_release", controller=controller)

    def release_all_inputs(controller: str = "both"):
        """Release all buttons and reset joystick to center."""
        _log_action("release_all_inputs", controller=controller)
        return _executor.call("release_all_inputs", controller=controller)

    def get_controller_state(controller: str):
        """Get current input state of a controller (buttons, joystick, trigger)."""
        _log_action("get_controller_state", controller=controller)
        return _executor.call("get_controller_state", controller=controller)

    # =========================================================================
    # VISION
    # =========================================================================

    def inspect_surroundings():
        """Take a screenshot of the current view."""
        _log_action("inspect_surroundings")
        return _executor.call("inspect_surroundings")

    def locate_object(object_description: str):
        """Find an object and return its normalized center coordinates."""
        _log_action("locate_object", description=object_description)
        res = _executor.call("inspect_surroundings")
        data = json.loads(res).get("data")
        if not data:
            return "Failed to capture image"
        img_bytes = base64.b64decode(data)
        boxes = _grounder.ground_object(img_bytes, object_description)
        if not boxes:
            return f"Could not find '{object_description}'"
        results = []
        for box in boxes:
            y1, x1, y2, x2 = box['box_2d']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            results.append(f"Found at Center(x={cx:.2f}, y={cy:.2f})")
        return "; ".join(results)

    def capture_video(duration: float = 3.0):
        """Capture a short video clip and auto-save frames to disk."""
        _log_action("capture_video", duration=duration)
        res_str = _executor.call("capture_video", duration=duration)
        try:
            res = json.loads(res_str)
            if res.get("type") == "video":
                timestamp = datetime.now().strftime("%H%M%S")
                vid_dir = LOG_DIR / "videos" / f"video_{timestamp}"
                vid_dir.mkdir(parents=True, exist_ok=True)
                frames = res.get("frames", [])
                for i, b64 in enumerate(frames):
                    with open(vid_dir / f"frame_{i:04d}.jpg", "wb") as f:
                        f.write(base64.b64decode(b64))
                print(f"Video saved to: {vid_dir}")
                get_logger().info(f"Video saved to {vid_dir}")
                return f"Video captured and saved to {vid_dir} ({len(frames)} frames)."
        except Exception as e:
            get_logger().error(f"Failed to auto-save video: {e}")
        return res_str

    def explore_environment(
        max_stations: int = 120,
        map_size_m: float = 20.0,
        grid_res_m: float = 0.10,
        station_spacing_m: float = 1.25,
        rotate_probability: float = 0.0,
        forward_move_m: float = 1.0,
        rotate_step_degrees: float = 20.0,
        depth_baseline_move_m: float = 0.0,
        obstacle_min_height_ratio: float = 0.30,
        obstacle_max_height_ratio: float = 0.90,
        treat_above_obstacle_band_as_blocking: bool = True,
        obstacle_min_distance_m: float = 0.3,
        require_known_free_forward_path: bool = True,
        split_stereo_capture: bool = True,
        stereo_eye_separation_m: float = 0.064,
        min_moves_before_mapping: int = 0,
        forward_depth_safety_margin_m: float = 0.35,
        forward_depth_corridor_width_ratio: float = 0.45,
        forward_depth_min_close_fraction: float = 0.05,
        forward_depth_relative_close_ratio: float = 0.70,
        depth_stride: int = 3,
        max_rays_per_observation: int = 8000,
        debug_output: str = "summary",
        export_ply: bool = True,
        save_ply_each_move: bool = True,
        depth_engine: str = "fast_foundationstereo",
        foundationstereo_repo: str = "",
        foundationstereo_checkpoint: str = "",
        foundationstereo_scale: float = 1.0,
        foundationstereo_valid_iters: int = 8,
        foundationstereo_max_disp: int = 192,
        rotate_to_frontier: bool = True,
        frontier_heading_lookahead_m: float = 2.0,
        frontier_unknown_radius_m: float = 1.25,
        frontier_novelty_weight: float = 0.08,
        frontier_visited_penalty: float = 25.0,
        avoid_visited_forward: bool = True,
        visited_revisit_unknown_radius_m: float = 1.25,
    ):
        """
        Explore the VR environment, avoid obstacles with current depth, then build a 2D occupancy map.

        The first min_moves_before_mapping movement actions do not build or use
        the 2D occupancy grid. During that warmup, every forward step is gated by
        the current depth image; if close geometry is detected in the forward
        corridor, the agent rotates instead. After the threshold, it starts
        rebuilding the occupancy map and can return to start with A*.
        """
        _log_action(
            "explore_environment",
            max_stations=max_stations,
            map_size_m=map_size_m,
            grid_res_m=grid_res_m,
            station_spacing_m=station_spacing_m,
            rotate_probability=rotate_probability,
            forward_move_m=forward_move_m,
            rotate_step_degrees=rotate_step_degrees,
            depth_baseline_move_m=depth_baseline_move_m,
            obstacle_min_height_ratio=obstacle_min_height_ratio,
            obstacle_max_height_ratio=obstacle_max_height_ratio,
            treat_above_obstacle_band_as_blocking=treat_above_obstacle_band_as_blocking,
            obstacle_min_distance_m=obstacle_min_distance_m,
            require_known_free_forward_path=require_known_free_forward_path,
            split_stereo_capture=split_stereo_capture,
            stereo_eye_separation_m=stereo_eye_separation_m,
            min_moves_before_mapping=min_moves_before_mapping,
            forward_depth_safety_margin_m=forward_depth_safety_margin_m,
            forward_depth_corridor_width_ratio=forward_depth_corridor_width_ratio,
            forward_depth_min_close_fraction=forward_depth_min_close_fraction,
            forward_depth_relative_close_ratio=forward_depth_relative_close_ratio,
            depth_stride=depth_stride,
            max_rays_per_observation=max_rays_per_observation,
            debug_output=debug_output,
            export_ply=export_ply,
            save_ply_each_move=save_ply_each_move,
            depth_engine=depth_engine,
            foundationstereo_repo=foundationstereo_repo,
            foundationstereo_checkpoint=foundationstereo_checkpoint,
            foundationstereo_scale=foundationstereo_scale,
            foundationstereo_valid_iters=foundationstereo_valid_iters,
            foundationstereo_max_disp=foundationstereo_max_disp,
            rotate_to_frontier=rotate_to_frontier,
            frontier_heading_lookahead_m=frontier_heading_lookahead_m,
            frontier_unknown_radius_m=frontier_unknown_radius_m,
            frontier_novelty_weight=frontier_novelty_weight,
            frontier_visited_penalty=frontier_visited_penalty,
            avoid_visited_forward=avoid_visited_forward,
            visited_revisit_unknown_radius_m=visited_revisit_unknown_radius_m,
        )
        cfg = OccupancyExploreConfig(
            max_stations=int(max_stations),
            map_size_m=float(map_size_m),
            grid_res_m=float(grid_res_m),
            station_spacing_m=float(station_spacing_m),
            rotate_probability=min(1.0, max(0.0, float(rotate_probability))),
            forward_move_m=max(0.0, float(forward_move_m)),
            rotate_step_degrees=float(rotate_step_degrees),
            depth_baseline_move_m=max(0.0, float(depth_baseline_move_m)),
            obstacle_min_height_ratio=max(0.0, float(obstacle_min_height_ratio)),
            obstacle_max_height_ratio=min(1.0, float(obstacle_max_height_ratio)),
            treat_above_obstacle_band_as_blocking=bool(treat_above_obstacle_band_as_blocking),
            obstacle_min_distance_m=max(0.0, float(obstacle_min_distance_m)),
            require_known_free_forward_path=bool(require_known_free_forward_path),
            split_stereo_capture=bool(split_stereo_capture),
            stereo_eye_separation_m=max(0.0, float(stereo_eye_separation_m)),
            min_moves_before_mapping=max(0, int(min_moves_before_mapping)),
            forward_depth_safety_margin_m=max(0.0, float(forward_depth_safety_margin_m)),
            forward_depth_corridor_width_ratio=max(
                0.05,
                min(1.0, float(forward_depth_corridor_width_ratio)),
            ),
            forward_depth_min_close_fraction=max(
                0.0,
                min(1.0, float(forward_depth_min_close_fraction)),
            ),
            forward_depth_relative_close_ratio=max(
                0.05,
                min(1.0, float(forward_depth_relative_close_ratio)),
            ),
            depth_stride=max(1, int(depth_stride)),
            max_rays_per_observation=max(0, int(max_rays_per_observation)),
            debug_output=str(debug_output).lower(),
            export_ply=bool(export_ply),
            save_ply_each_move=bool(save_ply_each_move),
            depth_engine=str(depth_engine),
            foundationstereo_repo=str(foundationstereo_repo),
            foundationstereo_checkpoint=str(foundationstereo_checkpoint),
            foundationstereo_scale=max(0.05, float(foundationstereo_scale)),
            foundationstereo_valid_iters=max(1, int(foundationstereo_valid_iters)),
            foundationstereo_max_disp=max(16, int(foundationstereo_max_disp)),
            rotate_to_frontier=bool(rotate_to_frontier),
            frontier_heading_lookahead_m=max(0.0, float(frontier_heading_lookahead_m)),
            frontier_unknown_radius_m=max(0.0, float(frontier_unknown_radius_m)),
            frontier_novelty_weight=max(0.0, float(frontier_novelty_weight)),
            frontier_visited_penalty=max(0.0, float(frontier_visited_penalty)),
            avoid_visited_forward=bool(avoid_visited_forward),
            visited_revisit_unknown_radius_m=max(0.0, float(visited_revisit_unknown_radius_m)),
        )
        explorer = OccupancyExplorer(_executor, LOG_DIR, cfg)
        stop_event = getattr(_agent_ref, "stop_execution", None)
        return explorer.explore(stop_event=stop_event)

    # =========================================================================
    # OBJECT TRACKING
    # =========================================================================

    def track_object(object_description: str):
        """
        Track an object in a 3-second video using SAM 2.
        1. Capture video → 2. Ground in first frame → 3. Track with SAM.
        """
        _log_action("track_object", description=object_description)
        logger = get_logger()

        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 2) is not available."

        print("Capturing video...")
        res_str = _executor.call("capture_video", duration=3.0)
        if "Error" in res_str and not res_str.startswith("{"):
            return res_str

        try:
            res = json.loads(res_str)
            if res.get("type") != "video":
                return f"Error capturing video: {res_str[:100]}"
            frames = res.get("frames", [])
            if not frames:
                return "Error: No frames in video."
        except json.JSONDecodeError:
            return f"Error parsing video response: {res_str[:100]}"

        timestamp = datetime.now().strftime("%H%M%S")
        temp_dir = LOG_DIR / "tracking" / f"temp_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        saved_frames = []
        for i, b64 in enumerate(frames):
            path = temp_dir / f"frame_{i:04d}.jpg"
            img_data = base64.b64decode(b64)
            try:
                with Image.open(io.BytesIO(img_data)) as img:
                    img.save(path, quality=95)
            except Exception as e:
                print(f"Warning: PIL failed to clean frame {i}: {e}. Saving raw.")
                with open(path, "wb") as f:
                    f.write(img_data)
            saved_frames.append(path)

        print("Locating object in first frame...")
        with open(saved_frames[0], "rb") as f:
            first_frame_data = f.read()

        boxes = _grounder.ground_object(first_frame_data, object_description)
        if not boxes:
            return f"Could not find '{object_description}' in the first frame to start tracking."

        init_box = boxes[0]['box_2d']
        print("Running SAM 2 Tracking...")
        video_output = _tracker.track(str(temp_dir), init_box, object_description)
        print(f"\n[SUCCESS] Tracking Video Saved to: {video_output}\n")
        logger.info(f"Tracking Video Saved to: {video_output}")
        return f"Tracking completed. Output video: {video_output}"

    def create_tracking_video(object_description: str):
        """Alias for track_object — creates a segmented tracking video."""
        return track_object(object_description)

    def track_multiple_items(object_names: List[str]):
        """
        Track multiple objects simultaneously using SAM 3.
        Example: ["red cup", "keyboard", "blue pen"]
        """
        _log_action("track_multiple_items", objects=object_names)

        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 3) is not available."

        print(f"Capturing video to track: {object_names}...")
        res_str = _executor.call("capture_video", duration=3.0)
        try:
            res = json.loads(res_str)
            frames = res.get("frames", [])
            if not frames:
                return "Error: No frames captured."
        except Exception:
            return "Error parsing video data."

        timestamp = datetime.now().strftime("%H%M%S")
        temp_dir = LOG_DIR / "tracking" / f"multi_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        saved_frames = []
        for i, b64 in enumerate(frames):
            path = temp_dir / f"frame_{i:04d}.jpg"
            img_data = base64.b64decode(b64)
            try:
                with Image.open(io.BytesIO(img_data)) as img:
                    img.save(path, quality=95)
            except Exception:
                with open(path, "wb") as f:
                    f.write(img_data)
            saved_frames.append(path)

        print("Locating objects in first frame...")
        with open(saved_frames[0], "rb") as f:
            first_frame_data = f.read()

        initial_data = _grounder.ground_multiple_objects(first_frame_data, object_names)
        if not initial_data:
            return f"Could not locate any of the requested objects: {object_names}"

        print(f"Found {len(initial_data)} objects: {list(initial_data.keys())}")
        print("Running SAM 3 Multi-Tracking...")
        result = _tracker.track_multi_objects(str(temp_dir), initial_data)

        if "error" in result:
            return f"Tracking failed: {result['error']}"

        video_path = result.get("video_path")
        print(f"\n[SUCCESS] Multi-Tracking Video: {video_path}\n")
        get_logger().info(f"Multi-Tracking Video: {video_path}")
        return f"Tracking Complete. Video saved to {video_path}"

    # =========================================================================
    # VISUAL SERVOING
    # =========================================================================

    def visual_servo_to_object(
        object_description: str,
        controller: str,
        ray_description: str = "blue VR controller ray"
    ):
        """
        Rotate the controller so its ray aligns with the target object.
        Uses Qwen for initial grounding, then SAM + PID for closed-loop control.
        """
        if ray_description == "blue VR controller ray":
            ray_description = (
                "blue VR controller ray of the left controller"
                if controller == "controller1"
                else "blue VR controller ray of the right controller"
            )

        _log_action("visual_servo_to_object", description=object_description,
                    controller=controller, ray=ray_description)
        logger = get_logger()
        servo_start_ts = time.perf_counter()
        last_overlay_frame = None
        last_overlay_iter = -1
        timing_data: Dict[str, List[float]] = {}

        def _record_timing(step_name: str, elapsed_s: float):
            timing_data.setdefault(step_name, []).append(max(0.0, elapsed_s))

        def _format_timing_breakdown() -> str:
            if not timing_data:
                return ""

            stage_order = [
                "init_pose",
                "initial_capture",
                "grounding",
                "initial_bbox_locate",
                "loop_capture",
                "loop_segmentation",
                "loop_mask_postprocess",
                "loop_visualization",
                "loop_pid_compute",
                "loop_rotate_call",
                "loop_reground",
            ]

            lines = []
            for stage in stage_order:
                values = timing_data.get(stage, [])
                if not values:
                    continue
                total_s = sum(values)
                count = len(values)
                avg_ms = (total_s / count) * 1000.0
                lines.append(
                    f"- {stage}: total={total_s:.3f}s avg={avg_ms:.2f}ms n={count}"
                )

            if not lines:
                return ""
            return "Timing breakdown:\n" + "\n".join(lines)

        def _with_benchmark(message: str) -> str:
            elapsed_s = time.perf_counter() - servo_start_ts
            summary = f"{message} (took {elapsed_s:.2f}s)"
            breakdown = _format_timing_breakdown()
            if breakdown:
                summary = f"{summary}\n{breakdown}"
            return summary

        def _save_last_overlay_snapshot(reason: str) -> str:
            nonlocal last_overlay_frame, last_overlay_iter
            if not CV2_AVAILABLE or last_overlay_frame is None:
                return ""

            try:
                tracking_dir = LOG_DIR / "tracking"
                tracking_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "_", reason).strip("_").lower() or "end"
                iter_suffix = f"_iter_{last_overlay_iter}" if last_overlay_iter >= 0 else ""
                out_path = tracking_dir / f"servo_last_overlay_{timestamp}_{safe_reason}{iter_suffix}.jpg"
                ok = cv2.imwrite(str(out_path), last_overlay_frame)
                if ok:
                    logger.info(f"Final servo overlay snapshot saved: {out_path}")
                    return str(out_path)
            except Exception as e:
                logger.warning(f"Failed to save final servo overlay snapshot: {e}")
            return ""

        def _with_benchmark_and_overlay(message: str, reason: str) -> str:
            overlay_path = _save_last_overlay_snapshot(reason)
            if overlay_path:
                message = f"{message} Final overlay: {overlay_path}"
            return _with_benchmark(message)

        def _expand_box_in_pid_direction(
            box_xywh: List[float],
            pid_dx: float,
            pid_dy: float,
            frame_w: int,
            frame_h: int,
        ) -> List[float]:
            """
            Expand SAM prompt box toward the PID motion direction.
            Keeps a small symmetric margin and then extends toward movement.
            """
            x, y, bw, bh = [float(v) for v in box_xywh]
            x1 = x
            y1 = y
            x2 = x + max(1.0, bw)
            y2 = y + max(1.0, bh)

            base_pad = max(2.0, 0.05 * max(bw, bh))
            x1 -= base_pad
            y1 -= base_pad
            x2 += base_pad
            y2 += base_pad

            # PID error magnitude (pixels) determines directional expansion amount.
            extend_x = max(4.0, min(abs(pid_dx) * 0.35, max(12.0, bw * 0.8)))
            extend_y = max(4.0, min(abs(pid_dy) * 0.35, max(12.0, bh * 0.8)))

            if pid_dx > 0:
                x2 += extend_x
            elif pid_dx < 0:
                x1 -= extend_x

            if pid_dy > 0:
                y2 += extend_y
            elif pid_dy < 0:
                y1 -= extend_y

            x1 = max(0.0, min(x1, float(frame_w - 1)))
            y1 = max(0.0, min(y1, float(frame_h - 1)))
            x2 = max(x1 + 1.0, min(x2, float(frame_w)))
            y2 = max(y1 + 1.0, min(y2, float(frame_h)))

            return [x1, y1, x2 - x1, y2 - y1]

        if not _using_api_backend() and (not _tracker or not _tracker.available):
            return _with_benchmark("Error: Object Tracking (SAM 3) is not available.")

        Kp_YAW = 0.01
        Kp_PITCH = 0.01
        MAX_ITER = 100
        TOLERANCE_PX = 15

        # 1. Get initial pose
        pose_start_ts = time.perf_counter()
        curr_pitch = curr_yaw = curr_roll = 0.0
        status = _executor.call("get_current_pose", device=controller)
        try:
            if "Rotation: [" in status:
                rot_str = status.split("Rotation: [")[1].split("]")[0]
                rot_str = rot_str.replace("np.float64(", "").replace(")", "")
                curr_pitch, curr_yaw, curr_roll = map(float, rot_str.split(","))
        except Exception as e:
            _record_timing("init_pose", time.perf_counter() - pose_start_ts)
            return _with_benchmark(f"Failed to parse initial pose: {e}")
        _record_timing("init_pose", time.perf_counter() - pose_start_ts)

        # 2. Capture initial image.
        initial_capture_start_ts = time.perf_counter()
        if _should_use_mss_capture():
            try:
                pil_img_init, img_bytes = _capture_mss_frame()
                w, h = pil_img_init.size
            except Exception as e:
                get_logger().warning(f"Initial MSS capture failed, falling back to driver: {e}")
                try:
                    pil_img_init, img_bytes = _capture_driver_frame()
                    w, h = pil_img_init.size
                except Exception as e2:
                    _record_timing("initial_capture", time.perf_counter() - initial_capture_start_ts)
                    return _with_benchmark(f"Initial capture failed (MSS + driver): {e2}")
        else:
            try:
                pil_img_init, img_bytes = _capture_driver_frame()
                w, h = pil_img_init.size
            except Exception as e:
                _record_timing("initial_capture", time.perf_counter() - initial_capture_start_ts)
                return _with_benchmark(f"Initial capture failed (driver): {e}")
        _record_timing("initial_capture", time.perf_counter() - initial_capture_start_ts)

        # 3. Ground targets
        targets = {"ray": ray_description, "logo": object_description}
        grounding_start_ts = time.perf_counter()
        grounding_results = _grounder.ground_multiple_objects(img_bytes, list(targets.values()))
        _record_timing("grounding", time.perf_counter() - grounding_start_ts)

        bbox_locate_start_ts = time.perf_counter()
        current_boxes = {}
        all_found = True
        for key, desc in targets.items():
            if desc in grounding_results:
                ymin, xmin, ymax, xmax = grounding_results[desc]
                current_boxes[key] = [xmin * w, ymin * h,
                                      (xmax - xmin) * w, (ymax - ymin) * h]
            else:
                print(f"[{key}] '{desc}' NOT found by Gemini.")
                all_found = False
        _record_timing("initial_bbox_locate", time.perf_counter() - bbox_locate_start_ts)

        if not all_found:
            return _with_benchmark(
                f"Failed to find both the controller ray and '{object_description}'. "
                "CAUTION: The ray might be occluding the object if they are already aligned. "
                "Verify alignment manually or try a different viewing angle."
            )

        print("Initial grounding successful. Starting control loop.")

        # 4. PID control loop
        prev_dist = float('inf')
        divergence_count = 0
        dist = float('inf')

        for i in range(MAX_ITER):
            if _agent_ref and _agent_ref.stop_execution.is_set():
                return _with_benchmark_and_overlay("Visual Servoing Stopped by User.", "stopped")

            # Capture frame.
            loop_capture_start_ts = time.perf_counter()
            if _should_use_mss_capture():
                try:
                    pil_img_loop, img_bytes_loop = _capture_mss_frame()
                    img_cv = cv2.cvtColor(np.array(pil_img_loop), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"MSS capture error in loop: {e}. Falling back to driver frame.")
                    try:
                        pil_img_loop, img_bytes_loop = _capture_driver_frame()
                        img_cv = cv2.cvtColor(np.array(pil_img_loop), cv2.COLOR_RGB2BGR)
                    except Exception as e2:
                        print(f"Driver capture fallback failed in loop: {e2}")
                        _record_timing("loop_capture", time.perf_counter() - loop_capture_start_ts)
                        break
            else:
                try:
                    pil_img_loop, img_bytes_loop = _capture_driver_frame()
                    img_cv = cv2.cvtColor(np.array(pil_img_loop), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Driver capture failed in loop: {e}")
                    _record_timing("loop_capture", time.perf_counter() - loop_capture_start_ts)
                    break
            _record_timing("loop_capture", time.perf_counter() - loop_capture_start_ts)

            points = {}
            masks_for_viz = {}
            boxes_to_track = {
                key: current_boxes[key] for key in targets.keys() if key in current_boxes
            }

            segmentation_start_ts = time.perf_counter()
            try:
                predicted_masks = _segment_boxes_with_backend(
                    pil_img=pil_img_loop,
                    boxes_by_key=boxes_to_track,
                    prompts_by_key=targets,
                    frame_w=w,
                    frame_h=h,
                )
            except Exception as e:
                print(f"Segmentation backend error: {e}")
                predicted_masks = {}
            _record_timing("loop_segmentation", time.perf_counter() - segmentation_start_ts)

            postprocess_start_ts = time.perf_counter()
            for key, desc in targets.items():
                if key not in boxes_to_track:
                    continue
                mask = predicted_masks.get(key)

                if mask is None:
                    print(f"[{key}] Lost tracking. Attempting reground...")
                    reground_start_ts = time.perf_counter()
                    try:
                        reground_res = _grounder.ground_multiple_objects(img_bytes_loop, [desc])
                        if desc in reground_res:
                            ymin, xmin, ymax, xmax = reground_res[desc]
                            current_boxes[key] = [xmin * w, ymin * h,
                                                  (xmax - xmin) * w, (ymax - ymin) * h]
                            print(f"[{key}] Reground Successful.")
                        else:
                            del current_boxes[key]
                    except Exception as e:
                        print(f"[{key}] Reground Error: {e}")
                        del current_boxes[key]
                    _record_timing("loop_reground", time.perf_counter() - reground_start_ts)
                    continue

                masks_for_viz[key] = mask
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    current_boxes[key] = [cmin, rmin, cmax - cmin, rmax - rmin]
                    if key == "ray":
                        ys, xs = np.where(mask)
                        if len(ys) > 0:
                            idx = np.argmin(ys)
                            points[key] = (xs[idx], ys[idx])
                    elif key == "logo":
                        M = cv2.moments(mask.astype(np.uint8))
                        if M["m00"] != 0:
                            points[key] = (int(M["m10"] / M["m00"]),
                                           int(M["m01"] / M["m00"]))
            _record_timing("loop_mask_postprocess", time.perf_counter() - postprocess_start_ts)

            # Visualize masks
            viz_start_ts = time.perf_counter()
            for key, mask in masks_for_viz.items():
                color = (np.array([255, 100, 0] if key == "ray" else [0, 255, 100],
                                  dtype=np.uint8))
                overlay = img_cv.copy()
                overlay[mask] = color
                cv2.addWeighted(overlay, 0.35, img_cv, 0.65, 0, img_cv)
            _record_timing("loop_visualization", time.perf_counter() - viz_start_ts)

            if CV2_AVAILABLE:
                last_overlay_frame = img_cv.copy()
                last_overlay_iter = i

            # PID control
            if "ray" in points and "logo" in points:
                pid_compute_start_ts = time.perf_counter()
                rx, ry = points["ray"]
                lx, ly = points["logo"]
                dx = lx - rx
                dy = ly - ry
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > prev_dist + 50.0:
                    divergence_count += 1
                else:
                    divergence_count = 0
                prev_dist = dist

                if divergence_count >= 3:
                    _record_timing("loop_pid_compute", time.perf_counter() - pid_compute_start_ts)
                    return _with_benchmark_and_overlay(
                        "Visual Servoing Aborted: Divergence detected.",
                        "divergence",
                    )

                # Save debug image
                cv2.line(img_cv, (rx, ry), (lx, ly), (0, 255, 255), 2)
                last_overlay_frame = img_cv.copy()
                last_overlay_iter = i
                timestamp = datetime.now().strftime("%H%M%S")
                debug_path = LOG_DIR / "tracking" / f"servo_{timestamp}_iter_{i}.jpg"
                if CV2_AVAILABLE:
                    cv2.imwrite(str(debug_path), img_cv)

                if dist < TOLERANCE_PX:
                    _record_timing("loop_pid_compute", time.perf_counter() - pid_compute_start_ts)
                    msg = _with_benchmark_and_overlay(
                        f"Visual Servoing Complete. Aligned with {object_description} (Error: {dist:.2f}px).",
                        "aligned",
                    )
                    logger.info(msg)
                    return msg

                curr_yaw += -Kp_YAW * dx
                curr_pitch += -Kp_PITCH * dy
                _record_timing("loop_pid_compute", time.perf_counter() - pid_compute_start_ts)
                rotate_start_ts = time.perf_counter()
                _executor.call("rotate_device", device=controller,
                               pitch=curr_pitch, yaw=curr_yaw, roll=curr_roll)
                _record_timing("loop_rotate_call", time.perf_counter() - rotate_start_ts)

                # After each PID move, expand SAM's target box in the commanded direction.
                if "logo" in current_boxes:
                    current_boxes["logo"] = _expand_box_in_pid_direction(
                        current_boxes["logo"],
                        pid_dx=dx,
                        pid_dy=dy,
                        frame_w=w,
                        frame_h=h,
                    )
            else:
                if "ray" in current_boxes and "logo" in current_boxes:
                    continue
                return _with_benchmark_and_overlay(
                    "Lost tracking of one or both objects during loop. Stopping.",
                    "lost_tracking",
                )

        if dist < 50.0:
            return _with_benchmark_and_overlay(
                f"Visual Servoing finished. Aligned within {dist:.1f}px (acceptable).",
                "acceptable",
            )
        return _with_benchmark_and_overlay(
            f"Visual servoing finished max iterations ({MAX_ITER}). Final error: {dist:.1f}.",
            "max_iterations",
        )

    # =========================================================================
    # VIRTUAL KEYBOARD TYPING
    # =========================================================================

    def type_text(text: str, controller: str = "controller2"):
        """
        Type text on a virtual keyboard using visual servoing.
        Grounds all unique characters once, then PID-aligns to each key and presses trigger.
        """
        _log_action("type_text", text=text, controller=controller)
        logger = get_logger()

        if not _using_api_backend() and (not _tracker or not _tracker.available):
            return "Error: Object Tracking (SAM 3) is not available."
        if not text:
            return "Error: No text provided to type."

        Kp_YAW = 0.02
        Kp_PITCH = 0.02
        MAX_ITER_PER_CHAR = 50
        TOLERANCE_PX = 8

        # Get initial pose
        curr_pitch = curr_yaw = curr_roll = 0.0
        status = _executor.call("get_current_pose", device=controller)
        try:
            if "Rotation: [" in status:
                rot_str = status.split("Rotation: [")[1].split("]")[0]
                rot_str = rot_str.replace("np.float64(", "").replace(")", "")
                curr_pitch, curr_yaw, curr_roll = map(float, rot_str.split(","))
        except Exception as e:
            return f"Failed to parse initial pose: {e}"

        # Capture initial image
        res = _executor.call("inspect_surroundings")
        if isinstance(res, str) and res.startswith("Error"):
            return f"Capture failed: {res}"
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            pil_img_init = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img_init.size
        except Exception as e:
            return f"Initial image parsing failed: {e}"

        # Ground all unique characters + ray in one call
        unique_chars = list(set(text.lower()))
        chars_list = ", ".join([f'"{c}"' for c in unique_chars])
        side = "left controller" if controller == "controller1" else "right controller"
        prompt = f"""
Find the following keyboard keys in the image: {chars_list}
Also find the "blue VR controller ray of the {side}".

You MUST return the answer in the following JSON format:
{{
    "thinking": "Describe what you see and where the keys are located...",
    "keys": {{
        "a": [ymin, xmin, ymax, xmax],
        "b": [ymin, xmin, ymax, xmax]
    }},
    "controller_ray": [ymin, xmin, ymax, xmax]
}}

Rules:
1. ymin, xmin, ymax, xmax must be normalized coordinates (0 to 1).
2. Include ONLY the keys you can confidently locate.
3. "controller_ray" is the bounding box of the blue VR controller ray.
"""
        try:
            out_buffer = io.BytesIO()
            pil_img_init.save(out_buffer, format="JPEG")
            clean_image_data = out_buffer.getvalue()

            base64_img = base64.b64encode(clean_image_data).decode('utf-8')
            response = _grounder.client.chat.completions.create(
                model=_grounder.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                temperature=0.2,
                extra_body={
                   "chat_template_kwargs": {"enable_thinking": False}  
                },
                response_format={"type": "json_object"}
            )

            resp_text = response.choices[0].message.content
            if not resp_text:
                return "Error: Qwen returned no grounding results." 

            if not isinstance(resp_text, str):
                if isinstance(resp_text, list):
                    text_parts = []
                    for part in resp_text:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and isinstance(part.get("text"), str):
                            text_parts.append(part["text"])
                    resp_text = "\n".join(text_parts)
                else:
                    resp_text = str(resp_text)

            match = re.search(r"```(?:json)?\s*(.*?)\s*```", resp_text, re.IGNORECASE | re.DOTALL)
            if match:
                resp_text = match.group(1)

            resp_text = resp_text.strip()
            if resp_text.lower().startswith("json\n"):
                resp_text = resp_text[5:].strip()

            grounding_data = json.loads(resp_text)
        except Exception as e:
            logger.error(f"Keyboard grounding failed: {e}")
            return f"Error grounding keyboard keys: {e}"

        grounded_keys = grounding_data.get("keys", {})
        ray_box_norm = grounding_data.get("controller_ray")
        if not ray_box_norm:
            return "Error: Could not find the controller ray in the image."

        def norm_to_pixel_box(norm_box):
            ymin, xmin, ymax, xmax = norm_box
            if any(c > 1.0 for c in norm_box):
                ymin, xmin, ymax, xmax = [c / 1000.0 for c in norm_box]
            return [xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h]

        key_boxes = {k: norm_to_pixel_box(v) for k, v in grounded_keys.items()}
        ray_box = norm_to_pixel_box(ray_box_norm)

        missing_chars = [c for c in unique_chars if c not in grounded_keys]
        if missing_chars:
            print(f"Warning: Could not find keys for: {missing_chars}")

        typed_chars = []
        failed_chars = []

        for char_idx, char in enumerate(text.lower()):
            print(f"\n=== Typing '{char}' ({char_idx + 1}/{len(text)}) ===")
            if char not in key_boxes:
                failed_chars.append(char)
                continue

            target_box = key_boxes[char]
            current_ray_box = ray_box.copy()
            converged = False
            final_dist = 0

            for i in range(MAX_ITER_PER_CHAR):
                if _agent_ref and _agent_ref.stop_execution.is_set():
                    return "Typing Stopped by User."

                res = _executor.call("inspect_surroundings")
                if isinstance(res, str) and res.startswith("Error"):
                    break
                try:
                    data = json.loads(res).get("data")
                    img_bytes_loop = base64.b64decode(data)
                    pil_img_loop = Image.open(io.BytesIO(img_bytes_loop)).convert("RGB")
                    img_cv = cv2.cvtColor(np.array(pil_img_loop), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Image parse error: {e}")
                    break

                points = {}
                boxes_to_track = {"ray": current_ray_box, "key": target_box}
                updated_boxes = {}
                masks_for_viz = {}
                prompts_by_key = {
                    "ray": "VR controller ray",
                    "key": f"keyboard key {char}",
                }

                try:
                    predicted_masks = _segment_boxes_with_backend(
                        pil_img=pil_img_loop,
                        boxes_by_key=boxes_to_track,
                        prompts_by_key=prompts_by_key,
                        frame_w=w,
                        frame_h=h,
                    )
                except Exception as e:
                    print(f"Segmentation backend error: {e}")
                    predicted_masks = {}

                for key, box in boxes_to_track.items():
                    mask = predicted_masks.get(key)

                    if mask is None:
                        continue

                    masks_for_viz[key] = mask
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        updated_boxes[key] = [cmin, rmin, cmax - cmin, rmax - rmin]
                        if key == "ray":
                            ys, xs = np.where(mask)
                            if len(ys) > 0:
                                idx = np.argmin(ys)
                                points[key] = (xs[idx], ys[idx])
                        elif key == "key":
                            M = cv2.moments(mask.astype(np.uint8))
                            if M["m00"] != 0:
                                points[key] = (int(M["m10"] / M["m00"]),
                                               int(M["m01"] / M["m00"]))

                for key, mask in masks_for_viz.items():
                    color = np.array([255, 100, 0] if key == "ray" else [0, 255, 100],
                                     dtype=np.uint8)
                    overlay = img_cv.copy()
                    overlay[mask] = color
                    cv2.addWeighted(overlay, 0.35, img_cv, 0.65, 0, img_cv)

                if "ray" in updated_boxes:
                    current_ray_box = updated_boxes["ray"]
                if "key" in updated_boxes:
                    target_box = updated_boxes["key"]

                if "ray" in points and "key" in points:
                    rx, ry = points["ray"]
                    kx, ky = points["key"]
                    dx = kx - rx
                    dy = ky - ry
                    dist = math.sqrt(dx * dx + dy * dy)
                    final_dist = dist

                    if dist < TOLERANCE_PX:
                        converged = True
                        break

                    curr_yaw += -Kp_YAW * dx
                    curr_pitch += -Kp_PITCH * dy
                    _executor.call("rotate_device", device=controller,
                                   pitch=curr_pitch, yaw=curr_yaw, roll=curr_roll)
                    time.sleep(0.05)
                else:
                    break

            if converged:
                _executor.call("click_button", controller=controller,
                               button="trigger", duration=0.1)
                time.sleep(0.15)
                typed_chars.append(char)
                print(f"Typed '{char}' successfully!")
            else:
                print(f"Failed to align to '{char}' (final error: {final_dist:.2f}px)")
                failed_chars.append(char)

        result = f"Typed {len(typed_chars)}/{len(text)} characters: '{''.join(typed_chars)}'"
        if failed_chars:
            result += f". Failed: {failed_chars}"
        logger.info(result)
        return result

    # =========================================================================
    # WHITE CANE ACCESSIBILITY
    # =========================================================================

    def white_cane_describe():
        """Immediately capture and describe the scene for a blind user."""
        _log_action("white_cane_describe")
        if not _white_cane:
            return "Error: White cane assistant not available."
        if not _white_cane.active:
            return "White cane mode is not active. Say 'white cane' to activate."
        return _white_cane.get_immediate_help()

    def white_cane_set_goal(goal: str):
        """Set or update the navigation goal for white cane mode."""
        _log_action("white_cane_set_goal", goal=goal)
        if not _white_cane:
            return "Error: White cane assistant not available."
        _white_cane.current_goal = goal
        return f"White cane goal updated: {goal}"

    # =========================================================================
    # UX / HELP
    # =========================================================================

    def provide_help():
        """Provide context-aware help based on current menu state."""
        _log_action("provide_help")
        if not _agent_ref:
            return "Agent reference not available."
        from .config import VoiceMenuState
        state = _agent_ref.menu_state
        if state == VoiceMenuState.MAIN_MENU:
            msg = "You are in the main menu. You can ask me to navigate, describe surroundings, identify objects, or click on different objects in the scene."
        elif state == VoiceMenuState.WHITE_CANE_MENU:
            msg = "White cane mode. You can update your goal, ask for a description, or say stop to exit."
        elif state == VoiceMenuState.CONFIRMATION:
            msg = f"I need you to confirm if you want to {_agent_ref.pending_action['description']}. Say confirm or cancel."
        else:
            msg = "I am ready. Say menu for options, or just tell me what to do."
        if _white_cane and hasattr(_white_cane, 'audio'):
            _white_cane.audio.speak(msg)
            return f"Provided Help: {msg}"
        print(f"Help: {msg}")
        return f"Provided Help (Printed): {msg}"

    def provide_tutorial():
        """Provide a tutorial/introduction to the agent."""
        _log_action("provide_tutorial")
        msg = ("I am your VR assistant. You can give me commands like 'find the keys' or "
               "'describe the room'. Say 'menu' to see structured options. If you get lost, say 'help'.")
        if _white_cane and hasattr(_white_cane, 'audio'):
            _white_cane.audio.speak(msg)
            return f"Provided Tutorial: {msg}"
        print(f"Tutorial: {msg}")
        return f"Provided Tutorial (Printed): {msg}"

    def provide_options():
        """List available options based on current menu state."""
        _log_action("provide_options")
        if not _agent_ref:
            return "Agent reference not available."
        from .config import VoiceMenuState
        state = _agent_ref.menu_state
        if state == VoiceMenuState.MAIN_MENU:
            msg = "Options: Navigate, Describe, Identify, Repeat, Help."
        elif state == VoiceMenuState.WHITE_CANE_MENU:
            msg = "Options: Goal, Help, Stop, Disable."
        elif state == VoiceMenuState.CONFIRMATION:
            msg = "Options: Confirm, Cancel."
        else:
            msg = "Options: Menu, White Cane, Stop, Help."
        if _white_cane and hasattr(_white_cane, 'audio'):
            _white_cane.audio.speak(msg)
            return f"Provided Options: {msg}"
        print(f"Options: {msg}")
        return f"Provided Options (Printed): {msg}"

    # =========================================================================
    # UTILITY
    # =========================================================================

    def finish_task(summary: str):
        """Call when the user's request is fully completed."""
        _log_action("finish_task", summary=summary)
        return f"Task Completed: {summary}"

    def get_connection_status():
        """Check VR driver connection status."""
        _log_action("get_connection_status")
        return _executor.call("get_connection_status")

    def kill_address():
        """Kill the process using the configured TCP port (fixes 'Address already in use')."""
        _log_action("kill_address")
        return _executor.call("kill_address")

    # =========================================================================
    # RETURN ALL TOOLS
    # =========================================================================
    return [
        # Movement & Orientation
        start_bridge, move_relative, move_absolute, teleport, rotate_device, get_current_pose,
        # Vision
        inspect_surroundings, locate_object, capture_video, explore_environment,
        # Tracking
        track_object, track_multiple_items, visual_servo_to_object,
        create_tracking_video, type_text,
        # White Cane Accessibility
        white_cane_describe, white_cane_set_goal,
        # Controller Positioning
        reset_controller_positions, reset_controller_orientation,
        position_controller_relative_to_headset, open_menu_sequence,
        # Button / Input Controls
        press_button, release_button, click_button, set_trigger,
        set_joystick, move_joystick_direction, click_trackpad_direction,
        perform_grab, perform_release, release_all_inputs, get_controller_state,
        # Utility
        finish_task, get_connection_status, kill_address,
        provide_help, provide_tutorial, provide_options,
    ]
