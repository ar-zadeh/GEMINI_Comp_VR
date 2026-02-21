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
import time
import base64
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List

from PIL import Image
from pydantic import BaseModel, Field

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from google.genai import types
except ImportError:
    pass

from .config import LOG_DIR
from .logger import get_logger

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
    Call once during GeminiAgent.__init__.
    """
    global _executor, _grounder, _tracker, _white_cane, _describer, _agent_ref
    _executor = executor
    _grounder = grounder
    _tracker = tracker
    _white_cane = white_cane
    _describer = describer
    _agent_ref = agent_ref

    # ── Shared helper ─────────────────────────────────────────────────────────

    def _log_action(tool_name, **kwargs):
        get_logger().info(f"[TOOL] {tool_name}({kwargs})")
        print(f"Action: {tool_name} {kwargs}")

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
        Uses Gemini for initial grounding, then SAM + PID for closed-loop control.
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

        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 3) is not available."

        Kp_YAW = 0.02
        Kp_PITCH = 0.02
        MAX_ITER = 100
        TOLERANCE_PX = 15

        # 1. Get initial pose
        curr_pitch = curr_yaw = curr_roll = 0.0
        status = _executor.call("get_current_pose", device=controller)
        try:
            if "Rotation: [" in status:
                rot_str = status.split("Rotation: [")[1].split("]")[0]
                rot_str = rot_str.replace("np.float64(", "").replace(")", "")
                curr_pitch, curr_yaw, curr_roll = map(float, rot_str.split(","))
        except Exception as e:
            return f"Failed to parse initial pose: {e}"

        # 2. Capture initial image
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

        # 3. Ground targets
        targets = {"ray": ray_description, "logo": object_description}
        grounding_results = _grounder.ground_multiple_objects(img_bytes, list(targets.values()))

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

        if not all_found:
            return (
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
                return "Visual Servoing Stopped by User."

            # Capture
            res = _executor.call("inspect_surroundings")
            if isinstance(res, str) and res.startswith("Error"):
                break
            try:
                data = json.loads(res).get("data")
                img_bytes_loop = base64.b64decode(data)
                pil_img_loop = Image.open(io.BytesIO(img_bytes_loop)).convert("RGB")
                img_cv = cv2.cvtColor(np.array(pil_img_loop), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Image parse error in loop: {e}")
                break

            # SAM tracking — process both objects with geometric prompts only
            inference_state = _tracker.processor.set_image(pil_img_loop)
            points = {}
            masks_for_viz = {}
            save_debug = (i % 15 == 0)  # only visualize every 15 iters

            for key, desc in targets.items():
                if key not in current_boxes:
                    continue
                box_x, box_y, box_w, box_h = current_boxes[key]
                box_input_xywh = _tracker.torch.tensor(
                    [box_x, box_y, box_w, box_h]).view(-1, 4)
                box_input_cxcywh = _tracker.box_xywh_to_cxcywh(box_input_xywh)
                norm_box_cxcywh = _tracker.normalize_bbox(
                    box_input_cxcywh, w, h).flatten().tolist()

                _tracker.processor.reset_all_prompts(inference_state)
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

                if mask is None:
                    print(f"[{key}] Lost tracking. Attempting reground...")
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
                    continue

                if save_debug:
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

            # Visualize masks — only every 15 iterations
            if save_debug:
                for key, mask in masks_for_viz.items():
                    color = (np.array([255, 100, 0] if key == "ray" else [0, 255, 100],
                                      dtype=np.uint8))
                    overlay = img_cv.copy()
                    overlay[mask] = color
                    cv2.addWeighted(overlay, 0.35, img_cv, 0.65, 0, img_cv)

            # PID control
            if "ray" in points and "logo" in points:
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
                    return "Visual Servoing Aborted: Divergence detected."

                # Save debug image — only every 15 iterations
                if save_debug:
                    cv2.line(img_cv, (rx, ry), (lx, ly), (0, 255, 255), 2)
                    timestamp = datetime.now().strftime("%H%M%S")
                    debug_path = LOG_DIR / "tracking" / f"servo_{timestamp}_iter_{i}.jpg"
                    if CV2_AVAILABLE:
                        cv2.imwrite(str(debug_path), img_cv)

                if dist < TOLERANCE_PX:
                    msg = f"Visual Servoing Complete. Aligned with {object_description} (Error: {dist:.2f}px)."
                    logger.info(msg)
                    return msg

                curr_yaw += -Kp_YAW * dx
                curr_pitch += -Kp_PITCH * dy
                _executor.call("rotate_device", device=controller,
                               pitch=curr_pitch, yaw=curr_yaw, roll=curr_roll)
            else:
                if "ray" in current_boxes and "logo" in current_boxes:
                    continue
                return "Lost tracking of one or both objects during loop. Stopping."

        if dist < 50.0:
            return f"Visual Servoing finished. Aligned within {dist:.1f}px (acceptable)."
        return f"Visual servoing finished max iterations ({MAX_ITER}). Final error: {dist:.1f}."

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

        if not _tracker or not _tracker.available:
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
    "keys": [
        {{"key": "a", "coordinates": [ymin, xmin, ymax, xmax]}},
        {{"key": "b", "coordinates": [ymin, xmin, ymax, xmax]}}
    ],
    "controller_ray": [ymin, xmin, ymax, xmax]
}}

Rules:
1. ymin, xmin, ymax, xmax must be normalized coordinates (0 to 1).
2. Include ONLY the keys you can confidently locate.
3. "controller_ray" is the bounding box of the blue VR controller ray.
"""

        class KeyDetection(BaseModel):
            key: str = Field(description="Single keyboard key label.")
            coordinates: List[float] = Field(
                description="Normalized coordinates [ymin, xmin, ymax, xmax]."
            )

        class KeyboardGroundingResponse(BaseModel):
            keys: List[KeyDetection] = Field(
                description="Detected keys with their normalized bounding boxes."
            )
            controller_ray: List[float] = Field(
                description="Normalized coordinates [ymin, xmin, ymax, xmax] for the blue VR controller ray."
            )

        try:
            out_buffer = io.BytesIO()
            pil_img_init.save(out_buffer, format="JPEG")
            clean_image_data = out_buffer.getvalue()

            response = _grounder.client.models.generate_content(
                model=_grounder.model_name,
                contents=[types.Content(role="user", parts=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=clean_image_data))
                ])],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": KeyboardGroundingResponse,
                    "thinking_config": {"thinking_budget": 0},
                }
            )

            if not response.candidates or not response.candidates[0].content.parts:
                return "Error: Gemini returned no grounding results."

            parsed_response = response.parsed
            if not parsed_response:
                parsed_response = KeyboardGroundingResponse.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Keyboard grounding failed: {e}")
            return f"Error grounding keyboard keys: {e}"

        grounded_keys = {
            item.key: item.coordinates
            for item in parsed_response.keys
            if item.key and item.coordinates and len(item.coordinates) == 4
        }
        ray_box_norm = parsed_response.controller_ray
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

                inference_state = _tracker.processor.set_image(pil_img_loop)
                points = {}
                boxes_to_track = {"ray": current_ray_box, "key": target_box}
                updated_boxes = {}
                masks_for_viz = {}

                for key, box in boxes_to_track.items():
                    box_x, box_y, box_w, box_h = box
                    box_input_xywh = _tracker.torch.tensor(
                        [box_x, box_y, box_w, box_h]).view(-1, 4)
                    box_input_cxcywh = _tracker.box_xywh_to_cxcywh(box_input_xywh)
                    norm_box_cxcywh = _tracker.normalize_bbox(
                        box_input_cxcywh, w, h).flatten().tolist()

                    _tracker.processor.reset_all_prompts(inference_state)
                    text_desc = "VR controller ray" if key == "ray" else f"keyboard key {char}"
                    inference_state = _tracker.processor.set_text_prompt(text_desc, inference_state)
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
            msg = "White cane mode. Tell me what help you need, ask for a description, or say stop to exit."
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
            msg = "Options: Help, Stop, Disable."
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
        inspect_surroundings, locate_object, capture_video,
        # Tracking
        track_object, track_multiple_items, visual_servo_to_object,
        create_tracking_video, type_text,
        # White Cane Accessibility
        white_cane_describe,
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
