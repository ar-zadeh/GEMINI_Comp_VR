#!/usr/bin/env python3
"""
Gemini VR Agent - Simplified Single-Agent Architecture
Uses Google GenAI SDK for native tool calling and state management.
"""

import os
import shlex
import json
import time
import base64
import traceback
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFile
import io
import math
import shutil
import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Allow loading truncated images (fixes issues with simple MJPEG streams)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import Object Tracker
try:
    from object_tracker import ObjectTracker
except ImportError:
    ObjectTracker = None
    print("Warning: ObjectTracker import failed. Tracking will be disabled.")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install google-genai: pip install google-genai")
    exit(1)

# OpenCV for drawing (optional but preferred)
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# --- CONFIG ---
GEMINI_MODEL = "gemini-3-flash-preview"  # User requested "Flash 3.0" (mapping to latest 2.0 Flash Exp)
LOG_DIR = Path("agent_logs")
SHOW_VISION_PREVIEW = False

# ============================================================================
# UTILS
# ============================================================================

class AgentLogger:
    """Simple logger setup."""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{self.session_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("VRAgent")
        
    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg) 
    def action(self, tool, args, result):
        self.info(f"[ACTION] {tool}({json.dumps(args)}) -> {str(result)[:200]}")

_logger = None
def get_logger():
    global _logger
    if not _logger: _logger = AgentLogger(LOG_DIR)
    return _logger

# ============================================================================
# VISUAL GROUNDING
# ============================================================================

class VisualGrounder:
    """Handles object detection using Gemini."""
    def __init__(self, client, model_name: str, log_dir: Path):
        self.client = client
        self.model_name = model_name
        self.log_dir = log_dir / "grounding"
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def ground_multiple_objects(self, image_data: bytes, object_names: List[str]) -> Dict[str, List[float]]:
        logger = get_logger()
        objects_str = ", ".join(object_names)
        logger.info(f"Grounding Multiple: '{objects_str}'")
        
        prompt = f"""
        Find the following objects in the image: {objects_str}.
        
        You MUST return the answer in the following JSON format:
        {{
            "thinking": "Reasoning about the scene...",
            "detections": [
                {{
                    "label": "exact_object_name_from_list",
                    "coordinates": [ymin, xmin, ymax, xmax]
                }}
            ]
        }}
        
        1. ymin, xmin, ymax, xmax must be normalized coordinates (0 to 1).
        2. Only return objects you are confident you see.
        """
        
        try:
            # 1. robust image loading (fixes Corrupt JPEG errors)
            try:
                pil_img = Image.open(io.BytesIO(image_data))
                # Convert to RGB to ensure compatibility
                pil_img = pil_img.convert("RGB")
                # Save to a clean buffer
                out_buffer = io.BytesIO()
                pil_img.save(out_buffer, format="JPEG")
                clean_image_data = out_buffer.getvalue()
            except Exception as e:
                logger.warning(f"Image cleaning failed, using raw bytes: {e}")
                clean_image_data = image_data

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=clean_image_data))
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", 

                )
            )
            
            # 2. Robust JSON Parsing (Strips Markdown)
            if not response.candidates:
                logger.error(f"Gemini returned NO candidates. Finish reason: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'Unknown'}")
                # Log full response for debugging
                logger.debug(f"Full Response: {response}")
                return {}

            if not response.candidates[0].content or not response.candidates[0].content.parts:
                 logger.error("Gemini candidate content is empty.")
                 return {}

            text = response.candidates[0].content.parts[0].text
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("\n", 1)[0]
            if text.startswith("json"):
                text = text[4:]
            
            try:
                data = json.loads(text.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini: {e}. Text: {text}")
                return {}
            
            results = {}
            valid_boxes_for_draw = []

            for det in data.get("detections", []):
                label = det.get("label")
                coords = det.get("coordinates")
                
                if label and coords and len(coords) == 4:
                    # Handle 0-1000 scale correction
                    if any(c > 1.0 for c in coords):
                        coords = [c / 1000.0 for c in coords]
                    
                    results[label] = coords
                    valid_boxes_for_draw.append({"label": label, "box_2d": coords})

            if valid_boxes_for_draw:
                self._draw_and_save(clean_image_data, valid_boxes_for_draw, f"multi_{len(results)}_objs")
            else:
                logger.warning(f"Gemini returned no detections. Raw text: {text[:100]}...")
                
            return results

        except Exception as e:
            logger.error(f"Multi-Grounding failed: {e}")
            return {}

    def ground_object(self, image_data: bytes, object_description: str) -> List[Dict]:
        # Update single grounding to use the same robust cleaning/parsing logic if you wish
        # For now, we can just wrap the new multi-grounder for simplicity:
        res_dict = self.ground_multiple_objects(image_data, [object_description])
        if object_description in res_dict:
            return [{"box_2d": res_dict[object_description], "label": object_description}]
        return []

    def _draw_and_save(self, image_data: bytes, boxes: List[Dict], description: str):
        logger = get_logger()
        timestamp = datetime.now().strftime("%H%M%S")
        filename = self.log_dir / f"ground_{timestamp}_{description.replace(' ','_')}.jpg"
        
        if CV2_AVAILABLE:
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None: return 
                
                h, w = img.shape[:2]
                colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]

                for i, box in enumerate(boxes):
                    y1, x1, y2, x2 = box['box_2d']
                    label = box.get('label', description)
                    color = colors[i % len(colors)]
                    
                    p1 = (int(x1*w), int(y1*h))
                    p2 = (int(x2*w), int(y2*h))
                    cv2.rectangle(img, p1, p2, color, 2)
                    cv2.putText(img, label, (p1[0], max(20, p1[1]-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imwrite(str(filename), img)
                logger.info(f"Saved grounding to {filename}")
            except Exception as e:
                logger.error(f"CV2 draw failed: {e}")
# ============================================================================
# EXECUTOR
# ============================================================================

class DirectMCPExecutor:
    """Loads mcp_server.py dynamically."""
    def __init__(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
        if not spec: raise ImportError("Could not find mcp_server.py")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules["mcp_server"] = self.module
        spec.loader.exec_module(self.module)
        
    def call(self, tool: str, **kwargs):
        func = getattr(self.module, tool)
        return func(**kwargs)

# ============================================================================
# TOOL FUNCTIONS (Wrapped for Gemini)
# ============================================================================

# Global instances
_executor = None
_grounder = None
_tracker = None

def _get_tools(executor, grounder, tracker):
    """Define tools as a list of callables."""
    global _executor, _grounder, _tracker
    _executor = executor
    _grounder = grounder
    _tracker = tracker
    
    def _log_action(tool_name, **kwargs):
        get_logger().info(f"[TOOL] {tool_name}({kwargs})")
        print(f"Action: {tool_name} {kwargs}")

    def start_bridge():
        """Start the VR bridge connection."""
        _log_action("start_bridge")
        return _executor.call("start_vr_bridge")
        
    def move_relative(device: str, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move a device relative to current position. dz: -ve is forward, +ve is back."""
        _log_action("move_relative", device=device, dx=dx, dy=dy, dz=dz)
        return _executor.call("move_relative", device=device, dx=dx, dy=dy, dz=dz)
        
    def teleport(device: str, x: float, y: float, z: float):
        """Teleport to exact coordinates."""
        _log_action("teleport", device=device, x=x, y=y, z=z)
        return _executor.call("teleport", device=device, x=x, y=y, z=z)
        
    def rotate_device(device: str, pitch: float, yaw: float, roll: float):
        """Rotate device (degrees)."""
        _log_action("rotate_device", device=device, pitch=pitch, yaw=yaw, roll=roll)
        return _executor.call("rotate_device", device=device, pitch=pitch, yaw=yaw, roll=roll)
        
    def inspect_surroundings():
        """Take a picture."""
        _log_action("inspect_surroundings")
        return _executor.call("inspect_surroundings")
        
    def locate_object(object_description: str):
        """Find an object and return its center coordinates."""
        _log_action("locate_object", description=object_description)
        # 1. Capture
        res = _executor.call("inspect_surroundings")
        data = json.loads(res).get("data")
        if not data: return "Failed to capture image"
        
        # 2. Ground
        img_bytes = base64.b64decode(data)
        boxes = _grounder.ground_object(img_bytes, object_description)
        
        if not boxes: return f"Could not find '{object_description}'"
        
        # 3. Format
        results = []
        for box in boxes:
            y1, x1, y2, x2 = box['box_2d']
            cx, cy = (x1+x2)/2, (y1+y2)/2
            results.append(f"Found at Center(x={cx:.2f}, y={cy:.2f})")
            
        return "; ".join(results)
    def track_multiple_items(object_names: List[str]):
        """
        Track multiple objects simultaneously in a video.
        Example input: ["red cup", "keyboard", "blue pen"]
        """
        _log_action("track_multiple_items", objects=object_names)
        
        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 3) is not available."

        # 1. Capture Video
        print(f"Capturing video to track: {object_names}...")
        res_str = _executor.call("capture_video", duration=3.0)
        
        # Parse video response
        try:
            res = json.loads(res_str)
            frames = res.get("frames", [])
            if not frames: return "Error: No frames captured."
        except:
            return "Error parsing video data."

        # Save Frames
        timestamp = datetime.now().strftime("%H%M%S")
        temp_dir = LOG_DIR / "tracking" / f"multi_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        for i, b64 in enumerate(frames):
            path = temp_dir / f"frame_{i:04d}.jpg"
            img_data = base64.b64decode(b64)
            # Robust Save
            try:
                with Image.open(io.BytesIO(img_data)) as img:
                    img.save(path, quality=95)  # High quality JPEG
            except:
                with open(path, "wb") as f: f.write(img_data)
            saved_frames.append(path)

        # Grounding
        print("Locating objects in first frame...")
        with open(saved_frames[0], "rb") as f:
            first_frame_data = f.read()

        # Call the new multi-grounder
        # Returns Dict: {'cup': [y,x,y,x], 'keyboard': [y,x,y,x]}
        initial_data = _grounder.ground_multiple_objects(first_frame_data, object_names)
        if not initial_data:
            print(f"FAILED: Grounding found 0 objects. Expected: {object_names}")
            # This return string tells the agent to try fallback
            return f"Could not locate any of the requested objects: {object_names}"
            
        print(f"Found {len(initial_data)} objects: {list(initial_data.keys())}")

        # Tracking
        print("Running SAM 3 Multi-Tracking...")
        result = _tracker.track_multi_objects(str(temp_dir), initial_data)
        
        if "error" in result:
            return f"Tracking failed: {result['error']}"
            
        video_path = result.get("video_path")
        print(f"\n[SUCCESS] Multi-Tracking Video: {video_path}\n")
        get_logger().info(f"Multi-Tracking Video: {video_path}")
        
        return f"Tracking Complete. Video saved to {video_path}"
    def finish_task(summary: str):
        """Call this when the user's request is fully completed."""
        _log_action("finish_task", summary=summary)
        return f"Task Completed: {summary}"

    def get_connection_status():
        """Check VR driver connection."""
        _log_action("get_connection_status")
        return _executor.call("get_connection_status")

    def kill_address():
        """
        Kills the process using the configured TCP port (default 5555).
        Use this when the server fails to start with "Address already in use".
        """
        _log_action("kill_address")
        return _executor.call("kill_address")

    def capture_video(duration: float = 3.0):
        """Capture a short video clip (sequence of frames) and save it."""
        _log_action("capture_video", duration=duration)
        res_str = _executor.call("capture_video", duration=duration)
        
        # Auto-save logic
        try:
            res = json.loads(res_str)
            if res.get("type") == "video":
                timestamp = datetime.now().strftime("%H%M%S")
                # Create a video directory
                vid_dir = LOG_DIR / "videos" / f"video_{timestamp}"
                vid_dir.mkdir(parents=True, exist_ok=True)
                
                # Save frames
                frames = res.get("frames", [])
                for i, b64 in enumerate(frames):
                    with open(vid_dir / f"frame_{i:04d}.jpg", "wb") as f:
                        f.write(base64.b64decode(b64))
                
                print(f"Video saved to: {vid_dir}")
                get_logger().info(f"Video saved to {vid_dir}")
                
                # Return a summary to the agent, not the huge base64 blob to save tokens
                return f"Video captured and saved to {vid_dir} ({len(frames)} frames)."
        except Exception as e:
            get_logger().error(f"Failed to auto-save video: {e}")
            
        return res_str

    def track_object(object_description: str):
        """
        Track an object in a video.
        1. Captures a 3-second video.
        2. Finds the object in the first frame.
        3. Tracks it using SAM 2.
        """
        _log_action("track_object", description=object_description)
        logger = get_logger()
        
        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 2) is not available."

        # 1. Capture Video
        print("Capturing video...")
        res_str = _executor.call("capture_video", duration=3.0)
        if "Error" in res_str and not res_str.startswith("{"): return res_str
        
        try:
            res = json.loads(res_str)
            if res.get("type") != "video": return f"Error capturing video: {res_str[:100]}"
            frames = res.get("frames", [])
            if not frames: return "Error: No frames in video."
        except json.JSONDecodeError:
            return f"Error parsing video response: {res_str[:100]}"

        # 2. Save Frames to Temp Dir
        timestamp = datetime.now().strftime("%H%M%S")
        temp_dir = LOG_DIR / "tracking" / f"temp_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        for i, b64 in enumerate(frames):
            path = temp_dir / f"frame_{i:04d}.jpg"
            # Decode the base64 string to get the JPEG byte data. 
            # This is required because the server sends images as base64 strings in JSON.
            img_data = base64.b64decode(b64)
            
            # Use PIL to save, which handles/fixes truncated JPEGs gracefully.
            # This "cleans" the file structure so OpenCV won't complain about corruption.
            try:
                with Image.open(io.BytesIO(img_data)) as img:
                    img.save(path, quality=95)  # High quality JPEG
            except Exception as e:
                # Fallback if image is too broken even for PIL
                print(f"Warning: PIL failed to clean frame {i}: {e}. Saving raw.")
                with open(path, "wb") as f:
                    f.write(img_data)
            
            saved_frames.append(path)
            
        # 3. Ground in First Frame
        print("Locating object in first frame...")
        # Re-read the first frame from disk to ensure we use the 'fixed' version
        with open(saved_frames[0], "rb") as f:
            first_frame_data = f.read()

        boxes = _grounder.ground_object(first_frame_data, object_description)
        
        if not boxes:
            return f"Could not find '{object_description}' in the first frame to start tracking."
            
        # Select best box (first one)
        init_box = boxes[0]['box_2d']
        
        # 4. Track
        print("Running SAM 2 Tracking...")
        video_output = _tracker.track(str(temp_dir), init_box, object_description)
        
        # Explicitly print the location for the user
        print(f"\n[SUCCESS] Tracking Video Saved to: {video_output}\n")
        get_logger().info(f"Tracking Video Saved to: {video_output}")
        
        return f"Tracking completed. Output video: {video_output}"

    def create_tracking_video(object_description: str):
        """
        Creates a segmented video of the specified object.
        This function captures a video, locates the object, and then generates
        a video with the object segmented and tracked.
        """
        return track_object(object_description)

    def visual_servo_to_object(object_description: str):
        """
        Rotates the controller (modifies yaw/pitch) so that the 'blue VR controller ray' aligns with the center of the specified target object.
        Use this when the user asks to 'align', 'point', or 'aim' the controller or its ray at something.
        """
        _log_action("visual_servo_to_object", description=object_description)
        logger = get_logger()

        if not _tracker or not _tracker.available:
            return "Error: Object Tracking (SAM 3) is not available."

        # Config
        Kp_YAW = 0.05   # Gain for Horizontal (Yaw) - Increased for faster convergence
        Kp_PITCH = 0.05 # Gain for Vertical (Pitch) - Increased for faster convergence
        MAX_ITER = 100
        TOLERANCE_PX = 5  # Stop if within this many pixels
        
        # 1. Get Initial Pose
        print("Getting initial pose of controller2...")
        curr_pitch = 0.0
        curr_yaw = 0.0
        curr_roll = 0.0
        
        status = _executor.call("get_current_pose", device="controller2")
        try:
            if "Rotation: [" in status:
                rot_str = status.split("Rotation: [")[1].split("]")[0]
                curr_pitch, curr_yaw, curr_roll = map(float, rot_str.split(","))
                print(f"Initial Rotation: Pitch={curr_pitch}, Yaw={curr_yaw}, Roll={curr_roll}")
        except Exception as e:
            msg = f"Failed to parse initial pose: {e}"
            logger.error(msg)
            return msg

        # 2. Capture Initial Image for Grounding
        print("Capturing initial image for grounding...")
        res = _executor.call("inspect_surroundings")
        if isinstance(res, str) and res.startswith("Error"): return f"Capture failed: {res}"
        
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            # Use PIL to clean/load
            pil_img_init = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img_init.size
        except Exception as e:
            return f"Initial image parsing failed: {e}"

        # 3. Ground Targets (Gemini - Once)
        targets = {
            "ray": "blue VR controller ray",
            "logo": object_description
        }
        current_boxes = {}
        
        print(f"Grounding '{targets['ray']}' and '{targets['logo']}'...")
        # We can use the multi-object grounder
        grounding_results = _grounder.ground_multiple_objects(img_bytes, list(targets.values()))
        
        all_found = True
        for key, desc in targets.items():
            if desc in grounding_results:
                # Normalized [ymin, xmin, ymax, xmax] -> Pixel [x, y, w, h]
                ymin, xmin, ymax, xmax = grounding_results[desc]
                box_x, box_y = xmin * w, ymin * h
                box_w, box_h = (xmax - xmin) * w, (ymax - ymin) * h
                current_boxes[key] = [box_x, box_y, box_w, box_h]
            else:
                print(f"[{key}] '{desc}' NOT found by Gemini.")
                all_found = False
        
        if not all_found:
             return f"Failed to find both the controller ray and '{object_description}' in the initial view. Cannot start servoing."
             
        print("Initial grounding successful. Starting control loop.")

        # 4. Control Loop
        for i in range(MAX_ITER):
            print(f"\n--- Servo Iteration {i+1}/{MAX_ITER} ---")
            
            # A. Capture New Image
            res = _executor.call("inspect_surroundings")
            if isinstance(res, str) and res.startswith("Error"):
                print("Capture failed in loop.")
                break
                
            try:
                data = json.loads(res).get("data")
                img_bytes_loop = base64.b64decode(data)
                pil_img_loop = Image.open(io.BytesIO(img_bytes_loop)).convert("RGB")
                img_cv = cv2.cvtColor(np.array(pil_img_loop), cv2.COLOR_RGB2BGR) # For Vis
            except Exception as e:
                print(f"Image parse error in loop: {e}")
                break

            # B. SAM Tracking (Update state with new image)
            inference_state = _tracker.processor.set_image(pil_img_loop)
            
            points = {}
            
            for key, desc in targets.items():
                if key not in current_boxes: continue # Should not happen unless lost
                
                box_x, box_y, box_w, box_h = current_boxes[key]
                
                # Format box for SAM
                box_input_xywh = _tracker.torch.tensor([box_x, box_y, box_w, box_h]).view(-1, 4)
                box_input_cxcywh = _tracker.box_xywh_to_cxcywh(box_input_xywh)
                norm_box_cxcywh = _tracker.normalize_bbox(box_input_cxcywh, w, h).flatten().tolist()
                
                _tracker.processor.reset_all_prompts(inference_state)
                inference_state = _tracker.processor.add_geometric_prompt(
                    state=inference_state, box=norm_box_cxcywh, label=True
                )
                
                # Get Mask
                mask = None
                if "masks" in inference_state and inference_state["masks"] is not None:
                    m = inference_state["masks"].detach().cpu().numpy() > 0.5
                    if m.ndim == 4: mask = m[0, 0]
                    elif m.ndim == 3: mask = m[0]
                
                if mask is None:
                    print(f"[{key}] Lost tracking (no mask).")
                    del current_boxes[key]
                    continue
                    
                # Update Box & Extract Point
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    current_boxes[key] = [cmin, rmin, cmax - cmin, rmax - rmin]
                    
                    # Points logic
                    if key == "ray":
                        # Ray Tip = Min Y (Highest point visually)
                        ys, xs = np.where(mask)
                        if len(ys) > 0:
                            idx = np.argmin(ys) 
                            points[key] = (xs[idx], ys[idx])
                            cv2.circle(img_cv, (xs[idx], ys[idx]), 5, (0, 0, 255), -1) # Red Tip
                    elif key == "logo":
                        # Centroid
                        M = cv2.moments(mask.astype(np.uint8))
                        if M["m00"] != 0:
                            cx = int(M["m10"]/M["m00"])
                            cy = int(M["m01"]/M["m00"])
                            points[key] = (cx, cy)
                            cv2.circle(img_cv, (cx, cy), 5, (0, 255, 0), -1) # Green center
                else:
                    print(f"[{key}] Mask empty.")
                    del current_boxes[key]
            
            # C. PID & Control
            if "ray" in points and "logo" in points:
                rx, ry = points["ray"]
                lx, ly = points["logo"]
                
                dx = lx - rx
                dy = ly - ry
                dist = math.sqrt(dx*dx + dy*dy)
                
                print(f"Error: dx={dx}, dy={dy}, dist={dist:.2f}")
                
                # Save debug image with telemetry
                cv2.line(img_cv, (rx, ry), (lx, ly), (0, 255, 255), 2)
                
                # Prepare info text
                # We calculate PID terms here just for display before applying them
                d_yaw_disp = Kp_YAW * dx
                d_pitch_disp = Kp_PITCH * dy
                info_text = f"Err: {dist:.1f}px | dYaw: {d_yaw_disp:.2f} | dPitch: {d_pitch_disp:.2f}"
                
                # Draw text background
                (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_cv, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
                cv2.putText(img_cv, info_text, (15, 15 + th), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                timestamp = datetime.now().strftime("%H%M%S")
                debug_path = LOG_DIR / "tracking" / f"servo_{timestamp}_iter_{i}.jpg"
                cv2.imwrite(str(debug_path), img_cv)
                logger.info(f"Saved servo debug image: {debug_path}")
                
                if dist < TOLERANCE_PX:
                    return f"Visual Servoing Complete. Aligned with {object_description} (Error: {dist:.2f}px)."
                
                # PID
                # INVERTED LOGIC: Previous test showed error increasing with positive gain.
                # If dx < 0 (Target Left), we need to move Left. 
                # Observation showed Yaw- moved Right. So we need Yaw+ to move Left.
                # So if dx is negative, we need dYaw positive -> Invert sign.
                d_yaw = -Kp_YAW * dx
                d_pitch = -Kp_PITCH * dy
                
                curr_yaw += d_yaw
                curr_pitch += d_pitch
                
                print(f"Adjusting: dYaw={d_yaw:.2f}, dPitch={d_pitch:.2f} -> New Yaw={curr_yaw:.2f}, Pitch={curr_pitch:.2f}")
                
                _executor.call("rotate_device", device="controller2", 
                               pitch=curr_pitch, yaw=curr_yaw, roll=curr_roll)
                               
                # Short wait for physical movement
                time.sleep(2)
                
                # Check actual rotation
                status_check = _executor.call("get_current_pose", device="controller2")
                try:
                    if "Rotation: [" in status_check:
                        rot_str = status_check.split("Rotation: [")[1].split("]")[0]
                        p, y, r = map(float, rot_str.split(","))
                        print(f"Current controller rotation values: Pitch={p:.2f}, Yaw={y:.2f}, Roll={r:.2f}")
                except Exception as e:
                    print(f"Could not read back pose: {e}")
                
            else:
                 return f"Lost tracking of one or both objects during loop. Stopping."

        return f"Visual servoing finished max iterations ({MAX_ITER}). Final error: {dist if 'dist' in locals() else 'Unknown'}."

    return [
        start_bridge, move_relative, teleport, rotate_device, 
        inspect_surroundings, locate_object, capture_video, 
        track_object, track_multiple_items, visual_servo_to_object, 
        create_tracking_video, finish_task, get_connection_status, kill_address
    ]
# ============================================================================
# AGENT
# ============================================================================

class GeminiAgent:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key: raise ValueError("GEMINI_API_KEY not set")
        
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        self.logger = get_logger()
        
        self.executor = DirectMCPExecutor()
        self.grounder = VisualGrounder(self.client, GEMINI_MODEL, LOG_DIR)
        
        if ObjectTracker:
            self.tracker = ObjectTracker(LOG_DIR)
        else:
            self.tracker = None
        
        # Define tools
        self.tools = _get_tools(self.executor, self.grounder, self.tracker)
        
        # Create Chat Session
        system_prompt = """You are an expert VR Agent.
        You control a headset and controllers.
        
        RULES:
        1. Always locate objects before interacting with them.
        2. Use `locate_object` for single items.
        3. Use `track_multiple_items` if the user asks to track specifically multiple things (e.g., "Track the cup and the bottle").
        4. Use `visual_servo_to_object` to ALIGN or POINT the controller/ray at an object.
        5. TRANSLATE 2D coordinates to 3D moves:
           - Object is Left (x < 0.5) -> Move Left (dx < 0).
           - Object is Right (x > 0.5) -> Move Right (dx > 0).
        6. BE DECISIVE. Do not keep looking for the same thing. Find it, then MOVE.
        7. When done, call `finish_task`.
        """
        
        self.chat = self.client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=self.tools,
                max_output_tokens=2048,
                temperature=0.5,
                automatic_function_calling=dict(disable=False) 
            )
        )
        
        # Start bridge immediately
        self.executor.call("start_vr_bridge")
        
    def reset_chat(self):
        """Reset the chat session."""
        print("Resetting chat session...")
        self.chat = self.client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=self.chat._config.system_instruction,
                tools=self.tools,
                max_output_tokens=2048,
                temperature=0.5,
                automatic_function_calling=dict(disable=False) 
            )
        )
        self.logger.info("Chat session reset.")
        print("Chat reset.")

    def print_status(self):
        """Print current agent status."""
        print(f"\n--- Status ---")
        print(f"Model: {GEMINI_MODEL}")
        print(f"Log Dir: {LOG_DIR}")
        try:
            status = self.executor.call("get_connection_status")
            print(f"VR Bridge: {status}")
        except Exception as e:
            print(f"VR Bridge: Error getting status ({e})")
        print("--------------")

    def run(self, user_input: str):
        self.logger.info(f"User: {user_input}")
        print("\nAgent is thinking...")
        
        try:
            # Send message and let SDK handle tool loop
            response = self.chat.send_message(user_input)
            print(f"\nAgent: {response.text}")
            self.logger.info(f"Agent: {response.text}")
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            traceback.print_exc()

    def handle_direct_command(self, user_input: str):
        """
        Parses and executes a direct command in the format ((function arg1 arg2 ...))
        Arguments are automatically converted to int/float/bool/None if possible.
        """
        try:
            # Strip (( and ))
            content = user_input[2:-2].strip()
            if not content:
                print("Empty direct command.")
                return

            # Parse with shlex (handles quoted strings)
            parts = shlex.split(content)
            func_name = parts[0]
            raw_args = parts[1:]
            
            # Convert args
            args = []
            for arg in raw_args:
                if arg.lower() == 'true':
                    args.append(True)
                elif arg.lower() == 'false':
                    args.append(False)
                elif arg.lower() == 'none':
                    args.append(None)
                else:
                    try:
                        if '.' in arg:
                            args.append(float(arg))
                        else:
                            args.append(int(arg))
                    except ValueError:
                        args.append(arg) # Keep as string
            
            print(f"Direct Execution: {func_name}({args})")
            self.logger.info(f"Direct Execution: {func_name}({args})")

            # 1. Check Agent Tools (Wrapped functions)
            # self.tools is a list of callables
            tool_func = next((t for t in self.tools if t.__name__ == func_name), None)
            
            if tool_func:
                # Introspect to map args if needed, or just pass *args
                # Since our tools are simple python functions, *args usually works 
                # if the user provided them in order.
                import inspect
                sig = inspect.signature(tool_func)
                
                # Simple binding attempt
                try:
                    bound_args = sig.bind(*args)
                    bound_args.apply_defaults()
                    res = tool_func(*bound_args.args, **bound_args.kwargs)
                    print(f"Result: {res}")
                    self.logger.info(f"Result: {res}")
                    return
                except TypeError as e:
                    print(f"Argument mismatch for tool '{func_name}': {e}")
                    return

            # 2. Check MCP Server directly (underlying functions)
            # This allows calling functions not exposed as agent tools
            if hasattr(self.executor.module, func_name):
                func = getattr(self.executor.module, func_name)
                try:
                    res = func(*args)
                    print(f"Result: {res}")
                    self.logger.info(f"Result: {res}")
                    return
                except Exception as e:
                     print(f"Error executing MCP function '{func_name}': {e}")
                     return

            print(f"Error: Function '{func_name}' not found in Agent Tools or MCP Server.")

        except Exception as e:
            print(f"Failed to execute direct command: {e}")
            traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    agent = GeminiAgent()
    print("VR Agent Ready. Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input: continue
            
            cmd = user_input.lower()
            if cmd in ['quit', 'exit']:
                break
            elif cmd == 'reset':
                agent.reset_chat()
                continue
            elif cmd == 'status':
                agent.print_status()
                continue
                
            # Direct Command Check
            if user_input.startswith("((") and user_input.endswith("))"):
                agent.handle_direct_command(user_input)
                continue

            agent.run(user_input)
        except KeyboardInterrupt:
            break