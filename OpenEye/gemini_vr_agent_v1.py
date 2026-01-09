#!/usr/bin/env python3
"""
Gemini VR Agent - Simplified Single-Agent Architecture
Uses Google GenAI SDK for native tool calling and state management.
"""

import os
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
import shutil

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
SHOW_VISION_PREVIEW = True

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
        
    def ground_object(self, image_data: bytes, object_description: str) -> List[Dict]:
        logger = get_logger()
        logger.info(f"Grounding: '{object_description}'")
        
        prompt = f"""
        Find this object: {object_description}
        
        You MUST return the answer in the following JSON format:
        {{
            "thinking": "Reasoning about where the object is in the image...",
            "coordinates": [ymin, xmin, ymax, xmax]
        }}
        
        ymin, xmin, ymax, xmax must be normalized coordinates (0 to 1).
        If multiple instances are found, return the most prominent one.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data))
                    ])
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
            )
            
            # Parse result
            text = response.candidates[0].content.parts[0].text
            data = json.loads(text)
            
            # Log thinking
            if "thinking" in data:
                logger.info(f"[Grounding Thought] {data['thinking']}")
                print(f"Grounding Thought: {data['thinking']}")
            
            # Extract coordinates (handle single list)
            coords = data.get("coordinates", [])
            boxes = []
            if coords and len(coords) == 4 and isinstance(coords[0], (int, float)):
                # Found single box
                boxes.append({"box_2d": coords, "label": object_description})
                # User request: Print and Log coordinates
                coord_msg = f"Found Coordinates: {coords}"
                print(coord_msg)
                logger.info(coord_msg)
            
            # Draw and save
            self._draw_and_save(image_data, boxes, object_description)
            return boxes
        except Exception as e:
            logger.error(f"Grounding failed: {e}")
            return []

    def _draw_and_save(self, image_data: bytes, boxes: List[Dict], description: str):
        logger = get_logger()
        timestamp = datetime.now().strftime("%H%M%S")
        filename = self.log_dir / f"ground_{timestamp}_{description.replace(' ','_')}.jpg"
        
        if CV2_AVAILABLE:
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                for box in boxes:
                    # ymin, xmin, ymax, xmax
                    y1, x1, y2, x2 = box['box_2d']
                    p1 = (int(x1*w), int(y1*h))
                    p2 = (int(x2*w), int(y2*h))
                    cv2.rectangle(img, p1, p2, (0,0,0), 3)
                    cv2.putText(img, description, (p1[0], max(20, p1[1]-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                
                cv2.imwrite(str(filename), img)
                if SHOW_VISION_PREVIEW:
                    cv2.imshow("Grounding", img)
                    cv2.waitKey(2000)
                logger.info(f"Saved grounding to {filename}")
                return
            except Exception as e:
                logger.error(f"CV2 draw failed: {e}")

        # Fallback to PIL (omitted for brevity, assume CV2 for now or easy add back)

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
    
    def finish_task(summary: str):
        """Call this when the user's request is fully completed."""
        _log_action("finish_task", summary=summary)
        return f"Task Completed: {summary}"

    def get_connection_status():
        """Check VR driver connection."""
        _log_action("get_connection_status")
        return _executor.call("get_connection_status")

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
                    img.save(path)
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

    return [start_bridge, move_relative, teleport, rotate_device, inspect_surroundings, locate_object, capture_video, track_object, create_tracking_video, finish_task, get_connection_status]

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
        1. Always locate objects before interacting with them if you don't know where they are.
        2. Use `locate_object` to find things. It gives you 2D image coordinates (0.0-1.0).
        3. TRANSLATE 2D coordinates to 3D moves:
           - Object is Left (x < 0.5) -> Move Left (dx < 0).
           - Object is Right (x > 0.5) -> Move Right (dx > 0).
           - Object is High (y < 0.5) -> Move Up (dy > 0).
        4. BE DECISIVE. Do not keep looking for the same thing. Find it, then MOVE.
        5. When done, call `finish_task`.
        6. Coordinate System:
          X: Left(-)/Right(+)
          Y: Down(-)/Up(+)
          Z: Forward(-)/Back(+)
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
                
            agent.run(user_input)
        except KeyboardInterrupt:
            break
