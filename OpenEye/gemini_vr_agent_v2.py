#!/usr/bin/env python3
"""
Gemini VR Agent V2 - System 1 (Visual Grounding) + System 2 (SAM3 Tracking + Control)
"""

import os
import json
import time
import base64
import traceback
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image, ImageFile
import io

# Import Control Logic
try:
    from control_logic import calculate_control_action
except ImportError:
    print("Error: control_logic.py not found.")
    exit(1)

# Import Object Tracker
try:
    from object_tracker import ObjectTracker
except ImportError:
    ObjectTracker = None
    print("Warning: ObjectTracker import failed.")

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

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIG ---
GEMINI_MODEL = "gemini-3-flash-preview" 
# Note: Using Flash Exp as requested for speed/multimodal, labeled as '3' in prompt but mapped to latest.
LOG_DIR = Path("agent_logs_v2")

# ============================================================================
# UTILS
# ============================================================================

class AgentLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{self.session_id}.log"
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(self.log_file), logging.StreamHandler()])
        self.logger = logging.getLogger("VRAgentV2")
        
    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def action(self, tool, args, result):
        self.info(f"[ACTION] {tool} -> {str(result)[:200]}")

_logger = None
def get_logger():
    global _logger
    if not _logger: _logger = AgentLogger(LOG_DIR)
    return _logger

# ============================================================================
# VISUAL GROUNDING (System 1)
# ============================================================================

class VisualGrounder:
    def __init__(self, client, model_name: str, log_dir: Path):
        self.client = client
        self.model_name = model_name
        self.log_dir = log_dir / "grounding"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
    def ground_multiple(self, image_data: bytes, objects: List[str]) -> Dict[str, List[float]]:
        """
        Finds multiple objects in one pass.
        Returns: {'label': [ymin, xmin, ymax, xmax], ...}
        """
        logger = get_logger()
        logger.info(f"Grounding multiple: {objects}")
        
        obj_str = ", ".join(objects)
        prompt = f"""
        Find these objects: {obj_str}
        
        Return JSON format:
        {{
            "found_objects": [
                {{
                    "label": "exact label from list",
                    "box_2d": [ymin, xmin, ymax, xmax] 
                }}
            ]
        }}
        ymin, xmin, ymax, xmax must be 0-1.
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
            
            data = json.loads(response.candidates[0].content.parts[0].text)
            results = {}
            for item in data.get("found_objects", []):
                # Normalize label to match input (basic cleanup)
                for req_obj in objects:
                    if req_obj.lower() in item["label"].lower():
                         results[req_obj] = item["box_2d"]
            
            logger.info(f"Grounding results: {results.keys()}")
            return results
        except Exception as e:
            logger.error(f"Grounding failed: {e}")
            return {}

# ============================================================================
# EXECUTOR
# ============================================================================

class DirectMCPExecutor:
    def __init__(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
        if not spec: raise ImportError("Could not find mcp_server.py")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules["mcp_server"] = self.module
        spec.loader.exec_module(self.module)
        
    def call(self, tool: str, **kwargs):
        return getattr(self.module, tool)(**kwargs)

# ============================================================================
# TOOLS
# ============================================================================

_executor = None
_grounder = None
_tracker = None

def _get_tools(executor, grounder, tracker):
    global _executor, _grounder, _tracker
    _executor = executor
    _grounder = grounder
    _tracker = tracker
    
    # --- Basic Wrappers ---
    def start_bridge(): return _executor.call("start_vr_bridge")
    def get_connection_status(): return _executor.call("get_connection_status")
    def move_relative(dx: float = 0, dy: float = 0, dz: float = 0):
        # Default to controller1 (Right usually)
        return _executor.call("move_relative", device="controller1", dx=dx, dy=dy, dz=dz)
    def click_trigger():
        return _executor.call("click_button", controller="controller1", button="trigger")
    def finish_task(summary: str): return f"Task Completed: {summary}"

    def align_and_click(effector_desc: str, target_desc: str):
        """
        Autonomous skill: Aligns the effector with the target and clicks.
        Uses System 1 (Grounding) + System 2 (Tracking) + Control Logic.
        """
        logger = get_logger()
        logger.info(f"Aligning {effector_desc} -> {target_desc}")
        
        MAX_STEPS = 5
        
        for step in range(MAX_STEPS):
            # 1. Capture 1 Frame (Snapshot for decision)
            res_str = _executor.call("capture_video", duration=0.5) # Short burst
            try:
                res = json.loads(res_str)
                frames = res.get("frames", [])
                if not frames: return "Error: No vision."
                
                # Save first frame to temp
                timestamp = datetime.now().strftime("%H%M%S")
                temp_dir = LOG_DIR / "temp" / f"step_{step}_{timestamp}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save just the first frame for grounding
                img_path = temp_dir / "frame_0000.jpg"
                img_bytes = base64.b64decode(frames[0])
                with open(img_path, "wb") as f: f.write(img_bytes)
                
            except Exception as e:
                return f"Vision Error: {e}"

            # 2. System 1: Grounding
            # We treat the single frame as a 'video' of length 1 for consistency
            boxes = _grounder.ground_multiple(img_bytes, [effector_desc, target_desc])
            
            # Print Grounding Results for Debugging
            print(f"Grounding found: {boxes.keys()}")

            if len(boxes) < 2:
                print(f"Align Failed: Needed 2, found {len(boxes)}. Keys: {list(boxes.keys())}")
                # Try simple retry logic or just continue hoping SAM can track? No, need boxes.
                return f"Alignment Failed: Could not find both objects. Found: {list(boxes.keys())}"
            
            # 3. System 2: SAM Tracking (Refining centroid)
            # We run tracking on this single frame (or short burst) to get precise mask centroids
            track_res = _tracker.track_multi_objects(str(temp_dir), boxes)
            
            # Print Tracking Results for Debugging
            if "telemetry" in track_res and track_res["telemetry"]:
                print(f"Tracking Telemetry: {track_res['telemetry'][0]}")

            if "error" in track_res:
                return f"Tracking Error: {track_res['error']}"
                
            telemetry = track_res.get("telemetry", [])
            
            # 4. Control Logic
            action = calculate_control_action(telemetry, effector_desc, target_desc)
            
            logger.info(f"Step {step}: {action}")
            print(f"Step {step}: {action}")

            if not action["action_needed"]:
                # Success!
                print("Target Aligned. Clicking...")
                # Use the executor to actually click
                _executor.call("click_button", controller="controller1", button="trigger")
                return f"Success: Aligned and clicked {target_desc}."
            
            # Execute Move
            # NOTE: We must ensure we are passing valid arguments to the MCP tool
            # The tool definition is: move_relative(device: str, dx: float = 0, dy: float = 0, dz: float = 0)
            dx = action.get("dx", 0.0)
            dy = action.get("dy", 0.0)
            
            print(f"Exec Move: dx={dx:.3f}, dy={dy:.3f}")
            _executor.call("move_relative", device="controller1", dx=dx, dy=dy, dz=0)
            
            # Important: Wait for driver to process and scene to update
            time.sleep(1.0) 
            
        return "Failed: Max steps reached without alignment."

    return [start_bridge, get_connection_status, move_relative, click_trigger, align_and_click, finish_task]

# ============================================================================
# AGENT SETUP
# ============================================================================

class GeminiAgentV2:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})
        self.executor = DirectMCPExecutor()
        self.grounder = VisualGrounder(self.client, GEMINI_MODEL, LOG_DIR)
        self.tracker = ObjectTracker(LOG_DIR) if ObjectTracker else None
        
        self.tools = _get_tools(self.executor, self.grounder, self.tracker)
        
        system_prompt = """You are an autonomous VR Agent.
        Your goal is to manipulate the environment.
        
        MAJOR SKILL: `align_and_click`
        Use this tool to physically interact with buttons or objects. 
        You do not need to calculate coordinates yourself. Just identify the "Hand/Pointer" and the "Target".
        
        Example: "Click the red button" 
        -> align_and_click(effector_desc="VR controller laser", target_desc="red button")
        """
        
        self.chat = self.client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=self.tools,
                temperature=0.5
            )
        )
        self.executor.call("start_vr_bridge")

    def run(self, user_input: str):
        print("\nAgent V2 is thinking...")
        try:
            response = self.chat.send_message(user_input)
            print(f"\nAgent: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    agent = GeminiAgentV2()
    print("VR Agent V2 Ready (System 2 Control).")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input: continue
            if user_input.lower() == 'quit': break
            agent.run(user_input)
        except KeyboardInterrupt:
            break
