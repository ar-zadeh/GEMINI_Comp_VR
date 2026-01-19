#!/usr/bin/env python3
import os
import json
import time
import base64
import math
import io
import sys
import importlib.util
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List

LOG_DIR = Path("logs")

# Load Custom Modules
try:
    from object_tracker import ObjectTracker
except ImportError:
    print("Error: object_tracker.py not found. Tracking disabled.")
    sys.exit(1)

load_dotenv()

## Object Tracking Schema
class DetectedObject(BaseModel):
    label: str = Field(description="The name of the detected object.")
    box_2d: list[int] = Field(
        ..., description="2D Bounding box in [ymin, xmin, ymax, xmax] format. Use 0-1000 scale."
    )

class SceneDetection(BaseModel):
    detections: List[DetectedObject]

class RealTimeTracker:
    def __init__(self):
        # 1. Setup Gemini (Only used ONCE at the start to find the object)
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={'api_version': 'v1alpha'})
        self.model_name = "gemini-3-flash-preview"

        # 2. Setup MCP (VR Bridge)
        spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
        self.mcp = importlib.util.module_from_spec(spec)
        sys.modules["mcp_server"] = self.mcp
        spec.loader.exec_module(self.mcp)
        self.mcp.start_vr_bridge()

        # 3. Setup SAM Tracker
        print("Loading SAM Tracker... (This allows the 5090 to stretch its legs)")
        self.tracker = ObjectTracker(log_dir=Path("logs"))
        print("Tracker Loaded.")

    def get_image(self):
        """Helper to get and decode image from VR."""
        res = json.loads(self.mcp.inspect_surroundings())
        img_bytes = base64.b64decode(res["data"])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return pil_img
        # return img_bytes

    def find_initial_box(self, object_names: List[str], pil_img):
        """Finds multiple objects at once.
        Returns a dict: {"logo": [x,y,w,h], "ray": [x,y,w,h]} in xywh format.
        """
        name_str = ", ".join(object_names)
        print(f"[SKIPPING] Asking Gemini to find: '{name_str}'...")
        
        
        # prompt = f"""Return bounding box for: {name_str}. 
        # JSON format: {{ "box_2d": [ymin, xmin, ymax, xmax] }} (0-1000 scale)"""

        prompt = f"Find the following objects: {name_str}"
        
        # Convert image for API
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        
        try:
            # response = self.client.models.generate_content(
            #     model=self.model_name,
            #     contents=[types.Content(role="user", parts=[
            #         types.Part(text=prompt),
            #         types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_byte_arr.getvalue()))
            #     ])],
            #     config={
            #         "response_mime_type": "application/json",
            #         "response_json_schema": SceneDetection.model_json_schema()
            #     }
            # )
            
            # result = SceneDetection.model_validate_json(response.text)
            # print("Gemini found the following objects:", result)
            # w, h = pil_img.size
            # found_objects = {}
            
            # for det in result.detections:
            #     # Normalize 0-1000 -> Pixels
            #     ymin, xmin, ymax, xmax = [c / 1000.0 for c in det.box_2d]
            #     # Convert to xywh for SAM
            #     box_xywh = [xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h]
            #     found_objects[det.label] = box_xywh
            #     print(f" - Found '{det.label}': {det.box_2d}")
            
            found_objects = {
                'steam_logo': [526.08, 490.32, 97.91999999999999, 61.559999999999995], 
                'blue VR controller ray': [1034.88, 606.96, 437.76, 471.9599999999999]
            }
            print(found_objects)

            # Quickly save this image for debugging
            debug_dir = LOG_DIR / "debug_frames"
            os.makedirs(debug_dir, exist_ok=True)
            vis_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            for obj_name, box in found_objects.items():
                bx, by, bw, bh = map(int, box)
                cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(vis_img, obj_name, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, "initial_detection.png"), vis_img)

            return found_objects
        except Exception as e:
            print(f"Gemini failed: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"[DEBUG] Raw response: {response.text}")
            return None

    def track_loop(self, target_name):
        # Setup Debugging
        debug_dir = LOG_DIR / "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)


        ray_name = "blue VR controller ray"
        track_list = [target_name, ray_name]

        # 1. Initialization Phase
        print("--- Phase 1: Acquiring Target ---")
        pil_img = self.get_image()
        initial_boxes = self.find_initial_box(track_list, pil_img)

        if target_name not in initial_boxes or ray_name not in initial_boxes:
            print(f"Error: Could not find both objects. Found: {list(initial_boxes.keys())}")
            print("Tip: Ensure the controller ray is visible in the headset view.")
            return

        print(f"Targets acquired! Servoing '{ray_name}' -> '{target_name}'")
        print("Press Ctrl+C to stop.")

        # Calculate Static Target Center (once)
        t_box = initial_boxes[target_name]
        # Box format is [x, y, w, h]
        tx = t_box[0] + (t_box[2] / 2)
        ty = t_box[1] + (t_box[3] / 2)
        print(f"Target Locked at Screen Coords: ({tx:.0f}, {ty:.0f})")

        # Setup dynamic Ray Tracking
        current_ray_box = initial_boxes[ray_name]
        # Control Gains (Tune these!)
        KP_YAW = 0.01   # Lower is smoother, Higher is faster/jittery
        KP_PITCH = 0.01
        
        # Get initial rotation to control relative movement
        pose_str = self.mcp.get_current_pose(device="controller2")
        rot_part = pose_str.split("Rotation: [")[1].split("]")[0]
        curr_pitch, curr_yaw, _ = map(float, rot_part.split(","))

        # 2. High-Speed Servo Loop
        frame_count = 0
        trigger_held = False
        while True:
            frame_count += 1
            loop_start = time.time()
            
            # A. Capture
            pil_img = self.get_image()
            w, h = pil_img.size

            vis_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # B. Inference (SAM)
            # We use the previous box as a prompt for the new frame
            inference_state = self.tracker.processor.set_image(pil_img)

            # Debug Visualize The Box (before it goes into SAM)
            bx, by, bw, bh = map(int, current_ray_box)
            cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

            ## also visualize the target
            # Debug Visualize the Target Box
            bx, by, bw, bh = map(int, t_box)
            cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
           
            # Format box
            box_tensor = self.tracker.torch.tensor(current_ray_box).view(-1, 4)
            cxcywh = self.tracker.box_xywh_to_cxcywh(box_tensor)
            norm_box = self.tracker.normalize_bbox(cxcywh, w, h).flatten().tolist()
            
            self.tracker.processor.reset_all_prompts(inference_state)
            inf_state = self.tracker.processor.add_geometric_prompt(
                state=inference_state, box=norm_box, label=True
            )
            
            # Extract Mask & Update Box
            raw_masks = inf_state["masks"]

            # Safety check
            if raw_masks is None or raw_masks.numel() == 0:
                mask = np.array([]) # Create empty array to force lost tracking
            else:
                # Convert to numpy
                mask = (raw_masks.detach().cpu().numpy() > 0.5).squeeze()

                # Handle dimensionality mismatch    
                if mask.ndim == 4: mask = mask[0, 0]
                elif mask.ndim == 3: 
                    if mask.shape[0] > 0:
                        mask = mask[0]
                    else:
                        mask = np.array([]) # Create empty array to force lost tracking
            
            if mask.size > 0 and np.any(mask):
                ys, xs = np.where(mask)
                min_x, max_x = np.min(xs), np.max(xs)
                min_y, max_y = np.min(ys), np.max(ys)
                # Update current_box for next frame (this is the "Tracking" part)
                current_ray_box = [min_x, min_y, max_x - min_x, max_y - min_y]

                # Calculate The Tip of the Ray
                idx = np.argmin(ys)
                rx = xs[idx]
                ry = ys[idx]
                
                # Debug Visualize The Tip (tracked by SAM)
                cv2.circle(vis_img, (rx, ry), 10, (0, 0, 255), -1) # Red for Ray Tip

                ## Also draw the line
                cv2.line(vis_img, (rx, ry), (int(tx), int(ty)), (0, 255, 255), 2) # Green line to Target

                # C. Control (PID)
                err_x = tx - rx
                err_y = ty - ry

                # Apply control
                curr_yaw += (-KP_YAW * err_x)
                curr_pitch += (-KP_PITCH * err_y)
                
                # --- SAFETY CLAMP ---
                # Stop the loop if we have rotated too far from the start (e.g., +/- 45 degrees)
                # This prevents the "out the window" issue.
                if abs(curr_yaw) > 45 or abs(curr_pitch) > 45:
                    print(f"[ABORT] Runaway rotation detected! Yaw: {curr_yaw:.1f}, Pitch: {curr_pitch:.1f}")
                    break

                self.mcp.rotate_device(device="controller2", pitch=curr_pitch, yaw=curr_yaw, roll=0)


                ## Hit Detection
                # 1. Unpack Target Box
                # t_box format is [x, y, w, h]
                t_x, t_y, t_w, t_h = map(int, t_box)

                # 2. Check if Tip is Inside Target Box
                # Logic: Is rx between left and right sides? AND Is ry between top and bottom sides?
                tip_inside_x = (rx >= t_x) and (rx <= t_x + t_w)
                tip_inside_y = (ry >= t_y) and (ry <= t_y + t_h)
                
                target_hit = tip_inside_x and tip_inside_y


                # Debug Info
                dist = np.sqrt(err_x**2 + err_y**2)
                
                # Debug a 15 pixel radius circle around the target
                status_color = (0, 255, 0) if target_hit else (0, 0, 255)
                status_text = "TARGET HIT" if target_hit else f"Dist: {dist:.1f}"
                
                cv2.putText(vis_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # Draw the target box again clearly to see the relationship
                cv2.rectangle(vis_img, (t_x, t_y), (t_x + t_w, t_y + t_h), status_color, 2)

                cv2.imwrite(os.path.join(debug_dir, f"{frame_count:03d}.png"), vis_img)
                print(f"Frame {frame_count}: Ray ({rx},{ry}) Inside Target? {target_hit}")
                
                if dist < 15 or target_hit:
                    print(f"Target Centered! ({rx}, {ry}) Dist: {dist:.1f}")
                    if not trigger_held:
                        print("Pulling Trigger...")
                        self.mcp.click_button(controller="controller2", button="trigger", duration=0.55)
                        trigger_held = True
                        
            else:
                print("Lost tracking!")
                if trigger_held:
                    print("Target Lost! Releasing Trigger.")
                    trigger_held = False
                cv2.imwrite(os.path.join(debug_dir, f"{frame_count:03d}.png"), vis_img)
                print(f"Saved debug frame. Lost tracking at frame {frame_count:03d}.png")

                # Don't move, just try to find it in the same spot next frame

            # D. Latency Measurement
            loop_end = time.time()
            fps = 1.0 / (loop_end - loop_start)
            print(f"FPS: {fps:.1f} | Latency: {(loop_end-loop_start)*1000:.1f}ms")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python realtime_track.py 'red cube'")
    else:
        bot = RealTimeTracker()
        bot.track_loop(sys.argv[1])