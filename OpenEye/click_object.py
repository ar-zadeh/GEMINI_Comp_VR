#!/usr/bin/env python3
import os
import json
import time
import base64
import math
import io
import importlib.util
import sys
import cv2
from datetime import datetime
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path

# Load Custom Modules
try:
    from object_tracker import ObjectTracker
except ImportError:
    print("Error: object_tracker.py not found. Tracking disabled.")
    sys.exit(1)

load_dotenv()

class VRController:
    """Minimal interface for VR interaction and Visual Servoing."""
    
    def __init__(self):
        # 1. Setup Gemini
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={'api_version': 'v1alpha'})
        self.model_name = "gemini-3-flash-preview"

        # 2. Setup MCP (VR Bridge)
        spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
        if not spec: raise ImportError("Could not find mcp_server.py")
        self.mcp = importlib.util.module_from_spec(spec)
        sys.modules["mcp_server"] = self.mcp
        spec.loader.exec_module(self.mcp)
        self.mcp.start_vr_bridge()

        # 3. Setup Tracker
        self.tracker = ObjectTracker(log_dir=Path("logs"))
        if not self.tracker.available: raise RuntimeError("SAM/Tracker not available.")

    def _ground_objects(self, image_data, object_names):
        """Uses Gemini to find bounding boxes [ymin, xmin, ymax, xmax]."""
        prompt = f"""Find these objects: {', '.join(object_names)}. 
        Return JSON: {{ "detections": [ {{ "label": "name", "coordinates": [ymin, xmin, ymax, xmax] }} ] }}"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data))
                ])],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.candidates[0].content.parts[0].text)
            results = {}
            for det in data.get("detections", []):
                # Handle 0-1000 scale if necessary, otherwise assume 0-1
                coords = [c/1000.0 if c > 1.0 else c for c in det['coordinates']]
                results[det['label']] = coords
            return results
        except Exception as e:
            print(f"Grounding failed: {e}")
            return {}

    def click_object(self, target_name):
        """Main Servo Loop: Ground -> Track -> Align -> Click."""
        print(f"--- Starting Visual Servo for: '{target_name}' ---")
        
        # Config
        KP_YAW, KP_PITCH = 0.05, 0.05
        TOLERANCE_PX = 15
        MAX_ITER = 50
        PATIENCE_LIMIT = 20 # allows for consecutive lost frames
        MIN_DIST_DELTA = 2.0 # minimum pixel change rquired to consider "movement"

        # Setup Debug Logging
        debug_dir = os.path.join(os.getcwd(), "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 1. Get Initial Pose & Image
        pose_str = self.mcp.get_current_pose(device="controller2")
        curr_pitch, curr_yaw, _ = map(float, pose_str.split("Rotation: [")[1].split("]")[0].split(","))
        
        res = json.loads(self.mcp.inspect_surroundings())
        img_bytes = base64.b64decode(res["data"])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = pil_img.size

        # 2. Ground Initial Targets (Ray + Target)
        targets = {"ray": "blue VR controller ray", "target": target_name}
        boxes_norm = self._ground_objects(img_bytes, list(targets.values()))
        
        current_boxes = {}
        for k, v in targets.items():
            if v in boxes_norm:
                ymin, xmin, ymax, xmax = boxes_norm[v]
                current_boxes[k] = [xmin * w, ymin * h, (xmax-xmin)*w, (ymax-ymin)*h] # xywh
            else:
                print(f"Failed to locate {v}")
                return

        # 3. Control Loop
        last_dist = None
        consecutive_lost = 0
        last_known_boxes = current_boxes.copy()

        for i in range(MAX_ITER):
            # Capture new frame
            res = json.loads(self.mcp.inspect_surroundings())
            img_bytes = base64.b64decode(res["data"])

            last_img_bytes = img_bytes
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Visualization setup
            vis_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Update Tracking (SAM)
            inference_state = self.tracker.processor.set_image(pil_img)
            points = {}
            
            for k, desc in targets.items():
                # 1. Use last known box if current is missing (Patience Logic)
                # This safely gets the box from either current OR backup
                search_box = current_boxes.get(k, last_known_boxes.get(k))
                
                # 2. If completely unknown, skip
                if not search_box:
                    print(f"[{k}] No box available to track.")
                    continue

                # Draw Search Box (Blue)
                bx, by, bw, bh = map(int, search_box)
                cv2.rectangle(vis_img, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)

                # Format box for SAM
                box_tensor = self.tracker.torch.tensor(search_box).view(-1, 4)
                cxcywh = self.tracker.box_xywh_to_cxcywh(box_tensor)
                norm_box = self.tracker.normalize_bbox(cxcywh, w, h).flatten().tolist()
                
                self.tracker.processor.reset_all_prompts(inference_state)

                inf_state = self.tracker.processor.add_geometric_prompt(
                    state=inference_state, 
                    box=norm_box, 
                    label=True
                )
                # Extract Mask & New Box
                mask = (inf_state["masks"].detach().cpu().numpy() > 0.5).squeeze()

                # Safety check if SAM returned nothing, skip this object
                if mask.size == 0:
                    print(f"[{k}] SAM lost the object (empty mask).")
                    continue
                if mask.ndim == 4: mask = mask[0, 0]
                elif mask.ndim == 3: mask = mask[0]
                if not np.any(mask): 
                    print(f"[{k}] SAM lost the object (no mask).")
                    continue
                
                ys, xs = np.where(mask)
                # current_boxes[k] = [np.min(xs), np.min(ys), np.max(xs)-np.min(xs), np.max(ys)-np.min(ys)]
                
                PADDING = 1 # Add small buffer for movement
                
                min_x, max_x = np.min(xs), np.max(xs)
                min_y, max_y = np.min(ys), np.max(ys)
                
                new_x = max(0, min_x - PADDING)
                new_y = max(0, min_y - PADDING)
                new_w = min(w - new_x, (max_x - min_x) + (PADDING * 2))
                new_h = min(h - new_y, (max_y - min_y) + (PADDING * 2))
                
                new_box = [new_x, new_y, new_w, new_h]
                
                current_boxes[k] = new_box       
                last_known_boxes[k] = new_box

                
                # Define Point of Interest (Tip for ray, Center for target)
                if k == "ray":
                    # Ray Tip: Lowest Y (if ray points down) or Highest Y?
                    idx = np.argmin(ys)
                    pt = (xs[idx], ys[idx]) # Topmost point
                    points[k] = pt
                    cv2.circle(vis_img, points[k], 5, (0, 0, 255), -1) # Red for Ray

                else: 
                    points[k] = (int(np.mean(xs)), int(np.mean(ys)))     # Centroid
                    cv2.circle(vis_img, points[k], 5, (0, 255, 0), -1) # Green for target

                

            # Calculate Error & Move
            if "ray" in points and "target" in points:
                # Found both, reset patience
                consecutive_lost = 0

                rx, ry = points["ray"]
                tx, ty = points["target"]

                # Draw Line
                cv2.line(vis_img, (rx, ry), (tx, ty), (0, 255, 255), 2)

                dx, dy = tx - rx, ty - ry
                dist = math.sqrt(dx**2 + dy**2)
                
                text = f"Dist: {dist:.1f}px"
                cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f'Status {text}')

                # Save Debug Frame
                cv2.imwrite(os.path.join(debug_dir, f"iter_{i:03d}.png"), vis_img)
                print(f'Saved to {os.path.join(debug_dir, f"iter_{i:03d}.png")}')
                print(f"Iter {i}: Dist={dist:.1f}px")
                
                if dist < TOLERANCE_PX:
                    print("Target Aligned. Clicking.")
                    self.mcp.click_button(controller="controller2", button="trigger")
                    return

                # Stagnation check
                if last_dist is not None and abs(dist - last_dist) < MIN_DIST_DELTA:
                    print(f"Iter {i}: Distance unchanged ({dist:.1f} vs {last_dist:.1f}). Waiting for update...")
                    time.sleep(0.1) 
                    continue # Skip PID update, loop again to get fresh state
                
                last_dist = dist
                # PID Update
                curr_yaw += (KP_YAW * dx)
                curr_pitch += (KP_PITCH * dy)
                self.mcp.rotate_device(device="controller2", pitch=curr_pitch, yaw=curr_yaw, roll=0)
            else:
                # TRACKING LOST
                consecutive_lost += 1
                print(f"Iter {i}: Tracking lost ({consecutive_lost}/{PATIENCE_LIMIT})")
                
                # Save the frame anyway so we see WHY it was lost
                cv2.putText(vis_img, "TRACKING LOST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(debug_dir, f"iter_{i:03d}_lost.png"), vis_img)
                print(f'Saved to {os.path.join(debug_dir, f"iter_{i:03d}_lost.png")}')
                if consecutive_lost >= PATIENCE_LIMIT:
                    print("Exceeded patience limit. Exiting.")
                    return
                
                # DO NOT MOVE ANYTHING this frame. Just wait for next one.
                time.sleep(0.2) # Allow time for VR to update

            

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python click_object.py 'object name'")
    else:
        bot = VRController()
        bot.click_object(sys.argv[1])