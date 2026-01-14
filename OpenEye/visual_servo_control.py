#!/usr/bin/env python3
import os
import json
import base64
import time
import math
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
from dotenv import load_dotenv

# Imports
from gemini_vr_agent import DirectMCPExecutor, VisualGrounder, GEMINI_MODEL, LOG_DIR
from object_tracker import ObjectTracker
try:
    from google import genai
except ImportError:
    pass

# Config
OUTPUT_DIR = Path("servo_logs")
OUTPUT_DIR.mkdir(exist_ok=True)
Kp_YAW = 0.02   # Gain for Horizontal (Yaw)
Kp_PITCH = 0.02 # Gain for Vertical (Pitch)
MAX_ITER = 15
TOLERANCE_PX = 20  # Stop if within this many pixels

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    
    executor = DirectMCPExecutor()
    grounder = VisualGrounder(client, GEMINI_MODEL, OUTPUT_DIR)
    tracker = ObjectTracker(OUTPUT_DIR)
    
    if not tracker.available:
        print("Error: SAM3 not available.")
        return

    # 1. Start Bridge
    print(executor.call("start_vr_bridge"))
    
    # Wait for connection
    print("Waiting for VR driver connection (10s)...")
    connected = False
    for _ in range(10):
        status = executor.call("get_connection_status")
        if "Connected" in status:
            print(f"Driver connected: {status}")
            connected = True
            break
        time.sleep(1)
        
    if not connected:
        print("Error: No VR driver connected.")
        return

    time.sleep(1) # Settle

    # 2. Get Initial Pose of Right Controller (controller2)
    # We need to track our current rotation state
    curr_pitch = 0.0
    curr_yaw = 0.0
    curr_roll = 0.0
    
    status = executor.call("get_current_pose", device="controller2")
    # Parse status "controller2 - Position: [x, y, z], Rotation: [p, y, r]"
    try:
        if "Rotation: [" in status:
            rot_str = status.split("Rotation: [")[1].split("]")[0]
            curr_pitch, curr_yaw, curr_roll = map(float, rot_str.split(","))
            print(f"Initial Rotation: Pitch={curr_pitch}, Yaw={curr_yaw}, Roll={curr_roll}")
    except Exception as e:
        print(f"Failed to parse initial pose: {e}. Starting from 0,0,0")

    # State for tracking: key -> [x, y, w, h] (pixels)
    current_boxes = {}
    
    # 3. Control Loop
    for i in range(MAX_ITER):
        print(f"\n--- Iteration {i+1}/{MAX_ITER} ---")
        
        # A. Capture
        res = executor.call("inspect_surroundings")
        if isinstance(res, str) and res.startswith("Error"):
             print(f"Capture failed: {res}")
             break
             
        try:
            data = json.loads(res).get("data")
            img_bytes = base64.b64decode(data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]
        except Exception as e:
            print(f"Capture parsing failed: {e}")
            break

        # B. Find Objects (Gemini once -> SAM Tracking)
        targets = {
            "ray": "blue VR controller ray",
            "logo": "desert bus logo"
        }
        points = {}
        
        # SAM State
        inference_state = tracker.processor.set_image(img)
        
        # 1. Ensure we have boxes for all targets
        for key, desc in targets.items():
            if key not in current_boxes:
                # Grounding (Gemini) - Only if we don't have a box yet
                print(f"[{key}] Grounding with Gemini...")
                boxes = grounder.ground_object(img_bytes, desc)
                if boxes:
                    norm_box = boxes[0]['box_2d'] # ymin, xmin, ymax, xmax
                    ymin, xmin, ymax, xmax = norm_box
                    box_x, box_y = xmin * w, ymin * h
                    box_w, box_h = (xmax - xmin) * w, (ymax - ymin) * h
                    current_boxes[key] = [box_x, box_y, box_w, box_h]
                else:
                    print(f"[{key}] Not found by Gemini.")
        
        # 2. Track / Segment with SAM
        for key, desc in targets.items():
            if key not in current_boxes:
                continue
                
            box_x, box_y, box_w, box_h = current_boxes[key]
            
            # Input to SAM
            box_input_xywh = tracker.torch.tensor([box_x, box_y, box_w, box_h]).view(-1, 4)
            box_input_cxcywh = tracker.box_xywh_to_cxcywh(box_input_xywh)
            norm_box_cxcywh = tracker.normalize_bbox(box_input_cxcywh, w, h).flatten().tolist()
            
            tracker.processor.reset_all_prompts(inference_state)
            inference_state = tracker.processor.add_geometric_prompt(
                state=inference_state, box=norm_box_cxcywh, label=True
            )
            
            # Extract Mask
            mask = None
            if "masks" in inference_state and inference_state["masks"] is not None:
                m = inference_state["masks"].detach().cpu().numpy() > 0.5
                if m.ndim == 4: mask = m[0, 0]
                elif m.ndim == 3: mask = m[0]
            
            if mask is None:
                print(f"[{key}] SAM failed.")
                # Maybe remove from current_boxes to re-ground next time?
                # del current_boxes[key] 
                continue
            
            # Update Box for next frame (Tracking)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # New box
                current_boxes[key] = [cmin, rmin, cmax - cmin, rmax - rmin]
            else:
                print(f"[{key}] Mask empty, lost tracking.")
                del current_boxes[key]
                continue

            # Get Keypoint
            if key == "ray":
                # Tip (min y)
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    idx = np.argmin(ys)
                    points[key] = (xs[idx], ys[idx])
            elif key == "logo":
                # Center
                M = cv2.moments(mask.astype(np.uint8))
                if M["m00"] != 0:
                    points[key] = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        # C. Calculate Error & Control
        if "ray" in points and "logo" in points:
            rx, ry = points["ray"]
            lx, ly = points["logo"]
            
            dx = lx - rx  # Target X - Ray X
            dy = ly - ry  # Target Y - Ray Y
            dist = math.sqrt(dx*dx + dy*dy)
            
            print(f"Ray: ({rx}, {ry}), Logo: ({lx}, {ly})")
            print(f"Error: dx={dx}, dy={dy}, dist={dist:.2f}")
            
            # Save Vis
            cv2.line(img_cv, (rx, ry), (lx, ly), (0, 255, 255), 2)
            cv2.circle(img_cv, (rx, ry), 5, (0, 0, 255), -1)
            cv2.circle(img_cv, (lx, ly), 5, (0, 255, 0), -1)
            cv2.imwrite(str(OUTPUT_DIR / f"iter_{i:02d}_dist_{int(dist)}.jpg"), img_cv)
            
            if dist < TOLERANCE_PX:
                print("SUCCESS: Target Reached!")
                break
                
            # Control Logic
            # Screen X is left-right. Rotating Right Controller Yaw+ moves ray Right? 
            # Let's assume Yaw+ moves Ray Right (increasing Pixel X).
            # If dx > 0 (Logo is to the right), we need Ray to move Right. So Yaw+.
            # DeltaYaw = K * dx
            
            # Screen Y is top-down. Rotating Pitch- (Up) moves ray Up (decreasing Pixel Y)?
            # If dy > 0 (Logo is below Ray), we need Ray to move Down. 
            # If Pitch- moves Up, then Pitch+ moves Down.
            # So DeltaPitch = K * dy
            
            d_yaw = Kp_YAW * dx
            d_pitch = Kp_PITCH * dy # Inverted Y axis logic check: +dy means target is "lower" (higher pixel value). We want ray to go "down".
            
            # Apply
            # If +Pitch means Look Down (RotX+), and we want Ray to match Target.
            # RayY < TargetY (Ray is Higher/Smaller Y) -> dy = Target - Ray > 0.
            # We want Ray to go Down (Increase Y).
            # So we need +Pitch.
            # dy > 0 -> d_pitch > 0.
            # So we ADD d_pitch.
            curr_yaw += d_yaw
            curr_pitch += d_pitch 
            
            print(f"Adjustment: dYaw={d_yaw:.2f}, dPitch={d_pitch:.2f}")
            print(f"New Pose: Yaw={curr_yaw:.2f}, Pitch={curr_pitch:.2f}")
            
            executor.call("rotate_device", device="controller2", 
                          pitch=curr_pitch, yaw=curr_yaw, roll=curr_roll)
            
            time.sleep(0.5) # Wait for move
            
        else:
            print("Could not find both objects. Stopping.")
            break

if __name__ == "__main__":
    main()
