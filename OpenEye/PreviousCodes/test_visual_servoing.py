#!/usr/bin/env python3
import os
import json
import base64
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import io
from dotenv import load_dotenv

# Imports from existing agent files
try:
    from gemini_vr_agent import DirectMCPExecutor, VisualGrounder, GEMINI_MODEL, LOG_DIR
    from object_tracker import ObjectTracker
except ImportError:
    print("Error: Could not import agent modules. Make sure you are in the OpenEye directory.")
    exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install google-genai: pip install google-genai")
    exit(1)

# Configuration
OUTPUT_DIR = Path("vision_tests")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return

    # 1. Initialize Components
    print("Initializing components...")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    
    executor = DirectMCPExecutor()
    grounder = VisualGrounder(client, GEMINI_MODEL, OUTPUT_DIR)
    
    tracker = ObjectTracker(OUTPUT_DIR)
    if not tracker.available:
        print("Error: SAM3 (ObjectTracker) not available. Cannot proceed.")
        return

    # Start VR Bridge
    print("\nStarting VR Bridge...")
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
        print("Warning: No VR driver connected. 'inspect_surroundings' will likely fail.")
        # We continue anyway to show the error, or you could return here.

    # 1.5 Take two steps forward
    if connected:
        print("\n[Movement] Taking two steps forward...")
        # Step 1
        print(" - Step 1/2")
        executor.call("move_relative", device="headset", dz=-1)
        time.sleep(1.0)
        # Step 2
        print(" - Step 2/2")
        executor.call("move_relative", device="headset", dz=-0.5)
        time.sleep(1.0)

    # 2. Capture Image
    print("\n[1/4] Capturing screenshot...")
    res = executor.call("inspect_surroundings")
    
    # Debug response
    print(f"Raw response from inspect_surroundings: {str(res)[:100]}...")
    
    if isinstance(res, str) and res.startswith("Error"):
        print(f"Capture failed: {res}")
        return

    try:
        data = json.loads(res).get("data")
    except json.JSONDecodeError:
        print(f"Failed to parse JSON response: {res}")
        return

    if not data:
        print("Failed to capture image (no data)")
        return

    img_bytes = base64.b64decode(data)
    
    # Save original
    with open(OUTPUT_DIR / "test_input.jpg", "wb") as f:
        f.write(img_bytes)
    
    # Load for processing
    # CV2 format (BGR)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img_bgr.shape[:2]
    
    # PIL format (RGB) for SAM
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 3. Process Objects
    descriptions = ["blue VR controller ray", "desert bus logo"]
    results = {}

    # Initialize SAM Processor with the image ONE time
    print("Setting image in SAM3 processor...")
    inference_state = tracker.processor.set_image(img_pil)

    for desc in descriptions:
        print(f"\n[Processing] {desc}...")
        
        # A. Ground (Gemini)
        boxes = grounder.ground_object(img_bytes, desc)
        if not boxes:
            print(f" - Failed to ground '{desc}'")
            continue
            
        # Use first box
        # Box is [ymin, xmin, ymax, xmax] (normalized)
        norm_box = boxes[0]['box_2d']
        ymin, xmin, ymax, xmax = norm_box
        
        # Convert to pixels for SAM
        box_x = xmin * w
        box_y = ymin * h
        box_w = (xmax - xmin) * w
        box_h = (ymax - ymin) * h
        
        print(f" - Box found: {norm_box}")
        
        # B. Segment (SAM)
        # Prepare box input (cxcywh normalized)
        box_input_xywh = tracker.torch.tensor([box_x, box_y, box_w, box_h]).view(-1, 4)
        box_input_cxcywh = tracker.box_xywh_to_cxcywh(box_input_xywh)
        norm_box_cxcywh = tracker.normalize_bbox(box_input_cxcywh, w, h).flatten().tolist()
        
        tracker.processor.reset_all_prompts(inference_state)
        inference_state = tracker.processor.add_geometric_prompt(
            state=inference_state,
            box=norm_box_cxcywh,
            label=True
        )
        
        # Get mask
        mask = None
        if "masks" in inference_state and inference_state["masks"] is not None:
             mask_tensor = inference_state["masks"]
             # Shape is typically [N, M, H, W] where N=batch(1), M=masks(3)
             # or [N, 1, H, W]
             m = mask_tensor.detach().cpu().numpy() > 0.5
             
             print(f" - Mask shape: {m.shape}")
             
             if m.ndim == 4:
                 # [1, 3, H, W] -> take index 0 of dim 1
                 mask = m[0, 0] 
             elif m.ndim == 3:
                 # [3, H, W] -> take index 0
                 mask = m[0]
             elif m.ndim == 2:
                 mask = m
             else:
                print(f" - Unexpected mask shape: {m.shape}")

        if mask is None:
            print(" - SAM failed to generate mask")
            continue
            
        # C. Analyze Mask
        if desc == "blue VR controller ray":
            # Find point furthest from bottom (min y)
            # np.where returns (rows, cols) -> (y, x)
            ys, xs = np.where(mask)
            if len(ys) > 0:
                min_y_idx = np.argmin(ys) # Index of minimum y
                target_y = ys[min_y_idx]
                target_x = xs[min_y_idx]
                results[desc] = (target_x, target_y)
                print(f" - Ray Tip: ({target_x}, {target_y})")
            else:
                print(" - Mask empty")

        elif desc == "desert bus logo":
            # Find center
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results[desc] = (cx, cy)
                print(f" - Logo Center: ({cx}, {cy})")
            else:
                print(" - Mask empty")
        
        # Store mask for visualization
        results[f"{desc}_mask"] = mask

    # 4. Visualize
    print("\n[4/4] Visualizing...")
    vis_img = img_bgr.copy()
    
    # Map strict descriptions to colors
    colors = {
        "blue VR controller ray": (0, 0, 255),  # Red
        "desert bus logo": (0, 255, 0)     # Green
    }
    
    for desc_key, color in colors.items():
        # Check if we have results for this key
        if f"{desc_key}_mask" in results:
            mask = results[f"{desc_key}_mask"]
            # Overlay mask
            colored_mask = np.zeros_like(vis_img)
            colored_mask[mask] = color
            cv2.addWeighted(colored_mask, 0.5, vis_img, 1.0, 0, vis_img)
            
            # Draw contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color, 2)
            
        if desc_key in results:
            pt = results[desc_key]
            # Draw point
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), -1) # White bg
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 5, color, -1)     # Color inner
            
            # Label
            label_text = f"{desc_key}: {pt}"
            cv2.putText(vis_img, label_text, (int(pt[0])+10, int(pt[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_path = OUTPUT_DIR / "test_result.jpg"
    cv2.imwrite(str(out_path), vis_img)
    print(f"Result saved to {out_path}")

if __name__ == "__main__":
    main()
