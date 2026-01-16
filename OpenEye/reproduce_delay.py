
import os
import json
import time
import base64
import math
import io
import cv2
import numpy as np
from PIL import Image
from gemini_vr_agent import DirectMCPExecutor

def main():
    executor = DirectMCPExecutor()
    print("Starting VR Bridge...")
    executor.call("start_vr_bridge")
    
    print("Waiting for VR driver connection...")
    for i in range(20): # Wait up to 20s
        status = executor.call("get_connection_status")
        if "Connected" in status:
            print(f"Driver connected: {status}")
            break
        time.sleep(1)
        if i == 19:
            print("Timeout waiting for driver connection. Please ensure SteamVR is running.")
            return

    time.sleep(2)
    
    print("Getting initial pose...")
    # Rotate to 0,0,0 first
    executor.call("rotate_device", device="controller2", pitch=0, yaw=0, roll=0)
    time.sleep(2) 
    
    # Capture "Before"
    print("Capturing BEFORE image...")
    res = executor.call("inspect_surroundings")
    if "Error" in res:
        print(f"Error capturing: {res}")
        return
        
    # Rotate abruptly
    target_yaw = 45.0
    print(f"Sending rotate command (Yaw -> {target_yaw})...")
    t_start = time.time()
    executor.call("rotate_device", device="controller2", pitch=0, yaw=target_yaw, roll=0)
    t_sent = time.time()
    print(f"Command sent in {t_sent - t_start:.4f}s")
    
    # Poll for visual change
    print("Polling for visual change...")
    detected = False
    
    # We can detect change by easy diff with initial image
    data0 = json.loads(res).get("data")
    img0 = cv2.imdecode(np.frombuffer(base64.b64decode(data0), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    for i in range(50): # 5 seconds max (assuming 0.1s sleep)
        t_now = time.time()
        res_i = executor.call("inspect_surroundings")
        data_i = json.loads(res_i).get("data")
        img_i = cv2.imdecode(np.frombuffer(base64.b64decode(data_i), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Simple diff
        diff = cv2.absdiff(img0, img_i)
        score = np.sum(diff) / (img0.shape[0]*img0.shape[1])
        
        print(f"Time: {t_now - t_sent:.2f}s | Diff Score: {score:.2f}")
        
        if score > 5.0: # Threshold for "significant change"
            print(f"SUCCESS: Visual change detected after {t_now - t_sent:.2f} seconds!")
            detected = True
            break
            
        time.sleep(0.1)
        
    if not detected:
        print("FAILED: No significant visual change detected within 5 seconds.")

if __name__ == "__main__":
    main()
