#!/usr/bin/env python3
import os
import argparse
import time
import base64
import json
import threading
import requests
import re
import numpy as np
import io
from PIL import Image, ImageChops

import sys
import select
from datetime import datetime
try:
    import termios
    import tty
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False

# Ensure we can import from vr_agent
import sys
os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vr_agent.executor import DirectMCPExecutor
from keyboard_controller import KeyboardVRController

URL = "http://localhost:8000/v1/chat/completions"
# MODEL = "nvidia/Cosmos-Reason2-8b"
MODEL = "Firworks/Cosmos-Reason2-8B-nvfp4"

def calculate_similarity(image1_bytes, image2_bytes):
    if not image1_bytes or not image2_bytes:
        return 0.0
    img1 = Image.open(io.BytesIO(image1_bytes)).convert("RGB").resize((400, 300))
    img2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB").resize((400, 300))
    diff = ImageChops.difference(img1, img2)
    h = diff.histogram()
    sq = (value * ((idx % 256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = np.sqrt(sum_of_squares / float(img1.size[0] * img1.size[1]))
    # RMS is max 255. 0 means perfect match (100% similarity)
    sim = max(0.0, 1.0 - (rms / 255.0))
    return sim * 100.0  # Percentage

def parse_freespace_direction(content):
    content = content.upper()
    if "STRAIGHT" in content: return 0.0
    elif "LEFT" in content: return -25.0
    elif "RIGHT" in content: return 25.0
    return None

def parse_clock_direction(content):
    numbers = re.findall(r"\d+", content)
    for num_str in numbers:
        num = int(num_str)
        if num in [10, 11, 12, 1, 2]:
            mapping = {10: -30.0, 11: -15.0, 12: 0.0, 1: 15.0, 2: 30.0}
            return mapping[num]
    return None

class AutoWalker(threading.Thread):
    def __init__(self, mcp_executor):
        super().__init__(daemon=True)
        self.mcp = mcp_executor
        self.running = False
        self.ctrl = None
        
    def start_walking(self):
        self.running = True
        self.ctrl = KeyboardVRController(self.mcp.module)
        # Ensure we are in HEADSET mode (M key toggles)
        if self.ctrl.mode != 'headset':
            self.ctrl.mode = 'headset'
        print(f"[AutoWalker] Initialized. Mode: {self.ctrl.mode.upper()}")
        print(">>> Press the backtick key (`) at any time to TOGGLE MANUAL CONTROL. <<<")
        self.start()
        
    def stop(self):
        self.running = False
        if self.ctrl:
            self.ctrl.deactivate()
        
    def run(self):
        while self.running:
            # Only press 'w' if keyboard controller is NOT active (Auto-pilot)
            if not self.ctrl.active:
                self.ctrl._handle_char('w')
            time.sleep(1)

class ManualToggleListener(threading.Thread):
    def __init__(self, walker):
        super().__init__(daemon=True)
        self.walker = walker
        
    def run(self):
        if not TERMIOS_AVAILABLE:
            return
            
        fd = sys.stdin.fileno()
        while True:
            # Only listen for toggle if the controller is currently INACTIVE
            if not self.walker.ctrl.active:
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    # Non-blocking check for input
                    r, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch == '`':
                            # Restore terminal before activating (it will take over)
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                            print("\n[Manual Control] Triggered via backtick!")
                            self.walker.ctrl.activate()
                except Exception as e:
                    pass
                finally:
                    # Always try to restore cooked mode if we aren't active
                    if not self.walker.ctrl.active:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
            time.sleep(0.1)

def capture_frame(executor):
    res = executor.call("inspect_surroundings")
    try:
        data = json.loads(res).get("data")
        if data:
            return base64.b64decode(data)
    except:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Automated VR Navigation Benchmark")
    parser.add_argument("--strategy", type=str, choices=["freespace", "clockface"], default="clockface", help="Prompt strategy")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=1, help="1: 1 pic every X sec. 2: Video (F fps for T sec). 3: N pics every X sec")
    parser.add_argument("--interval", "-x", type=float, default=3.0, help="Interval (X) in seconds between queries")
    parser.add_argument("--duration", "-t", type=float, default=1.0, help="Duration (T) for video mode in seconds")
    parser.add_argument("--fps", "-f", type=int, default=2, help="Frames per second for video mode")
    parser.add_argument("--num_pics", "-n", type=int, default=3, help="Number of pictures (N) for mode 3")
    args = parser.parse_args()

    print(f"--- Starting Benchmark: {args.strategy} | Mode: {args.mode} ---")

    # Create logging directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_turning_log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"[Logging] Created directory: {log_dir}")
    log_file = os.path.join(log_dir, "responses.log")

    executor = DirectMCPExecutor()
    print("Starting VR Bridge...")
    executor.call("start_vr_bridge")
    time.sleep(2)

    input("\nVR Bridge initialized. Press Enter to start the benchmark loop...")

    walker = AutoWalker(executor)
    walker.start_walking()

    # Start the toggle listener
    listener = ManualToggleListener(walker)
    listener.start()

    if args.strategy == "freespace":
        prompt = "You are looking through the eyes of a person walking forward. Is the floor clear ahead for at least 3 steps? If not, is there more walkable floor space to the LEFT or RIGHT? Respond with: DIRECTION (LEFT, STRAIGHT, RIGHT) and a brief reason."
        parse_func = parse_freespace_direction
    else:
        prompt = "Imagine a clock face overlay on this first-person view. 12 o'clock is straight ahead, 10 o'clock is left, 2 o'clock is right. Which clock direction (10, 11, 12, 1, or 2) has the clearest and longest obstacle-free walking path? Respond with only the clock number."
        parse_func = parse_clock_direction

    history_buffer = []
    
    try:
        while True:
            # If manual control is active, skip the benchmark logic
            if walker.ctrl.active:
                time.sleep(1.0)
                continue

            print(f"\n[Benchmarking] Waiting {args.interval}s for next query...")
            time.sleep(args.interval)
            
            frames = []
            if args.mode == 1:
                frame = capture_frame(executor)
                if frame: frames.append(frame)
            elif args.mode == 2:
                # Video (T duration, F fps)
                total_frames = int(args.duration * args.fps)
                delay = 1.0 / args.fps
                for _ in range(total_frames):
                    frame = capture_frame(executor)
                    if frame: frames.append(frame)
                    time.sleep(delay)
            elif args.mode == 3:
                # N pictures
                for _ in range(args.num_pics):
                    frame = capture_frame(executor)
                    if frame: frames.append(frame)
                    time.sleep(1/5) # Short burst
            
            if not frames:
                print("[Error] Failed to capture frames")
                continue
                
            # Stuck detector check
            last_frame = frames[-1]
            if len(history_buffer) > 0:
                similarity = calculate_similarity(history_buffer[-1], last_frame)
                print(f"[Detector] Visual Similarity to last tick: {similarity:.2f}%")
                if similarity > 90.0:
                    print(">>> [STUCK DETECTED] The agent is highly likely stuck! <<<")
            
            history_buffer.append(last_frame)
            if len(history_buffer) > 5:
                history_buffer.pop(0)

            # Query the VLM
            b64_images = [base64.b64encode(f).decode('utf-8') for f in frames]
            
            messages = [{"role": "user", "content": []}]
            for b64 in b64_images:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            messages[0]["content"].append({"type": "text", "text": prompt})
            
            payload = {
                "model": MODEL,
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.1
            }
            
            print(f"--- Sending {len(frames)} frames to VLM ---")
            try:
                response = requests.post(URL, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                print(f"[AI Response]: {content}")
                
                # Log the response
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file, "a") as f:
                    f.write(f"[{timestamp}] {content}\n")
                
                # Apply turning if identified
                angle = parse_func(content)
                if angle is not None:
                    print(f"[Navigation] Suggests turning: {angle} degrees")
                    if angle != 0.0:
                        walker.ctrl._rotate_headset(dyaw=angle)
                else:
                    print("[Navigation] Failed to parse a valid direction.")
            except Exception as e:
                print(f"[Error querying VLM]: {e}")

    except KeyboardInterrupt:
        print("\nBenchmark stopped by user.")
    finally:
        walker.stop()
        print("Done.")

if __name__ == "__main__":
    main()
