#!/usr/bin/env python3
import time
import math
import sys
import importlib.util
from dotenv import load_dotenv

load_dotenv()


class SimpleVRController:
    def __init__(self):
        # Setup MCP (VR Bridge)
        spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
        if not spec: raise ImportError("Could not find mcp_server.py")
        self.mcp = importlib.util.module_from_spec(spec)
        sys.modules["mcp_server"] = self.mcp
        spec.loader.exec_module(self.mcp)
        self.mcp.start_vr_bridge()
        print("VR Bridge Connected.")

    def wander_and_click(self, duration_seconds=30):
        print(f"--- Starting Movement & Click Demo ({duration_seconds}s) ---")
        start_time = time.time()
        
        # Get initial center
        try:
            pose_str = self.mcp.get_current_pose(device="controller2")
            rot_part = pose_str.split("Rotation: [")[1].split("]")[0]
            center_pitch, center_yaw, _ = map(float, rot_part.split(","))
        except:
            center_pitch, center_yaw = 0.0, 0.0

        last_click_time = time.time()

        while (time.time() - start_time) < duration_seconds:
            elapsed = time.time() - start_time
            
            # 1. Move in Figure-8
            yaw_offset = math.sin(elapsed * -2.0) * -2
            pitch_offset = math.sin(elapsed * -4.0) * -1.0

            self.mcp.rotate_device(
                device="controller2", 
                pitch=center_pitch + pitch_offset, 
                yaw=center_yaw + yaw_offset, 
                roll=0
            )

            # 2. Click every 2 seconds
            if time.time() - last_click_time > 2.0:
                print(f"Clicking! (Time: {elapsed:.1f})")
                
                # We use the MCP tool to handle the press/release logic
                self.mcp.click_button(
                    controller="controller2", 
                    button="trigger", 
                    duration=0.2
                )
                last_click_time = time.time()

            time.sleep(0.01)

if __name__ == "__main__":
    bot = SimpleVRController()
    bot.wander_and_click(30)