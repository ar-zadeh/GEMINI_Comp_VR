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

    def trigger(self):
        """Helper to pull the controller trigger."""
        self.mcp.click_button(controller="controller2", button="trigger", duration=0.55)

    def bumper(self):
        """Helper to pull the controller bumper."""
        self.mcp.click_button(controller="controller2", button="grip", duration=0.55)
if __name__ == "__main__":
    
    bot = RealTimeTracker()
    for i in range(10, 0, -1):
        print(f"Starting in {i}...")
        bot.trigger()
        # bot.bumper()
        time.sleep(1.5)
    sys.exit(0)