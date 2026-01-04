import os
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

# Configure logging
logger = logging.getLogger("VRAgent.Tracker")

# Constants
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_tiny.yaml"
CHECKPOINT_NAME = "sam2_hiera_tiny.pt"
CHECKPOINT_DIR = Path("checkpoints")

class ObjectTracker:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "tracking"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.available = False
        self.predictor = None
        self.inference_state = None
        
        # Check imports
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
            self.available = True
            self.torch = torch
            self.build_sam2_video_predictor = build_sam2_video_predictor
        except ImportError as e:
            logger.warning(f"SAM 2 or Torch not available: {e}")
            self.available = False
            return

        # Prepare Checkpoint
        self._prepare_checkpoint()
        
        # Initialize Model
        if self.available:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    logger.warning("CUDA not available. SAM 2 will be slow.")
                
                checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME
                self.predictor = build_sam2_video_predictor(SAM2_CONFIG, str(checkpoint_path), device=device)
                logger.info("SAM 2 Model initialized.")
            except Exception as e:
                logger.error(f"Failed to init SAM 2: {e}")
                self.available = False

    def _prepare_checkpoint(self):
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        ckpt_path = CHECKPOINT_DIR / CHECKPOINT_NAME
        if not ckpt_path.exists():
            logger.info(f"Downloading SAM 2 checkpoint to {ckpt_path}...")
            try:
                # Use curl to download
                subprocess.run(["curl", "-o", str(ckpt_path), SAM2_CHECKPOINT_URL], check=True)
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {e}")
                self.available = False

    def track(self, video_path: str, initial_box: List[float], label: str) -> str:
        """
        Track an object in a video given an initial bounding box.
        video_path: Path to the video file (mp4 or directory of frames?) 
                    SAM 2 usually takes a directory of JPEGs.
        initial_box: [ymin, xmin, ymax, xmax] normalized.
        
        Returns: Path to the output video with tracking visualization.
        """
        if not self.available or not self.predictor:
            return "Error: SAM 2 not available."

        try:
            # 1. Process Video -> Frames Directory (SAM 2 expects this)
            # Assuming video_path is a directory of frames or we need to extract them.
            # For this agent, existing capture_video usually returns a list of base64 frames.
            # We will assume calling code saves them to a dir.
            
            video_dir = Path(video_path)
            if not video_dir.is_dir():
                return f"Error: {video_path} is not a directory of frames."
            
            # 2. Init State
            self.inference_state = self.predictor.init_state(video_path=str(video_dir))
            
            # 3. Add Prompt to Frame 0
            # Denormalize box
            # We need image dimensions. Read first frame.
            import cv2
            first_frame_path = sorted(list(video_dir.glob("*.jpg")))[0]
            img = cv2.imread(str(first_frame_path))
            h, w = img.shape[:2]
            
            ymin, xmin, ymax, xmax = initial_box
            box_xyxy = np.array([xmin * w, ymin * h, xmax * w, ymax * h], dtype=np.float32)
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=1,
                box=box_xyxy,
            )
            
            # 4. Propagate
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # 5. Visualize and Save
            # We will create a new video file
            output_path = self.log_dir / f"tracked_{label.replace(' ', '_')}.mp4"
            
            frame_files = sorted(list(video_dir.glob("*.jpg")))
            
            # Setup VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 10, (w, h)) # Assuming 10 fps
            
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(str(frame_file))
                
                if i in video_segments and 1 in video_segments[i]:
                    mask = video_segments[i][1][0] # shape [H, W]
                    
                    # Draw mask overlay
                    # Create red overlay
                    overlay = frame.copy()
                    overlay[mask] = [0, 0, 255] # BGR
                    
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    
                    # Draw bounding box around mask (optional)
                    # rows = np.any(mask, axis=1)
                    # cols = np.any(mask, axis=0)
                    # if rows.any() and cols.any():
                    #     rmin, rmax = np.where(rows)[0][[0, -1]]
                    #     cmin, cmax = np.where(cols)[0][[0, -1]]
                    #     cv2.rectangle(frame, (cmin, rmin), (cmax, rmax), (0, 255, 0), 2)
                
                cv2.putText(frame, f"Tracking: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
                
            out.release()
            self.predictor.reset_state(self.inference_state)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            import traceback
            traceback.print_exc()
            return f"Tracking failed: {e}"
