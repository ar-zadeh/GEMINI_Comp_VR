"""
SAM3 Object Tracker for VR Agent
Based on SAM3_test.py implementation
"""

import os
import io
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from PIL import Image

# Configure logging
logger = logging.getLogger("VRAgent.SAM3Tracker")

# GPU settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class SAM3ObjectTracker:
    """
    SAM3-based object tracker.
    Uses Gemini for bounding box detection, then SAM3 for segmentation.
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "tracking"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.available = False
        self.model = None
        self.processor = None
        
        # Try to import SAM3
        try:
            import sam3
            from sam3 import build_sam3_image_model
            from sam3.model.box_ops import box_xywh_to_cxcywh
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.visualization_utils import normalize_bbox, plot_results
            
            # Store imports
            self.sam3 = sam3
            self.build_sam3_image_model = build_sam3_image_model
            self.box_xywh_to_cxcywh = box_xywh_to_cxcywh
            self.Sam3Processor = Sam3Processor
            self.normalize_bbox = normalize_bbox
            self.plot_results = plot_results
            
            # Setup paths
            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            bpe_path = os.path.join("./", "assets", "bpe_simple_vocab_16e6.txt.gz")
            
            # Build model
            logger.info("Loading SAM3 model...")
            print("Loading SAM3 model...")
            self.model = build_sam3_image_model(bpe_path=bpe_path)
            self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
            
            self.available = True
            logger.info("SAM3 model loaded successfully!")
            print("SAM3 model loaded successfully!")
            
        except ImportError as e:
            logger.warning(f"SAM3 not available: {e}")
            print(f"Warning: SAM3 import failed: {e}. Tracking will be disabled.")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize SAM3: {e}")
            print(f"Error initializing SAM3: {e}")
            self.available = False
    
    def segment_with_box(self, image_data: bytes, box: List[float], label: str = "object") -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
        """
        Segment an object in an image using a bounding box.
        
        Args:
            image_data: JPEG image bytes
            box: [ymin, xmin, ymax, xmax] normalized coordinates (0-1)
            label: Label for the object
            
        Returns:
            Tuple of (mask array, visualization image) or (None, None) on failure
        """
        if not self.available:
            return None, None
        
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            width, height = pil_image.size
            
            # Convert from [ymin, xmin, ymax, xmax] normalized to [x1, y1, x2, y2] pixels
            ymin, xmin, ymax, xmax = box
            x1 = xmin * width
            y1 = ymin * height
            x2 = xmax * width
            y2 = ymax * height
            
            # Clamp coordinates
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            w = x2 - x1
            h = y2 - y1
            
            if w < 5 or h < 5:
                logger.warning("Box too small for segmentation")
                return None, None
            
            # Set image in processor
            inference_state = self.processor.set_image(pil_image)
            
            # Convert box to SAM3 format (cxcywh normalized)
            box_input_xywh = torch.tensor([x1, y1, w, h]).view(-1, 4)
            box_input_cxcywh = self.box_xywh_to_cxcywh(box_input_xywh)
            norm_box_cxcywh = self.normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
            
            # Reset and add box prompt
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.add_geometric_prompt(
                state=inference_state,
                box=norm_box_cxcywh,
                label=True
            )
            
            # Get masks
            masks = inference_state.get("masks", [])
            
            # Create visualization
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            self.plot_results(pil_image, inference_state)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            result_img = Image.open(buf).copy()
            plt.close('all')
            
            # Return first mask if available
            if masks and len(masks) > 0:
                mask = masks[0]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                return mask, result_img
            
            return None, result_img
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def segment_with_text(self, image_data: bytes, text_prompt: str) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
        """
        Segment an object in an image using a text prompt.
        
        Args:
            image_data: JPEG image bytes
            text_prompt: Text description of the object to segment
            
        Returns:
            Tuple of (mask array, visualization image) or (None, None) on failure
        """
        if not self.available:
            return None, None
        
        try:
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            inference_state = self.processor.set_image(pil_image)
            
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt.strip()
            )
            
            # Get masks
            masks = inference_state.get("masks", [])
            
            # Create visualization
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            self.plot_results(pil_image, inference_state)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            result_img = Image.open(buf).copy()
            plt.close('all')
            
            if masks and len(masks) > 0:
                mask = masks[0]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                return mask, result_img
            
            return None, result_img
            
        except Exception as e:
            logger.error(f"Text segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def track(self, video_path: str, initial_box: List[float], label: str) -> str:
        """
        Track an object through video frames.
        
        Args:
            video_path: Path to directory containing video frames (*.jpg)
            initial_box: [ymin, xmin, ymax, xmax] normalized coordinates for first frame
            label: Label for the tracked object
            
        Returns:
            Path to output video with tracking visualization
        """
        if not self.available:
            return "Error: SAM3 not available."
        
        try:
            import cv2
            
            video_dir = Path(video_path)
            if not video_dir.is_dir():
                return f"Error: {video_path} is not a directory of frames."
            
            frame_files = sorted(list(video_dir.glob("*.jpg")))
            if not frame_files:
                return "Error: No frames found in directory."
            
            # Read first frame dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            h, w = first_frame.shape[:2]
            
            # Segment first frame
            with open(frame_files[0], 'rb') as f:
                first_frame_data = f.read()
            
            mask, viz_img = self.segment_with_box(first_frame_data, initial_box, label)
            
            if mask is None:
                return f"Error: Failed to segment '{label}' in first frame."
            
            # Save visualization of first frame
            if viz_img:
                viz_path = self.log_dir / f"seg_{label.replace(' ', '_')}_frame0.png"
                viz_img.save(str(viz_path))
                logger.info(f"Saved first frame segmentation to {viz_path}")
            
            # Create output video
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = self.log_dir / f"tracked_{label.replace(' ', '_')}_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 10, (w, h))
            
            # Process each frame
            # For SAM3, we segment each frame independently using box tracking
            # In a real implementation, you might want to use motion estimation
            # or update the box based on previous mask
            
            current_box = initial_box
            
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(str(frame_file))
                
                # Read frame data
                with open(frame_file, 'rb') as f:
                    frame_data = f.read()
                
                # Segment this frame
                mask, _ = self.segment_with_box(frame_data, current_box, label)
                
                if mask is not None:
                    # Ensure mask is the right shape
                    if len(mask.shape) > 2:
                        mask = mask.squeeze()
                    
                    # Resize mask if needed
                    if mask.shape != (h, w):
                        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        mask = mask_resized > 0
                    
                    # Create overlay
                    overlay = frame.copy()
                    overlay[mask > 0] = [0, 0, 255]  # BGR - Red overlay
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    
                    # Update bounding box from mask for next frame
                    rows = np.any(mask > 0, axis=1)
                    cols = np.any(mask > 0, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        # Convert back to normalized [ymin, xmin, ymax, xmax]
                        current_box = [rmin/h, cmin/w, rmax/h, cmax/w]
                        # Draw bounding box
                        cv2.rectangle(frame, (cmin, rmin), (cmax, rmax), (0, 255, 0), 2)
                
                cv2.putText(frame, f"Tracking: {label}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
            
            out.release()
            logger.info(f"Tracking complete. Output saved to {output_path}")
            print(f"Tracking complete. Output: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            import traceback
            traceback.print_exc()
            return f"Tracking failed: {e}"
