import os
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger("VRAgent.Tracker")


class ObjectTracker:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "tracking"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.available = False
        self.processor = None
        self.model = None
        
        # Check imports
        try:
            import torch
            import sam3
            from sam3 import build_sam3_image_model
            from sam3.model.box_ops import box_xywh_to_cxcywh
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.visualization_utils import normalize_bbox
            
            self.torch = torch
            self.sam3 = sam3
            self.build_sam3_image_model = build_sam3_image_model
            self.box_xywh_to_cxcywh = box_xywh_to_cxcywh
            self.Sam3Processor = Sam3Processor
            self.normalize_bbox = normalize_bbox
            self.available = True
        except ImportError as e:
            logger.warning(f"SAM 3 or Torch not available: {e}")
            self.available = False
            return

        # Initialize Model
        if self.available:
            try:
                # Configure CUDA settings
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Get BPE path from sam3 assets
                sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
                bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
                
                # Build model and processor
                self.model = build_sam3_image_model(bpe_path=bpe_path)
                self.processor = Sam3Processor(self.model, confidence_threshold=0.85)
                logger.info("SAM 3 Model initialized.")
            except Exception as e:
                logger.error(f"Failed to init SAM 3: {e}")
                self.available = False

    def track(self, video_path: str, initial_box: List[float], label: str) -> str:
        """
        Track an object in a video given an initial bounding box.
        video_path: Path to the video file (mp4 or directory of frames).
                    Expects a directory of JPEGs.
        initial_box: [ymin, xmin, ymax, xmax] normalized.
        
        Returns: Path to the output video with tracking visualization.
        """
        if not self.available or not self.processor:
            return "Error: SAM 3 not available."

        try:
            import cv2
            
            video_dir = Path(video_path)
            if not video_dir.is_dir():
                return f"Error: {video_path} is not a directory of frames."
            
            frame_files = sorted(list(video_dir.glob("*.jpg")))
            if not frame_files:
                return f"Error: No JPEG frames found in {video_path}"
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            h, w = first_frame.shape[:2]
            
            # Convert initial_box from [ymin, xmin, ymax, xmax] normalized to [x, y, w, h] pixels
            ymin, xmin, ymax, xmax = initial_box
            box_x = xmin * w
            box_y = ymin * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h
            
            # Setup output video
            output_path = self.log_dir / f"tracked_{label.replace(' ', '_')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 10, (w, h))
            
            # Process each frame with SAM3
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(str(frame_file))
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Set image for SAM3
                inference_state = self.processor.set_image(pil_image)
                
                # Prepare box for SAM3: convert to normalized cxcywh format
                box_input_xywh = self.torch.tensor([box_x, box_y, box_w, box_h]).view(-1, 4)
                box_input_cxcywh = self.box_xywh_to_cxcywh(box_input_xywh)
                norm_box_cxcywh = self.normalize_bbox(box_input_cxcywh, w, h).flatten().tolist()
                
                # Add geometric prompt (box)
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box_cxcywh,
                    label=True
                )
                
                # Get mask from inference state if available
                try:
                    # Retrieve masks (boolean tensor: [N, 1, H, W])
                    if "masks" in inference_state and inference_state["masks"] is not None:
                        # masks is [N, 1, H, W], we want [H, W]
                        # Assuming batch size 1
                        mask_tensor = inference_state["masks"]
                        mask = mask_tensor.detach().cpu().numpy() > 0.5
                        
                        # Handle dimensions
                        if mask.ndim == 4: # N, 1, H, W
                            mask = mask[0, 0]
                        elif mask.ndim == 3: # 1, H, W or H, W, ?
                            mask = mask[0]
                        
                        # Draw mask overlay with reduced opacity to avoid gray hue
                        # Only apply if mask covers a reasonable area (not entire frame)
                        mask_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
                        if mask_ratio < 0.9:  # Skip overlay if mask covers >90% of frame
                            overlay = frame.copy()
                            color = [0, 255, 0]  # Green (BGR)
                            overlay[mask] = color
                            # Blend with 20% overlay, 80% original (reduced from 40/60)
                            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                        
                        # Update bounding box for next frame based on mask
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        if rows.any() and cols.any():
                            rmin, rmax = np.where(rows)[0][[0, -1]]
                            cmin, cmax = np.where(cols)[0][[0, -1]]
                            # Update box for next frame
                            box_x = cmin
                            box_y = rmin
                            box_w = cmax - cmin
                            box_h = rmax - rmin
                            # Draw bounding box
                            cv2.rectangle(frame, (cmin, rmin), (cmax, rmax), (0, 255, 0), 2)
                    else:
                        # If no mask, just draw the current box
                        cv2.rectangle(frame, 
                                    (int(box_x), int(box_y)), 
                                    (int(box_x + box_w), int(box_y + box_h)), 
                                    (0, 255, 0), 2)
                except Exception as mask_error:
                    logger.warning(f"Could not extract mask for frame {i}: {mask_error}")
                    # Draw current box as fallback
                    cv2.rectangle(frame, 
                                (int(box_x), int(box_y)), 
                                (int(box_x + box_w), int(box_y + box_h)), 
                                (0, 255, 0), 2)
                
                cv2.putText(frame, f"Tracking: {label}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
            
            out.release()
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            import traceback
            traceback.print_exc()
            return f"Tracking failed: {e}"
    def ground_multiple_objects(self, image_data: bytes, object_names: List[str]) -> Dict[str, List[float]]:
        """
        Grounds multiple objects in a single pass.
        Returns: Dict { "label": [ymin, xmin, ymax, xmax] }
        """
        logger = get_logger()
        objects_str = ", ".join(object_names)
        logger.info(f"Grounding Multiple: '{objects_str}'")
        
        prompt = f"""
        Find the following objects in the image: {objects_str}.
        
        You MUST return the answer in the following JSON format:
        {{
            "thinking": "Reasoning about the scene...",
            "detections": [
                {{
                    "label": "exact_object_name_from_list",
                    "coordinates": [ymin, xmin, ymax, xmax]
                }}
            ]
        }}
        
        1. ymin, xmin, ymax, xmax must be normalized coordinates (0 to 1).
        2. Only return objects you are confident you see.
        3. If an object appears multiple times, pick the most prominent one.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data))
                    ])
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
            )
            
            # Parse result
            text = response.candidates[0].content.parts[0].text
            data = json.loads(text)
            
            if "thinking" in data:
                logger.info(f"[Grounding Thought] {data['thinking']}")
            
            results = {}
            valid_boxes_for_draw = []

            for det in data.get("detections", []):
                label = det.get("label")
                coords = det.get("coordinates")
                
                # Sanity check
                if label and coords and len(coords) == 4:
                    # Handle 0-1000 scale if Gemini hallucinates that format
                    if any(c > 1.0 for c in coords):
                        coords = [c / 1000.0 for c in coords]
                    
                    results[label] = coords
                    valid_boxes_for_draw.append({"label": label, "box_2d": coords})

            # Draw and save visualization
            if valid_boxes_for_draw:
                self._draw_and_save(image_data, valid_boxes_for_draw, f"multi_{len(results)}_objs")
                
            return results

        except Exception as e:
            logger.error(f"Multi-Grounding failed: {e}")
            traceback.print_exc()
            return {}

    def _draw_and_save(self, image_data: bytes, boxes: List[Dict], description: str):
        # ... (Keep your existing _draw_and_save method exactly as it is) ...
        # Ensure it handles the list of boxes correctly (which your previous code did).
        logger = get_logger()
        timestamp = datetime.now().strftime("%H%M%S")
        filename = self.log_dir / f"ground_{timestamp}_{description}.jpg"
        
        if CV2_AVAILABLE:
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                
                colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]

                for i, box in enumerate(boxes):
                    y1, x1, y2, x2 = box['box_2d']
                    label = box.get('label', description)
                    
                    p1 = (int(x1*w), int(y1*h))
                    p2 = (int(x2*w), int(y2*h))
                    
                    color = colors[i % len(colors)]
                    
                    cv2.rectangle(img, p1, p2, color, 3)
                    cv2.putText(img, label, (p1[0], max(20, p1[1]-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imwrite(str(filename), img)
                logger.info(f"Saved grounding to {filename}")
            except Exception as e:
                logger.error(f"CV2 draw failed: {e}")
    def track_multi_objects(self, video_path: str, initial_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Track multiple objects in a video.
        video_path: Directory of frames.
        initial_data: Dict mapping 'label' -> [ymin, xmin, ymax, xmax] (normalized)

        Returns:
            Dict containing:
            - 'video_path': Path to visualization video
            - 'telemetry': List of Dicts (one per frame) with centroids:
               [{'label1': [cx, cy], 'label2': [cx, cy]}, ...]
        """
        if not self.available or not self.processor:
            return {"error": "SAM 3 not available."}

        try:
            import cv2
            
            video_dir = Path(video_path)
            frame_files = sorted(list(video_dir.glob("*.jpg")))
            if not frame_files:
                return {"error": "No frames found"}
            
            first_frame = cv2.imread(str(frame_files[0]))
            h, w = first_frame.shape[:2]
            
            # Setup output video
            timestamp = 0 # timestamp logic if needed
            output_path = self.log_dir / f"tracked_multi.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 10, (w, h))

            # Initialize states for each object
            # For simplicity, we'll re-run inference for each object per frame 
            # (Independent tracking) as SAM3 multi-object API is more complex to setup here.
            # We maintain current boxes for each object.
            
            # Convert normalized boxes to pixel [x, y, w, h]
            current_boxes = {}
            for label, box in initial_data.items():
                ymin, xmin, ymax, xmax = box
                current_boxes[label] = [
                    xmin * w, ymin * h, 
                    (xmax - xmin) * w, (ymax - ymin) * h
                ]
            
            telemetry_data = [] # List of frame data
            
            colors = {
                0: (0, 255, 0),   # Green
                1: (0, 0, 255),   # Red
                2: (255, 0, 0),   # Blue
                3: (0, 255, 255)  # Yellow
            }

            for i, frame_file in enumerate(frame_files):
                # Use PIL to read the file first to fix truncation/corruption, then feed to OpenCV
                pil_image_raw = Image.open(frame_file)
                pil_image = pil_image_raw.convert("RGB") # Ensure RGB for SAM
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV viz

                # We need to set image ONCE per frame
                inference_state = self.processor.set_image(pil_image)
                
                frame_telemetry = {}
                
                # For visualization blending
                vis_frame = frame.copy()
                
                for idx, (label, box_rect) in enumerate(current_boxes.items()):
                    box_x, box_y, box_w, box_h = box_rect
                    
                    # Prepare box
                    box_input_xywh = self.torch.tensor([box_x, box_y, box_w, box_h]).view(-1, 4)
                    box_input_cxcywh = self.box_xywh_to_cxcywh(box_input_xywh)
                    norm_box_cxcywh = self.normalize_bbox(box_input_cxcywh, w, h).flatten().tolist()
                    
                    # Run Inference
                    self.processor.reset_all_prompts(inference_state)
                    inference_state = self.processor.add_geometric_prompt(
                        state=inference_state,
                        box=norm_box_cxcywh,
                        label=True
                    )
                    
                    # Extract result
                    cx, cy = 0.0, 0.0
                    mask_found = False
                    
                    if "masks" in inference_state and inference_state["masks"] is not None:
                        mask_tensor = inference_state["masks"]
                        mask = mask_tensor.detach().cpu().numpy() > 0.5
                        if mask.ndim == 4: mask = mask[0, 0]
                        elif mask.ndim == 3: mask = mask[0]
                        
                        # Update box
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        
                        if rows.any() and cols.any():
                            rmin, rmax = np.where(rows)[0][[0, -1]]
                            cmin, cmax = np.where(cols)[0][[0, -1]]
                            
                            # Calcluate Centroid
                            M = cv2.moments(mask.astype(np.uint8))
                            if M["m00"] != 0:
                                cx = M["m10"] / M["m00"]
                                cy = M["m01"] / M["m00"]
                            else:
                                cx = (cmin + cmax) / 2
                                cy = (rmin + rmax) / 2
                            
                            # Update next box
                            current_boxes[label] = [cmin, rmin, cmax - cmin, rmax - rmin]
                            frame_telemetry[label] = [cx / w, cy / h] # Normalized
                            mask_found = True
                            
                            # Viz Mask
                            color = colors.get(idx % 4, (255, 255, 255))
                            vis_frame[mask] = color # Simple overlay
                            
                            # Viz Box & Label
                            cv2.rectangle(vis_frame, (cmin, rmin), (cmax, rmax), color, 2)
                            cv2.circle(vis_frame, (int(cx), int(cy)), 5, (255, 255, 255), -1)
                            cv2.putText(vis_frame, label, (cmin, rmin-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if not mask_found:
                         # Fallback to center of previous box
                         frame_telemetry[label] = [(box_x + box_w/2)/w, (box_y + box_h/2)/h]

                # Blend with reduced opacity to avoid gray hue (20% overlay, 80% original)
                cv2.addWeighted(vis_frame, 0.2, frame, 0.8, 0, frame)
                out.write(frame)
                telemetry_data.append(frame_telemetry)

            out.release()
            
            return {
                "video_path": str(output_path),
                "telemetry": telemetry_data
            }

        except Exception as e:
            logger.error(f"Multi-track error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
