"""
vr_agent/grounding.py
---------------------
VisualGrounder: object detection via Gemini structured output.
"""

import io
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from PIL import Image
from pydantic import BaseModel, Field

try:
    from google.genai import types
except ImportError:
    pass

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .config import MODEL_GROUNDING
from .logger import get_logger


class VisualGrounder:
    """Handles object detection using Gemini (structured bounding-box output)."""

    def __init__(self, client, log_dir: Path):
        self.client = client
        self.model_name = MODEL_GROUNDING
        self.log_dir = log_dir / "grounding"
        self.log_dir.mkdir(exist_ok=True, parents=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def ground_multiple_objects(
        self, image_data: bytes, object_names: List[str]
    ) -> Dict[str, List[float]]:
        """
        Detect multiple objects in one Gemini call.
        Returns {label: [ymin, xmin, ymax, xmax]} with normalized (0-1) coords.
        """
        logger = get_logger()
        objects_str = ", ".join(object_names)
        logger.info(f"Grounding Multiple (Gemini 3 Flash): '{objects_str}'")

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
        """

        class Detection(BaseModel):
            label: str = Field(description="The exact name of the object found.")
            coordinates: List[float] = Field(
                description="Normalized coordinates [ymin, xmin, ymax, xmax]."
            )

        class GroundingResponse(BaseModel):
            thinking: str = Field(description="Reasoning about the scene and object locations.")
            detections: List[Detection] = Field(description="List of detected objects.")

        try:
            # Clean / re-encode image
            try:
                pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
                out_buffer = io.BytesIO()
                pil_img.save(out_buffer, format="JPEG", quality=100)
                clean_image_data = out_buffer.getvalue()
            except Exception as e:
                logger.warning(f"Image cleaning failed: {e}")
                clean_image_data = image_data

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(
                            mime_type="image/jpeg", data=clean_image_data
                        ))
                    ])
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GroundingResponse,
                }
            )

            if not response.candidates:
                logger.error(
                    f"Gemini returned NO candidates. "
                    f"Finish reason: {getattr(response, 'prompt_feedback', 'Unknown')}"
                )
                return {}

            try:
                parsed_response = response.parsed
                if not parsed_response:
                    parsed_response = GroundingResponse.model_validate_json(response.text)

                results: Dict[str, List[float]] = {}
                valid_boxes_for_draw = []

                for det in parsed_response.detections:
                    label = det.label
                    coords = det.coordinates
                    if label and coords and len(coords) == 4:
                        if any(c > 1.0 for c in coords):
                            coords = [c / 1000.0 for c in coords]
                        results[label] = coords
                        valid_boxes_for_draw.append({"label": label, "box_2d": coords})

                if valid_boxes_for_draw:
                    self._draw_and_save(
                        clean_image_data, valid_boxes_for_draw,
                        f"multi_{len(results)}_objs"
                    )
                else:
                    logger.warning(
                        f"Gemini returned no detections. Raw text: {response.text[:100]}..."
                    )

                return results

            except Exception as e:
                logger.error(f"Failed to parse JSON with Pydantic: {e}. Text: {response.text}")
                return {}

        except Exception as e:
            logger.error(f"Multi-Grounding failed: {e}")
            return {}

    def ground_object(self, image_data: bytes, object_description: str) -> List[Dict]:
        """Single-object wrapper around ground_multiple_objects."""
        res_dict = self.ground_multiple_objects(image_data, [object_description])
        if object_description in res_dict:
            return [{"box_2d": res_dict[object_description], "label": object_description}]
        return []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_and_save(self, image_data: bytes, boxes: List[Dict], description: str):
        """Draw bounding boxes on image and save to log directory."""
        logger = get_logger()
        timestamp = datetime.now().strftime("%H%M%S")
        filename = self.log_dir / f"ground_{timestamp}_{description.replace(' ', '_')}.jpg"

        if not CV2_AVAILABLE:
            return

        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return

            h, w = img.shape[:2]
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]

            for i, box in enumerate(boxes):
                y1, x1, y2, x2 = box['box_2d']
                label = box.get('label', description)
                color = colors[i % len(colors)]
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                cv2.rectangle(img, p1, p2, color, 2)
                cv2.putText(
                    img, label, (p1[0], max(20, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            cv2.imwrite(str(filename), img)
            logger.info(f"Saved grounding to {filename}")
        except Exception as e:
            logger.error(f"CV2 draw failed: {e}")
