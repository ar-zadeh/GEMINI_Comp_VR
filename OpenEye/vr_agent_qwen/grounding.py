"""
vr_agent/grounding.py
---------------------
VisualGrounder: object detection via OpenAI-compatible structured output.
"""

import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

from PIL import Image
from pydantic import BaseModel, Field

import base64

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .config import MODEL_GROUNDING
from .logger import get_logger


class VisualGrounder:
    """Handles object detection using OpenAI-compatible VLMs (including Qwen)."""

    def __init__(self, client, log_dir: Path):
        self.client = client
        self.model_name = MODEL_GROUNDING
        self.log_dir = log_dir / "grounding"
        self.log_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _coerce_response_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text_val = part.get("text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _extract_json_candidate(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        if cleaned.lower().startswith("json\n"):
            return cleaned[5:].strip()

        return cleaned

    def _parse_detections_with_fallback(self, response_text: str) -> Tuple[List[Dict[str, Any]], str]:
        candidates = [self._extract_json_candidate(response_text), response_text.strip()]

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                detections = parsed.get("detections")
                if isinstance(detections, list):
                    return detections, "dict-detections"
                if all(k in parsed for k in ("label", "coordinates")):
                    return [parsed], "single-detection-dict"
            elif isinstance(parsed, list):
                return parsed, "top-level-list"

        # Last-resort recovery for loosely formatted outputs.
        object_pattern = re.compile(
            r'\{[^{}]*"label"\s*:\s*"([^"]+)"[^{}]*"coordinates"\s*:\s*\[([^\]]+)\][^{}]*\}',
            re.IGNORECASE | re.DOTALL,
        )
        detections: List[Dict[str, Any]] = []
        for label, coords_blob in object_pattern.findall(response_text):
            nums = re.findall(r"-?\d+(?:\.\d+)?", coords_blob)
            if len(nums) >= 4:
                detections.append(
                    {
                        "label": label.strip(),
                        "coordinates": [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])],
                    }
                )

        if detections:
            return detections, "regex-fallback"
        return [], "none"

    # ── Public API ────────────────────────────────────────────────────────────

    def ground_multiple_objects(
        self, image_data: bytes, object_names: List[str]
    ) -> Dict[str, List[float]]:
        """
        Detect multiple objects in one model call.
        Returns {label: [ymin, xmin, ymax, xmax]} with normalized (0-1) coords.
        """
        logger = get_logger()
        objects_str = ", ".join(object_names)
        logger.info(f"Grounding Multiple ({self.model_name}): '{objects_str}'")

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

            base64_img = base64.b64encode(clean_image_data).decode('utf-8')
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]
                    }
                ],
                temperature=0.2,
                extra_body={
                   "chat_template_kwargs": {"enable_thinking": False}  
                },
                response_format={"type": "json_object"}
            )
            
            response_text = self._coerce_response_text(response.choices[0].message.content)
            if not response_text:
                logger.error(
                    f"Model returned NO candidates. "
                    f"Finish reason: {getattr(response, 'prompt_feedback', 'Unknown')}"
                )
                return {}

            try:
                parsed_response = GroundingResponse.model_validate_json(self._extract_json_candidate(response_text))
                detections = [{"label": d.label, "coordinates": d.coordinates} for d in parsed_response.detections]
            except Exception:
                detections, parse_mode = self._parse_detections_with_fallback(response_text)
                logger.warning(f"Grounding JSON schema parse failed; fallback mode: {parse_mode}")

            results: Dict[str, List[float]] = {}
            valid_boxes_for_draw = []

            for det in detections:
                label = str(det.get("label", "")).strip()
                coords = det.get("coordinates")
                if label and coords and len(coords) == 4:
                    coords = [float(c) for c in coords]
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
                    f"Model returned no detections. Raw text: {response_text[:100]}..."
                )

            return results

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
