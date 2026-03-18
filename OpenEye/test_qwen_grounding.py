import os
import json
import base64
import argparse
import time
import statistics
import re
from PIL import Image
from openai import OpenAI
import io

def encode_image(image_path, target_height=540):
    """Load image, downscale to target height, and return base64 JPEG bytes."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size

        # Keep aspect ratio and avoid upscaling smaller images.
        if h > target_height:
            new_w = int(round(w * (target_height / float(h))))
            img = img.resize((new_w, target_height), Image.LANCZOS)
            print(f"[Image] Resized for model input: {w}x{h} -> {new_w}x{target_height}")
        else:
            print(f"[Image] Using original size for model input: {w}x{h}")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def draw_boxes(image_path, detections, output_path="grounding_result.jpg"):
    """Optionally draw the returned normalized boxes on the image if OpenCV is installed."""
    try:
        import cv2
        import numpy as np
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        for det in detections:
            label = det.get("label", "Object")
            coords = det.get("coordinates", [])
            if len(coords) == 4:
                ymin, xmin, ymax, xmax = coords
                
                # Rescale if not normalized
                if any(c > 1.0 for c in coords):
                    ymin, xmin, ymax, xmax = [c / 1000.0 for c in coords]
                    
                p1 = (int(xmin * w), int(ymin * h))
                p2 = (int(xmax * w), int(ymax * h))
                
                cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
                cv2.putText(img, label, (p1[0], max(20, p1[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, img)
        print(f"\n[+] Saved grounded image visualization to: {output_path}")
    except ImportError:
        print("OpenCV not installed. Cannot draw bounding boxes. (Run `pip install opencv-python`)")


def extract_json_candidate(content):
    """Extract likely JSON payload from raw model text."""
    if "```json" in content:
        return content.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in content:
        return content.split("```", 1)[1].split("```", 1)[0].strip()
    return content.strip()


def parse_detections_with_fallback(content):
    """Parse detections from strict JSON first, then recover from loose text."""
    candidates = [extract_json_candidate(content), content.strip()]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("detections"), list):
                    return parsed["detections"], "strict-json"
                if all(k in parsed for k in ("label", "coordinates")):
                    return [parsed], "single-object-json"
            if isinstance(parsed, list):
                return parsed, "top-level-list-json"
        except json.JSONDecodeError:
            pass

    # Fallback for non-structured answers that still contain object-like snippets.
    object_pattern = re.compile(
        r'\{[^{}]*"label"\s*:\s*"([^"]+)"[^{}]*"coordinates"\s*:\s*\[([^\]]+)\][^{}]*\}',
        re.IGNORECASE | re.DOTALL,
    )
    detections = []
    for label, coords_blob in object_pattern.findall(content):
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

def main():
    print("======================================================")
    print(" Qwen3.5 Visual Grounding Test harness")
    print(" Using endpoint: http://100.100.219.101:8001/v1")
    print("======================================================")

    parser = argparse.ArgumentParser(description="Test Qwen3's object grounding abilities.")
    parser.add_argument("--image", type=str, help="Path to the image file to analyze")
    parser.add_argument("--objects", type=str, help="Comma separated list of objects to find")
    parser.add_argument("--no-think", action="store_true", help="Ask the model to suppress chain-of-thought output")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum output tokens for the model response")
    args = parser.parse_args()

    image_path = args.image
    objects_str = args.objects
    no_think = args.no_think
    max_tokens = args.max_tokens

    if not image_path:
        image_path = input("Enter the path to the image file: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    if not objects_str:
        objects_str = input("What objects do you want to find? (e.g. 'a red cup, the keyboard'): ").strip()

    print(f"\n[1] Encoding image: {image_path}")
    base64_image = encode_image(image_path)

    # Initialize the OpenAI client pointing to the local server
    try:
        # client = OpenAI(
        #     base_url="http://100.100.219.101:8001/v1",
        #     api_key="sk-no-key-required",
        # )
        client = OpenAI(
            base_url="https://zippy-sarita-flabbier.ngrok-free.dev/v1", # Replace with your URL
            api_key="sk-no-key-required",
            default_headers={"ngrok-skip-browser-warning": "true"} # Bypasses the HTML warning
        )
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    # Instruction mapping exactly what the Gemini VisualGrounder asks for
    think_directive = "/no_think\n" if no_think else ""
    prompt = f"""
    {think_directive}
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

    1. ymin, xmin, ymax, xmax must be normalized coordinates (0.0 to 1.0).
    2. Only return objects you are confident you see.
    3. Do not think more than 15 words. Focus on directly answering the question with bounding boxes.
    """

    print(f"[2] Sending prompt to Qwen3 endpoint for grounding (max_tokens={max_tokens})...\n")
    
    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="qwen3",
            messages=[
                {
                    "role": "system",
                    "content": "You are a visual grounding assistant. Return ONLY valid JSON. Do not output markdown, code fences, or extra explanation.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=0.2, # Lower temp for more precise/deterministic bounding boxes
            extra_body={
               "chat_template_kwargs": {"enable_thinking": False}  
            },
            response_format={"type": "json_object"} if hasattr(client.chat.completions, 'response_format') else None
        )
        elapsed = time.perf_counter() - start
        # simple one-call benchmark list for compatibility with other harness
        try:
            rt_times.append(elapsed)
        except NameError:
            rt_times = [elapsed]
        avg = statistics.mean(rt_times) if rt_times else elapsed
        print(f"\n[Benchmark] API call RTT: {elapsed:.3f}s (avg {avg:.3f}s over {len(rt_times)} calls)")

        message = response.choices[0].message
        content = message.content or ""
        
        print("======== MODEL RESPONSE ========")
        if (not no_think) and hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"--- Thinking ---\n{message.reasoning_content}\n----------------")
            
        print(content)
        print("================================")
        
        detections, parse_mode = parse_detections_with_fallback(content)
        if detections:
            print(f"[Parser] Parsed {len(detections)} detection(s) using mode: {parse_mode}")
            draw_boxes(image_path, detections)
        else:
            print("Failed to parse detections from model output (strict + fallback modes).")
            print("The model response may not include label/coordinates in a recoverable format.")
            print("-------- RAW MODEL OUTPUT (UNPARSED) --------")
            print(content if content else "<empty>")
            print("---------------------------------------------")
            
    except Exception as e:
        print(f"Error during grounding request: {e}")
        # Some OpenAI-compatible servers include useful payload details in the exception.
        raw_error = getattr(e, "body", None) or getattr(e, "response", None)
        if raw_error:
            print("-------- RAW SERVER ERROR PAYLOAD --------")
            print(raw_error)
            print("-----------------------------------------")

if __name__ == "__main__":
    main()
