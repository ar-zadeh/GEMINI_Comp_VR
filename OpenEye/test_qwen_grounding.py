import os
import json
import base64
import argparse
from PIL import Image
from openai import OpenAI
import io

def encode_image(image_path):
    """Read an image and return base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

def main():
    print("======================================================")
    print(" Qwen3.5 Visual Grounding Test harness")
    print(" Using endpoint: http://100.100.219.101:8001/v1")
    print("======================================================")

    parser = argparse.ArgumentParser(description="Test Qwen3's object grounding abilities.")
    parser.add_argument("--image", type=str, help="Path to the image file to analyze")
    parser.add_argument("--objects", type=str, help="Comma separated list of objects to find")
    args = parser.parse_args()

    image_path = args.image
    objects_str = args.objects

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
        client = OpenAI(
            base_url="http://100.100.219.101:8001/v1",
            api_key="sk-no-key-required",
        )
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    # Instruction mapping exactly what the Gemini VisualGrounder asks for
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

    1. ymin, xmin, ymax, xmax must be normalized coordinates (0.0 to 1.0).
    2. Only return objects you are confident you see.
    """

    print(f"[2] Sending prompt to Qwen3 endpoint for grounding...\n")
    
    try:
        response = client.chat.completions.create(
            model="qwen3",
            messages=[
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
            temperature=0.2, # Lower temp for more precise/deterministic bounding boxes
            response_format={"type": "json_object"} if hasattr(client.chat.completions, 'response_format') else None
        )
        
        message = response.choices[0].message
        content = message.content or ""
        
        print("======== MODEL RESPONSE ========")
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"--- Thinking ---\n{message.reasoning_content}\n----------------")
            
        print(content)
        print("================================")
        
        # Try to parse the output and draw boxes
        try:
            # Simple json extraction if it returned markdown
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                
            parsed_data = json.loads(json_str)
            if "detections" in parsed_data:
                draw_boxes(image_path, parsed_data["detections"])
            else:
                print("No 'detections' key found in the parsed JSON.")
                
        except json.JSONDecodeError as je:
            print(f"Failed to parse model output as JSON: {je}")
            print("The model might not have followed the strict JSON format.")
            
    except Exception as e:
        print(f"Error during grounding request: {e}")

if __name__ == "__main__":
    main()
