import requests
import base64
import json
import re
from PIL import Image, ImageDraw, ImageFont

# Configuration
URL = "http://localhost:8000/v1/chat/completions"
MODEL = "nvidia/Cosmos-Reason2-8b"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_freespace_direction(content):
    """
    Parses 'DIRECTION (LEFT, STRAIGHT, RIGHT)'
    """
    content = content.upper()
    if "STRAIGHT" in content:
        return "STRAIGHT"
    elif "LEFT" in content:
        return "LEFT"
    elif "RIGHT" in content:
        return "RIGHT"
    return None

def visualize_freespace(image_path, direction, reason, output_path="output_freespace.png"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    try:
        font_dir = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 40)
        font_reason = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf", 25)
    except:
        font_dir = ImageFont.load_default()
        font_reason = ImageFont.load_default()

    if direction:
        text_dir = f"DIRECTION: {direction}"
        color = "lime" if direction == "STRAIGHT" else "orange"
        text_reason = f"REASON: {reason}"
    else:
        text_dir = "DIRECTION NOT FOUND"
        color = "red"
        text_reason = ""

    draw.text((52, 52), text_dir, fill="black", font=font_dir)
    draw.text((50, 50), text_dir, fill=color, font=font_dir)
    
    if text_reason:
        draw.text((52, 112), text_reason, fill="black", font=font_reason)
        draw.text((50, 110), text_reason, fill="white", font=font_reason)

    img.save(output_path)
    print(f"Visualization saved to {output_path}")

def main():
    image_path = "assets/image_15_35_34.png"
    base64_image = encode_image(image_path)

    prompt = """You are looking through the eyes of a person walking forward. Identify the walkable floor area in this image. Is the floor clear ahead for at least 3 steps? If not, is there more walkable floor space to the LEFT or RIGHT? Respond with: DIRECTION (LEFT, STRAIGHT, RIGHT) and a brief reason."""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": 150,
        "temperature": 0.3
    }

    print("--- Sending request to Cosmos Reason 8B (Free-Space Detection) ---")
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        print(f"\n[AI Response]: {content}")
        
        direction = parse_freespace_direction(content)
        # Extract reason: split by "REASON" or just take everything after the direction
        reason_match = re.search(r"REASON\s*[:\s]*(.*)", content, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            # Fallback extraction
            reason = content.replace("DIRECTION", "").replace("STRAIGHT", "").replace("LEFT", "").replace("RIGHT", "").replace("(", "").replace(")", "").strip()
            if reason.startswith(":") or reason.startswith("-"):
                reason = reason[1:].strip()

        if direction:
            print(f"\n[Parsed Direction]: {direction}")
            print(f"[Parsed Reason]: {reason}")
            visualize_freespace(image_path, direction, reason)
        else:
            print("\n[Error]: Could not parse direction from response.")
            visualize_freespace(image_path, None, "")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
