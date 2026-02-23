import requests
import base64
import json
import re
from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
URL = "http://localhost:8000/v1/chat/completions"
MODEL = "nvidia/Cosmos-Reason2-8b"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_clock_direction(content):
    """
    Parses the clock number (10, 11, 12, 1, 2) from response.
    """
    # Extract the first appearing number in the range [10, 11, 12, 1, 2]
    numbers = re.findall(r"\d+", content)
    for num_str in numbers:
        num = int(num_str)
        if num in [10, 11, 12, 1, 2]:
            return num
    return None

def map_clock_to_angle(clock_num):
    """
    Maps clock face numbers to rotation angles.
    10 o'clock -> -30° (left)
    11 o'clock -> -15° (slight left)
    12 o'clock -> 0° (straight)
    1 o'clock -> +15° (slight right)
    2 o'clock -> +30° (right)
    """
    mapping = {
        10: -30,
        11: -15,
        12: 0,
        1: 15,
        2: 30
    }
    return mapping.get(clock_num, None)

def visualize_clock(image_path, clock_num, angle, output_path):
    """Annotates the image with clock direction and corresponding angle."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 40)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 28)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    if clock_num is not None:
        # Main result
        text = f"CLOCK: {clock_num} o'clock"
        color = "yellow"
        draw.text((52, 52), text, fill="black", font=font_large)
        draw.text((50, 50), text, fill=color, font=font_large)

        # Angle mapping
        angle_text = f"Rotation: {angle:+d}°"
        draw.text((52, 102), angle_text, fill="black", font=font_small)
        draw.text((50, 100), angle_text, fill="lime", font=font_small)

        # Draw a simple clock face indicator
        center_x = width - 150
        center_y = 150
        radius = 80

        # Draw clock circle
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline="white", width=3
        )

        # Draw clock hand pointing to selected direction
        # Convert clock position to angle (12=0°, clockwise)
        clock_angles = {10: 300, 11: 330, 12: 0, 1: 30, 2: 60}
        hand_angle = clock_angles.get(clock_num, 0)
        import math
        hand_rad = math.radians(hand_angle - 90)  # Adjust so 0° points up
        hand_x = center_x + int(radius * 0.7 * math.cos(hand_rad))
        hand_y = center_y + int(radius * 0.7 * math.sin(hand_rad))

        draw.line((center_x, center_y, hand_x, hand_y), fill="lime", width=4)
        draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill="lime")

        # Label 12 o'clock
        draw.text((center_x - 8, center_y - radius - 25), "12", fill="white", font=font_small)

    else:
        text = "CLOCK DIRECTION NOT FOUND"
        color = "red"
        draw.text((52, 52), text, fill="black", font=font_large)
        draw.text((50, 50), text, fill=color, font=font_large)

    img.save(output_path)
    print(f"Visualization saved to {output_path}")

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    image_path = os.path.join(project_root, "assets", "image_15_35_34.png")
    output_path = os.path.join(script_dir, "outputs", "output_clockface.png")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base64_image = encode_image(image_path)

    prompt = """Imagine a clock face overlay on this first-person view. 12 o'clock is straight ahead, 10 o'clock is left, 2 o'clock is right. Which clock direction (10, 11, 12, 1, or 2) has the clearest and longest obstacle-free walking path? Respond with only the clock number."""

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
        "max_tokens": 50,
        "temperature": 0.1
    }

    print("=" * 80)
    print("CLOCKFACE NAVIGATION - Discrete Clock Position Mapping")
    print("=" * 80)
    print("--- Sending request to Cosmos Reason 8B ---")

    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        print(f"\n[AI Response]: {content}")

        clock_num = parse_clock_direction(content)
        if clock_num is not None:
            angle = map_clock_to_angle(clock_num)
            print(f"\n[Parsed Clock]: {clock_num} o'clock")
            print(f"[Mapped Angle]: {angle:+d} degrees")
            print(f"\n[Result]: Clock position successfully mapped to rotation angle")
            visualize_clock(image_path, clock_num, angle, output_path)
        else:
            print("\n[Error]: Could not parse clock number from response.")
            visualize_clock(image_path, None, None, output_path)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
