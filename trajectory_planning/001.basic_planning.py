## plans a path on the 2d occupancy map using cosmos models
# we'll test 2B, 8B nvfp4, and 8B (unquantized) models
from openai import OpenAI
import json
import os
import base64
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

def decode_json_points(text: str):
    """Parse coordinate points from text format"""
    try:
        # 清理markdown标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        
        # 解析JSON
        data = json.loads(text)
        points = []
        labels = []
        
        for item in data:
            if "point_2d" in item:
                x, y = item["point_2d"]
                
                # Deduplicate identical consecutive points
                if points and points[-1] == [x, y]:
                    continue
                    
                points.append([x, y])
                
                # 获取label，如果没有则使用默认值
                label = item.get("label", f"point_{len(points)}")
                labels.append(label)
        
        # Further prune collinear points (optional but helpful for "deduplication" of straight lines)
        if len(points) > 2:
            pruned_points = [points[0]]
            pruned_labels = [labels[0]]
            for i in range(1, len(points) - 1):
                p1 = points[i-1]
                p2 = points[i]
                p3 = points[i+1]
                
                # Check for collinearity (cross product of vectors p1p2 and p2p3)
                # Using a small epsilon for floating point/pixel variations
                area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area > 10: # Threshold for "straight enough"
                    pruned_points.append(p2)
                    pruned_labels.append(labels[i])
            
            pruned_points.append(points[-1])
            pruned_labels.append(labels[-1])
            return pruned_points, pruned_labels
            
        return points, labels
        
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        return [], []

def visualize_waypoints(img_path, points, labels, output_path):
    """Draw waypoints on the image and save"""
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        for i, (point, label) in enumerate(zip(points, labels)):
            # Map 0-1000 normalized to pixel coordinates
            px = int(point[0] * w / 1000)
            py = int(point[1] * h / 1000)
            
            # Draw point (circle)
            r = 5
            color = (255, 0, 0) # Red for waypoints
            if i == 0: color = (0, 255, 0) # Green for first
            if i == len(points) - 1: color = (0, 0, 255) # Blue for last
            
            draw.ellipse([px-r, py-r, px+r, py+r], fill=color, outline="white")
            draw.text((px + r + 2, py), label, fill="white")
            
            # Draw line to next point
            if i < len(points) - 1:
                next_px = int(points[i+1][0] * w / 1000)
                next_py = int(points[i+1][1] * h / 1000)
                draw.line([px, py, next_px, next_py], fill=(255, 255, 0), width=2)
                
        img.save(output_path)
        print(f"Visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error during visualization: {e}")

def inference_with_openai_api(img_url, prompt, model_name, min_pixels=64 * 32 * 32, max_pixels=9800* 32 * 32):
    if os.path.exists(img_url):
        with open(img_url, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    else:
        raise ValueError(f"Invalid image URL: {img_url}")
    
    client = OpenAI(
        api_key='api-key',
        base_url="http://localhost:8000/v1"
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        top_p=0.3,
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trajectory planning inference.")
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason2-2B", help="Model name to use")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt text")
    parser.add_argument("--prompt_id", type=str, required=True, help="The ID of the prompt for tracking")
    parser.add_argument("--img_path", type=str, default="./occupancy_grid.png", help="Path to the occupancy grid image")
    
    args = parser.parse_args()
    
    # Create output directory based on model name
    model_dir_name = args.model.replace("/", "_")
    output_dir = Path("results") / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_text_file = output_dir / f"{args.prompt_id}.txt"
    output_viz_file = output_dir / f"{args.prompt_id}_viz.png"
    
    print(f"Running inference with model: {args.model}")
    print(f"Prompt ID: {args.prompt_id}")
    
    try:
        response = inference_with_openai_api(args.img_path, args.prompt, args.model)
        
        # Save raw response
        with open(output_text_file, "w") as f:
            f.write(response)
        print(f"Response saved to: {output_text_file}")
        
        # Parse waypoints
        points, labels = decode_json_points(response)
        
        if points:
            print(f"Parsed {len(points)} waypoints. Visualizing...")
            visualize_waypoints(args.img_path, points, labels, output_viz_file)
        else:
            print("No valid waypoints parsed from response.")
            
        print("-" * 20)
        print(response)
        print("-" * 20)
        
    except Exception as e:
        print(f"Error during execution: {e}")
