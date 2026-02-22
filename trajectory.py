import argparse
import json
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image, ImageDraw


MODEL_ID = "gemini-robotics-er-1.5-preview"


def parse_json(json_output: str) -> str:
	lines = json_output.splitlines()
	for index, line in enumerate(lines):
		if line.strip() == "```json":
			json_output = "\n".join(lines[index + 1 :])
			json_output = json_output.split("```")[0]
			break
	return json_output.strip()


def get_image_resized(img_path: str, width: int = 960) -> Image.Image:
	image = Image.open(img_path)
	resized = image.resize(
		(width, int(width * image.size[1] / image.size[0])),
		Image.Resampling.LANCZOS,
	)
	return resized


def call_gemini_robotics_er(
	client: genai.Client,
	image: Image.Image,
	prompt: str,
	temperature: float,
) -> str:
	response = client.models.generate_content(
		model=MODEL_ID,
		contents=[image, prompt],
		config=types.GenerateContentConfig(
			temperature=temperature,
			thinking_config=types.ThinkingConfig(thinking_budget=0),
		),
	)
	if not response.text:
		raise ValueError("Gemini returned an empty response.")
	return parse_json(response.text)


def clamp(value: int, low: int, high: int) -> int:
	return max(low, min(high, value))


def normalize_box(box_2d: list[Any]) -> list[int] | None:
	if not isinstance(box_2d, list) or len(box_2d) != 4:
		return None
	if not all(isinstance(v, (int, float)) for v in box_2d):
		return None

	y1, x1, y2, x2 = [int(round(v)) for v in box_2d]
	y1 = clamp(y1, 0, 1000)
	x1 = clamp(x1, 0, 1000)
	y2 = clamp(y2, 0, 1000)
	x2 = clamp(x2, 0, 1000)

	if x1 > x2:
		x1, x2 = x2, x1
	if y1 > y2:
		y1, y2 = y2, y1
	if (x2 - x1) < 2 or (y2 - y1) < 2:
		return None
	return [y1, x1, y2, x2]


def normalize_point(point: list[Any]) -> list[float] | None:
	if not isinstance(point, list) or len(point) != 2:
		return None
	if not all(isinstance(v, (int, float)) for v in point):
		return None
	y, x = float(point[0]), float(point[1])
	if not (0 <= y <= 1000 and 0 <= x <= 1000):
		return None
	return [y, x]


def validate_response(raw_json: str) -> dict[str, Any]:
	parsed = json.loads(raw_json)
	if not isinstance(parsed, dict):
		return {
			"status": "not_found",
			"reason": "unexpected_json_shape",
			"destination": None,
			"obstacles": [],
			"suggested_path": [],
		}

	status = str(parsed.get("status", "")).lower()
	reason = str(parsed.get("reason", "")).strip() or "unspecified"

	destination = None
	destination_raw = parsed.get("destination")
	if isinstance(destination_raw, dict):
		box = normalize_box(destination_raw.get("box_2d", []))
		label = str(destination_raw.get("label", "destination")).strip() or "destination"
		if box is not None:
			destination = {"label": label, "box_2d": box}

	obstacles: list[dict[str, Any]] = []
	obstacles_raw = parsed.get("obstacles", [])
	if isinstance(obstacles_raw, list):
		for obstacle in obstacles_raw:
			if not isinstance(obstacle, dict):
				continue
			box = normalize_box(obstacle.get("box_2d", []))
			if box is None:
				continue
			label = str(obstacle.get("label", "obstacle")).strip() or "obstacle"
			why = str(obstacle.get("why", "")).strip()
			obstacles.append({"label": label, "box_2d": box, "why": why})

	suggested_path: list[dict[str, Any]] = []
	path_raw = parsed.get("suggested_path", [])
	if isinstance(path_raw, list):
		for item in path_raw:
			if not isinstance(item, dict):
				continue
			point = normalize_point(item.get("point", []))
			if point is None:
				continue
			label = str(item.get("label", "")).strip()
			suggested_path.append({"point": point, "label": label})

	if status not in {"found", "not_found"}:
		status = "found" if destination is not None else "not_found"

	if status == "found" and destination is None:
		status = "not_found"
		reason = "destination_missing"

	return {
		"status": status,
		"reason": reason,
		"destination": destination,
		"obstacles": obstacles,
		"suggested_path": suggested_path,
	}


def norm_box_to_px(box_2d: list[int], width: int, height: int) -> tuple[int, int, int, int]:
	y1, x1, y2, x2 = box_2d
	px_x1 = int((x1 / 1000.0) * width)
	px_y1 = int((y1 / 1000.0) * height)
	px_x2 = int((x2 / 1000.0) * width)
	px_y2 = int((y2 / 1000.0) * height)
	return px_x1, px_y1, px_x2, px_y2


def norm_point_to_px(point: list[float], width: int, height: int) -> tuple[int, int]:
	y, x = point
	px_x = int((x / 1000.0) * width)
	px_y = int((y / 1000.0) * height)
	return px_x, px_y


def annotate_scene(image: Image.Image, result: dict[str, Any]) -> Image.Image:
	draw = ImageDraw.Draw(image)
	width, height = image.size

	destination = result.get("destination")
	if isinstance(destination, dict):
		x1, y1, x2, y2 = norm_box_to_px(destination["box_2d"], width, height)
		draw.rectangle((x1, y1, x2, y2), outline=(50, 220, 50), width=5)
		draw.text((x1 + 5, max(5, y1 - 18)), f"DEST: {destination['label']}", fill=(50, 220, 50))

	for idx, obstacle in enumerate(result.get("obstacles", []), start=1):
		x1, y1, x2, y2 = norm_box_to_px(obstacle["box_2d"], width, height)
		draw.rectangle((x1, y1, x2, y2), outline=(240, 70, 70), width=4)
		draw.text((x1 + 5, max(5, y1 - 18)), f"OBS{idx}: {obstacle['label']}", fill=(240, 70, 70))

	path_points = [norm_point_to_px(item["point"], width, height) for item in result.get("suggested_path", [])]
	if len(path_points) > 1:
		draw.line(path_points, fill=(60, 170, 255), width=3)
		for idx, (px_x, px_y) in enumerate(path_points):
			draw.ellipse((px_x - 4, px_y - 4, px_x + 4, px_y + 4), fill=(255, 255, 0), outline=(0, 0, 0))
			draw.text((px_x + 6, px_y - 10), str(idx), fill=(255, 255, 255))

	if result["status"] != "found":
		draw.rectangle((10, 10, width - 10, 70), fill=(0, 0, 0))
		draw.text((20, 25), f"Destination not found: {result['reason']}", fill=(255, 255, 255))

	return image


def build_prompt(goal_text: str, max_obstacles: int, num_path_points: int) -> str:
	return (
		"You are a robot navigation scene interpreter. "
		f"User goal: \"{goal_text}\". "
		"Infer the target destination region in the image that best matches the user goal. "
		"Then identify only the significant physical obstacles on the most direct walkable route from current viewpoint (bottom-center foreground) to that destination. "
		"Do not include minor texture/shadow patterns as obstacles. "
		"Return obstacles that block or force path deviation. "
		f"Return at most {max_obstacles} obstacles. "
		f"Also provide an ordered suggested_path with exactly {num_path_points} points if destination is found. "
		"Return JSON only in this exact schema: "
		"{"
		"\"status\":\"found\"|\"not_found\","
		"\"reason\":\"short reason\","
		"\"destination\":{\"label\":\"...\",\"box_2d\":[ymin,xmin,ymax,xmax]}|null,"
		"\"obstacles\":[{\"label\":\"...\",\"box_2d\":[ymin,xmin,ymax,xmax],\"why\":\"...\"}],"
		"\"suggested_path\":[{\"point\":[y,x],\"label\":\"0\"}]"
		"}. "
		"All box coordinates and points must be normalized to 0-1000. "
		"If the destination is not visible, return status not_found with destination null and empty arrays."
	)


def default_output_paths(image_path: str) -> tuple[Path, Path]:
	image_file = Path(image_path)
	base = image_file.stem
	parent = image_file.parent if str(image_file.parent) != "" else Path(".")
	return (
		parent / f"{base}_navigation_result.json",
		parent / f"{base}_navigation_overlay.png",
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Detect destination and route obstacles with bounding boxes for a natural-language navigation goal."
	)
	parser.add_argument("--image", required=True, help="Path to the input image")
	parser.add_argument(
		"--goal",
		required=True,
		help='Natural-language goal, e.g. "I want to go behind the bar".',
	)
	parser.add_argument(
		"--api-key",
		default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
		help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.2,
		help="Sampling temperature for model response.",
	)
	parser.add_argument(
		"--max-obstacles",
		type=int,
		default=8,
		help="Maximum number of obstacle boxes to return.",
	)
	parser.add_argument(
		"--num-path-points",
		type=int,
		default=10,
		help="Number of suggested path points.",
	)
	parser.add_argument("--output-json", default=None, help="Output JSON file path.")
	parser.add_argument("--output-image", default=None, help="Output overlay image path.")

	args = parser.parse_args()

	if not args.api_key:
		raise ValueError("Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --api-key.")
	if args.max_obstacles < 1:
		raise ValueError("--max-obstacles must be at least 1.")
	if args.num_path_points < 2:
		raise ValueError("--num-path-points must be at least 2.")

	default_json, default_image = default_output_paths(args.image)
	output_json_path = Path(args.output_json) if args.output_json else default_json
	output_image_path = Path(args.output_image) if args.output_image else default_image

	client = genai.Client(api_key=args.api_key)
	image = get_image_resized(args.image)
	prompt = build_prompt(args.goal, args.max_obstacles, args.num_path_points)

	raw_json = call_gemini_robotics_er(client, image, prompt, args.temperature)
	result = validate_response(raw_json)

	output_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
	annotated = annotate_scene(image.convert("RGB"), result)
	annotated.save(output_image_path)

	print(f"Saved navigation JSON: {output_json_path.resolve()}")
	print(f"Saved overlay image: {output_image_path.resolve()}")
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
