from pathlib import Path
import argparse
import cv2

# Bounding boxes: (label, (x1, y1, x2, y2))
BOXES = [
    ("Steam logo", (471, 375, 513, 417)),
    ("VR laser", (421, 519, 994, 988)),
]

DEFAULT_IMAGE_CANDIDATES = [
    # Preferred location under this repo
    Path(__file__).parent / "agent_logs" / "videos" / "video_150152" / "frame_0029.jpg",
    # Fallback: same directory as this script
    Path(__file__).parent / "frame_0029.jpg",
]
DEFAULT_OUTPUT_NAME = "frame_0029_with_boxes.jpg"


def draw_boxes(image_path: Path, output_path: Path | None = None) -> Path:
    # Print the current files in the directory for quick debugging
    base_dir = image_path.parent
    print(f"Base directory: {base_dir}")
    try:
        files = list(base_dir.glob("*"))
        print(f"Files in directory: {[f.name for f in files]}")
    except Exception:
        print("Could not list base directory contents.")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(
            f"Could not read image at {image_path}. "
            f"Pass a valid path via --image, e.g.: \n"
            f"  python visualize_bboxes.py --image \"D:/path/to/frame_0029.jpg\""
        )

    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]  # BGR colors rotated per box

    for idx, (label, (x1, y1, x2, y2)) in enumerate(BOXES):
        color = colors[idx % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        text_origin = (x1, max(y1 - 10, 10))
        cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=2)

    # Determine output path
    if output_path is None:
        output_path = image_path.parent / DEFAULT_OUTPUT_NAME
    elif not output_path.is_absolute():
        output_path = image_path.parent / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to write output image to {output_path}")

    print(f"Wrote image with boxes to {output_path}")
    return output_path


def resolve_default_image() -> Path:
    for candidate in DEFAULT_IMAGE_CANDIDATES:
        if candidate.exists():
            return candidate
    # Return the first candidate even if it doesn't exist so error message is consistent
    return DEFAULT_IMAGE_CANDIDATES[0]


def main():
    parser = argparse.ArgumentParser(description="Visualize predefined bounding boxes on an image.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the input image. If omitted, tries known default locations.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path or filename (defaults next to input).",
    )
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else resolve_default_image()
    output_path = Path(args.out) if args.out else None

    draw_boxes(image_path=image_path, output_path=output_path)


if __name__ == "__main__":
    main()
