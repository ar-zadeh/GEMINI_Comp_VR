import os
import torch
from PIL import Image
from vismatch import get_matcher
from vismatch.viz import plot_matches

def find_pattern(image_path1, image_path2, crop_coords):
    print(f"Loading and cropping query image: {image_path1}")
    
    # Load and crop the first image
    try:
        img1_pil = Image.open(image_path1)
        # crop_coords is expected to be [x_min, y_min, x_max, y_max]
        cropped_img1 = img1_pil.crop(tuple(crop_coords))
        temp_crop_path = "temp_query_crop.jpg"
        cropped_img1.save(temp_crop_path)
        print(f"Saved cropped query image to {temp_crop_path}")
    except Exception as e:
        print(f"Error processing query image: {e}")
        return

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize matcher
    # Using 'xfeat-star' as a good default for xfeat based matching, or just 'xfeat' if available/preferred.
    # The doc mentioned 'xfeat-star' under semi-dense. Let's try 'xfeat-star'.
    # Note: If 'xfeat-star' fails, we might fall back to 'xfeat-lightglue' or just 'xfeat'.
    matcher_name = "xfeat-star" 
    print(f"Initializing matcher: {matcher_name}")
    try:
        matcher = get_matcher(matcher_name, device=device)
    except Exception as e:
        print(f"Error initializing matcher {matcher_name}: {e}")
        # Fallback attempt if specific model name issues arise
        print("Attempting fallback to 'xfeat-lightglue'...")
        try:
             matcher = get_matcher("xfeat-lightglue", device=device)
        except Exception as e2:
             print(f"Fallback failed: {e2}")
             return

    # Load images for matcher
    # Resize might be optional, but good for consistency/performance. 
    # However, for pattern matching precise sizes might matter. XFeat handles scales well?
    # Let's try without forcing resize first to keep original resolution if possible, 
    # or use a reasonable max size like 1024 or remain default.
    # Docs example uses resize=512. Let's default to standard behavior or user arg.
    # We'll stick to a reasonable size or None to keep original aspect.
    # matcher.load_image handles reading.
    
    try:
        print("Loading images into matcher...")
        img0 = matcher.load_image(temp_crop_path)
        img1 = matcher.load_image(image_path2)
        
        print("Running matcher...")
        result = matcher(img0, img1)
        
        print("Matching complete.")
        print("Result keys:", result.keys())
        
        if "num_inliers" in result:
            print(f"Number of inliers: {result['num_inliers']}")
        
        output_plot = "plot_matches.png"
        print(f"Saving visualization to {output_plot}...")
        plot_matches(img0, img1, result, save_path=output_plot)
        print("Done.")

        # Clean up temp file
        if os.path.exists(temp_crop_path):
            os.remove(temp_crop_path)

    except Exception as e:
        print(f"Error during matching execution: {e}")

if __name__ == "__main__":
    img1_path = "/home/amir/gemini_project_VR/GEMINI_Comp_VR/OpenEye/agent_logs_test/tracking/test_ref_track/frame_0000.jpg"
    img2_path = "/home/amir/gemini_project_VR/GEMINI_Comp_VR/OpenEye/agent_logs_test/tracking/test_ref_track/frame_0001.jpg"
    
    # [x_min, y_min, x_max, y_max]
    coords = [570, 542, 1000, 770]
    
    if not os.path.exists(img1_path):
        print(f"Error: Image 1 not found at {img1_path}")
    elif not os.path.exists(img2_path):
        print(f"Error: Image 2 not found at {img2_path}")
    else:
        find_pattern(img1_path, img2_path, coords)
