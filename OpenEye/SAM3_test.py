import os
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox, draw_box_on_image

# --- 1. Setup & Model Loading (Global Scope) ---
print("Loading SAM 3 Model... this may take a moment.")

# Define paths (adjust if your asset path is different)
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

# Configure CUDA settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Build Model and Processor
# These must be global so the Gradio function can see them
try:
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure the BPE file exists at: {bpe_path}")
    exit(1)

# Initialize global inference state
inference_state = None

# --- 2. Interaction Logic ---

# Global state to store clicks
click_state = {
    "points": [],
    "original_image": None
}

def reset_image_state(image_input):
    """
    Resets the state when a new image is uploaded/cleared.
    """
    global click_state, processor, inference_state
    print("Resetting state due to image upload/clear.")
    click_state["points"] = []
    click_state["original_image"] = image_input
    
    if image_input is not None:
        try:
            pil_image = Image.fromarray(image_input)
            inference_state = processor.set_image(pil_image)
        except Exception as e:
            print(f"Error initializing SAM3: {e}")
            
    return "Image state reset. Click Top-Left corner."

def run_segmentation(current_image):
    """
    Overlays the segmentation mask from inference_state onto the current image.
    """
    global inference_state
    if inference_state is None or "masks" not in inference_state:
        return current_image, "No segmentation available. Please trace a box first."

    # Retrieve masks (boolean tensor: [N, 1, H, W])
    masks = inference_state["masks"]
    
    if masks is None or masks.numel() == 0:
        return current_image, "No masks generated."
    
    # Convert to numpy (assuming single batch, single frame)
    masks_np = masks.detach().cpu().numpy()
    
    # Prepare base image
    if current_image is None:
        return None, "No image to display."
        
    pil_image = Image.fromarray(current_image).convert("RGBA")
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    
    # Apply each mask
    for i in range(masks_np.shape[0]):
        mask = masks_np[i, 0] # (H, W)
        
        # Color: Green with alpha
        color = (30, 255, 30, 100) 
        
        # Create mask image
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        
        # Create a solid color layer
        color_layer = Image.new('RGBA', pil_image.size, color)
        
        # Paste color layer into overlay using the mask
        overlay.paste(color_layer, (0, 0), mask_pil)
        
    # Composite overlay onto original image
    result = Image.alpha_composite(pil_image, overlay)
    
    return np.array(result.convert("RGB")), "Segmentation overlay applied."

def process_interaction(image_input, evt: gr.SelectData):
    """
    Handles user clicks.
    1st Click: marks the top-left corner.
    2nd Click: marks the bottom-right, draws box, runs SAM3.
    """
    global click_state, processor, inference_state

    if image_input is None:
        return None, "Please upload an image first."

    # Check if this is a new image
    is_new_image = False
    if click_state["original_image"] is None:
        is_new_image = True
    elif len(click_state["points"]) == 0 and not np.array_equal(click_state["original_image"], image_input):
        # Only treat as new image if we aren't tracking points
        is_new_image = True

    if is_new_image:
        print("New image detected (auto). Resetting state.")
        click_state["points"] = []
        click_state["original_image"] = image_input
        # Initialize SAM3 state for the new image
        pil_image = Image.fromarray(image_input)
        inference_state = processor.set_image(pil_image)

    # Record the clicked point (x, y)
    click_state["points"].append(evt.index)
    
    # --- State 1: First click (Top-Left) ---
    if len(click_state["points"]) == 1:
        # Draw a small marker to show where the user clicked
        temp_img = Image.fromarray(image_input).copy()
        draw = ImageDraw.Draw(temp_img)
        x, y = evt.index
        r = 5
        draw.ellipse((x-r, y-r, x+r, y+r), fill="red", outline="white")
        return np.array(temp_img), "Click the Bottom-Right corner to finish the box."

    # --- State 2: Second click (Bottom-Right) -> Run SAM3 ---
    elif len(click_state["points"]) == 2:
        p1 = click_state["points"][0]
        p2 = click_state["points"][1]
        
        # Calculate Box (x, y, w, h)
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        w = x2 - x1
        h = y2 - y1
        
        # Guard against zero-size boxes
        if w < 2 or h < 2:
            click_state["points"] = []
            return image_input, "Box too small! Please click two distinct corners."

        print(f"Processing box: {x1}, {y1}, {w}, {h}")

        # Prepare Box for SAM3
        width, height = Image.fromarray(image_input).size
        box_input_xywh = torch.tensor([float(x1), float(y1), float(w), float(h)]).view(-1, 4)
        box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
        norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()

        # Run Inference
        processor.reset_all_prompts(inference_state)
        inference_state = processor.add_geometric_prompt(
            state=inference_state, 
            box=norm_box_cxcywh, 
            label=True
        )
        
        # Draw the green box on the image to confirm tracking
        img_with_box = draw_box_on_image(Image.fromarray(image_input), box_input_xywh.flatten().tolist())
        
        # Reset points for next interaction
        click_state["points"] = []
        
        return np.array(img_with_box), "Object tracked! Click 2 points to track another object."

    return image_input, "Click Top-Left corner."

# --- 3. Launch App ---
if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## SAM 3 Interactive Object Tracking")
        gr.Markdown("Click the **Top-Left** and then the **Bottom-Right** of an object to define the box.")
        
        with gr.Row():
            input_img = gr.Image(label="Input Image", type="numpy")
        with gr.Row():
            segment_btn = gr.Button("Segment Object", variant="primary")
        
        status_text = gr.Textbox(label="Status", value="Upload an image, then click Top-Left corner.")
        
        # Bind the click event
        input_img.select(process_interaction, inputs=[input_img], outputs=[input_img, status_text])
        
        # Bind button event
        segment_btn.click(run_segmentation, inputs=[input_img], outputs=[input_img, status_text])

        # Bind reset events
        
        # Bind reset events
        input_img.upload(reset_image_state, inputs=[input_img], outputs=[status_text])
        input_img.clear(reset_image_state, inputs=[input_img], outputs=[status_text])

    demo.launch(share=True, debug=True)