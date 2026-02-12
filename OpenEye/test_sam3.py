import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
CKPT_DIR = "./checkpoints/sam3.pt"
# print(help(build_sam3_image_model))
model = build_sam3_image_model(checkpoint_path=CKPT_DIR)
processor = Sam3Processor(model)