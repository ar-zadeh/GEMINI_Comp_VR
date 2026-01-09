"""
Test script to verify SAM3 imports and basic functionality work correctly.
"""
import os
import sys

print("=" * 60)
print("SAM3 Import Test")
print("=" * 60)

# Test 1: Basic imports
print("\n[1] Testing basic imports...")
try:
    import torch
    print(f"    ✓ torch imported (version: {torch.__version__})")
    print(f"    ✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"    ✗ torch import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"    ✓ numpy imported (version: {np.__version__})")
except ImportError as e:
    print(f"    ✗ numpy import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print(f"    ✓ PIL imported")
except ImportError as e:
    print(f"    ✗ PIL import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"    ✓ cv2 imported (version: {cv2.__version__})")
except ImportError as e:
    print(f"    ✗ cv2 import failed: {e}")
    sys.exit(1)

# Test 2: SAM3 imports
print("\n[2] Testing SAM3 imports...")
try:
    import sam3
    print(f"    ✓ sam3 imported")
    print(f"    ✓ sam3 location: {sam3.__file__}")
except ImportError as e:
    print(f"    ✗ sam3 import failed: {e}")
    sys.exit(1)

try:
    from sam3 import build_sam3_image_model
    print(f"    ✓ build_sam3_image_model imported")
except ImportError as e:
    print(f"    ✗ build_sam3_image_model import failed: {e}")
    sys.exit(1)

try:
    from sam3.model.box_ops import box_xywh_to_cxcywh
    print(f"    ✓ box_xywh_to_cxcywh imported")
except ImportError as e:
    print(f"    ✗ box_xywh_to_cxcywh import failed: {e}")
    sys.exit(1)

try:
    from sam3.model.sam3_image_processor import Sam3Processor
    print(f"    ✓ Sam3Processor imported")
except ImportError as e:
    print(f"    ✗ Sam3Processor import failed: {e}")
    sys.exit(1)

try:
    from sam3.visualization_utils import normalize_bbox, draw_box_on_image
    print(f"    ✓ normalize_bbox imported")
    print(f"    ✓ draw_box_on_image imported")
except ImportError as e:
    print(f"    ✗ visualization_utils import failed: {e}")
    sys.exit(1)

# Test 3: Check BPE path
print("\n[3] Checking BPE asset path...")
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
print(f"    BPE path: {bpe_path}")
if os.path.exists(bpe_path):
    print(f"    ✓ BPE file exists")
else:
    print(f"    ✗ BPE file NOT found!")
    # Try alternative paths
    alt_paths = [
        os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(os.getcwd(), "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            print(f"    ✓ Found at alternative path: {alt}")
            bpe_path = alt
            break

# Test 4: Initialize model
print("\n[4] Testing model initialization...")
try:
    # Configure CUDA settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("    Building SAM3 model (this may take a moment)...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    print(f"    ✓ Model built successfully")
    print(f"    ✓ Model type: {type(model)}")
except Exception as e:
    print(f"    ✗ Model build failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Initialize processor
print("\n[5] Testing processor initialization...")
try:
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print(f"    ✓ Processor created successfully")
    print(f"    ✓ Processor type: {type(processor)}")
except Exception as e:
    print(f"    ✗ Processor creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test with a dummy image
print("\n[6] Testing with a dummy image...")
try:
    # Create a simple test image
    test_image = Image.new('RGB', (640, 480), color='blue')
    print(f"    ✓ Created test image: {test_image.size}")
    
    # Set image
    inference_state = processor.set_image(test_image)
    print(f"    ✓ set_image() succeeded")
    print(f"    ✓ inference_state type: {type(inference_state)}")
    
    # Check what attributes inference_state has
    print(f"    ✓ inference_state attributes: {dir(inference_state)}")
    
except Exception as e:
    print(f"    ✗ Image processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test box processing
print("\n[7] Testing box processing...")
try:
    # Test box conversion
    box_xywh = torch.tensor([100.0, 100.0, 200.0, 150.0]).view(-1, 4)
    print(f"    ✓ box_xywh: {box_xywh}")
    
    box_cxcywh = box_xywh_to_cxcywh(box_xywh)
    print(f"    ✓ box_cxcywh: {box_cxcywh}")
    
    norm_box = normalize_bbox(box_cxcywh, 640, 480)
    print(f"    ✓ normalized box: {norm_box}")
    
except Exception as e:
    print(f"    ✗ Box processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test adding geometric prompt
print("\n[8] Testing geometric prompt...")
try:
    norm_box_list = norm_box.flatten().tolist()
    print(f"    norm_box_list: {norm_box_list}")
    
    processor.reset_all_prompts(inference_state)
    print(f"    ✓ reset_all_prompts() succeeded")
    
    inference_state = processor.add_geometric_prompt(
        state=inference_state,
        box=norm_box_list,
        label=True
    )
    print(f"    ✓ add_geometric_prompt() succeeded")
    
    # Check inference_state for masks
    print(f"    Checking for masks in inference_state...")
    if hasattr(inference_state, 'masks'):
        print(f"    ✓ Has 'masks' attribute: {inference_state.masks}")
    else:
        print(f"    ✗ No 'masks' attribute")
        
    # List all attributes that might contain masks
    for attr in dir(inference_state):
        if not attr.startswith('_'):
            try:
                val = getattr(inference_state, attr)
                if hasattr(val, 'shape') or (isinstance(val, (list, tuple)) and len(val) > 0):
                    print(f"    Attribute '{attr}': {type(val)}")
            except:
                pass
                
except Exception as e:
    print(f"    ✗ Geometric prompt failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Check how to get segmentation mask
print("\n[9] Testing mask extraction...")
try:
    # Try to find how to get the mask from the processor
    print(f"    Processor methods: {[m for m in dir(processor) if not m.startswith('_')]}")
    
    # Check if there's a predict or segment method
    if hasattr(processor, 'predict'):
        print(f"    ✓ Has 'predict' method")
    if hasattr(processor, 'segment'):
        print(f"    ✓ Has 'segment' method")
    if hasattr(processor, 'get_mask'):
        print(f"    ✓ Has 'get_mask' method")
    if hasattr(processor, 'inference'):
        print(f"    ✓ Has 'inference' method")
        
except Exception as e:
    print(f"    ✗ Mask extraction check failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All basic tests completed!")
print("=" * 60)
