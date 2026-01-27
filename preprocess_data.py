# preprocess_data.py

import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import argparse

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help="Path to raw imagenette folder")
parser.add_argument('--output_dir', type=str, required=True, help="Base path for processed data")
parser.add_argument('--mode', type=str, choices=['edges', 'segmentation'], required=True)
args = parser.parse_args()

# SETUP DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running mode: {args.mode} on {device}")

# MODEL SETUP (Only for segmentation)
seg_model = None
seg_transform = None
if args.mode == 'segmentation':
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    seg_model = deeplabv3_resnet50(weights=weights).to(device)
    seg_model.eval()
    seg_transform = weights.transforms()

def process_edges(img_path, save_path):
    # Read Image
    img = cv2.imread(img_path)
    if img is None: return

    # Canny Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Stack to make 3-channel (so standard CNNs accept it)
    edges_rgb = np.stack([edges]*3, axis=-1)

    cv2.imwrite(save_path, edges_rgb)

def process_segmentation(img_path, save_path):
    # Load and Transform
    try:
        img_pil = Image.open(img_path).convert("RGB")
        original_np = np.array(img_pil)
    except:
        return

    # Model scales image up to ~520px internally
    input_tensor = seg_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(input_tensor)['out'][0]

    # Generate Mask (Class 0 is background)
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Resize mask back to original image size
    # cv2.resize expects (Width, Height), but numpy shape is (Height, Width)
    h, w = original_np.shape[:2]
    mask_resized = cv2.resize(output_predictions, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create Binary Mask (0=Background, 1=Object)
    mask = (mask_resized > 0).astype(np.uint8)

    # Apply Mask to Original Image
    mask_3ch = np.stack([mask]*3, axis=-1)
    foreground = original_np * mask_3ch

    # Convert back to BGR for OpenCV saving
    foreground_bgr = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, foreground_bgr)

# MAIN LOOP
splits = ['train', 'val']

for split in splits:
    split_path = os.path.join(args.input_dir, split)
    if not os.path.exists(split_path): continue

    # Iterate over classes
    classes = os.listdir(split_path)
    for cls in classes:
        cls_dir = os.path.join(split_path, cls)
        if not os.path.isdir(cls_dir): continue

        # Create Output Directory
        out_cls_dir = os.path.join(args.output_dir, args.mode, split, cls)
        os.makedirs(out_cls_dir, exist_ok=True)

        # Process Images
        images = os.listdir(cls_dir)
        print(f"Processing {split}/{cls}...")

        for img_name in images:
            src = os.path.join(cls_dir, img_name)
            dst = os.path.join(out_cls_dir, img_name)

            if args.mode == 'edges':
                process_edges(src, dst)
            elif args.mode == 'segmentation':
                process_segmentation(src, dst)

print("Processing Complete.")