# analyze_failures.py

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--test_mode', type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#  Point directly to 'val' folder
if args.test_mode in ['original', 'grayscale', 'occlusion']:
    base_dir = os.path.join(args.data_root, 'raw', 'imagenette2-160')
else:
    base_dir = os.path.join(args.data_root, 'processed', args.test_mode)

data_dir = os.path.join(base_dir, 'val')


dataset = datasets.ImageFolder(data_dir, trans)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Loaded {len(dataset.classes)} classes from {data_dir}")

# Load Model (ResNet18)
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# Find Failures
failures = []

print("Searching for failures...")
with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Find indices where predictions do NOT match labels
        wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]

        for idx in wrong_idx:
            if len(failures) < 5:
                # Save the failure: (Image Tensor, True Label, Pred Label)
                failures.append((inputs[idx].cpu(), labels[idx].item(), preds[idx].item()))

        if len(failures) >= 5:
            break

# Plot
print(f"Found {len(failures)} failures. Generating image...")
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
class_names = dataset.classes

for i, (img_tensor, true_idx, pred_idx) in enumerate(failures):
    # Un-normalize for display
    img = img_tensor.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    axes[i].imshow(img)
    axes[i].set_title(f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]}", color='red')
    axes[i].axis('off')

plt.suptitle(f'Failure Analysis: {args.test_mode}', fontsize=16)
plt.savefig('failure_examples.png')
print("Saved failure_examples.png")