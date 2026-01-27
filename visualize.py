# visualize.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--test_mode', type=str, required=True)
parser.add_argument('--model_arch', type=str, default='resnet18')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Setup Data
def get_transforms(mode):
    trans = [transforms.Resize((224, 224))]
    if mode == 'grayscale': trans.append(transforms.Grayscale(num_output_channels=3))
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(trans)

#  Point directly to 'val' folder
if args.test_mode in ['original', 'grayscale', 'occlusion']:
    base_dir = os.path.join(args.data_root, 'raw', 'imagenette2-160')
else:
    base_dir = os.path.join(args.data_root, 'processed', args.test_mode)

data_dir = os.path.join(base_dir, 'val')

dataset = datasets.ImageFolder(data_dir, get_transforms(args.test_mode))
loader = DataLoader(dataset, batch_size=32, shuffle=False)

print(f"Loaded {len(dataset.classes)} classes from {data_dir}")

# 2. Load Model
if args.model_arch == 'resnet18': model = models.resnet18()
else: model = models.resnet101()
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# 3. Get Predictions
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 4. Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.classes, yticklabels=dataset.classes, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Confusion Matrix: {args.test_mode}')
plt.savefig(f'confusion_matrix_{args.test_mode}.png')
print(f"Saved confusion_matrix_{args.test_mode}.png")