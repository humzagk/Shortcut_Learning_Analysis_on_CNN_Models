# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import argparse
import copy
import sys

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True, help="Root of dataset")
parser.add_argument('--model', type=str, choices=['resnet18', 'resnet101'], default='resnet18')
parser.add_argument('--train_mode', type=str, choices=['original', 'edges', 'segmentation', 'grayscale', 'occlusion'], required=True)
parser.add_argument('--test_mode', type=str, choices=['original', 'edges', 'segmentation', 'grayscale', 'occlusion'], required=True)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--output_dir', type=str, default='./results')
args = parser.parse_args()

# SETUP DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Exp: Train on {args.train_mode} -> Test on {args.test_mode} | Model: {args.model}")

# 1. DEFINE TRANSFORMS (Dynamic Augmentation)
def get_transforms(mode, is_training=False):
    trans_list = [transforms.Resize((224, 224))]

    # Mode-Specific Pre-processing
    if mode == 'grayscale':
        trans_list.append(transforms.Grayscale(num_output_channels=3))

    # Standard Tensors
    trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    # Occlusion (Cutout) - Applied after Normalization
    if mode == 'occlusion':
        # Probability 1.0 means we ALWAYS occlude something (since we want to test robustness)
        trans_list.append(transforms.RandomErasing(p=1.0, scale=(0.02, 0.2)))

    return transforms.Compose(trans_list)

# 2. DATA LOADER SETUP
def get_data_dir(root, mode):
    # Base paths
    if mode == 'original':
        base = os.path.join(root, 'raw', 'imagenette2-160')
    elif mode in ['edges', 'segmentation']:
        base = os.path.join(root, 'processed', mode)
    elif mode in ['grayscale', 'occlusion']:
        base = os.path.join(root, 'raw', 'imagenette2-160')
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return base

# Train Loader (Append '/train')
train_dir = os.path.join(get_data_dir(args.data_root, args.train_mode), 'train')
train_dataset = datasets.ImageFolder(train_dir, get_transforms(args.train_mode, is_training=True))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Test Loader (Append '/val')
test_dir = os.path.join(get_data_dir(args.data_root, args.test_mode), 'val')
test_dataset = datasets.ImageFolder(test_dir, get_transforms(args.test_mode, is_training=False))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Data Loaded. Classes: {num_classes}")

# 3. MODEL SETUP
if args.model == 'resnet18':
    model = models.resnet18(weights='IMAGENET1K_V1')
elif args.model == 'resnet101':
    model = models.resnet101(weights='IMAGENET1K_V1')

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 4. TRAINING LOOP
best_acc = 0.0

for epoch in range(args.epochs):
    print(f'Epoch {epoch+1}/{args.epochs}')

    # Train Phase
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    scheduler.step()

    # Test Phase
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Train Loss: {running_loss/len(train_dataset):.4f} | Test Acc: {epoch_acc:.4f} | Test F1: {epoch_f1:.4f}")

    # Save Best
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"{args.model}_Tr-{args.train_mode}_Te-{args.test_mode}.pth")
        torch.save(model.state_dict(), save_path)

print(f"Best Accuracy: {best_acc:.4f}")