#  Shortcut Learning Analysis in CNNs

> **Investigating the fragility of Deep Learning models: Do they see "Shapes" or just "Textures"?**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

##  Overview
State-of-the-art CNNs often achieve superhuman accuracy on benchmarks like ImageNet, yet they can fail catastrophically in real-world scenarios. This project investigates **"Shortcut Learning,"** revealing that standard models (like ResNet) function as **texture-matchers** rather than shape-recognizers.

Using the **Imagenette** dataset, this project quantifies the "Degradation Gap" that occurs when visual shortcuts (color, texture, background) are removed from an image.

## Experimental Design
We compared **ResNet-18** vs. **ResNet-101** across three distinct data modalities:
1.  **Original (RGB):** Standard baseline.
2.  **Edges (Shape-Only):** Texture removed using Canny Edge Detection.
3.  **Segmentation (Object-Only):** Background removed using DeepLabV3.

## Key Results
Our experiments revealed a massive reliance on texture. When texture was removed, the model's "intelligence" collapsed, proving it was relying on superficial statistics.

| Model | Train Data | Test Data | Accuracy | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** | Original | **Original** | **98.27%** | Near perfect baseline. |
| **ResNet-18** | Original | **Edges** | **26.98%** | **~71% Performance Crash.** |
| **ResNet-18** | Edges | Edges | **87.57%** | Control exp: Shape learning *is* possible. |

> **Conclusion:** Increasing model capacity (ResNet-101) provided **no significant improvement** in robustness, suggesting that architectural depth alone does not solve shortcut learning.

## 📂 Repository Structure
```bash
├── src/
│   ├── train.py             # Main training loop (ResNet18/101)
│   ├── preprocess_data.py   # Canny Edge & DeepLabV3 pipeline
│   ├── visualize.py         # Confusion Matrix generation
│   ├── analyze_failures.py  # Visual failure case extraction
│   └── parse_results.py     # Log parsing tool
├── slurm_scripts/           # HPC Job submission scripts
└── README.md
