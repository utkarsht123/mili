# Lightweight-MultiModal-DETR: Setup Guide

This guide provides instructions to set up and run this project, which trains a single object detection model on a combination of the AU-AIR (RGB) and HIT-UAV (Thermal) datasets.

## 1. Project Setup

- **Clone the repository:** `git clone <your-repo-url>`
- **Install dependencies:** `pip install -r requirements.txt`

## 2. Dataset Organization

Place your datasets in the `datasets/` folder following the structure outlined in the project documentation. Ensure `au-air` contains `rgb/` and `annotations/` folders, and `hit-uav-yolo` follows the standard YOLO format (`images/`, `labels/`, `dataset.yaml`).

## 3. Configuration

Review `config.yaml` to ensure all paths and parameters are correct, especially the `class_names` list which must align with your datasets' labels.

## 4. Train the Model

Run the main training script. This will combine both datasets, train the model, and save the best checkpoint to `lightweight_multimodal_detr_best.pth`.

```bash
python main.py
```
