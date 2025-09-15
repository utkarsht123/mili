import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import random
import yaml
import numpy as np

class AUAIR_RGB_Dataset(Dataset):
    """
    Final corrected loader for AU-AIR.
    This version fixes the 'config not defined' scope error.
    """
    def __init__(self, config, split='train'):
        # --- FIX #1: Make config accessible to the whole class ---
        self.config = config
        
        self.root_dir = self.config['data']['au_air_path']
        self.img_size = tuple(self.config['data']['img_size'])
        self.class_map = {name.lower(): i for i, name in enumerate(self.config['data']['class_names'])}
        self.rgb_dir = os.path.join(self.root_dir, 'rgb')
        annotation_file_path = os.path.join(self.root_dir, 'annotations.json')

        with open(annotation_file_path, 'r') as f:
            self.all_image_data = json.load(f)['annotations']
            
        initial_count = len(self.all_image_data)
        self.all_image_data = [
            ann for ann in self.all_image_data 
            if os.path.exists(os.path.join(self.rgb_dir, ann['image_name']))
        ]
        if len(self.all_image_data) < initial_count:
            print(f"WARNING: Skipped {initial_count - len(self.all_image_data)} entries due to missing image files.")

        subset_percentage = self.config['data']['au_air_subset_percentage']
        subset_size = int(len(self.all_image_data) * subset_percentage)
        random.seed(42)
        random.shuffle(self.all_image_data)
        self.all_image_data = self.all_image_data[:subset_size]

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.all_image_data)

    def __getitem__(self, idx):
        img_data = self.all_image_data[idx]
        filename = img_data['image_name']
        img_w, img_h = img_data['image_width:'], img_data['image_height']
        img_path = os.path.join(self.rgb_dir, filename)
        image = self.transform(Image.open(img_path).convert("RGB"))
        labels, boxes = [], []
        for ann in img_data['bbox']:
            class_id = ann['class']
            
            # --- FIX #2: Access config via self.config ---
            if 0 <= class_id < len(self.config['data']['class_names']):
                class_name = self.config['data']['class_names'][class_id]
                if class_name.lower() in self.class_map:
                    labels.append(self.class_map[class_name.lower()])
                    top, left, h, w = ann['top'], ann['left'], ann['height'], ann['width']
                    x1, y1, x2, y2 = left, top, left + w, top + h
                    boxes.append([(x1 + x2)/2/img_w, (y1 + y2)/2/img_h, w/img_w, h/img_h])

        target = {'labels': torch.tensor(labels, dtype=torch.int64), 'boxes': torch.tensor(boxes, dtype=torch.float32)}
        return image, target

# ... The HITUAV_Thermal_YOLO_Dataset class remains unchanged and correct ...
class HITUAV_Thermal_YOLO_Dataset(Dataset):
    def __init__(self, config, split='train'):
        self.root_dir = config['data']['hit_uav_path']
        self.img_size = tuple(config['data']['img_size'])
        with open(os.path.join(self.root_dir, 'dataset.yaml'), 'r') as f:
            yolo_config = yaml.safe_load(f)
        yolo_class_names = yolo_config['names']
        final_class_names = [name.lower() for name in config['data']['class_names']]
        self.yolo_to_model_class_map = {
            yolo_id: final_class_names.index(name.lower())
            for yolo_id, name in yolo_class_names.items()
            if name.lower() in final_class_names
        }
        self.image_dir = os.path.join(self.root_dir, 'images', split)
        self.label_dir = os.path.join(self.root_dir, 'labels', split)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        image = self.transform(Image.open(img_path).convert("RGB"))
        labels, boxes = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    yolo_id = int(parts[0])
                    if yolo_id in self.yolo_to_model_class_map:
                        labels.append(self.yolo_to_model_class_map[yolo_id])
                        boxes.append([float(p) for p in parts[1:]])
        target = {'labels': torch.tensor(labels, dtype=torch.int64), 'boxes': torch.tensor(boxes, dtype=torch.float32)}
        return image, target