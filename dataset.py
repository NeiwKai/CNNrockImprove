# dataset.py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os, json
from PIL import Image

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

def find_all_ann(root_dir, max_files=None):
    ann_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        # Skip 'segm' directory
        dirs[:] = [d for d in dirs if d != 'segm']

        for file in files:
            if file.endswith(".json") and file != "meta.json":
                ann_paths.append(os.path.join(subdir, file))
                if max_files and len(ann_paths) >= max_files:
                    return ann_paths
    return ann_paths

class MineralImage5k(Dataset):
    def __init__(self, root_dir, max_files):
        self.ann_files = find_all_ann(root_dir, max_files)

        # Transform image to the same
        self.transforms = A.Compose([
            A.Resize(height=512, width=512),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        self.label_map = {'stone': 1, 'rock': 2, 'mineral': 3, 'gem': 4, 'crystal': 5, 'mineral ore': 6}

    def __len__(self):
        return len(self.ann_files)
            
    def __getitem__(self, idx):
        ann_path = self.ann_files[idx]
        with open(ann_path) as f:
            ann = json.load(f)

        img_path = ann_path.replace(f"{os.sep}ann{os.sep}", f"{os.sep}img{os.sep}").replace(".json", "")
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)  # Albumentations expects numpy array

        boxes = []
        labels = []

        for obj in ann.get("objects", []):
            # (same bbox extraction as before)
            (x_min, y_min), (x_max, y_max) = obj["points"]["exterior"]
            boxes.append([x_min, y_min, x_max, y_max])
            class_name = obj.get("classTitle")
            class_id = self.label_map.get(class_name, 0)
            labels.append(class_id)

        if len(boxes) == 0:
            boxes = []
            labels = []

        transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
        img = transformed['image']
        if img.dtype == torch.uint8:
            img = img.float() / 255.0

        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        labels = torch.tensor(transformed['labels'], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.shape[0] > 0 else torch.tensor([]),
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64)
        }

        return img, target



