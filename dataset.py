# dataset.py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os, json
from PIL import Image

def find_all_ann(root_dir):
    ann_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json") and file != "meta.json":
                ann_paths.append(os.path.join(subdir, file))
    return ann_paths

class MineralImage5k(Dataset):
    def __init__(self, root_dir):
        self.ann_files = find_all_ann(root_dir)

        # Transform image to the same
        self.transforms = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ])

        self.label_map = {'stone': 1, 'rock': 2, 'mineral': 3, 'gem': 4, 'crystal': 5, 'mineral ore': 6}

    def __len__(self):
        return len(self.ann_files)
        
    def __getitem__(self, idx):
        ann_path = self.ann_files[idx]
        with open(ann_path) as f:
            ann = json.load(f)

        img_path = ann_path.replace("/ann/", "/img/").replace(".json", "")
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for obj in ann.get("objects", []):
            if "points" not in obj or "exterior" not in obj["points"]:
                print(f"Skip no points")
                continue

            exterior = obj["points"]["exterior"]
            if not exterior or len(exterior) != 2:
                print(f"Skip no exterior")
                continue

            (x_min, y_min), (x_max, y_max) = exterior
            boxes.append([x_min, y_min, x_max, y_max])

            class_name = obj.get("classTitle", "unknown")
            class_id = self.label_map.get(class_name, 0)
            labels.append(class_id)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Compute area
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.shape[0] > 0 else torch.tensor([])

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


