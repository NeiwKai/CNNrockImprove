# main.py

import dataset_tools as dtools
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import models
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

# Visualization
from PIL import Image
import matplotlib.pyplot as plt

# Cross-Validation
from sklearn.model_selection import KFold

# import other python
from core import compute_f1_score, train, val
from dataset import MineralImage5k

def main():
    # Download datasets
    dst_dir = os.path.expanduser('~/.cache/dataset-ninja')

    dataset_path = os.path.join(dst_dir, 'mineralimage5k')
    if not os.path.exists(dataset_path):
        dtools.download(dataset='MineralImage5k', dst_dir=dst_dir)
    else:
        print("Dataset already cached!")

    # Split datasets
    dataset = MineralImage5k(root_dir=dataset_path)
    print(len(dataset))

    # device init
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    model.to(device)

    # Tunning
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-6
    K_FOLDS = 5
    N_EPOCHS = 20
    BATCH_SIZE = 1

    # useful option init
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_metrics = []
    val_metrics = []

    k_fold = KFold(n_splits=K_FOLDS, shuffle=True)
    results = {}

    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, targets


    for fold, (train_idx, val_idx) in enumerate(k_fold.split(dataset)):
        print(f"\nFold {fold + 1}/{K_FOLDS}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        for epoch in range(N_EPOCHS):
            model.train()
            for images, targets in train_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                loss.backward()
                optimizer.step()

            # Validation loop here (similar)
            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    output = model(images)

if __name__ == "__main__":
    main()

