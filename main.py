# main.py

import dataset_tools as dtools
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import models
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from tqdm import tqdm

# Visualization
from PIL import Image
import matplotlib.pyplot as plt

# Cross-Validation
from sklearn.model_selection import KFold

# import other python
from dataset import MineralImage5k

# import .env
from dotenv import load_dotenv, dotenv_values

load_dotenv()

def main():
    # Download datasets
    dst_dir = os.path.expanduser(os.getenv("DATASET_DIR"))

    dataset_path = os.path.join(dst_dir, 'mineralimage5k')
    if not os.path.exists(dataset_path):
        dtools.download(dataset='MineralImage5k', dst_dir=dst_dir)
    else:
        print("Dataset already cached!")

    # Split datasets
    dataset = MineralImage5k(root_dir=dataset_path, max_files=2000)
    print(len(dataset))

    # device init
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)

    if os.path.exists(os.getenv("MODEL_PTH")):
        model.load_state_dict(torch.load(os.getenv("MODEL_PTH")))
        print("successfully load the model!!!")

    model.to(device)

    # Tunning
    LEARNING_RATE = 7e-5
    WEIGHT_DECAY = 1e-6
    K_FOLDS = 10
    N_EPOCHS = 20
    BATCH_SIZE = 256

    # useful option init
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
            print(f"\nEpoch {epoch + 1}/{N_EPOCHS}")
            # Train
            model.train()
            train_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)

            for images, targets in train_loader_tqdm:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                train_loss += batch_loss
                train_loader_tqdm.set_postfix(loss=f"{batch_loss:.4f}")

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Train Loss: {avg_train_loss:.4f}")

            metric = MeanAveragePrecision()
            # Validation
            model.eval()
            val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
            with torch.no_grad():
                for images, targets in val_loader_tqdm:
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    outputs = model(images)
                    # Move to CPU for torchmetrics
                    outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
                    targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

                    # Update mAP metric
                    metric.update(outputs, targets)
            # Compute final result
            result = metric.compute()
            print(f"mAP: {result['map']:.4f}")

    # Test
    # Step 1: Initialize metric
    map_metric = MeanAveragePrecision()
    test_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("Begin Testing...")
    # Step 2: Set model to evaluation mode
    model.eval()

    # Step 3: Loop through test data
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            # Convert to CPU for metric calculation
            outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]

            # Step 4: Update mAP metric
            map_metric.update(outputs, targets)

    # Step 5: Compute final mAP scores
    results = map_metric.compute()

    # Step 6: Print results
    print(f"mAP: {result['map']:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'model_brain.pth')


if __name__ == "__main__":
    main()

