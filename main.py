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
import matplotlib.patches as patches
import numpy as np

# Cross-Validation
from sklearn.model_selection import KFold

# import other python
from dataset import MineralImage5k


def plot_training_curves(train_losses, val_maps, fold_num):
    """Plot training loss and validation mAP curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss - Fold {fold_num}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot validation mAP
    ax2.plot(val_maps, 'r-', label='Validation mAP', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title(f'Validation mAP - Fold {fold_num}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_curves_fold_{fold_num}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cross_validation_results(all_fold_results):
    """Plot results across all folds"""
    fold_numbers = list(range(1, len(all_fold_results) + 1))
    final_maps = [results['final_map'] for results in all_fold_results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(fold_numbers, final_maps, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.axhline(y=np.mean(final_maps), color='red', linestyle='--', 
                label=f'Mean mAP: {np.mean(final_maps):.4f}')
    
    # Add value labels on bars
    for bar, map_val in zip(bars, final_maps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{map_val:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Fold')
    plt.ylabel('Final mAP')
    plt.title('Cross-Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, dataset, device, num_samples=6, confidence_threshold=0.5):
    """Visualize model predictions on sample images"""
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    class_names = ['background', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6']  # Update with your actual class names
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, target = dataset[idx]
            
            # Get prediction
            image_tensor = image.unsqueeze(0).to(device)
            prediction = model(image_tensor)[0]
            
            # Convert image to numpy for visualization
            if image.shape[0] == 3:  # RGB
                img_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = image.squeeze().cpu().numpy()
                img_np = np.stack([img_np] * 3, axis=-1)  # Convert to RGB
            
            # Denormalize if needed (adjust based on your preprocessing)
            img_np = np.clip(img_np, 0, 1)
            
            ax = axes[i]
            ax.imshow(img_np)
            
            # Draw ground truth boxes (green)
            if 'boxes' in target:
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                for box, label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor='green', 
                                           facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'GT: {class_names[label]}', 
                           color='green', fontsize=8, weight='bold')
            
            # Draw predictions (red)
            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_labels = prediction['labels'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()
            
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor=colors[label % len(colors)], 
                                           facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y2+15, f'{class_names[label]}: {score:.2f}', 
                           color=colors[label % len(colors)], fontsize=8, weight='bold')
            
            ax.set_title(f'Sample {idx}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_loss_components(loss_history):
    """Plot different components of the loss if available"""
    if not loss_history:
        return
    
    # Extract loss components
    epochs = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # If you track individual loss components, plot them separately
    total_losses = [sum(loss_dict.values()) for loss_dict in loss_history]
    plt.plot(epochs, total_losses, 'b-', linewidth=2, label='Total Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_components.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Download datasets
    dst_dir = os.path.expanduser('D:/Work/ICT_MU/work/Data_Science/dataset-ninja')

    dataset_path = os.path.join(dst_dir, 'mineralimage5k')
    if not os.path.exists(dataset_path):
        dtools.download(dataset='MineralImage5k', dst_dir=dst_dir)
    else:
        print("Dataset already cached!")

    # Split datasets
    dataset = MineralImage5k(root_dir=dataset_path, max_files=500)
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
    model.to(device)

    # Tunning
    LEARNING_RATE = 8e-5 #default 7e-5
    WEIGHT_DECAY = 2e-6 # default 1e-6
    K_FOLDS = 2 #default 5
    N_EPOCHS = 5 # default 20
    BATCH_SIZE = 64 # default 64

    # useful option init
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_metrics = []
    val_metrics = []

    all_fold_results = []
    all_train_losses = []
    all_mAPs_scores = []
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

            all_train_losses.append(avg_train_loss)
            # Store training metrics

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
            all_mAPs_scores.append(result['map'])
            all_fold_results.append({
                'train_losses': all_train_losses[-N_EPOCHS:],  # Last N_EPOCHS entries
                'val_maps': all_mAPs_scores[-N_EPOCHS:],       # Last N_EPOCHS entries
                'final_map': all_mAPs_scores[-1].item() if hasattr(all_mAPs_scores[-1], 'item') else all_mAPs_scores[-1]
                })
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
    
    
    # Step 7: Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, dataset, device, num_samples=6, confidence_threshold=0.5)
    
    plt.figure(figsize=(10, 6))

     # Plot 1: Training curves for all folds
    plt.subplot(2, 2, 1)
    for i, fold_result in enumerate(all_fold_results):
        plt.plot(fold_result['train_losses'], label=f'Fold {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss - All Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Validation mAP for all folds
    plt.subplot(2, 2, 2)
    for i, fold_result in enumerate(all_fold_results):
        plt.plot(fold_result['val_maps'], label=f'Fold {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Validation mAP')
    plt.title('Validation mAP - All Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final mAP per fold
    plt.subplot(2, 2, 3)
    fold_nums = list(range(1, len(all_mAPs_scores) + 1))
    final_maps = [r.item() if hasattr(r, 'item') else r for r in all_mAPs_scores]
    plt.bar(fold_nums, final_maps, alpha=0.7, color='skyblue')
    plt.xlabel('Fold')
    plt.ylabel('Final mAP')
    plt.title('Final mAP per Fold')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f"Mean mAP: {np.mean(final_maps):.4f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"Std mAP: {np.std(final_maps):.4f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Final Test mAP: {results['map']:.4f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Number of Folds: {K_FOLDS}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Epochs per Fold: {N_EPOCHS}", fontsize=12, transform=plt.gca().transAxes)
    plt.title('Training Summary')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    best_val_loss = min(all_train_losses)
    best_mAP_score = max([r.item() if hasattr(r, 'item') else r for r in all_mAPs_scores])
    
    # Save the model state dictionary
    torch.save({'model_state_dict': model.state_dict(), 'train_loss': best_val_loss, 'mAP_score': best_mAP_score}, "D:\Work\LLM_project\CNNrockImprove\CNNrockImprove\models_saved\model.pth")
    print("Model Saved")

if __name__ == "__main__":
    main()

