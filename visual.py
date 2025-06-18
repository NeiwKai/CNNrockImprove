import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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

