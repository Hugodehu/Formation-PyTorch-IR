from matplotlib import patches, pyplot as plt

def visualize_prediction(image, true_boxes, pred_boxes, ListNameModels):
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for boxes, ax, model in zip(pred_boxes, axes, ListNameModels):
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            
        # Draw true boxes in green
        for i, box in enumerate(true_boxes["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            # Add class label
            label = true_boxes["labels"][i].item()
            ax.text(x1, y1, str(label), verticalalignment='top', color='green', fontsize=12, weight='bold')
        
        # Draw predicted boxes in red
            
        for i, box in enumerate(boxes["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Add class label
            label = boxes["labels"][i].item()
            ax.text(x1, y1, str(label), verticalalignment='top', color='red', fontsize=12, weight='bold')
        
        # Add model name in the top right corner
        ax.text(
            0.95, 0.05, model, 
            verticalalignment='top', horizontalalignment='right', 
            transform=ax.transAxes,
            color='white', fontsize=14, weight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3')
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(precisions, recalls):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()
