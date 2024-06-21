from matplotlib import patches, pyplot as plt

def visualize_prediction(images, true_boxes, pred_boxes):
    for indx, image in enumerate(images):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        
        # Draw true boxes in green
        for i, box in enumerate(true_boxes[indx]["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            # Add class label
            label = true_boxes[indx]["labels"][i].item()
            ax.text(x1, y1, str(label), verticalalignment='top', color='green', fontsize=12, weight='bold')
        
        # Draw predicted boxes in red
        for i, box in enumerate(pred_boxes[indx]["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Add class label
            label = pred_boxes[indx]["labels"][i].item()
            ax.text(x1, y1, str(label), verticalalignment='top', color='red', fontsize=12, weight='bold')
        
        plt.title("Predictions vs Ground Truth")
        plt.axis('off')
        plt.show()

def plot_precision_recall_curve(precisions, recalls):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()
