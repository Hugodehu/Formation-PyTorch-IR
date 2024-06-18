from matplotlib import patches, pyplot as plt


def visualize_prediction(image, true_boxes, pred_boxes):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image[0].permute(1, 2, 0).cpu().numpy())
    
    # Draw true boxes in green
    for box in true_boxes[0]["boxes"]:
        x1, y1, x2, y2 = box.cpu()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    
    # Draw predicted boxes in red
    for box in pred_boxes[0]["boxes"]:
        x1, y1, x2, y2 = box.cpu()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.title("test")
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