from matplotlib import patches, pyplot as plt
import torch

def show_comparison_image_models(listModels, dataloader, targetsOut, ListLablesImage, device):
    with torch.no_grad():
        count = 0
        for models in listModels:
            for prediction in models:
                for output in prediction:
                    pred_boxes = output['boxes']
                    scores = output['scores']
                    # Filter out low-confidence boxes
                    pred_boxes = pred_boxes[scores >= 0.5]
                    scores = scores[scores >= 0.5]
                    output['boxes'] = pred_boxes
                    output['scores'] = scores

        for images, labels in dataloader:
            images = [img.to(device) for img in images]
            for idx, img in enumerate(images):
                listPred = []
                i = 0
                while i < len(listModels):
                    listPred.append(listModels[i][count][idx])
                    i += 1
                visualize_prediction(img, targetsOut[count][idx], listPred, ListLablesImage)
            count += 1


def visualize_prediction(image, true_boxes, pred_boxes, ListNameModels):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
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
        ax.title.set_text(model)
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
