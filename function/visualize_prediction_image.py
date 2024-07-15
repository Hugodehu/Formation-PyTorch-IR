from matplotlib import patches, pyplot as plt
import torch

def show_comparison_image_models(listModels, dataloader, targetsOut, ListLablesImage, device, threshold=0.5):
    with torch.no_grad():
        for models in listModels:
            for prediction in models:
                for output in prediction:
                    pred_boxes = output['boxes']
                    scores = output['scores']
                    # Filter out low-confidence boxes
                    pred_boxes = pred_boxes[scores >= threshold]
                    scores = scores[scores >=  threshold]
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
                visualize_prediction(img, targetsOut[count][idx], listPred, ListLablesImage, len(listModels))
            count += 1


def show_comparison_image_models_for_map_inferior(listModels, dataloader, ListtargetsOut, ListLablesImage, device, threshold=0.5, listIndexImagesWithMapInferiorForWBF=[]):
    with torch.no_grad():
        count = 0
        for models in listModels:
            for prediction in models:
                for output in prediction:
                    pred_boxes = output['boxes']
                    scores = output['scores']
                    # Filter out low-confidence boxes
                    pred_boxes = pred_boxes[scores >= threshold]
                    scores = scores[scores >=  threshold]
                    output['boxes'] = pred_boxes
                    output['scores'] = scores

        for images, labels in dataloader:
            images = [img.to(device) for img in images]
            if(count in listIndexImagesWithMapInferiorForWBF):
                for idx, img in enumerate(images):
                    listPred = []
                    i = 0
                    while i < len(listModels):
                        listPred.append(listModels[i][listIndexImagesWithMapInferiorForWBF.index(count)][idx])
                        i += 1
                    visualize_prediction(img, ListtargetsOut, listPred, ListLablesImage, len(listModels), imageidx= idx, count=listIndexImagesWithMapInferiorForWBF.index(count))
            count += 1

def visualize_prediction(image, true_boxes, pred_boxes, ListNameModels, num_models, imageidx, count):
    fig, axes = plt.subplots(1, num_models, figsize=(20, 10))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for idx, (boxes, ax, model) in enumerate(zip(pred_boxes, axes, ListNameModels)):
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            
        # Draw true boxes in green
        for i, box in enumerate(true_boxes[idx][count][imageidx]["boxes"]):
            height, width = image.shape[1:]
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            # Add class label
            label = true_boxes[idx][count][imageidx]["labels"][i].item()
            ax.text(x1, y1, f"{label}", verticalalignment='top', color='green', fontsize=12, weight='bold')
        
        # Draw predicted boxes in red
            
        for i, box in enumerate(boxes["boxes"]):
            height, width = image.shape[1:]
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Add class label
            label = boxes["labels"][i].item()
            conf = boxes["scores"][i].item()
            ax.text(x1, y1, f"{label}; {conf:00.0000}", verticalalignment='top', color='red', fontsize=12, weight='bold')
        
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
