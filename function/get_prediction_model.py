import torch
from function.visualize_prediction_image import visualize_prediction

def convert_coco_to_targets(coco_labels, device):
    targets = []
    for lables in coco_labels:
        boxes = []
        category_id = []
        for lbl in lables:
            xmin, ymin, width, height = lbl['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])  # Extract bounding box coordinates [x_min, y_min, width, height]
            category_id.append(lbl['category_id'])  # Extract category ID
        boxesTensor = torch.tensor(boxes, dtype=torch.float32)
        category_idTensor = torch.tensor(category_id, dtype=torch.int64)
        targets.append({'boxes': boxesTensor.to(device), 'labels': category_idTensor.to(device)})
    return targets

def getPredictionModel(model, dataloader, device):
    model.eval()
    prediction = []
    targetsOut = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = [img.to(device) for img in images]

            if(labels[0].__class__ == list):
                targets = convert_coco_to_targets(labels, device)
            else:
                targets = [{'boxes': lbl["boxes"].to(device), 'labels': lbl["labels"].to(device)} for lbl in labels]
            outputs = model(images)
            prediction.append(outputs)
            targetsOut.append(targets)
            # visualize_prediction(images,targets, outputs )
    return prediction, targetsOut