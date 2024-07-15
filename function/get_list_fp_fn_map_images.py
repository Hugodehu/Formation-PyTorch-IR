from matplotlib import pyplot as plt
import torch
from function.calculate_IoU import bb_iou_array, compute_iou
from torchmetrics.detection import MeanAveragePrecision
import copy

def getListFPFNMapImages(prediction, targetsOut, iou_threshold=0.5, threshold=0.001, goodPrediction=False):
    """
    Evaluates the performance of a model by calculating precision and recall metrics.

    Args:
        prediction (list): List of predicted outputs from the model.
        targetsOut (list): List of target outputs.
        iou_threshold (float, optional): IoU threshold for considering a prediction as a true positive. Defaults to 0.5.
        threshold (float, optional): Confidence threshold for filtering out low-confidence predictions. Defaults to 0.5.

    Returns:
        tuple: A tuple containing precision, recall, number of target boxes, and number of predicted boxes.
    """
    true_positives_total = 0
    false_positives_total = 0
    false_negatives_total = 0
    total_iou = 0
    Target_num_boxes = 0
    num_boxes = 0
    List_Map = []
    with torch.no_grad():
        temp_outputs = []
        temp_outputs = copy.deepcopy(prediction)
        temp_targets = []
        temp_targets = copy.deepcopy(targetsOut)
        for id,(outputs, targets) in enumerate(zip(temp_outputs, temp_targets)):

            for idx, (target, output) in enumerate(zip(targets, outputs)):
                ImageMap = MeanAveragePrecision()
                ImageMap.update([output], [target])
                map = ImageMap.compute()
                List_Map.append(map)
                true_positives = 0
                false_positives = 0
                false_negatives = 0
                target_boxes = target['boxes']
                target_labels = target['labels']
                pred_boxes = output['boxes']
                scores = output['scores']
                # Filter out low-confidence boxes
                pred_boxes = pred_boxes[scores >= threshold]
                labels = output['labels'][scores >= threshold]
                scores = scores[scores >= threshold]

                output['boxes'] = pred_boxes
                output['scores'] = scores
                output['labels'] = labels

                num_boxes += len(pred_boxes)
                Target_num_boxes += len(target_boxes)

                if len(pred_boxes) == 0:
                    false_negatives_total += len(target_boxes)
                    false_negatives += len(target_boxes)
                    continue
                
                if(len(target_boxes) == 0):
                    false_positives_total += len(pred_boxes)
                    false_positives += len(pred_boxes)
                    continue
                

                iou_matrix = compute_iou(target_boxes, pred_boxes)
                test = bb_iou_array(torch.tensor([0.2766, 0.3702, 0.3482, 0.4539]), torch.tensor([0.2115, 0.4003, 0.2418, 0.4194]))
                test = bb_iou_array(torch.tensor([0.2766, 0.3702, 0.3482, 0.4539]), torch.tensor([0.2476, 0.4021, 0.2573, 0.4263]))
                pred_boxes_to_keep = torch.ones(len(pred_boxes), dtype=torch.bool)
                for j in range(len(pred_boxes)):
                    max_iou, _ = iou_matrix[:, j].max(0)
                    max_iou_idx = iou_matrix[:, j].argmax(0)
                    if max_iou.item() >= iou_threshold and target_labels[max_iou_idx] == labels[j]:
                        true_positives_total += 1
                        true_positives += 1
                        if not goodPrediction:
                            pred_boxes_to_keep[j] = False
                    else:
                        false_positives_total += 1
                        false_positives += 1
                        if goodPrediction:
                            pred_boxes_to_keep[j] = False
                
                
                output["boxes"] = pred_boxes[pred_boxes_to_keep]
                output["labels"] = labels[pred_boxes_to_keep]
                output["scores"] = scores[pred_boxes_to_keep]


                target_boxes_to_keep = torch.ones(len(target_boxes), dtype=torch.bool)
                for i in range(len(target_boxes)):
                    max_iou, _ = iou_matrix[i].max(0)
                    total_iou += max_iou.item()
                    if max_iou.item() < iou_threshold:
                        false_negatives_total += 1
                        false_negatives += 1
                    else:
                        if not goodPrediction:
                            target_boxes_to_keep[i] = False
                
                target["boxes"] = target_boxes[target_boxes_to_keep]
                target["labels"] = target["labels"][target_boxes_to_keep]

    
    return List_Map, temp_outputs, temp_targets


def plot_precision_recall_curve(mAPClasses, classes):
    """
    Trace la courbe de précision en fonction du rappel.

    Args:
        precisions (list): Liste des précisions.
        recalls (list): Liste des rappels.
    """
    mAPClassesArray = mAPClasses.cpu().numpy()
    classesArray = classes.cpu().numpy()
    # Traçage de la courbe de précision-rappel
    plt.figure(figsize=(10, 6))
    plt.scatter(classesArray, mAPClassesArray, color='b', marker='o')

    # Ajout des labels et du titre
    plt.xlabel('Classes')
    plt.ylabel('mAP')
    plt.title('Nuage de Points de la mAP par Classe')
    plt.grid(True)

    # Optionnel : Ajout des labels pour chaque point
    for i, txt in enumerate(mAPClassesArray):
        plt.annotate(f'{txt:.2f}', (classes[i], mAPClassesArray[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()