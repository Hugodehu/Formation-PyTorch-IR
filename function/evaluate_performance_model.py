import torch
from function.calculate_IoU import compute_iou
from torchmetrics.detection import MeanAveragePrecision

def evaluate_performance_model(prediction, targetsOut, iou_threshold=0.5, threshold=0.5):
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
    imageNumber = 1
    listoutputs = []
    listtargets = []
    with torch.no_grad():
        for outputs, targets in zip(prediction, targetsOut):
            for idx, (target, output) in enumerate(zip(targets, outputs)):
                true_positives = 0
                false_positives = 0
                false_negatives = 0
                target_boxes = target['boxes']
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
                for j in range(len(pred_boxes)):
                    max_iou, _ = iou_matrix[:, j].max(0)
                    if max_iou.item() >= iou_threshold:
                        true_positives_total += 1
                        true_positives += 1
                    else:
                        false_positives_total += 1
                        false_positives += 1

                for i in range(len(target_boxes)):
                    max_iou, _ = iou_matrix[i].max(0)
                    total_iou += max_iou.item()
                    if max_iou.item() < iou_threshold:
                        false_negatives_total += 1
                        false_negatives += 1

                listoutputs.append(output)
                listtargets.append(target)

                # accuracy = true_positives / (true_positives + false_positives + false_negatives) * 100
                # precisionImage = true_positives / (true_positives + false_positives)
                # recallImage = true_positives / (true_positives + false_negatives)
                # print(f"Image {imageNumber} : Accuracy: {accuracy:.2f}, Precision: {precisionImage:.4f}, Recall: {recallImage:.4f}, Number of target boxes: {len(target_boxes)}, Number of predicted boxes: {len(pred_boxes)}, TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}\n")
                # imageNumber += 1

            # metric = MeanAveragePrecision()
            # metric.update(outputs, targets)
            # mAP = metric.compute()
    metric = MeanAveragePrecision()
    metric.update(listoutputs, listtargets)
    mAP = metric.compute()  
    map = mAP['map'].item()*100 if mAP['map'] >=0 else 0       
    precision = true_positives_total / (true_positives_total + false_positives_total) if true_positives_total + false_positives_total > 0 else 0 # représente la proportion de prédictions correctes parmi les prédictions totales
    recall = true_positives_total / (true_positives_total + false_negatives_total) if true_positives_total + false_negatives_total > 0 else 0 # représente la proportion de prédictions correctes parmi les vrais positifs
    
    print(f"Nombre de boîtes de l'image optimal: {Target_num_boxes}")
    print(f"Nombre de boîtes de l'image prédit: {num_boxes}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"mAP: {map: .4f}\n")
    return precision, recall, Target_num_boxes, num_boxes, map
