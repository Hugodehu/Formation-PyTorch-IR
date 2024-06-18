import torch
from function.compute_iou import compute_iou
from function.mAP import mean_average_precision

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
    
    precisions = []
    recalls = []
    precisionsTest = []
    recallsTest = []

    with torch.no_grad():
        for outputs, targets in zip(prediction, targetsOut):
            for target, output in zip(targets, outputs):
                true_positives = 0
                false_positives = 0
                false_negatives = 0
                target_boxes = target['boxes']
                pred_boxes = output['boxes']
                scores = output['scores']

                # Filter out low-confidence boxes
                pred_boxes = pred_boxes[scores >= threshold]
                
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
                
                for i in range(len(target_boxes)):
                    max_iou, _ = iou_matrix[i].max(0)
                    total_iou += max_iou.item()
                    if max_iou.item() >= iou_threshold:
                        true_positives_total += 1
                        true_positives += 1

                    else:
                        false_negatives_total += 1
                        false_negatives += 1

                for j in range(len(pred_boxes)):
                    max_iou, _ = iou_matrix[:, j].max(0)
                    if max_iou.item() < iou_threshold:
                        false_positives_total += 1
                        false_positives += 1

                # # Calculate precision and recall for this image
                # precision = true_positives / (true_positives + false_positives)
                # recall = true_positives / (true_positives + false_negatives)
                # precisions.append(precision)
                # recalls.append(recall)

                # precision = true_positives_total / (true_positives_total + false_positives_total) 
                # recall = true_positives_total / (true_positives_total + false_negatives_total) 
                # precisionsTest.append(precision)
                # recallsTest.append(recall)
            # mAP, AP =  mean_average_precision(outputs, targets)
            # print(f"mAP: {mAP:.4f}, AP: {AP}")

                
    precision = true_positives_total / (true_positives_total + false_positives_total) # représente la proportion de prédictions correctes parmi les prédictions positives
    recall = true_positives_total / (true_positives_total + false_negatives_total) # représente la proportion de prédictions correctes parmi les vrais positifs
    
    print(f"Nombre de boîtes de l'image optimal: {Target_num_boxes}")
    print(f"Nombre de boîtes de l'image prédit: {num_boxes}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}\n")
    return precision, recall, Target_num_boxes, num_boxes
