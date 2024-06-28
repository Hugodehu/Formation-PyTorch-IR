import torch
from torchvision.ops import nms

from function.merge_prediction.regroup_predictions_filter_low_confidence_box import RegroupPredictionsFilterLowConfidenceBox


def merge_predictions_with_nms(predictionsModel1, predictionsModel2, iou_threshold=0.5, threshold=0.5):
    output = []
    for predictions, predictions2 in zip(predictionsModel1, predictionsModel2):
        merged_predictions = []
        for pred1, pred2 in zip(predictions, predictions2):
            pred1_boxes = pred1['boxes']
            pred1_scores = pred1['scores']
            pred1_labels = pred1['labels']

            pred1keep = nms(pred1_boxes, pred1_scores, iou_threshold)
            
            pred1_boxes = pred1_boxes[pred1keep]
            pred1_scores = pred1_scores[pred1keep]
            pred1_labels = pred1_labels[pred1keep]

            pred1Temp = {'boxes': pred1_boxes, 'labels': pred1_labels, 'scores': pred1_scores}

            pred2_boxes = pred2['boxes']
            pred2_scores = pred2['scores']
            pred2_labels = pred2['labels']

            pred2keep = nms(pred2_boxes, pred2_scores, iou_threshold)
            
            pred2_boxes = pred2_boxes[pred2keep]
            pred2_scores = pred2_scores[pred2keep]
            pred2_labels = pred2_labels[pred2keep]
            pred2Temp = {'boxes': pred2_boxes, 'labels': pred2_labels, 'scores': pred2_scores}

            combined_boxes, combined_scores, combined_labels = RegroupPredictionsFilterLowConfidenceBox(pred1Temp, pred2Temp, threshold)
            
            # Apply NMS to the combined boxes
            keep = nms(combined_boxes, combined_scores, iou_threshold)
            
            merged_boxes = combined_boxes[keep]
            merged_scores = combined_scores[keep]
            merged_labels = combined_labels[keep]
            
            merged_predictions.append({'boxes': merged_boxes, 'labels': merged_labels, 'scores': merged_scores})
        output.append(merged_predictions)
    return output