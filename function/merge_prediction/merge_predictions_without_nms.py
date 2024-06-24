import torch

from function.merge_prediction.regroup_predictions_filter_low_confidence_box import RegroupPredictionsFilterLowConfidenceBox

def merge_predictions_without_nms(predictionsModel1, predictionsModel2, threshold=0.5):
    output = []
    for predictions1, predictions2 in zip(predictionsModel1, predictionsModel2):
        merged_predictions = []
        for pred1, pred2 in zip(predictions1, predictions2):
            combined_boxes, combined_scores, combined_labels = RegroupPredictionsFilterLowConfidenceBox(pred1, pred2, threshold)
            
            merged_predictions.append({'boxes': combined_boxes, 'labels': combined_labels, 'scores': combined_scores})
        output.append(merged_predictions)
    return output
