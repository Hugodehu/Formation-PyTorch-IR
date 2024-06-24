import numpy as np
import torch
import function.calculate_IoU as calculate_IoU
from function.merge_prediction.regroup_predictions_filter_low_confidence_box import RegroupPredictionsFilterLowConfidenceBox

def merge_predictions_with_wbf(predictionsModel1, predictionsModel2, IoU_threshold=0.5, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else"cpu"
    regrouped_predictions = regroup_predictions(predictionsModel1, predictionsModel2, threshold)
    fusionListList = []
    clusterListList = []
    clusterList = []

    for predictions in regrouped_predictions:
        fusionList = []
        clusterList = []
        
        for prediction in predictions:
            fusion = {'boxes': [], 'scores': [], 'labels': []}
            cluster = {'boxes': [], 'scores': [], 'labels': []}
            boxes = prediction['boxes']
            scores = prediction['scores']
            labels = prediction['labels']
            fusionboxes = fusion['boxes']
            for idx, box in enumerate(boxes):
                matched = False
                for idfusion, fusionbox in enumerate(fusionboxes):
                    iou = calculate_IoU.bb_iou_array(box, fusionbox)
                    if iou > IoU_threshold:
                        cluster["boxes"][idfusion].append(box.cpu().numpy())
                        cluster["scores"][idfusion].append(scores[idx].item())
                        cluster["labels"][idfusion].append(labels[idx].item())
                        matched = True

                        T = cluster["boxes"][idfusion]
                        S = cluster["scores"][idfusion]
                        new_score, new_box = update_box_values(S, T, device)

                        fusion['scores'][idfusion] = new_score
                        fusion['boxes'][idfusion] = new_box
                        break
                if not matched:
                    cluster['boxes'].append([box.cpu().numpy()])
                    cluster['scores'].append([scores[idx].item()])
                    cluster['labels'].append([labels[idx].item()])
                    fusion['scores'].append(scores[idx].item())
                    fusion['boxes'].append(box.cpu().numpy())
                    fusion['labels'].append(labels[idx].item())
            
            fusionList.append({'boxes': torch.tensor(fusion['boxes'], device=device), 'scores': torch.tensor(fusion['scores'], device=device), 'labels': torch.tensor(fusion['labels'], device=device)})
            clusterList.append(cluster)
        fusionListList.append(fusionList)
        clusterListList.append(clusterList)

        for idxFusions, fusions in enumerate(fusionListList):
            for idxFusion, fusion in enumerate(fusions):
                scores = fusion['scores']
                T = len(fusion["scores"])
                min_T_N = min(T, 2)
                for idx, score in enumerate(scores):
                    T = len(clusterListList[idxFusions][idxFusion]["boxes"][idx])
                    min_T_N = min(T, 2)                
                    # if method == 1:
                    score = score * (min_T_N / 2)
                    # elif method == 2:
                    #     F[idx]['score'] = F[idx]['score'] * (T / N)
    
    return fusionListList
            

def regroup_predictions(predictionsModel1, predictionsModel2, threshold):
    output = []
    for predictions1, predictions2 in zip(predictionsModel1, predictionsModel2):
        merged_predictions = []
        for pred1, pred2 in zip(predictions1, predictions2):
            combined_boxes, combined_scores, combined_labels = RegroupPredictionsFilterLowConfidenceBox(pred1, pred2, threshold)

            merged_predictions.append({'boxes': combined_boxes, 'labels': combined_labels, 'scores': combined_scores})
        output.append(merged_predictions)
    return output


def update_box_values(S, T, device):
    """
    Update box values based on accumulated boxes in T.
    """
    boxes_tensors = [torch.tensor(box, device=device) for box in T]
    T_boxes =  torch.stack(boxes_tensors)
    T_scores = torch.tensor(S, device=device)
    
    new_score = T_scores.mean().item()
    new_x1 = (T_scores * T_boxes[:, 0]).sum().item() / T_scores.sum().item()
    new_y1 = (T_scores * T_boxes[:, 1]).sum().item() / T_scores.sum().item()
    new_x2 = (T_scores * T_boxes[:, 2]).sum().item() / T_scores.sum().item()
    new_y2 = (T_scores * T_boxes[:, 3]).sum().item() / T_scores.sum().item()
    
    return new_score, [new_x1, new_y1, new_x2, new_y2]
