import warnings
import numpy as np
import torch
from function.merge_prediction.regroup_predictions_filter_low_confidence_box import RegroupPredictionsFilterLowConfidenceBox

import function.calculate_IoU as calculate_IoU

def merge_predictions_with_wbf(predictionsModel1, predictionsModel2, IoU_threshold=0.5, threshold=0.5, number_of_models=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    regrouped_predictions = regroup_predictions(predictionsModel1, predictionsModel2, threshold)
    fusionListList = []
    clusterListList = []

    for predictions in regrouped_predictions:
        fusionList = []
        clusterList = []

        for prediction in predictions:
            fusion = {'boxes': [], 'scores': [], 'labels': []}
            cluster = {'boxes': [], 'scores': [], 'labels': []}
            predBoxes = prediction['boxes']
            predScores = prediction['scores']
            predLabels = prediction['labels']
            predScores_inx = torch.argsort(predScores, descending=True)
            predBoxes = predBoxes[predScores_inx]
            predScores = predScores[predScores_inx]
            predLabels = predLabels[predScores_inx]
            fusionboxes = fusion['boxes']

            for idx, predBox in enumerate(predBoxes):
                matched = False

                for idfusion, fusionbox in enumerate(fusionboxes):
                    iou = calculate_IoU.bb_iou_array(predBox, fusionbox)
                    sameLabel = predLabels[idx] == fusion['labels'][idfusion]

                    if iou > IoU_threshold and sameLabel.item():
                        cluster["boxes"][idfusion].append(predBox.cpu().numpy())
                        cluster["scores"][idfusion].append(predScores[idx].item())
                        cluster["labels"][idfusion].append(predLabels[idx].item())

                        max_score_cluster = max(cluster["scores"][idfusion])
                        if(max_score_cluster > 0.75):
                            #get score below 0.15
                            idxs = torch.argsort(torch.tensor(cluster["scores"][idfusion]), descending=False)
                            for idx in idxs:
                                if cluster["scores"][idfusion][idx] < 0.15:
                                    cluster["scores"][idfusion].pop(idx)
                                    cluster["boxes"][idfusion].pop(idx)
                                    cluster["labels"][idfusion].pop(idx)
                                    break
                                
                        matched = True

                        T = cluster["boxes"][idfusion]
                        S = cluster["scores"][idfusion]
                        new_score, new_box = update_box_values(S, T, device)

                        fusion['scores'][idfusion] = new_score
                        fusion['boxes'][idfusion] = new_box
                        break

                if not matched:
                    cluster['boxes'].append([predBox.cpu().numpy()])
                    cluster['scores'].append([predScores[idx].item()])
                    cluster['labels'].append([predLabels[idx].item()])
                    fusion['scores'].append(predScores[idx].item())
                    fusion['boxes'].append(predBox.cpu().numpy())
                    fusion['labels'].append(predLabels[idx].item())

            for idx, score in enumerate(fusion['scores']):
                if score > 1:
                    warnings.warn("Scores should be between 0 and 1")

                T = len(cluster['scores'][idx])
                min_T_N = min(T, number_of_models)
                newscore = score * (min_T_N / number_of_models)
                fusion['scores'][idx] = newscore

            idxs = torch.argsort(torch.tensor(fusion['scores']), descending=True)
            boxes_np = np.array(fusion['boxes'])
            scores_np = np.array(fusion['scores'])
            labels_np = np.array(fusion['labels'])
            fusion['boxes'] = torch.tensor(boxes_np[idxs], device=device)
            fusion['scores'] = torch.tensor(scores_np[idxs], device=device)
            fusion['labels'] = torch.tensor(labels_np[idxs], device=device)

            fusionList.append(fusion)
            clusterList.append(cluster)

    # for idxFusions, fusions in enumerate(fusionListList):
    #     for idxFusion, fusion in enumerate(fusions):
    #         predScores = fusion['scores']
    #         for idx, score in enumerate(predScores):
    #             T = len(clusterListList[idxFusions][idxFusion]["boxes"][idx])
    #             min_T_N = min(T, number_of_models)        
    #             score = score * (min_T_N / number_of_models)
    #             fusion['scores'][idx] = score

    #         for score in predScores:
    #             if score > 1:
    #                 warnings.warn("Scores should be between 0 and 1")

    #         idxs = torch.argsort(predScores, descending=True)
    #         fusion['scores'] = predScores[idxs]
    #         fusion['boxes'] = fusion['boxes'][idxs]
    #         fusion['labels'] = fusion['labels'][idxs]

    #         fusion['scores'] = fusion['scores'].to(device)
    #         fusion['boxes'] = fusion['boxes'].to(device)
    #         fusion['labels'] = fusion['labels'].to(device)
        fusionListList.append(fusionList)
        clusterListList.append(clusterList)

    return fusionListList


def regroup_predictions(predictionsModel1, predictionsModel2, threshold=0.5):
    output = []

    for predictions1, predictions2 in zip(predictionsModel1, predictionsModel2):
        merged_predictions = []

        for pred1, pred2 in zip(predictions1, predictions2):
            combined_boxes, combined_scores, combined_labels = RegroupPredictionsFilterLowConfidenceBox(pred1, pred2, threshold=threshold)
            merged_predictions.append({'boxes': combined_boxes, 'labels': combined_labels, 'scores': combined_scores})

        output.append(merged_predictions)

    return output


def update_box_values(S, T, device):
    """
    Update box values based on accumulated boxes in T.
    """
    T_boxes = torch.tensor(np.array(T), device=device)
    T_scores = torch.tensor(np.array(S), device=device)

    new_score = T_scores.mean().item()
    new_x1 = (T_scores * T_boxes[:, 0]).sum().item() / T_scores.sum().item()
    new_y1 = (T_scores * T_boxes[:, 1]).sum().item() / T_scores.sum().item()
    new_x2 = (T_scores * T_boxes[:, 2]).sum().item() / T_scores.sum().item()
    new_y2 = (T_scores * T_boxes[:, 3]).sum().item() / T_scores.sum().item()

    return new_score, [new_x1, new_y1, new_x2, new_y2]
