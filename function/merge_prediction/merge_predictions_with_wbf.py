import numpy as np
import torch
import function.compute_iou as compute_iou

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
                    iou = bb_iou_array(box, fusionbox)
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
            boxes1, scores1, labels1 = pred1['boxes'], pred1['scores'], pred1['labels']
            boxes2, scores2, labels2 = pred2['boxes'], pred2['scores'], pred2['labels']
            
            # Combine boxes and scores
            combined_boxes = torch.cat((boxes1, boxes2), dim=0)
            combined_scores = torch.cat((scores1, scores2), dim=0)
            combined_labels = torch.cat((labels1, labels2), dim=0)

            # Filter out low-confidence boxes
            high_conf_idx = combined_scores >= threshold
            combined_boxes = combined_boxes[high_conf_idx]
            combined_scores = combined_scores[high_conf_idx]
            combined_labels = combined_labels[high_conf_idx]
            
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

def bb_iou_array(boxes, new_box):
        # bb interesection over union
        boxX1, boxY1, boxX2, boxY2 = boxes.cpu().numpy()
        new_boxX1, new_boxY1, new_boxX2, new_boxY2 = new_box
        xA = np.maximum(boxX1, new_boxX1)
        yA = np.maximum(boxY1, new_boxY1)
        xB = np.minimum(boxX2, new_boxX2)
        yB = np.minimum(boxY2, new_boxY2)

        # compute the area of intersection rectangle
        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxX2 - boxX1) * (boxY2 - boxY1)
        boxBArea = (new_boxX2 - new_boxX1) * (new_boxY2 - new_boxY1)

        # compute the intersection over union by taking the intersection
        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou