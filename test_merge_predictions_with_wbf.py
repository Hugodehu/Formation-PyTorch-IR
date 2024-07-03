import pytest
import torch
import ensemble_boxes 
from function.merge_prediction.merge_predictions_with_wbf import merge_predictions_with_wbf

# Fixture to set the device
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def test_empty_predictions():
    predictionsModel1 = []
    predictionsModel2 = []
    fusionListList = merge_predictions_with_wbf(predictionsModel1, predictionsModel2)
    assert len(fusionListList) == 0

def test_non_empty_predictions_with_iou_threshold(device):
    predictionsModel1 = [
        [
            {
                'boxes': torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                'scores': torch.tensor([0.9], dtype=torch.float32),
                'labels': torch.tensor([1])
            }
        ]
    ]
    predictionsModel2 = [
        [
            {
                'boxes': torch.tensor([[20, 20, 30, 30]], dtype=torch.float32),
                'scores': torch.tensor([0.8], dtype=torch.float32),
                'labels': torch.tensor([2])
            }
        ]
    ]
    fusionListList = merge_predictions_with_wbf(predictionsModel1, predictionsModel2)
    assert len(fusionListList) == 1
    assert len(fusionListList[0]) == 1
    assert torch.allclose(fusionListList[0][0]['boxes'].to(torch.float32), torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], device=device, dtype=torch.float32))
    assert torch.allclose(fusionListList[0][0]['scores'].to(torch.float32), torch.tensor([0.45, 0.4], device=device, dtype=torch.float32))
    assert torch.allclose(fusionListList[0][0]['labels'].to(torch.int), torch.tensor([1, 2], device=device ).to(torch.int))

def test_multiple_predictions_with_different_iou_threshold(device):
    predictionsModel1 = [
        [
            {
                'boxes': torch.tensor([[0, 0, 0.10, 0.10]], dtype=torch.float32),
                'scores': torch.tensor([0.9], dtype=torch.float32),
                'labels': torch.tensor([1])
            }
        ],
        [
            {
                'boxes': torch.tensor([[0.20, 0.20, 0.30, 0.30]], dtype=torch.float32),
                'scores': torch.tensor([0.8], dtype=torch.float32),
                'labels': torch.tensor([2])
            }
        ]
    ]
    predictionsModel2 = [
        [
            {
                'boxes': torch.tensor([[0, 0, 0.08, 0.12]], dtype=torch.float32),
                'scores': torch.tensor([0.7], dtype=torch.float32),
                'labels': torch.tensor([1])
            }
        ],
        [
            {
                'boxes': torch.tensor([[0.40, 0.40, 0.50, 0.50]], dtype=torch.float32),
                'scores': torch.tensor([0.6], dtype=torch.float32),
                'labels': torch.tensor([2])
            }
        ]
    ]

    fusionListList = merge_predictions_with_wbf(predictionsModel1, predictionsModel2)
    boxes1 = torch.tensor([[[0, 0, 0.10, 0.10]], [[0, 0, 0.08, 0.12]]], device=device, dtype=torch.float32)
    scores1 = torch.tensor([[0.9], [0.7]], device=device, dtype=torch.float32)
    labels1 = torch.tensor([[1], [1]], device=device).to(torch.int)
    boxes2 = torch.tensor([[[0.20, 0.20, 0.30, 0.30]], [[0.40, 0.40, 0.50, 0.50]]], device=device, dtype=torch.float32)
    scores2 = torch.tensor([[0.8], [0.6]], device=device, dtype=torch.float32)
    labels2 = torch.tensor([[2], [2]], device=device).to(torch.int)
    testwbf1Boxes, testwbf1Scores, testwbf1Lables = ensemble_boxes.weighted_boxes_fusion(boxes_list=boxes1.cpu().numpy(), scores_list=scores1.cpu().numpy(), labels_list=labels1.cpu().numpy(), iou_thr=0.5, skip_box_thr=0.0, conf_type='avg')
    testwbf2Boxes, testwbf2Scores, testwbf2Lalbles = ensemble_boxes.weighted_boxes_fusion(boxes_list=boxes2.cpu().numpy(), scores_list=scores2.cpu().numpy(), labels_list=labels2.cpu().numpy(), iou_thr=0.5, skip_box_thr=0.0, conf_type='avg')
    assert len(fusionListList) == 2
    assert len(fusionListList[0]) == 1
    assert len(fusionListList[1]) == 1
    assert torch.allclose(fusionListList[0][0]['boxes'].to(torch.float32), torch.tensor([0, 0, 0.09125, 0.10875], device=device, dtype=torch.float32))   
    assert torch.allclose(fusionListList[0][0]['scores'].to(torch.float32), torch.tensor([0.8], device=device, dtype=torch.float32))
    assert torch.allclose(fusionListList[0][0]['labels'].to(torch.int), torch.tensor([1], device=device).to(torch.int))
    assert torch.allclose(fusionListList[1][0]['boxes'].to(torch.float32), torch.tensor([[0.20, 0.20, 0.30, 0.30], [0.40, 0.40, 0.50, 0.50]], device=device, dtype=torch.float32))
    assert torch.allclose(fusionListList[1][0]['scores'].to(torch.float32), torch.tensor([0.4, 0.3], device=device, dtype=torch.float32))
    assert torch.allclose(fusionListList[1][0]['labels'].to(torch.int), torch.tensor([2, 2], device=device).to(torch.int))
    assert torch.allclose(torch.tensor(testwbf2Boxes, device=device).to(torch.float32), fusionListList[1][0]['boxes'].to(torch.float32))
    assert torch.allclose(torch.tensor(testwbf2Scores, device=device).to(torch.float32), fusionListList[1][0]['scores'].to(torch.float32))
    assert torch.allclose(torch.tensor(testwbf2Lalbles, device=device).to(torch.int), fusionListList[1][0]['labels'].to(torch.int))
    assert torch.allclose(torch.tensor(testwbf1Boxes, device=device).to(torch.float32), fusionListList[0][0]['boxes'].to(torch.float32))
    assert torch.allclose(torch.tensor(testwbf1Scores, device=device).to(torch.float32), fusionListList[0][0]['scores'].to(torch.float32))
    assert torch.allclose(torch.tensor(testwbf1Lables, device=device).to(torch.int), fusionListList[0][0]['labels'].to(torch.int))



