import torch
from function.merge_prediction.merge_predictions_with_wbf import merge_predictions_with_wbf

# Test case 1: Empty predictions
predictionsModel1 = []
predictionsModel2 = []
fusionListList = merge_predictions_with_wbf(predictionsModel1, predictionsModel2)
assert len(fusionListList) == 0

# Test case 2: Non-empty predictions with IoU threshold of 0.5 and threshold of 0.5
predictionsModel1 = [
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([1])
        }
    ]
]
predictionsModel2 = [
    [
        {
            'boxes': torch.tensor([[20, 20, 30, 30]]),
            'scores': torch.tensor([0.8]),
            'labels': torch.tensor([2])
        }
    ]
]
device  = "cuda" if torch.cuda.is_available() else "cpu"
fusionListList = merge_predictions_with_wbf(predictionsModel1, predictionsModel2)
assert len(fusionListList) == 1
assert len(fusionListList[0]) == 1
assert torch.all(torch.eq(fusionListList[0][0]['boxes'], torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], device=device)))
assert torch.all(torch.eq(fusionListList[0][0]['scores'], torch.tensor([0.9, 0.8], device=device)))
assert torch.all(torch.eq(fusionListList[0][0]['labels'], torch.tensor([1, 2], device=device)))

# Test case 3: Multiple predictions with different IoU threshold and threshold
predictionsModel1 = [
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([1])
        }
    ],
    [
        {
            'boxes': torch.tensor([[20, 20, 30, 30]]),
            'scores': torch.tensor([0.8]),
            'labels': torch.tensor([2])
        }
    ]
]
predictionsModel2 = [
    [
        {
            'boxes': torch.tensor([[0, 0, 8, 12]]),
            'scores': torch.tensor([0.7]),
            'labels': torch.tensor([1])
        }
    ],
    [
        {
            'boxes': torch.tensor([[40, 40, 50, 50]]),
            'scores': torch.tensor([0.6]),
            'labels': torch.tensor([2])
        }
    ]
]
fusionListList = merge_predictions_with_wbf(predictionsModel1, predictionsModel2)
device  = "cuda" if torch.cuda.is_available() else "cpu"
assert len(fusionListList) == 2
assert len(fusionListList[0]) == 1
assert len(fusionListList[1]) == 1
assert torch.all(torch.eq(fusionListList[0][0]['boxes'], torch.tensor([[0, 0, 9.125, 10.875]], device=device)))
assert torch.all(torch.eq(fusionListList[0][0]['scores'], torch.tensor([0.8], device=device)))
assert torch.all(torch.eq(fusionListList[0][0]['labels'], torch.tensor([1], device=device)))
assert torch.all(torch.eq(fusionListList[1][0]['boxes'], torch.tensor([[20, 20, 30, 30], [40, 40, 50, 50]], device=device)))
assert torch.all(torch.eq(fusionListList[1][0]['scores'], torch.tensor([0.8, 0.6], device=device)))
assert torch.all(torch.eq(fusionListList[1][0]['labels'], torch.tensor([2, 2], device=device)))

print("All test cases passed!")