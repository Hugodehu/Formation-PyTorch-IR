import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch


class CustomDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # Load annotations
        with open(json_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.img_dir, self.annotations[idx]['name'])
        image = Image.open(img_name)
        
        # Get annotations
        labels = self.annotations[idx]['labels']
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
         # Extract bounding boxes and convert them to a tensor
        boxes = []
        cocoNames = []
        for label in labels:
            box = label['box2d']
            boxes.append([box['x1'], box['y1'], box['x2'], box['y2']])
            className = label['category']
            cocoNames.append(convertClassesBdd100KToClassesCOCO(className))

        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(cocoNames, dtype=torch.int64)
        labels = {'boxes': boxes_tensor, 'labels': labels_tensor}
        sample = {'image': image, 'labels': labels}
        
        return sample


def collate_fn_for_bdd100K(batch):
    """ 
    La fonction collate_fn prend un lot de données, sépare les images et les annotations, et les retourne sous forme de listes. 
    Cette fonction permet de gérer les lots où les annotations ont des tailles variables.
    """
    images = [item['image'] for item in batch]
    labels = [item['labels'] for item in batch]
    return images, labels


def collate_fn_for_coco(batch):
    """ 
    La fonction collate_fn prend un lot de données, sépare les images et les annotations, et les retourne sous forme de listes. 
    Cette fonction permet de gérer les lots où les annotations ont des tailles variables.
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

ClassesCoco= {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
}

ClassesBdd100k = {
    1: "pedestrian",
    2: "rider",
    3: "car",
    4: "truck",
    5: "bus",
    6: "train",
    7: "motorcycle",
    8: "bicycle",
    9: "traffic light",
    10: "traffic sign",
}

def convertClassesBdd100KToClassesCOCO(classes):
    if(classes == "pedestrian"):
        return 1
    if(classes == "rider"):
        return 1
    if(classes == "car"):
        return 3
    if(classes == "truck"):
        return 8
    if(classes == "bus"):
        return 6
    if(classes == "train"):
        return 7
    if(classes == "motorcycle"):
        return 4
    if(classes == "bicycle"):
        return 2
    if(classes == "traffic light"):
        return 10
    if(classes == "traffic sign"):
        return 12
    else:
        return -3