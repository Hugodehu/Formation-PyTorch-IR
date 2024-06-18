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
        for label in labels:
            box = label['box2d']
            boxes.append([box['x1'], box['y1'], box['x2'], box['y2']])
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        
        sample = {'image': image, 'labels': boxes_tensor}
        
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