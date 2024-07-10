from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms
from torchvision.transforms import ToTensor
from classes.custom_dataset_bdd100k import CustomDataset, collate_fn_for_bdd100K, collate_fn_for_coco

from classes.custom_dataset_bdd100k import CustomDataset

def initialiseDataloader(isSubset, subsetSize=100):
    """
    Initialize and return dataloaders for BDD100K and COCO datasets.

    Returns:
        BDD100K_dataloader (torch.utils.data.DataLoader): Dataloader for BDD100K dataset.
        Coco_dataloader (torch.utils.data.DataLoader): Dataloader for COCO dataset.
    """
    # Usage
    img_dir_bdd100K = 'data/bdd100K/bdd100K/images/100K/val'
    json_file_bdd100K = 'data/bdd100k/bdd100K/labels/det_20/det_val.json'
    BDD100K_dataset = CustomDataset(json_file=json_file_bdd100K, img_dir=img_dir_bdd100K, transform=ToTensor())

    img_dir_coco = 'data/coco/val2017'
    json_file_coco = 'data/coco/annotations/instances_val2017.json'
    transform = transforms.Compose([transforms.ToTensor()])
    COCO_dataset = datasets.CocoDetection(root=img_dir_coco, annFile=json_file_coco, transform=transform,)

    # Créer un sous-ensemble contenant seulement les 10 premières images
    subset_indices = list(range(subsetSize))
    BDD100Ksubset_dataset = Subset(BDD100K_dataset, subset_indices)

    batch_size = 1 # 1 images per batch

    if(isSubset):
        # Create data loaders.
        BDD100KSubset_dataloader = DataLoader(BDD100Ksubset_dataset, batch_size=batch_size, collate_fn=collate_fn_for_bdd100K)

        COCOsubset_dataset = Subset(COCO_dataset, subset_indices)

        Cocosubset_dataloader = DataLoader(COCOsubset_dataset, batch_size=batch_size, collate_fn=collate_fn_for_coco)

        return BDD100KSubset_dataloader, Cocosubset_dataloader


    else:
        BDD100K_dataloader = DataLoader(BDD100K_dataset, batch_size=batch_size, collate_fn=collate_fn_for_bdd100K)

        Coco_dataloader = DataLoader(COCO_dataset, batch_size=batch_size, collate_fn=collate_fn_for_coco)

        return BDD100K_dataloader, Coco_dataloader