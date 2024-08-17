import argparse
import os
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_train_data(args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    common_transform = transforms.Compose([
        transforms.Resize((args.img_width, args.img_height)),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)  
    ])

    # 定义对real图像的额外数据增强
    real_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.RandomCrop(100, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 0.3)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        common_transform  
    ])

    fake_transform = transforms.Compose([
        common_transform  
    ])

    image_datasets = {
        "train": datasets.ImageFolder(
            os.path.join(args.dataset, "trainset"),
            transform=real_transform if os.path.basename(os.path.normpath(args.dataset)) == "real" else fake_transform
        ),
        "val": datasets.ImageFolder(
            os.path.join(args.dataset, "valset"),
            transform=common_transform
        ),
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0
        ),
        "val": DataLoader(
            image_datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0
        ),
    }

    dataset_sizes = {
        "train": len(image_datasets["train"]),
        "val": len(image_datasets["val"]),
    }

    return dataloaders, dataset_sizes