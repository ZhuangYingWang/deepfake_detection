import argparse
import os
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_train_data(args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(args.dataset, "trainset"), data_transform),
        "val": datasets.ImageFolder(os.path.join(args.dataset, "valset"), data_transform),
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0  # 原来是8，改为0，减少并行加载数据时的内存使用
        ),
        "val": DataLoader(
            image_datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0  #
        ),
    }

    dataset_sizes = {
        "train": len(image_datasets["train"]),
        "val": len(image_datasets["val"]),
    }

    return dataloaders, dataset_sizes
