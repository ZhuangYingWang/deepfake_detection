import argparse
import os
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_train_data(args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 定义数据增强的变换
    data_transform = transforms.Compose([
        # 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        # 随机垂直翻转
        transforms.RandomVerticalFlip(p=0.5),
        # 随机旋转，例如随机旋转0到180度
        transforms.RandomRotation(degrees=180),
        # 随机裁剪，例如裁剪到100x100的大小
        transforms.RandomCrop(100, padding=4),
        # 随机调整亮度、对比度、饱和度和色调
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # 高斯模糊，例如半径为2，sigma为0.3
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 0.3)),
        # 随机擦除，例如擦除面积为20%的区域
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        # 调整图像大小
        transforms.Resize((args.img_width, args.img_height)),
        # 转换为张量
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize(mean, std)
    ])

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
