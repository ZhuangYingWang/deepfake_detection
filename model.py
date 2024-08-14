import argparse

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn


def get_model(args: argparse.Namespace):
    net = EfficientNet.from_name(args.model_name).to(args.device)
    net._fc.out_features = args.output_dim
    # net = models.res
    net = net.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_dict = {
        "net": net,
        "criterion": criterion,
        "optimizer_ft": optimizer_ft,
        "exp_lr_scheduler": exp_lr_scheduler
    }

    return model_dict
