import argparse

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn


def get_model(args: argparse.Namespace, load_model_path: str = None):
    net = EfficientNet.from_name(args.model_name).to(args.device)
    net._fc = nn.Linear(in_features=net._fc.in_features, out_features=args.output_dim)
    net = net.to(args.device)
    if load_model_path is not None:
        model_state_dict = torch.load(load_model_path, map_location=args.device,
                                      weights_only=True)
        net.load_state_dict(model_state_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.99)
    epoch = args.epoch

    model_dict = {
        "model": net,
        "criterion": criterion,
        "optimizer": optimizer_ft,
        "scheduler": exp_lr_scheduler,
        "num_epochs": epoch
    }

    return model_dict
