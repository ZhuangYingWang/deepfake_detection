import argparse
from thop import profile
import timm
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import models
from torch import nn
from utils.loss import FocalLoss


def get_model(args: argparse.Namespace, load_model_path: str = None):
    model_name = args.model_name
    if model_name == "xception":
        net = timm.create_model('xception', pretrained=True)
        net.reset_classifier(num_classes=2)
        net = net.to(args.device)
    elif model_name == "efficientnet":
        net = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        # net = EfficientNet.from_name("efficientnet-b7").to(args.device)
        # net._fc = nn.Linear(in_features=net._fc.in_features, out_features=args.output_dim)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, args.output_dim)
        net = net.to(args.device)
    else:
        raise ValueError(f"Model Can't Recognize -> {model_name}")

    # log model param
    inputs = torch.randn(args.batch_size, 3, args.img_width, args.img_height).to(args.device)
    flops, params = profile(net, inputs=(inputs,))
    args.logger.info(f"FLOPs: {flops / 1e9} G")
    args.logger.info(f"Params: {params / 1e6} M")

    if load_model_path is not None:
        model_state_dict = torch.load(load_model_path, map_location=args.device,
                                      weights_only=True)
        net.load_state_dict(model_state_dict)

    weight = torch.tensor([args.fake_weight, args.real_weight], dtype=torch.float32).to(args.device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = FocalLoss(weight)
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
