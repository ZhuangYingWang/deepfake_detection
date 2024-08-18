import argparse
import os

import torch

from config import get_config
from dataset import get_test_data
from model import get_model


def test(args: argparse.Namespace):
    model_dict = get_model(args, "weight3.0/0.5991_epoch0.pt")
    model = model_dict["model"]

    model.eval()

    test_loader, _ = get_test_data(args)

    predictions = []

    with torch.no_grad():
        for inputs, img_paths in test_loader:
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            scores = probabilities[:, 0].cpu().numpy()

            for img_path, score in zip(img_paths, scores):
                img_filename = os.path.basename(img_path)
                predictions.append(f"{img_filename},{score}")

    with open("prediction.txt", "w") as f:
        for pred in predictions:
            f.write(pred + "\n")

    print("Predictions have been written to prediction.txt")


if __name__ == '__main__':
    config = get_config()
    test(config)
