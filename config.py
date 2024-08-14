import argparse
import logging
import os

import configargparse
import torch


def get_config() -> argparse.Namespace:
    parser = configargparse.ArgumentParser(
        description="Deepfake Detection"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="dataset dir path")

    # device
    parser.add_argument("--device", type=str, default=device,
                        help="cpu or cuda")

    # train
    parser.add_argument("--model_name", type=str, default="efficientnet-b7",
                        help="load pretrain model name")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="train learning rate")
    parser.add_argument("--batch_size", type=int, default="1024",
                        help="train batch size")
    parser.add_argument("--epoch", type=int, default="100",
                        help="train epoch number")
    parser.add_argument("--output_dim", type=int, default=2,
                        help="model output dim")

    # log
    parser.add_argument('--model_save_step', type=int, default=500,
                        help="model save step")
    parser.add_argument('--save_checkpoint_num', type=int, default=3,
                        help="save checkpoint number")
    parser.add_argument('--print_step', type=int, default=100,
                        help="log print step")
    parser.add_argument('--log_dir', type=str, default='log',
                        help='log directory')

    args = parser.parse_args()

    # create log dir
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    print("Log dir: ", args.log_dir)

    # create logger
    logging_path: str = str(os.path.join(args.log_dir, "log.txt"))
    logging.basicConfig(filename=logging_path, level=logging.DEBUG,
                        filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    args.logger = logging.getLogger(__name__)

    return args
