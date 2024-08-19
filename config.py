import argparse
import logging
import os
import time

import configargparse
import torch

from utils.file_util import create_dirs


def get_config() -> argparse.Namespace:
    parser = configargparse.ArgParser(
        description="Deepfake Detection"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    parser.add_argument("--dataset", type=str, default="data_phase1/phase1",
                        help="dataset dir path")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="load dataset number of workers")

    # device
    parser.add_argument("--device", type=str, default=device,
                        help="cpu or cuda")

    # train
    parser.add_argument("--img_width", type=int, default=128,
                        help="train image resize width")
    parser.add_argument("--img_height", type=int, default=128,
                        help="train image resize height")
    parser.add_argument("--model_name", type=str, default="efficientnet-b5",
                        help="load pretrain model name")
    parser.add_argument("--init_lr", type=float, default=5e-4,
                        help="train init learning rate")
    parser.add_argument("--final_lr", type=float, default=5e-5,
                        help="train final learning rate")
    parser.add_argument("--batch_size", type=int, default="16",
                        help="train batch size")
    parser.add_argument("--epoch", type=int, default=5,
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
    parser.add_argument('--log_dir', type=str, default='new_logs',
                        help='log directory')

    args = parser.parse_args()

    # create log dir
    log_file_tiemstamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    args.log_dir = os.path.join(args.log_dir, log_file_tiemstamp)
    create_dirs(args.log_dir)
    print("Log dir: ", args.log_dir)

    # create logger
    logging.basicConfig(filename=os.path.join(args.log_dir, f"log.txt"), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    args.logger = logger

    # create weight dir
    args.weight_dir = os.path.join(args.log_dir, "weight")
    create_dirs(args.weight_dir)

    return args
