import argparse
import logging
import os
import time
import configargparse
import torch


def get_config() -> argparse.Namespace:
    
    # create logger
    log_file_tiemstamp = time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime(time.time()))
    logging.basicConfig(filename=f"new_logs/log_{log_file_tiemstamp}.txt", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    parser = configargparse.ArgumentParser(
        description="Deepfake Detection"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    parser.add_argument("--dataset", type=str, default="data_phase1/phase1",
                        help="dataset dir path")

    # device
    parser.add_argument("--device", type=str, default=device,
                        help="cpu or cuda")

    # train
    parser.add_argument("--model_name", type=str, default="efficientnet-b7",
                        help="load pretrain model name")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="train learning rate")
    parser.add_argument("--batch_size", type=int, default="16",
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
    parser.add_argument('--log_dir', type=str, default='new_logs',
                        help='log directory')
    parser.add_argument('--logger', type=str, default=logger,
                        help='log directory')

    args = parser.parse_args()

    # create log dir
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir, exist_ok=True)
    print("Log dir: ", args.log_dir)

    return args
