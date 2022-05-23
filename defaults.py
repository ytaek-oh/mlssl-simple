import argparse
import json
import os
from pathlib import Path

import torch

from utils import set_seed


def default_argument_parser():
    parser = argparse.ArgumentParser(description="")

    # algorithm
    parser.add_argument("--algo-name")
    parser.add_argument("--pos-conf-thr", default=0.5, type=float)
    parser.add_argument("--neg-conf-thr", default=0.5, type=float)
    parser.add_argument("--ul-loss-weight", default=1.0, type=float)
    parser.add_argument("--ul-batch-ratio", default=1, type=int)
    parser.add_argument("--rampup-ratio", default=0.4, type=float)  # ul

    # model
    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--pretrained", default="", type=str, choices=["", "imagenet"])
    parser.add_argument("--weights", default="", type=str)
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--ema-decay", default=0.99, type=float)

    # data
    parser.add_argument("--data-root", default="./datasets", type=str)
    parser.add_argument("--download-dataset", default=False, type=bool)  # only for voc
    parser.add_argument("--percent-labels", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--inference-batch-size", default=128, type=int)

    # solver
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--num-epochs", default=270, type=int)
    parser.add_argument("--lr", default=0.4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)

    # scheduler
    parser.add_argument(
        "--scheduler", default="cosine", type=str, choices=["none", "linear", "cosine"]
    )
    parser.add_argument("--decay-epochs", default=[90], nargs='*', type=int)
    parser.add_argument("--decay-gamma", default=0.1, type=float)
    parser.add_argument("--cos-ratio", default=7, type=int)
    parser.add_argument("--warmup-epochs", default=5, type=int)

    # etc
    parser.add_argument("--eval-only", default=False, type=bool)
    parser.add_argument("--print-iters", default=40, type=int)
    parser.add_argument("--eval-periods", default=5, type=int)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--memo", default="", type=str)

    # reproducability
    parser.add_argument("--cudnn-benchmark", default=False, type=bool)
    parser.add_argument("--cudnn-deterministic", default=True, type=bool)
    parser.add_argument("--seed", default=1, type=int)
    return parser


def default_setup(args):
    print("Command Line Args:", args)

    # reproducability
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    if not args.eval_only:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"experiment dir: {args.output_dir}")

    # save config
    save_path = os.path.join(args.output_dir, "config.txt")
    with open(save_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
    print("Full config saved to {}".format(save_path))
