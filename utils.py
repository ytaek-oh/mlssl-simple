import datetime
import os
import random
import shutil

import numpy as np
import torch


def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
        return x
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        assert isinstance(x, int) or isinstance(x, float)
        return np.asarray(x)


def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        # Assume obj is a Tensor or other type that supports .to(device)
        return obj.to(device)


def get_eta(elapsed_time, epoch, num_epochs):
    eta_seconds = (num_epochs - epoch) * elapsed_time
    return str(datetime.timedelta(seconds=int(eta_seconds)))


def save_checkpoint(save_path, state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        target_path = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, target_path)


def load_checkpoint(args):
    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.resume, map_location="cpu")
        keys = list(checkpoint.keys())
        print("loaded checkpoint from {} including keys: {}".format(args.weights, keys))
        if "best_mAP" in keys:
            print(
                "mAP: {:.4f} at epoch: {:02d} (best: {:.4f})".format(
                    checkpoint["mAP"], checkpoint["epoch"], checkpoint["best_mAP"]
                )
            )
            print()
        return checkpoint
    else:
        print("No checkpoint found at {}".format(args.weights))
        raise ValueError
