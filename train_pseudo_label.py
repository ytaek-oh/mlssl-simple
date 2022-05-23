import functools
import os
import time

import torch
import torch.nn as nn

from data.voc import build_voc_datasets
from defaults import default_argument_parser, default_setup
from engine import train_loops, validate
from loss_util import partial_cross_entropy
from metrics import MetricMeter
from model import build_model
from scheduler import build_scheduler
from utils import load_checkpoint, move_to


def main(args):
    best_mAP = 0.

    loader_labeled, loader_unlabeled, loader_test = build_voc_datasets(args)
    num_steps_per_epoch = len(loader_unlabeled)

    model = build_model(args)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd
    )
    scheduler = build_scheduler(args, optimizer, num_steps_per_epoch)

    if args.weights:
        ckpt = load_checkpoint(args)
        model.load_state_dict(ckpt["state_dict"])
        if not args.eval_only:  # resume the training
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            args.start_epoch = ckpt["epoch"]
            best_mAP = ckpt["best_mAP"]
    if args.eval_only:
        validate(args, loader_test, model)
        return

    train_one_epoch_fn = functools.partial(
        _train_one_epoch,
        iter_labeled=iter(loader_labeled),
        iter_unlabeled=iter(loader_unlabeled),
        criterion=nn.BCEWithLogitsLoss().cuda(),
        num_steps_per_epoch=num_steps_per_epoch
    )
    train_loops(
        args, train_one_epoch_fn, model, optimizer, scheduler, loader_test, best_mAP=best_mAP
    )


def _train_one_epoch(
    args,
    model,
    optimizer,
    scheduler,
    # defined by partial
    iter_labeled=None,
    iter_unlabeled=None,
    criterion=None,
    num_steps_per_epoch=None,
    ema_model=None
):
    model.train()
    metric_meters = MetricMeter()
    for i in range(num_steps_per_epoch):
        start_time = time.time()
        l_imgs, labels = move_to(next(iter_labeled), "cuda")
        ul_imgs, _ = move_to(next(iter_unlabeled), "cuda")

        num_labels = labels.size(0)
        inputs_cat = torch.cat([l_imgs, ul_imgs], dim=0)
        logits_cat = model(inputs_cat)
        l_logits, ul_logits = logits_cat[:num_labels], logits_cat[num_labels:]
        l_loss = criterion(l_logits, labels.float())

        # ul loss
        with torch.no_grad():
            p = ul_logits.detach().sigmoid()
            pseudo_label = p.ge(0.5).float()
            pl_valid_mask = torch.bitwise_or(p.ge(args.pos_conf_thr), p.le(args.neg_conf_thr))
        ul_loss = partial_cross_entropy(
            ul_logits, pseudo_label, loss_weight=args.ul_loss_weight, valid_mask=pl_valid_mask
        )
        loss = l_loss + ul_loss

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log some metrics
        metric_dict = {"l_loss": l_loss, "ul_loss": ul_loss, "batch_time": time.time() - start_time}
        metric_meters.update(metric_dict)

        # aggregate
        if ((i + 1) % args.print_iters == 0) or ((i + 1) == num_steps_per_epoch):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                "{steps},  {metrics},  {lr}".format(
                    steps=f"Steps: {i+1}/{num_steps_per_epoch}",
                    metrics=metric_meters,
                    lr=f"lr: {current_lr:.4f}"
                )
            )


def _get_experiment_name(args):
    names = []
    names.append(f"{args.dataset}_{args.percent_labels}p")
    names.append(f"ep_{args.num_epochs}")
    names.append(f"{args.algo_name}_pconf_{args.pos_conf_thr}_nconf_{args.neg_conf_thr}")
    names.append(f"lr_{args.lr}")
    if args.scheduler == "cosine":
        names.append(f"cos_{args.cos_ratio}")
    if args.memo:
        names.append(args.memo)
    names.append(f"seed_{args.seed}")
    return "_".join(names)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.algo_name = "PseudoLabel"  # set algorithm name
    args.dataset = "voc"
    args.output_dir = os.path.join(args.output_dir, _get_experiment_name(args))
    default_setup(args)
    main(args)
