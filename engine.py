import time

import numpy as np
import torch

from metrics import MetricMeter, Writer
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import get_eta, move_to, save_checkpoint, to_numpy


def train_loops(
    args, train_epoch_fn, model, optimizer, scheduler, loader_test, ema_model=None, best_mAP=0.
):
    writer = Writer(args.output_dir)
    total_epoch_time = 0.
    for epoch in range(args.start_epoch, args.num_epochs):
        start_time = time.time()
        print(f"Starting train epochs: {epoch + 1}/{args.num_epochs}")
        train_epoch_fn(args, model, optimizer, scheduler, ema_model=ema_model)

        if ((epoch + 1) % args.eval_periods == 0) or ((epoch + 1) == args.num_epochs):
            results = validate(args, loader_test, model)
            results.update({"epoch": epoch + 1})
            mAP = results["macro_aurpc"]

            # remember the best mAP and save the checkpoint
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint(
                args.output_dir, {
                    'epoch': epoch + 1,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'ema_model': ema_model.state_dict() if ema_model is not None else None,
                    'mAP': mAP,
                    'best_mAP': best_mAP,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best
            )

            # write metrics
            writer.write(results)

        # output eta
        total_epoch_time += (time.time() - start_time)
        avg_epoch_time = total_epoch_time / (epoch + 1)
        eta = get_eta(avg_epoch_time, epoch + 1, args.num_epochs)
        print(
            "[Epoch {}/{}]  Total elapsed: {:.1f} seconds, ETA: {}".format(
                epoch + 1, args.num_epochs, total_epoch_time, eta
            )
        )
        print()

    # end of epoch
    print(f"Best mAP: {best_mAP:.4f}, Last mAP: {mAP:.4f}")


def do_evaluate(all_predictions, all_labels):
    # mesaure test performances
    macro_aurpc = average_precision_score(all_labels, all_predictions, average="macro")
    micro_aurpc = average_precision_score(all_labels, all_predictions, average="micro")

    macro_auroc = roc_auc_score(all_labels, all_predictions, average="macro")
    micro_auroc = roc_auc_score(all_labels, all_predictions, average="micro")

    # print results
    print(
        "[Evaluation] {aurpc} | {auroc}".format(
            aurpc=f"macro AURPC: {macro_aurpc:.4f}, micro AURPC: {micro_aurpc:.4f}",
            auroc=f"macro AUROC: {macro_auroc:.4f}, micro AUROC: {micro_auroc:.4f}"
        )
    )
    results = {}
    results.update({"macro_aurpc": macro_aurpc, "micro_aurpc": micro_aurpc})
    results.update({"macro_auroc": macro_auroc, "micro_auroc": micro_auroc})
    return results


def validate(args, data_loader, model):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    metric_meters = MetricMeter()

    pred_list = []
    label_list = []
    print(f"Starting evaluation for {len(data_loader)} batches...")
    for i, (images, labels) in enumerate(data_loader):
        start_time = time.time()
        with torch.no_grad():
            if torch.cuda.is_available():
                images, labels = move_to((images, labels), "cuda")
            preds = model(images)
            loss = criterion(preds, labels.float())

            pred_list.append(to_numpy(preds))
            label_list.append(to_numpy(labels))

            # log some metrics
            metric_dict = {"loss": loss, "batch_time": time.time() - start_time}
            metric_meters.update(metric_dict)

        if ((i + 1) % 20 == 0) or ((i + 1) == len(data_loader)):
            _eta = get_eta(metric_meters.meters["batch_time"].avg, i + 1, len(data_loader))
            print(
                "{steps},  {eta},  {metrics}".format(
                    steps=f"Steps: {i+1}/{len(data_loader)}", eta=_eta, metrics=metric_meters
                )
            )

    pred_list = np.concatenate(pred_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    results = do_evaluate(pred_list, label_list)
    results.update({"eval_loss": metric_meters.meters["loss"].avg})

    return results
