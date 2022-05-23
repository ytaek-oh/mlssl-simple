import torch
from torch.utils.data import DataLoader
from torchvision.datasets.voc import VOCDetection

from .transforms import get_image_transforms
from .utils import InfiniteSampler, Subset, split_indices

CLASS_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor'
)


def _voc_target_tfm(target):
    labels = [0 for _ in range(len(CLASS_NAMES))]
    names = list(set([obj["name"] for obj in target["annotation"]["object"]]))
    for obj in names:
        labels[CLASS_NAMES.index(obj)] = 1
    return torch.Tensor(labels).long()


def build_voc_datasets(args):
    assert len(CLASS_NAMES) == args.num_classes

    # get transforms
    tfm_labeled, tfm_unlabeled, tfm_eval = get_image_transforms(args, _voc_target_tfm)

    # split indices for labeled and unlabeled data
    voc_train_all = VOCDetection(
        args.data_root, image_set="trainval", year="2007", download=args.download_dataset
    )
    inds_labeled, inds_unlabeled = split_indices(len(voc_train_all), args.percent_labels, args.seed)

    # build labeled train data and loader
    voc_train_labeled = Subset(voc_train_all, inds_labeled, transforms=tfm_labeled)
    loader_labeled = DataLoader(
        voc_train_labeled,
        batch_size=args.batch_size,
        sampler=InfiniteSampler(len(voc_train_labeled), seed=args.seed),
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers
    )

    # build unlabeled train data and loader
    loader_unlabeled = None
    if tfm_unlabeled is not None:
        voc_train_unlabeled = Subset(voc_train_all, inds_unlabeled, transforms=tfm_unlabeled)
        loader_unlabeled = DataLoader(
            voc_train_unlabeled,
            batch_size=args.ul_batch_ratio * args.batch_size,
            sampler=InfiniteSampler(len(voc_train_unlabeled), seed=args.seed),
            drop_last=True,
            shuffle=False,
            num_workers=args.num_workers
        )

    # build test data and loader
    voc_test = VOCDetection(
        args.data_root,
        image_set="test",
        year="2007",
        download=args.download_dataset,
        transforms=tfm_eval
    )
    loader_test = DataLoader(
        voc_test,
        batch_size=args.inference_batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers
    )
    return loader_labeled, loader_unlabeled, loader_test
