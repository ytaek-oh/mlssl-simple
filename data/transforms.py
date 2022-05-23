import torchvision.transforms as TF
from torchvision.datasets.vision import StandardTransform

_IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _get_unlabeled_image_transforms(args):
    normalize_tfm = TF.Normalize(_IMAGENET_DEFAULT_MEAN, _IMAGENET_DEFAULT_STD)
    if args.algo_name == "Supervised":
        _tfm_unlabeled = None

    elif args.algo_name == "PseudoLabel":
        _tfm_unlabeled = TF.Compose(
            [
                TF.Resize(256),
                TF.RandomCrop(224),
                TF.RandomHorizontalFlip(),
                TF.ToTensor(), normalize_tfm
            ]  # same as labeled tfm (weak)
        )

    elif args.algo_name == "MeanTeacher":
        _tfm_unlabeled = GeneralizedSSLTransform(
            [
                TF.Compose(
                    [
                        TF.Resize(256),
                        TF.RandomCrop(224),
                        TF.RandomHorizontalFlip(),
                        TF.ToTensor(), normalize_tfm
                    ]  # weak augmentations
                ),
                TF.Compose(
                    [
                        TF.Resize(256),
                        TF.RandomCrop(224),
                        TF.RandomHorizontalFlip(),
                        TF.ToTensor(), normalize_tfm
                    ]  # weak augmentations
                )
            ]
        )

    else:
        print(f"{args.algo_name} is unidentifed algorithm name.")
        raise NotImplementedError
    return _tfm_unlabeled


def get_image_transforms(args, tfm_target):
    normalize_tfm = TF.Normalize(_IMAGENET_DEFAULT_MEAN, _IMAGENET_DEFAULT_STD)

    # labeled data transforms
    _tfm_labeled = TF.Compose(
        [
            TF.Resize(256),
            TF.RandomCrop(224),
            TF.RandomHorizontalFlip(),
            TF.ToTensor(), normalize_tfm
        ]
    )
    tfm_labeled = StandardTransform(transform=_tfm_labeled, target_transform=tfm_target)

    # unlabeled data transforms
    _tfm_unlabeled = _get_unlabeled_image_transforms(args)
    tfm_unlabeled = None
    if _tfm_unlabeled is not None:
        tfm_unlabeled = StandardTransform(transform=_tfm_unlabeled, target_transform=tfm_target)

    # eval transforms
    _tfm_eval = TF.Compose([TF.Resize(256), TF.CenterCrop(224), TF.ToTensor(), normalize_tfm])
    tfm_eval = StandardTransform(transform=_tfm_eval, target_transform=tfm_target)

    return tfm_labeled, tfm_unlabeled, tfm_eval


class GeneralizedSSLTransform:
    _repr_indent = 4

    def __init__(self, transforms: list) -> None:
        assert len(transforms) > 0
        self.transforms = transforms

    def __call__(self, img):
        results = []
        for t in self.transforms:
            results.append(t(img))
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def __repr__(self) -> str:
        head = f"{self.__class__.__name__}:"
        body = [
            " " * self._repr_indent + f"({i}): {repr(tfm)}"
            for i, tfm in enumerate(self.transforms)
        ]
        lines = [head] + body
        return "\n".join(lines)
