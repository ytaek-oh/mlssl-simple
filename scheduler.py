import math
from bisect import bisect_right

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from typing import List


def build_scheduler(args, optimizer, num_steps_per_epoch=None):
    if args.scheduler == "cosine":
        scheduler = WarmupCosineLRFixMatch(
            optimizer,
            args.num_epochs,
            cos_ratio=args.cos_ratio,
            warmup_iters=args.warmup_epochs,
            num_steps_per_epoch=num_steps_per_epoch
        )
    elif args.scheduler == "linear":
        scheduler = WarmupMultiStepLR(
            optimizer,
            args.decay_epochs,
            gamma=args.decay_gamma,
            warmup_iters=args.warmup_epochs,
            num_steps_per_epoch=num_steps_per_epoch
        )
    elif args.scheduler == "none":
        scheduler = WarmupMultiStepLR(
            optimizer,
            args.decay_epochs,
            gamma=1.0,  # identity learning rates
            warmup_iters=args.warmup_epochs,
            num_steps_per_epoch=num_steps_per_epoch
        )
    else:
        print(f"{args.scheduler} scheduler is unidentified.")
        raise NotImplementedError
    return scheduler


class WarmupMultiStepLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        *,
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        num_steps_per_epoch=None
    ) -> None:
        if num_steps_per_epoch is not None:  # epoch-based to iter-based
            milestones = [m * num_steps_per_epoch for m in milestones]
            warmup_iters *= num_steps_per_epoch

        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma**bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLRFixMatch(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        cos_ratio: int = 7,
        last_epoch: int = -1,
        num_steps_per_epoch=None
    ):
        if num_steps_per_epoch is not None:  # epoch-based to iter-based
            max_iters *= num_steps_per_epoch
            warmup_iters *= num_steps_per_epoch

        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.cos_decay = float(cos_ratio) / (2 * (cos_ratio + 1)) * math.pi
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )

        return [
            base_lr * warmup_factor * math.cos(self.cos_decay * (self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:

    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
