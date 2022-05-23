from copy import deepcopy

import torch
import torch.nn as nn


def build_model(args):
    model = _build_torchvision_model(args.model, pretrained=args.pretrained)
    classifier = nn.Linear(model.fc.in_features, args.num_classes)
    setattr(model, "fc", classifier)  # replace with the target classifier

    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def _build_torchvision_model(name, pretrained=None):
    import torchvision

    # get constructor and last layer names
    if name in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
        constructor_name = name
    else:
        raise ValueError(f'Torchvision model {name} not recognized')

    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    pretrained = pretrained == "imagenet"
    model = constructor(pretrained=pretrained, progress=True)
    return model


class EMAModel(nn.Module):

    def __init__(self, model: nn.Module, ema_decay: float):
        super().__init__()
        ema_model = deepcopy(model).cuda()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.train()

    def update(self, model):
        # parameter update
        for emp_p, p in zip(self.ema_model.parameters(), model.parameters()):
            emp_p.data = self.ema_decay * emp_p.data + (1 - self.ema_decay) * p.data

        # buffer update (i.e., running mean in BN)
        for emp_p, p in zip(self.ema_model.buffers(), model.buffers()):
            emp_p.data = self.ema_decay * emp_p.data + (1 - self.ema_decay) * p.data

    def forward(self, x, **kwargs):
        return self.ema_model(x, **kwargs)

    def train(self):
        self.ema_model.train()

    def eval(self):
        self.ema_model.eval()
