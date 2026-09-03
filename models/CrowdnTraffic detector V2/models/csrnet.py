"""
CSRNet architecture (Li, Zhang, Chen 2018 — "CSRNet: Dilated Convolutional
Neural Networks for Understanding the Highly Congested Scenes").

This definition matches the widely-shared reference implementation
(e.g. https://github.com/leeyeehoo/CSRNet-pytorch), so pretrained .pth
checkpoints from that architecture load directly via load_state_dict.

Structure:
    frontend  -> first 10 conv layers of VGG16 (feature extractor)
    backend   -> 6 dilated conv layers (context aggregation, no downsampling)
    output    -> 1x1 conv producing a single-channel density map

The output density map is at 1/8 the input resolution (3 max-pools in the
frontend). Summing all values in the raw output approximates the object
count in the scene; this holds regardless of the map's spatial resolution
because of how density-map ground truth is generated during training.
"""

import torch
import torch.nn as nn

try:
    from torchvision import models as tv_models
except ImportError:
    tv_models = None


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    FRONTEND_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
    BACKEND_CFG = [512, 512, 512, 256, 128, 64]

    def __init__(self, load_imagenet_weights: bool = False):
        super().__init__()
        self.seen = 0
        self.frontend = make_layers(self.FRONTEND_CFG)
        self.backend = make_layers(self.BACKEND_CFG, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_imagenet_weights:
            self._init_frontend_from_vgg16()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _init_frontend_from_vgg16(self):
        """Only useful if you plan to TRAIN CSRNet from scratch. Not needed
        for inference with a pretrained CSRNet .pth checkpoint."""
        if tv_models is None:
            raise ImportError("torchvision is required to load VGG16 weights.")
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        vgg_items = list(vgg.features.state_dict().items())
        own_state = self.frontend.state_dict()
        own_keys = list(own_state.keys())
        for i in range(len(own_keys)):
            own_state[own_keys[i]] = vgg_items[i][1]
        self.frontend.load_state_dict(own_state)


def load_csrnet(weights_path: str, device: str = "cpu") -> CSRNet:
    """Load a pretrained CSRNet checkpoint for inference.

    Older CSRNet checkpoints (e.g. the commonly-shared leeyeehoo pretrained
    weights) were saved before PyTorch's newer, stricter `weights_only=True`
    default (introduced in PyTorch 2.6) existed, so they fail that check even
    though they only contain ordinary tensors. weights_only=False restores
    the old (pre-2.6) loading behavior. Only do this for checkpoints you
    trust the source of, since it can execute arbitrary code embedded in a
    malicious pickle file.
    """
    model = CSRNet()
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Some published checkpoints store {'state_dict': ...}, others store the
    # raw state_dict directly. Handle both.
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model