"""Small latent encoders. F/K channel adapters remain part of the loss encoder."""
import torch
from torch import nn
from torch.nn import functional as F


STAGES = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')


def _groups(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(outgoing), outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(outgoing), outgoing))
        self.skip = (nn.Identity() if incoming == outgoing and stride == 1 else
                     nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False))

    def forward(self, x):
        return F.silu(self.main(x) + self.skip(x))


class CompactEncoder(nn.Module):
    """BCHW four-channel latent -> five BCHW feature maps.

    GroupNorm is per image; changing reference-bank/batch partition cannot change
    a sample's features through BatchNorm statistics. With teacher_channels, the
    retained 1x1 heads are identical in the feature and geometry arms. Their
    forward/backward time and output sizes must be counted in benchmarks.
    """
    def __init__(self, width=64, blocks=2, teacher_channels=None):
        super().__init__()
        if not isinstance(width, int) or width < 1 or not isinstance(blocks, int) or blocks < 1:
            raise ValueError('Positive integer width and blocks required')
        self.width, self.blocks = width, blocks
        self._backbone_channels = dict(zip(STAGES, (width, width, 2*width, 4*width, 8*width)))
        if teacher_channels is not None:
            if set(teacher_channels) != set(STAGES) or any(
                    not isinstance(v, int) or v < 1 for v in teacher_channels.values()):
                raise ValueError('teacher_channels must give positive channels for all five stages')
        self._channels = dict(teacher_channels or self._backbone_channels)
        self.stem = nn.Sequential(nn.Conv2d(4, width, 3, padding=1, bias=False),
                                  nn.GroupNorm(_groups(width), width), nn.SiLU())
        self.stages = nn.ModuleList()
        incoming = width
        for index, outgoing in enumerate((width, 2*width, 4*width, 8*width)):
            layers = [ResidualBlock(incoming, outgoing, 1 if index == 0 else 2)]
            layers += [ResidualBlock(outgoing, outgoing) for _ in range(blocks - 1)]
            self.stages.append(nn.Sequential(*layers))
            incoming = outgoing
        self.adapters = nn.ModuleDict({
            name: nn.Identity() if teacher_channels is None else
            nn.Conv2d(self._backbone_channels[name], self._channels[name], 1)
            for name in STAGES})

    @property
    def channels(self):
        return dict(self._channels)

    @property
    def backbone_channels(self):
        return dict(self._backbone_channels)

    def raw_features(self, x):
        if x.ndim != 4 or x.shape[1] != 4 or min(x.shape[-2:]) < 8:
            raise ValueError('Expected BCHW four-channel latents with spatial size at least 8')
        x = self.stem(x)
        result = {'conv1': x}
        for index, stage in enumerate(self.stages, 1):
            x = stage(x)
            result[f'layer{index}'] = x
        return result

    def forward(self, x):
        return {key: self.adapters[key](value) for key, value in self.raw_features(x).items()}


def collect_features(maps, positions, target_sizes=None):
    """Expose spatial samples and global moments as [position,sample,channel].

    The SAME position indices are applied to all episode samples. Global moments
    are computed over the full map (never only the selected spatial positions).
    Optional resizing defines the map before both sampling and moment extraction.
    """
    if not maps or set(maps) != set(positions):
        raise ValueError('One position-index tensor is required for each feature map')
    if target_sizes is not None and set(target_sizes) != set(maps):
        raise ValueError('target_sizes must cover every map')
    result = {}
    for name, value in maps.items():
        if value.ndim != 4:
            raise ValueError('Feature maps must be BCHW')
        if target_sizes is not None:
            size = tuple(target_sizes[name])
            if len(size) != 2 or min(size) < 1:
                raise ValueError('Invalid target spatial grid')
            if tuple(value.shape[-2:]) != size:
                value = F.interpolate(value, size=size, mode='bilinear', align_corners=False)
        indices = torch.as_tensor(positions[name], device=value.device)
        if indices.ndim != 1 or indices.numel() < 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('Spatial positions must be a nonempty one-dimensional integer tensor')
        if torch.any(indices < 0) or torch.any(indices >= value.shape[-2]*value.shape[-1]):
            raise ValueError('Spatial position outside feature grid')
        flat = value.float().flatten(2)  # N,C,H*W
        result[f'{name}/spatial'] = flat.index_select(2, indices.long()).permute(2, 0, 1)
        mean = flat.mean(-1)
        result[f'{name}/mean'] = mean.unsqueeze(0)
        result[f'{name}/std'] = ((flat - mean[..., None]).square().mean(-1) + 1e-6).sqrt().unsqueeze(0)
    return result


class _Up(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.project = nn.Conv2d(incoming, outgoing, 1)
        self.refine = ResidualBlock(outgoing + skip, outgoing)

    def forward(self, x, skip):
        x = self.project(F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False))
        return self.refine(torch.cat((x, skip), 1))


class ReconstructionModel(nn.Module):
    """Masked-reconstruction control; only .encoder survives generator integration."""
    def __init__(self, width=64, blocks=2):
        super().__init__()
        self.encoder = CompactEncoder(width=width, blocks=blocks)
        self.decoder = nn.ModuleList((_Up(8*width, 4*width, 4*width),
                                      _Up(4*width, 2*width, 2*width),
                                      _Up(2*width, width, width),
                                      _Up(width, width, width)))
        self.output = nn.Conv2d(width, 4, 1)

    def forward(self, x):
        maps = self.encoder(x)
        value = maps['layer4']
        for layer, skip in zip(self.decoder, ('layer3', 'layer2', 'layer1', 'conv1')):
            value = layer(value, maps[skip])
        return self.output(value)


def masked_reconstruction_loss(model, x, mask_ratio=.5, patch_size=4,
                               generator=None, return_details=False):
    """Per-image MSE on masked coordinates only; patches are contiguous in latents."""
    if x.ndim != 4 or x.shape[1] != 4 or not 0 < mask_ratio <= 1:
        raise ValueError('Expected BCHW four-channel latents and 0 < mask_ratio <= 1')
    if not isinstance(patch_size, int) or patch_size < 1 or any(d % patch_size for d in x.shape[-2:]):
        raise ValueError('Patch size must divide both spatial dimensions')
    b, channels, h, w = x.shape
    mask = torch.rand((b, 1, h//patch_size, w//patch_size), device=x.device,
                      generator=generator) < mask_ratio
    # An empty random mask carries no reconstruction signal. Force one patch
    # in such a sample, without modifying the caller's latent tensor.
    empty = ~mask.flatten(1).any(1)
    mask[empty, 0, 0, 0] = True
    mask = mask.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
    prediction = model(x * (~mask))
    if prediction.shape != x.shape:
        raise ValueError('Reconstruction shape differs from input')
    loss = (((prediction.float() - x.float()).square() * mask).sum((1,2,3)) /
            (mask.sum((1,2,3)) * channels)).mean()
    if return_details:
        return loss, {'masked_fraction': mask.float().mean().detach()}
    return loss
