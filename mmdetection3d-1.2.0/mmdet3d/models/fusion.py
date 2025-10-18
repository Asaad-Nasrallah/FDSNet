import torch
import torch.nn as nn

class SimpleFusion(nn.Module):
    """Concat (N,C,H,W) features -> 1x1 conv -> BN -> ReLU."""
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.proj = None
        self.bn = None
        self.act = nn.ReLU(inplace=True)

    def _build(self, in_channels):
        self.proj = nn.Conv2d(in_channels, self.out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, *feats):
        feats = [f for f in feats if f is not None]
        if not feats:
            return None
        x = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        if self.proj is None:
            self._build(x.shape[1])
        return self.act(self.bn(self.proj(x)))

