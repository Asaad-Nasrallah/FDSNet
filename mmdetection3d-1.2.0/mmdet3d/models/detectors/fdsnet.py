from typing import Dict, Optional, List, Tuple
import torch
from torch import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.models import MVXFasterRCNN
from ..fusion import SimpleFusion

@MODELS.register_module()
class FDSNet(MVXFasterRCNN):
    """Drop-in detector for MMDetection3D.
    - Works as LiDAR-only baseline out of the box.
    - If batch dict provides 'radar_bev' and/or 'img_bev' (N,C,H,W),
      fuses them with the deepest LiDAR BEV before the neck.
    """
    def __init__(self, *args, fusion_out_channels: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion = SimpleFusion(fusion_out_channels) if fusion_out_channels else None

    def extract_feat(self, batch_inputs_dict: Dict) -> Tuple[List[Tensor], Optional[List[Tensor]]]:
        x = self.extract_feat_pts(batch_inputs_dict)  # Tensor or List[Tensor]
        lidar_feats = [x] if isinstance(x, torch.Tensor) else list(x)

        radar_bev: Optional[Tensor] = batch_inputs_dict.get('radar_bev', None)
        img_bev:   Optional[Tensor] = batch_inputs_dict.get('img_bev', None)

        if self.fusion is None or (radar_bev is None and img_bev is None):
            feats = self.pts_neck(lidar_feats) if self.with_pts_neck else lidar_feats
            return feats, None

        base = lidar_feats[-1]
        extras = []
        for feat in (radar_bev, img_bev):
            if feat is None:
                extras.append(None); continue
            if feat.shape[-2:] != base.shape[-2:]:
                feat = torch.nn.functional.interpolate(feat, base.shape[-2:], mode='nearest')
            extras.append(feat)

        fused = self.fusion(base, *extras)
        lidar_feats[-1] = fused
        feats = self.pts_neck(lidar_feats) if self.with_pts_neck else lidar_feats
        return feats, None
