# Make mmengine import your package (since we use the same 'mmdet3d' namespace)
custom_imports = dict(imports=['mmdet3d'], allow_failed_imports=False)
default_scope = 'mmdet3d'

# Use LiDAR+Camera+Radar; it still runs if extra modalities are absent
input_modality = dict(use_lidar=True, use_camera=True, use_radar=True)

# Reuse your existing SECOND/PointPillars parts; only change the type and add fusion channels
model = dict(
    type='FDSNet',               # <- your registered detector
    fusion_out_channels=256,     # enables 1x1 conv after concat
    # keep your pts_voxel_encoder / pts_middle_encoder / pts_backbone / pts_neck / pts_bbox_head from a base config
)

# Point data_root to your NuScenes root
data_root = '/path/to/mmdetection3d/data/nuscenes/'
metainfo = dict(classes=[
    'car','truck','trailer','bus','construction_vehicle',
    'bicycle','motorcycle','pedestrian','traffic_cone','barrier'
])

# (you can inherit the rest from an existing MMDet3D config)
