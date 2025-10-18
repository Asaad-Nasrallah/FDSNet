# FDSNet: Dynamic Multimodal Fusion Stage Selection for Autonomous Driving via Feature Disagreement Scoring
## Abstract
Robust and efficient 3D perception is critical for autonomous vehicles operating in complex environments. Multi-sensor fusion,
such as Camera+LiDAR, Camera+Radar, or all three modalities, significantly enhances scene understanding, However, most
existing frameworks fuse data at a fixed stage, categorized as early fusion (raw data level), mid fusion (intermediate feature
level), or late fusion (detection output level), neglecting semantic consistency across modalities. This static strategy may
result in performance degradation or unnecessary computation under sensor misalignment or noise. In this work, we propose
FDSNet (Feature Disagreement Score Network), a dynamic fusion framework that adaptively selects the fusion stage based
on measured semantic consistency across sensor modalities. Each sensor stream (Camera, LiDAR, Radar) independently
extracts mid-level features, which are then transformed into a common Bird’s Eye View (BEV) representation, ensuring spatial
alignment across modalities. To assess agreement, a Feature Disagreement Score (FDS) is computed at each BEV location
by measuring statistical deviation across modality features. These local scores are aggregated into a global FDS value,
which is compared against threshold to determine the fusion strategy. A low FDS, indicating strong semantic consistency
across modalities, triggers mid-level fusion for computational efficiency, whereas a high FDS value activates late fusion to
preserve detection robustness under cross-modal disagreement. We evaluate FDSNet on the nuScenes dataset across
multiple configurations: Camera+Radar, Camera+LiDAR, and Camera+Radar+LiDAR. Experimental results demonstrate that
FDSNet achieves consistent improvements over recent multimodal baselines, with gains of up to +3.0% in NDS and +2.6% in
mAP on the validation set, and +2.1% in NDS and +1.6% in mAP on the test set, highlighting that dynamic stage selection
provides both robustness and quantifiable advantages over static fusion strategies.
## Quick Start
### Prerequisites 
#### Step 0. Download and install Miniconda from the [official website.](https://www.anaconda.com/docs/main)
#### Step 1. Create a conda environment and activate it.
```
conda create --name FDSNet python=3.8 -y
conda activate FDSNet
```
#### Step 2. Install required libraries and dependencies.
- [**Python** = 3.8.20](https://www.python.org/)
- [**CUDA** = 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive)
- [**cuDNN** = 8.2](https://developer.nvidia.com/cudnn)
- [**PyTorch** = 1.10.0](https://pytorch.org/get-started/previous-versions/)
- [**TorchVision** = 0.11.1](https://pytorch.org/vision/stable/index.html)
- [**OpenCV** = 4.11.0](https://opencv.org/releases/)
- [**MMEngine** = 0.10.7](https://github.com/open-mmlab/mmengine)
- [**MMCV** = 1.4.0](https://github.com/open-mmlab/mmcv)
- [**MMDetection3D** = 1.2.0](https://github.com/open-mmlab/mmdetection3d)

### Installation
#### Step 0. Install FDSNet
```
git clone https://github.com/Asaad-Nasrallah/FDSNet.git
cd mmdetection3d-1.2.0
pip install -v -e .
```
### Data preparation
#### nuScenes
##### Step 0. Download [nuScenes Dataset]([https://www.cvlibs.net/datasets/kitti/](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/nuscenes_det.md)).
The directory will be as follows.
```
├── mmdet3d
├── tools
├── configs
├── data
│ ├── nuscenes
│ │ ├── maps
│ │ ├── samples
│ │ ├── sweeps
│ │ ├── v1.0-test
│ │ ├── v1.0-trainval
│ │ ├── nuscenes_database
│ │ ├── nuscenes_infos_train.pkl
│ │ ├── nuscenes_infos_val.pkl
│ │ ├── nuscenes_infos_test.pkl
│ │ └── nuscenes_dbinfos_train.pkl
```
### Test FDSNet (LiDAR + Radar + Camera Fusion) on the NuScenes Dataset and Evaluate mAP.
```
python tools/test.py
configs/fdsnet/fdsnet_fusion_3sensors_nuscenes.py
checkpoints/fdsnet_fusion_3sensors_nuscenes.pth
--eval mAP
```
### Benchmark on NuScenes (Multi-Sensor Fusion)

| Method | Input | NDS↑ | mAP↑ | mATE↓ | mASE↓ | mAOE↓ | mAVE↓ | mAAE↓ |
|:--------------------------|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KPConvPillars | R | 13.9 | 4.9 | 0.823 | 0.428 | 0.607 | 2.081 | 1.000 |
| CenterFusion | C+R | 44.9 | 32.6 | 0.631 | 0.261 | 0.516 | 0.614 | 0.115 |
| RCBEV | C+R | 46.0 | 40.6 | 0.484 | 0.257 | 0.587 | 0.702 | 0.140 |
| MVFusion | C+R | 51.7 | 45.3 | 0.569 | 0.246 | 0.379 | 0.781 | 0.128 |
| CRAFT | C+R | 52.3 | 41.1 | 0.467 | 0.268 | 0.456 | 0.519 | 0.118 |
| BEVFormer | C | 56.9 | 49.0 | 0.582 | 0.275 | 0.375 | 0.378 | 0.132 |
| PETRv2 | C | 58.2 | 49.0 | 0.561 | 0.243 | 0.343 | 0.343 | 0.120 |
| BEVDepth | C | 60.5 | 51.5 | 0.446 | 0.247 | 0.372 | 0.324 | 0.135 |
| SOLOFusion | C+R | 62.0 | 51.5 | 0.453 | 0.270 | 0.334 | 0.273 | 0.137 |
| CRN | C+R | 62.4 | 57.5 | 0.514 | 0.254 | 0.331 | 0.260 | 0.110 |
| SparseBEV | C | 55.6 | 48.5 | 0.448 | 0.244 | 0.332 | 0.246 | 0.112 |
| StreamPETR | C | 63.0 | 55.0 | 0.493 | 0.244 | 0.343 | 0.243 | 0.123 |
| RCBEVDet | C+R | 63.9 | 56.0 | 0.390 | 0.234 | 0.362 | 0.259 | 0.113 |
| PolarFusion | C+L | 74.5 | 74.5 | - | - | - | - | - |
| IS-Fusion | C+L | 75.2 | 73.0 | - | - | - | - | - |
| **FDSNet (Ours)** | C+R | 64.4 | 56.7 | 0.402 | 0.236 | 0.341 | 0.248 | 0.114 |
| **FDSNet (Ours)** | C+L | 76.8 | 74.9 | 0.384 | 0.227 | 0.322 | 0.233 | 0.107 |
| **FDSNet (Ours)** | C+L+R | **78.2** | **76.1** | **0.371** | **0.222** | **0.315** | **0.225** | **0.105** |
### Object Detection Results
<img src="images/Results.png" alt="Object Detection Results" width="1000" height="550">





