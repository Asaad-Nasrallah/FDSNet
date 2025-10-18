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
The code is built with following libraries:
- [**Python** = 3.8.20](https://www.python.org/)
- [**CUDA** = 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive)
- [**cuDNN** = 8.2](https://developer.nvidia.com/cudnn)
- [**PyTorch** = 1.10.0](https://pytorch.org/get-started/previous-versions/)
- [**TorchVision** = 0.11.1](https://pytorch.org/vision/stable/index.html)
- [**OpenCV** = 4.11.0](https://opencv.org/releases/)
- [**MMEngine** = 0.10.7](https://github.com/open-mmlab/mmengine)
- [**MMCV** = 1.4.0](https://github.com/open-mmlab/mmcv)
- [**MMDetection3D** = 1.2.0](https://github.com/open-mmlab/mmdetection3d)




