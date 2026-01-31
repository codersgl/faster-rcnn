# Faster R-CNN Implementation in PyTorch

A PyTorch implementation of Faster R-CNN for object detection on PASCAL VOC 2007 dataset. This repository provides a complete training pipeline with modern tooling for experiment management and visualization.

## Features

- End-to-end training of Region Proposal Network (RPN) and Fast R-CNN detector
- Hydra configuration system for flexible experiment management
- Real-time logging with TensorBoard (losses, metrics, bounding box visualizations)
- Modular architecture for easy extension and debugging

## Quick Start

### Installation

if you installed uv.

```
uv init
uv sync
```

### Dataset Setup

Download and extract PASCAL VOC 2007 dataset:

```bash
# Example structure after extraction
/path/to/VOCdevkit/
└── VOC2007/
    ├── Annotations/
    ├── JPEGImages/
    └── ImageSets/
```

### Training

```bash
python scripts/train.py data.root_dir=/path/to/VOCdevkit
```

or if you installed uv,
```bash
uv run scripts/train.py data.root_dir=/path/to/VOCdevkit
```

### Monitoring

```bash
tensorboard --logdir runs/
```

## Project Structure

```
src/faster_rcnn/
├── configs/          # Hydra configuration files
├── data/             # Dataset loaders and transforms
├── engine/           # Training and validation loops
├── models/           # Backbone, RPN, ROI heads
└── utils/            # Metrics, visualization, utilities
```

## Configuration

The project uses Hydra for hierarchical configuration. Key parameters can be overridden via command line:

```bash
# Example overrides
python scripts/train.py training.batch_size=8 training.optimizer.lr=0.001
```

## Evaluation

Evaluate a trained model:

```bash
python scripts/val.py +checkpoint=path/to/model.pth
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- hydra-core
- tensorboard
- opencv-python
- matplotlib
- loguru
- tqdm

## References

1. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. _Advances in Neural Information Processing Systems_, 28.

2. Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) Challenge. _International Journal of Computer Vision_, 88(2), 303-338.
