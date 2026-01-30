# Faster R-CNN Reproduction

This repository contains a PyTorch reproduction of **Faster R-CNN** (Region Proposal Network + Fast R-CNN) for object detection, specifically tailored for the PASCAL VOC 2007 dataset.

## Features

- **End-to-End Training**: Complete pipeline for training RPN and Fast R-CNN detector simultaneously.
- **Hydra Configuration**: Flexible and hierarchical experiment configuration management.
- **TensorBoard Integration**: Real-time logging of losses, learning rates, mAP scores, and **predicted bounding box visualizations**.
- **Modular Architecture**: Clean separation of data loading, model components, training engine, and utilities.

## Project Structure

```
.
├── src/faster_rcnn/
│   ├── configs/        # Hydra configuration files (model, data, training)
│   ├── data/           # Dataset wrappers, transforms, and collate functions
│   ├── engine/         # Training logic and epoch loops
│   ├── models/         # Model components (Backbone, RPN, ROI Heads)
│   └── utils/          # Metrics (mAP), visualization, box ops, losses
├── scripts/
│   ├── train.py        # Main training entry point
│   └── val.py          # Standalone validation/evaluation entry point
└── runs/               # Default output directory for experiments (managed by Hydra)
```

## Dataset

This project is configured for the **PASCAL VOC 2007** dataset.

1.  **Download** the VOC2007 Train/Val and Test data.
2.  **Extract** them. The structure should look like:
    ```
    /path/to/VOCdevkit/
    └── VOC2007/
        ├── Annotations/
        ├── JPEGImages/
        ├── ImageSets/
        └── ...
    ```
3.  **Configure**: You can either edit `src/faster_rcnn/configs/data/PascalVOC2007.yaml` or pass the path via command line arguments (see Usage below).

## Usage

### 1. Training

To start training with the default configuration:

```bash
python scripts/train.py
```

To specify the dataset root directory explicitly:

```bash
python scripts/train.py data.root_dir=/path/to/VOCdevkit
```

To run an experiment with a specific output directory (recommended for organizing runs):

```bash
python scripts/train.py hydra.run.dir=experiments/exp_vgg16_run1
```

**What happens during training?**
*   The script initializes the model (VGG16 backbone by default).
*   Logs are written to the output directory (default: `runs/YYYY-MM-DD_HH-MM-SS`).
*   TensorBoard logs are saved to the same directory.
*   Validation runs periodically (controlled by `training.val_interval`), calculating mAP and saving the best model.
*   **Sample predictions** (images with bounding boxes) are logged to TensorBoard during validation.

### 2. Visualization (TensorBoard)

To view training progress, losses, and visualized predictions:

```bash
tensorboard --logdir runs/
# or pointing to your specific experiment folder
tensorboard --logdir experiments/
```

Navigate to the **IMAGES** tab in TensorBoard to see how the model performs on validation data during training.

### 3. Evaluation

To evaluate a saved checkpoint on the validation set:

```bash
python scripts/val.py +checkpoint=path/to/model_epoch_10.pth
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration. The main config file is `src/faster_rcnn/configs/config.yaml`.

Key overrides examples:

*   **Batch Size**: `training.batch_size=8`
*   **Learning Rate**: `training.optimizer.lr=0.001`
*   **Epochs**: `training.epochs=20`
*   **Device**: `environment.device=cpu` (default is cuda)

## Project Status

- [x] **Data Loading**
    - [x] VOCDataset implementation
    - [x] Robust XML parsing & class indexing
    - [x] Data augmentation & transforms
- [x] **Model Architecture**
    - [x] VGG16 Backbone
    - [x] Region Proposal Network (RPN)
    - [x] ROI Pooling & Heads
    - [x] Prediction/Inference logic (`predict` method)
- [x] **Training Engine**
    - [x] RPN Loss (Cls + Reg) & R-CNN Loss (Cls + Reg)
    - [x] Optimizer & Scheduler setup
    - [x] TensorBoard integration
- [x] **Tools**
    - [x] mAP Calculation (VOC metric)
    - [x] Visualization utilities (drawing boxes, denormalization)
    - [x] Checkpoint saving/loading

## Requirements

*   Python 3.8+
*   PyTorch
*   Torchvision
*   Hydra-core
*   OpenCV
*   TensorBoard
*   Matplotlib
*   Loguru
*   Tqdm