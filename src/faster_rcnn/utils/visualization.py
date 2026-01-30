import random
from typing import List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def random_colors(N: int, bright: bool = True) -> List[tuple]:
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: tuple(int(x * 255) for x in plt.cm.hsv(c[0])[:3]), hsv))
    random.shuffle(colors)
    return colors


def denormalize_image(
    image: torch.Tensor, mean: List[float], std: List[float]
) -> torch.Tensor:
    """
    Denormalize an image tensor using mean and std.

    Args:
        image: (C, H, W) tensor
        mean: list of means for each channel
        std: list of stds for each channel

    Returns:
        (C, H, W) tensor
    """
    # Clone to avoid modifying original
    image = image.clone()

    mean = torch.as_tensor(mean, dtype=image.dtype, device=image.device)
    std = torch.as_tensor(std, dtype=image.dtype, device=image.device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    return image * std + mean


def draw_boxes(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    boxes: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
    score_thresh: float = 0.5,
    colors: Optional[List[tuple]] = None,
    line_thickness: int = 2,
    font_size: int = 12,
) -> np.ndarray:
    """
    Draw bounding boxes on image using OpenCV.

    Args:
        image: Image to draw on. Can be:
               - torch.Tensor: (C, H, W), float [0, 1] or uint8 [0, 255]
               - np.ndarray: (H, W, C), usually RGB
               - PIL.Image
        boxes: Bounding boxes [N, 4] (x1, y1, x2, y2).
        labels: Class indices [N].
        scores: Confidence scores [N].
        class_names: List of class names.
        score_thresh: Threshold to filter boxes.
        colors: List of colors for each class.
        line_thickness: Thickness of bounding box lines.

    Returns:
        np.ndarray: Image with boxes drawn (H, W, C), RGB.
    """
    # 1. Convert image to numpy uint8 RGB
    if isinstance(image, torch.Tensor):
        # Assuming (C, H, W)
        if image.ndim == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        elif image.ndim == 2:
            image = image.cpu().numpy()

        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

    elif isinstance(image, Image.Image):
        image = np.array(image)

    # Make contiguous and ensure RGB
    image = np.ascontiguousarray(image)
    if image.shape[-1] == 4:  # RGBA
        image = image[..., :3]

    # 2. Prepare data
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    num_boxes = boxes.shape[0]
    if num_boxes == 0:
        return image

    if colors is None:
        # Generate colors based on number of classes or arbitrary 80
        N = len(class_names) if class_names else 80
        colors = random_colors(N)

    # 3. Draw
    for i in range(num_boxes):
        if scores is not None and scores[i] < score_thresh:
            continue

        box = boxes[i].astype(int)
        x1, y1, x2, y2 = box

        # Get class info
        label = int(labels[i]) if labels is not None else 0
        score = scores[i] if scores is not None else 1.0

        # Choose color
        color = colors[label % len(colors)]

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # Draw label
        if class_names is not None and 0 <= label < len(class_names):
            class_name = class_names[label]
            text = f"{class_name}: {score:.2f}"
        else:
            text = f"{label}: {score:.2f}"

        # Calculate text size
        # scale font size relative to thickness
        font_scale = font_size / 30.0
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Draw text background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            color,
            -1,
        )

        # Draw text (white)
        cv2.putText(
            image,
            text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image


def plot_image(image: np.ndarray, title: Optional[str] = None):
    """Plot an image using matplotlib."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def save_image(image: np.ndarray, path: str):
    """Save image using OpenCV (expects RGB input, converts to BGR for saving)."""
    # Convert RGB to BGR for OpenCV
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
