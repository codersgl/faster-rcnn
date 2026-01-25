import torch


def encode_boxes(gt_boxes, anchor_boxes, eps=1e-6):
    """Encode bounding boxes using anchor boxes.

    Args:
        gt_boxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4) or (batch_size, N, 4).
        anchor_boxes (torch.Tensor): Anchor boxes with shape (N, 4) or (batch_size, N, 4).
        eps (float): Small value to avoid division by zero and log(0).

    Returns:
        torch.Tensor: Encoded bounding boxes with same shape as inputs.
    """
    assert gt_boxes.shape == anchor_boxes.shape, (
        f"Shape mismatch: {gt_boxes.shape} vs {anchor_boxes.shape}"
    )

    gt_x_center = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2.0
    gt_y_center = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2.0
    gt_w = gt_boxes[..., 2] - gt_boxes[..., 0] + eps
    gt_h = gt_boxes[..., 3] - gt_boxes[..., 1] + eps

    anchor_x_center = (anchor_boxes[..., 0] + anchor_boxes[..., 2]) / 2.0
    anchor_y_center = (anchor_boxes[..., 1] + anchor_boxes[..., 3]) / 2.0
    anchor_w = anchor_boxes[..., 2] - anchor_boxes[..., 0] + eps
    anchor_h = anchor_boxes[..., 3] - anchor_boxes[..., 1] + eps

    # 计算编码值
    # t_x = (gt_x_center - anchor_x_center) / anchor_w
    # t_y = (gt_y_center - anchor_y_center) / anchor_h
    # t_w = log(gt_w / anchor_w)
    # t_h = log(gt_h / anchor_h)
    t_x = (gt_x_center - anchor_x_center) / anchor_w
    t_y = (gt_y_center - anchor_y_center) / anchor_h
    t_w = torch.log(gt_w / anchor_w)
    t_h = torch.log(gt_h / anchor_h)

    # 返回编码后的框
    if gt_boxes.dim() == 2:
        return torch.stack([t_x, t_y, t_w, t_h], dim=1)
    else:
        return torch.stack([t_x, t_y, t_w, t_h], dim=-1)


def decode_boxes(encoded_boxes, anchor_boxes, eps=1e-6):
    """Decode encoded bounding boxes using anchor boxes.

    Args:
        encoded_boxes (torch.Tensor): Encoded bounding boxes with shape (N, 4) or (batch_size, N, 4).
        anchor_boxes (torch.Tensor): Anchor boxes with shape (N, 4) or (batch_size, N, 4).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Decoded bounding boxes with same shape as inputs.
    """
    assert encoded_boxes.shape == anchor_boxes.shape, (
        f"Shape mismatch: {encoded_boxes.shape} vs {anchor_boxes.shape}"
    )

    if encoded_boxes.dim() == 2:
        t_x = encoded_boxes[:, 0]
        t_y = encoded_boxes[:, 1]
        t_w = encoded_boxes[:, 2]
        t_h = encoded_boxes[:, 3]
    else:
        t_x = encoded_boxes[..., 0]
        t_y = encoded_boxes[..., 1]
        t_w = encoded_boxes[..., 2]
        t_h = encoded_boxes[..., 3]

    anchor_x_center = (anchor_boxes[..., 0] + anchor_boxes[..., 2]) / 2.0
    anchor_y_center = (anchor_boxes[..., 1] + anchor_boxes[..., 3]) / 2.0
    anchor_w = anchor_boxes[..., 2] - anchor_boxes[..., 0] + eps
    anchor_h = anchor_boxes[..., 3] - anchor_boxes[..., 1] + eps

    # 计算解码后的中心点和宽高
    # x_center = t_x * anchor_w + anchor_x_center
    # y_center = t_y * anchor_h + anchor_y_center
    # w = exp(t_w) * anchor_w
    # h = exp(t_h) * anchor_h
    x_center = t_x * anchor_w + anchor_x_center
    y_center = t_y * anchor_h + anchor_y_center
    w = torch.exp(t_w) * anchor_w
    h = torch.exp(t_h) * anchor_h

    x1 = x_center - w / 2.0
    y1 = y_center - h / 2.0
    x2 = x_center + w / 2.0
    y2 = y_center + h / 2.0

    if encoded_boxes.dim() == 2:
        return torch.stack([x1, y1, x2, y2], dim=1)
    else:
        return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.types.Tensor, boxes2: torch.types.Tensor):
    """Compute the iou between box1 and box2
    Args:
        boxes1(torch.types.Tensor): (N, 4)[x1, y1, x2, y2]
        boxes2(torch.types.Tensor): (M, 4)[x1, y1, x2, y2]
    Return:
        iou(torch.types.Tensor): (N, M) IoU Matrix

    """
    assert boxes1.device == boxes2.device, "Must be in same device"

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[1])  # [N,]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[1])  # [M,]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M,]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)  # [N, M]

    return iou
