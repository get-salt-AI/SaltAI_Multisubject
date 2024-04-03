import torch
import numpy as np
from PIL import Image


def tensor_to_pil(img_tensor, batch_index=0):
    """
    Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
    """
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def batch_tensor_to_pil(img_tensor):
    """
    Convert tensor of shape [batch_size, channels, height, width] to a list of PIL Images
    """
    return [tensor_to_pil(img_tensor, i) for i in range(img_tensor.shape[0])]


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - bbox1: List representing the first bounding box [x_min, y_min, x_max, y_max]
    - bbox2: List representing the second bounding box [x_min, y_min, x_max, y_max]

    Returns:
    - IoU: Intersection over Union between bbox1 and bbox2
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    area_bbox1 = (x2 - x1) * (y2 - y1)
    area_bbox2 = (x4 - x3) * (y4 - y3)
    union_area = area_bbox1 + area_bbox2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


def head_to_batch_dim(head_size: int, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
    """
    Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
    the number of heads initialized while constructing the `Attention` class.

    Args:
        tensor (`torch.Tensor`): The tensor to reshape.
        out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
            reshaped to `[batch_size * heads, seq_len, dim // heads]`.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    if tensor.ndim == 3:
        batch_size, seq_len, dim = tensor.shape
        extra_dim = 1
    else:
        batch_size, extra_dim, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3)

    if out_dim == 3:
        tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

    return tensor


def batch_to_head_dim(head_size: int, tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
    is the number of heads initialized while constructing the `Attention` class.

    Args:
        tensor (`torch.Tensor`): The tensor to reshape.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor
