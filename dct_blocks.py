import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RegionDCTResult:
    coefficients: np.ndarray
    block_mask: np.ndarray
    bbox: tuple
    padded_shape: tuple
    block_size: int


def build_region_mask(parsing_anno, target_classes):
    """根据指定的人脸解析类别生成二值区域 mask。"""
    return np.isin(parsing_anno, list(target_classes)).astype(np.uint8)


def _mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _pad_to_block_size(array, block_size):
    height, width = array.shape[:2]
    padded_height = int(np.ceil(height / block_size) * block_size)
    padded_width = int(np.ceil(width / block_size) * block_size)

    if array.ndim == 2:
        padded = np.zeros((padded_height, padded_width), dtype=array.dtype)
        padded[:height, :width] = array
    else:
        padded = np.zeros((padded_height, padded_width, array.shape[2]), dtype=array.dtype)
        padded[:height, :width, :] = array
    return padded


def split_into_blocks(array, block_size=8):
    """把已补齐的二维数组切分为 (rows, cols, block_size, block_size)。"""
    height, width = array.shape
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError("array height and width must be divisible by block_size")

    return array.reshape(
        height // block_size,
        block_size,
        width // block_size,
        block_size,
    ).swapaxes(1, 2)


def dct_transform_blocks(blocks, center=True):
    """对每个 8x8 分块执行 OpenCV DCT。"""
    coefficients = np.empty(blocks.shape, dtype=np.float32)
    for row in range(blocks.shape[0]):
        for col in range(blocks.shape[1]):
            block = blocks[row, col].astype(np.float32)
            if center:
                block = block - 128.0
            coefficients[row, col] = cv2.dct(block)
    return coefficients


def compute_region_dct(image_rgb, parsing_anno, target_classes, block_size=8, center=True):
    """
    提取指定人脸区域，按 8x8 分块后计算 DCT。

    bbox 内非目标区域会置为 0；返回的 block_mask 标记至少包含一个目标像素的分块。
    """
    image_rgb = np.asarray(image_rgb)
    parsing_anno = np.asarray(parsing_anno)
    if image_rgb.shape[:2] != parsing_anno.shape[:2]:
        raise ValueError("image and parsing annotation must have the same height and width")

    mask = build_region_mask(parsing_anno, target_classes)
    bbox = _mask_bbox(mask)
    if bbox is None:
        empty_blocks = np.empty((0, 0, block_size, block_size), dtype=np.float32)
        return RegionDCTResult(empty_blocks, np.empty((0, 0), dtype=bool), None, (0, 0), block_size)

    x0, y0, x1, y1 = bbox
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    region_gray = np.where(mask > 0, gray, 0).astype(np.float32)
    cropped_region = region_gray[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]

    padded_region = _pad_to_block_size(cropped_region, block_size)
    padded_mask = _pad_to_block_size(cropped_mask, block_size)
    blocks = split_into_blocks(padded_region, block_size)
    mask_blocks = split_into_blocks(padded_mask, block_size)
    block_mask = np.any(mask_blocks > 0, axis=(2, 3))
    coefficients = dct_transform_blocks(blocks, center=center)

    return RegionDCTResult(
        coefficients=coefficients,
        block_mask=block_mask,
        bbox=bbox,
        padded_shape=padded_region.shape[:2],
        block_size=block_size,
    )


def _dct_magnitude_image(coefficients):
    if coefficients.size == 0:
        return None
    rows, cols, block_h, block_w = coefficients.shape
    magnitude = np.log1p(np.abs(coefficients)).swapaxes(1, 2).reshape(rows * block_h, cols * block_w)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _draw_block_grid(image_rgb, bbox, block_size):
    grid_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if bbox is None:
        return grid_image

    x0, y0, x1, y1 = bbox
    cv2.rectangle(grid_image, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1)
    for x in range(x0, x1, block_size):
        cv2.line(grid_image, (x, y0), (x, y1 - 1), (0, 255, 255), 1)
    for y in range(y0, y1, block_size):
        cv2.line(grid_image, (x0, y), (x1 - 1, y), (0, 255, 255), 1)
    return grid_image


def save_region_dct_outputs(image_rgb, parsing_anno, target_classes, save_path, block_size=8):
    """
    在 save_path 同目录保存 DCT 系数和轻量可视化结果。

    输出文件：
    - *_dct_blocks.npz：DCT 系数、分块 mask、bbox、补齐后尺寸。
    - *_blocks.jpg：关键区域 bbox 和 8x8 分块可视化。
    - *_dct.jpg：归一化后的 DCT 系数幅值可视化。
    """
    result = compute_region_dct(image_rgb, parsing_anno, target_classes, block_size=block_size)

    root, _ = os.path.splitext(save_path)
    np.savez_compressed(
        root + "_dct_blocks.npz",
        coefficients=result.coefficients,
        block_mask=result.block_mask,
        bbox=np.array(result.bbox if result.bbox is not None else [], dtype=np.int32),
        padded_shape=np.array(result.padded_shape, dtype=np.int32),
        block_size=np.array(result.block_size, dtype=np.int32),
    )

    grid_image = _draw_block_grid(image_rgb, result.bbox, result.block_size)
    cv2.imwrite(root + "_blocks.jpg", grid_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    dct_image = _dct_magnitude_image(result.coefficients)
    if dct_image is not None:
        cv2.imwrite(root + "_dct.jpg", dct_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return result
