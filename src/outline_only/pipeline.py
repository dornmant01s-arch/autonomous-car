from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from rembg import remove


@dataclass
class OutlineConfig:
    silhouette_thickness: int = 4
    simplify_epsilon_ratio: float = 0.003

    use_occlusion_lines: bool = True
    inner_thickness: int = 1
    canny1: int = 80
    canny2: int = 160
    inner_min_area_ratio: float = 0.006

    alpha_threshold: int = 10
    morph_kernel: int = 5


def _odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def remove_background_rgba(image_bytes: bytes) -> np.ndarray:
    cut = remove(image_bytes)
    img = Image.open(io.BytesIO(cut)).convert("RGBA")
    return np.array(img)


def extract_mask_from_alpha(rgba: np.ndarray, alpha_threshold: int) -> np.ndarray:
    alpha = rgba[:, :, 3]
    mask = (alpha > alpha_threshold).astype(np.uint8) * 255
    return mask


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def extract_inner_contours(rgba: np.ndarray, mask: np.ndarray, cfg: OutlineConfig):
    h, w = mask.shape
    area = h * w

    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.canny1, cfg.canny2)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    k = _odd(max(3, cfg.morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    inner, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = area * cfg.inner_min_area_ratio

    inners = []
    for c in inner:
        a = cv2.contourArea(c)
        L = cv2.arcLength(c, False)
        if a >= min_area and L >= 0.02 * (h + w):
            inners.append(c)

    return inners


def render_outline_png(mask, inner_contours, cfg: OutlineConfig):
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)

    # silhouette from mask boundary
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)

    if cfg.silhouette_thickness > 1:
        kk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.silhouette_thickness, cfg.silhouette_thickness),
        )
        boundary = cv2.dilate(boundary, kk)

    out[boundary > 0] = (255, 255, 255, 255)

    if cfg.use_occlusion_lines and inner_contours:
        cv2.drawContours(out, inner_contours, -1, (255, 255, 255, 255), cfg.inner_thickness)

    return out


def convert_bytes_to_outline_png(image_bytes: bytes, cfg: OutlineConfig) -> np.ndarray:
    rgba = remove_background_rgba(image_bytes)
    mask = extract_mask_from_alpha(rgba, cfg.alpha_threshold)
    mask = smooth_mask(mask)

    inner = extract_inner_contours(rgba, mask, cfg) if cfg.use_occlusion_lines else []

    out = render_outline_png(mask, inner, cfg)
    return out


