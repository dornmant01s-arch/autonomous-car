
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from rembg import remove


@dataclass
class OutlineConfig:
    # silhouette
    silhouette_thickness: int = 4
    simplify_epsilon_ratio: float = 0.003  # bigger => simpler

    # occlusion / inner edges
    use_occlusion_lines: bool = True
    inner_thickness: int = 1
    canny1: int = 80
    canny2: int = 160
    inner_min_area_ratio: float = 0.003  # remove tiny inner fragments
    morph_kernel: int = 5  # odd recommended

    # mask threshold
    alpha_threshold: int = 10


def _ensure_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def remove_background_rgba(image_bytes: bytes) -> np.ndarray:
    """
    Returns RGBA image as numpy array after background removal.
    """
    cut = remove(image_bytes)
    img = Image.open(io.BytesIO(cut)).convert("RGBA")
    return np.array(img)


def extract_mask_from_alpha(rgba: np.ndarray, alpha_threshold: int) -> np.ndarray:
    alpha = rgba[:, :, 3]
    mask = (alpha > alpha_threshold).astype(np.uint8) * 255
    return mask

def smooth_mask(mask: np.ndarray, close_k: int = 9, open_k: int = 5, blur: int = 5) -> np.ndarray:
    # close: 찢김/구멍 메우기, open: 잔털 제거
    close_k = max(3, close_k) | 1
    open_k = max(3, open_k) | 1
    blur = max(3, blur) | 1

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))

    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k_open,  iterations=1)
    m = cv2.GaussianBlur(m, (blur, blur), 0)

    # 다시 이진화(경계 부드럽게)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def find_main_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contour found. Try a different image or lower alpha_threshold.")
    return max(contours, key=cv2.contourArea)


def simplify_contour(contour: np.ndarray, epsilon_ratio: float) -> np.ndarray:
    eps = float(epsilon_ratio) * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, eps, True)


def extract_inner_contours(
    rgba: np.ndarray,
    mask: np.ndarray,
    cfg: OutlineConfig,
) -> list[np.ndarray]:
    h, w = mask.shape[:2]
    area = h * w

    # Use RGB from RGBA
    rgb = rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, cfg.canny1, cfg.canny2)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    k = _ensure_odd(max(3, int(cfg.morph_kernel)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    inner, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = area * float(cfg.inner_min_area_ratio)
    inners = [for c in inner:
    a = cv2.contourArea(c)
    L = cv2.arcLength(c, False)
    if a >= min_area and L >= 0.02 * (h + w):  # 길이도 기준
        inners.append(c)]
    
    return inners


def render_outline_png(mask, silhouette_contour, inner_contours, cfg):
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)

    # silhouette: mask boundary using morphological gradient
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)  # 경계만 255

    # 두께 만들기(팽창)
    if cfg.silhouette_thickness > 1:
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.silhouette_thickness, cfg.silhouette_thickness))
        boundary = cv2.dilate(boundary, kk, iterations=1)

    out[boundary > 0] = (255, 255, 255, 255)

    # inner lines
    if cfg.use_occlusion_lines and inner_contours:
        cv2.drawContours(out, inner_contours, -1, (255,255,255,255), cfg.inner_thickness)

    return out



def convert_bytes_to_outline_png(image_bytes: bytes, cfg: OutlineConfig) -> np.ndarray:
    rgba = remove_background_rgba(image_bytes)
    mask = extract_mask_from_alpha(rgba, cfg.alpha_threshold)
    mask = smooth_mask(mask, close_k=11, open_k=5, blur=5)

    main = find_main_contour(mask)
    main = simplify_contour(main, cfg.simplify_epsilon_ratio)

    inner = extract_inner_contours(rgba, mask, cfg) if cfg.use_occlusion_lines else []
    out = render_outline_png(mask, main, inner, cfg)
    return out

