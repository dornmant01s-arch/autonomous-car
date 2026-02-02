from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from .pipeline import OutlineConfig, convert_bytes_to_outline_png

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="outline-only",
        description="Convert images to outline-only line art (silhouette + optional occlusion lines).",
    )

    p.add_argument("input", type=str, help="Input file or folder")
    p.add_argument("-o", "--output", type=str, default="out", help="Output file or folder")

    p.add_argument("--no-occlusion", action="store_true", help="Disable inner/occlusion lines")

    p.add_argument("--sil-thick", type=int, default=4, help="Silhouette thickness")
    p.add_argument("--in-thick", type=int, default=1, help="Inner line thickness")

    p.add_argument("--eps", type=float, default=0.003, help="Silhouette simplify epsilon ratio")
    p.add_argument("--canny1", type=int, default=80)
    p.add_argument("--canny2", type=int, default=160)
    p.add_argument("--inner-min", type=float, default=0.006)
    p.add_argument("--alpha-th", type=int, default=10)
    p.add_argument("--morph", type=int, default=5)

    return p.parse_args()


def iter_images(path: Path):
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in SUPPORTED:
            yield p


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    cfg = OutlineConfig(
        silhouette_thickness=args.sil_thick,
        simplify_epsilon_ratio=args.eps,
        use_occlusion_lines=not args.no_occlusion,
        inner_thickness=args.in_thick,
        canny1=args.canny1,
        canny2=args.canny2,
        inner_min_area_ratio=args.inner_min,
        alpha_threshold=args.alpha_th,
        morph_kernel=args.morph,
    )

    imgs = list(iter_images(in_path))
    if not imgs:
        raise SystemExit("No images found.")

    # single file
    if in_path.is_file():
        if out_path.suffix.lower() == ".png":
            out_file = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / (in_path.stem + "_outline.png")

        with open(in_path, "rb") as f:
            b = f.read()

        out = convert_bytes_to_outline_png(b, cfg)
        cv2.imwrite(str(out_file), out)
        print(f"Saved: {out_file}")
        return

    # folder batch
    out_path.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(imgs, desc="Converting"):
        rel = img_path.relative_to(in_path)
        target_dir = out_path / rel.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        out_file = target_dir / (img_path.stem + "_outline.png")

        with open(img_path, "rb") as f:
            b = f.read()

        out = convert_bytes_to_outline_png(b, cfg)
        cv2.imwrite(str(out_file), out)



