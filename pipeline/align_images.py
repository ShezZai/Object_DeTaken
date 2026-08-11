#!/usr/bin/env python3
"""Align two images and save them in the same coordinate system."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".png"
    return Path(f"{input_path.stem}_aligned{suffix}")


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def align_to_reference(
    image: np.ndarray, reference: np.ndarray, alignment_max_size: int = 800
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate alignment on small copies, then warp the full-resolution image."""
    height, width = reference.shape[:2]
    if image.shape[:2] != reference.shape[:2]:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    scale = min(1.0, alignment_max_size / max(width, height))
    alignment_width = max(1, round(width * scale))
    alignment_height = max(1, round(height * scale))
    alignment_size = (alignment_width, alignment_height)

    if alignment_size == (width, height):
        small_image = image
        small_reference = reference
    else:
        small_image = cv2.resize(image, alignment_size, interpolation=cv2.INTER_AREA)
        small_reference = cv2.resize(
            reference, alignment_size, interpolation=cv2.INTER_AREA
        )

    image_gray = (
        cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    )
    reference_gray = (
        cv2.cvtColor(small_reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
        / 255.0
    )
    small_transform = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        1000,
        1e-6,
    )

    try:
        correlation, small_transform = cv2.findTransformECC(
            reference_gray,
            image_gray,
            small_transform,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error as error:
        raise ValueError(f"OpenCV could not align the images: {error}") from error

    # Convert the thumbnail-space affine matrix back to full-resolution
    # coordinates: full_transform = inverse(scale) * small_transform * scale.
    scale_x = alignment_width / width
    scale_y = alignment_height / height
    scale_matrix = np.array(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    small_affine = np.vstack((small_transform, [0.0, 0.0, 1.0]))
    full_transform = np.linalg.inv(scale_matrix) @ small_affine @ scale_matrix

    aligned = cv2.warpAffine(
        image,
        full_transform[:2].astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    valid_mask = cv2.warpAffine(
        np.full((height, width), 255, dtype=np.uint8),
        full_transform[:2].astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    print(
        f"Alignment correlation: {correlation:.6f} "
        f"(calculated at {alignment_width}x{alignment_height})"
    )
    return aligned, valid_mask


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the largest axis-aligned rectangle containing only valid pixels."""
    height, width = mask.shape
    column_heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_box = (0, 0, width, height)

    for bottom in range(height):
        column_heights = np.where(mask[bottom] != 0, column_heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(width + 1):
            current_height = int(column_heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] > current_height:
                rectangle_start, rectangle_height = stack.pop()
                area = rectangle_height * (column - rectangle_start)
                if area > best_area:
                    best_area = area
                    best_box = (
                        rectangle_start,
                        bottom - rectangle_height + 1,
                        column,
                        bottom + 1,
                    )
                start = rectangle_start
            if not stack or stack[-1][1] < current_height:
                stack.append((start, current_height))

    if best_area == 0:
        raise ValueError("alignment produced no shared valid image area")
    return best_box


def save_rgb(image: np.ndarray, path: Path) -> None:
    result = Image.fromarray(image.astype(np.uint8), mode="RGB")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        result.save(path, format="JPEG", quality=95, subsampling=0)
    else:
        result.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align picture 1 to picture 2 and save both aligned images."
    )
    parser.add_argument("pic1", type=Path)
    parser.add_argument("pic2", type=Path)
    parser.add_argument("--output1", type=Path, help="aligned picture 1 output path")
    parser.add_argument("--output2", type=Path, help="aligned picture 2 output path")
    parser.add_argument(
        "--alignment-max-size",
        type=int,
        default=800,
        help="maximum thumbnail width or height used for alignment (default: 800)",
    )
    args = parser.parse_args()

    for path in (args.pic1, args.pic2):
        if not path.is_file():
            parser.error(f"image does not exist or is not a file: {path}")
    args.output1 = args.output1 or default_output_path(args.pic1)
    args.output2 = args.output2 or default_output_path(args.pic2)
    if args.output1.resolve() == args.output2.resolve():
        parser.error("--output1 and --output2 must be different files")
    if args.alignment_max_size < 32:
        parser.error("--alignment-max-size must be at least 32")
    return args


def main() -> None:
    args = parse_args()
    try:
        pic1 = load_rgb(args.pic1)
        pic2 = load_rgb(args.pic2)
        pic1_aligned, valid_mask = align_to_reference(
            pic1, pic2, args.alignment_max_size
        )
        left, top, right, bottom = largest_valid_rectangle(valid_mask)
        pic1_cropped = pic1_aligned[top:bottom, left:right]
        pic2_cropped = pic2[top:bottom, left:right]
        save_rgb(pic1_cropped, args.output1)
        save_rgb(pic2_cropped, args.output2)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    width, height = pic2_cropped.shape[1], pic2_cropped.shape[0]
    print(f"Shared valid crop: left={left}, top={top}, right={right}, bottom={bottom}")
    print(f"Saved aligned picture 1 crop ({width}x{height}) to {args.output1}")
    print(f"Saved aligned picture 2 crop ({width}x{height}) to {args.output2}")


if __name__ == "__main__":
    main()
