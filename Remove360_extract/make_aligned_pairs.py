#!/usr/bin/env python3
"""Pair before/after Remove360 images whose viewpoints overlap 90%+.

Walks every <scene>/<object> folder containing train/ (before) and test/
(after) images. Each after image is matched against the before images with
SIFT features; candidate pairs are aligned with a RANSAC homography and
accepted when the two frames cover at least the overlap threshold of each
other and the warped content agrees photometrically. Accepted pairs are
cropped to their shared valid rectangle and written per object as
<output>/scene/<scene>/<object>/before_<n>.jpg / after_<n>.jpg (numbering
restarts at 0 in each object folder), with a pairs.csv manifest at the output
root describing each pair's source files and scores.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

# Roughly a fifth of Remove360's images were truncated by a few hundred bytes
# when the author uploaded them (all cut at exactly 2.75 MiB); the loss is a
# few bottom-edge pixels, so decode them instead of failing.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class Features(NamedTuple):
    gray: np.ndarray
    keypoints: tuple
    descriptors: np.ndarray | None


class Match(NamedTuple):
    before_path: Path
    homography: np.ndarray
    overlap: float
    inliers: int
    similarity: float


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def load_rgb_or_warn(path: Path) -> np.ndarray | None:
    try:
        return load_rgb(path)
    except OSError as error:
        print(f"warning: skipping unreadable image {path}: {error}")
        return None


def valid_content_crop(
    image: np.ndarray, minimum_fraction: float = 0.15
) -> np.ndarray | None:
    """Crop off the uniform gray tail that truncated JPEGs decode to.

    Returns None when less than minimum_fraction of the image holds real
    content. Intact images are returned unchanged.
    """
    height = image.shape[0]
    probe = cv2.cvtColor(
        cv2.resize(image, (512, min(512, height)), interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2GRAY,
    )
    row_variance = probe.astype(np.float32).var(axis=1)
    dead = row_variance < 2.0
    trailing_dead = 0
    for is_dead in dead[::-1]:
        if not is_dead:
            break
        trailing_dead += 1
    if trailing_dead == 0:
        return image
    valid_fraction = 1.0 - trailing_dead / len(dead)
    # Trim a small safety margin: the boundary row decodes only partially.
    valid_end = round(valid_fraction * height) - max(8, height // 200)
    if valid_fraction < minimum_fraction or valid_end < 64:
        return None
    return image[:valid_end]


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def find_object_directories(root: Path) -> list[Path]:
    """Return directories that contain both train/ and test/ image folders."""
    found = []
    for train_directory in sorted(root.rglob("train")):
        object_directory = train_directory.parent
        if (object_directory / "test").is_dir():
            found.append(object_directory)
    return found


def thumbnail_gray(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def extract_features(sift, image: np.ndarray, max_size: int) -> Features:
    gray = thumbnail_gray(image, max_size)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return Features(gray, keypoints, descriptors)


def good_matches(matcher, first: Features, second: Features) -> list:
    """Lowe-ratio-filtered descriptor matches from first to second."""
    if first.descriptors is None or second.descriptors is None:
        return []
    if len(first.descriptors) < 2 or len(second.descriptors) < 2:
        return []
    pairs = matcher.knnMatch(first.descriptors, second.descriptors, k=2)
    return [m for m, n in pairs if m.distance < 0.75 * n.distance]


def frame_coverage(
    homography: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> float:
    """Fraction of the target frame covered by the warped source frame."""
    height, width = source_shape
    mask = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        homography,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    return float(np.count_nonzero(mask)) / mask.size


def masked_similarity(
    warped: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    """Zero-mean normalized correlation over the valid (mask) pixels."""
    valid = mask != 0
    if np.count_nonzero(valid) < 100:
        return 0.0
    first = warped[valid].astype(np.float32)
    second = target[valid].astype(np.float32)
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def evaluate_candidate(
    before: Features,
    after: Features,
    matches: list,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, int, float] | None:
    """RANSAC a homography and score it; None when the pair is unusable."""
    if len(matches) < args.min_inliers:
        return None
    source = np.float32(
        [before.keypoints[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    target = np.float32(
        [after.keypoints[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        source, target, cv2.RANSAC, 4.0
    )
    if homography is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.sum())
    if inliers < args.min_inliers:
        return None

    forward = frame_coverage(homography, before.gray.shape, after.gray.shape)
    backward = frame_coverage(
        np.linalg.inv(homography), after.gray.shape, before.gray.shape
    )
    overlap = min(forward, backward)
    if overlap < args.overlap:
        return None

    warped = cv2.warpPerspective(
        before.gray,
        homography,
        (after.gray.shape[1], after.gray.shape[0]),
        flags=cv2.INTER_LINEAR,
    )
    valid_mask = cv2.warpPerspective(
        np.full(before.gray.shape, 255, dtype=np.uint8),
        homography,
        (after.gray.shape[1], after.gray.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    similarity = masked_similarity(warped, after.gray, valid_mask)
    if similarity < args.min_similarity:
        return None
    return homography, overlap, inliers, similarity


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned rectangle containing only valid (non-zero) pixels."""
    height, width = mask.shape
    column_heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_box = (0, 0, 0, 0)

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
        raise ValueError("no valid area shared by the aligned pair")
    return best_box


def scale_homography_to_full(
    homography: np.ndarray,
    before_small: tuple[int, int],
    before_full: tuple[int, int],
    after_small: tuple[int, int],
    after_full: tuple[int, int],
) -> np.ndarray:
    """Convert a thumbnail-space homography to full-resolution coordinates."""

    def scaling(small: tuple[int, int], full: tuple[int, int]) -> np.ndarray:
        return np.array(
            [
                [small[1] / full[1], 0.0, 0.0],
                [0.0, small[0] / full[0], 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    return np.linalg.inv(scaling(after_small, after_full)) @ homography @ scaling(
        before_small, before_full
    )


def save_jpeg(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image, mode="RGB").save(
        path, format="JPEG", quality=95, subsampling=0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="dataset root containing <scene>/<object>/train|test (default: script folder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output folder for aligned pairs (default: <root>/aligned)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.90,
        help="minimum mutual frame coverage to accept a pair (default: 0.90)",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=40,
        help="minimum RANSAC inlier matches (default: 40)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.45,
        help="minimum photometric correlation of the aligned pair (default: 0.45)",
    )
    parser.add_argument(
        "--shortlist",
        type=int,
        default=3,
        help="before-image candidates tried per after image (default: 3)",
    )
    parser.add_argument(
        "--alignment-max-size",
        type=int,
        default=800,
        help="thumbnail size used for matching and alignment (default: 800)",
    )
    parser.add_argument(
        "--sift-features",
        type=int,
        default=800,
        help="maximum SIFT features per image (default: 800)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only process the first N after images per object (for testing)",
    )
    parser.add_argument(
        "--objects",
        default=None,
        help=(
            "comma-separated object paths relative to the root (e.g. "
            "backyard/playhouse); other objects' existing pairs are kept"
        ),
    )
    args = parser.parse_args()

    if not 0.0 < args.overlap <= 1.0:
        parser.error("--overlap must be between 0 and 1")
    if args.shortlist < 1:
        parser.error("--shortlist must be at least 1")
    if args.alignment_max_size < 64:
        parser.error("--alignment-max-size must be at least 64")
    args.output = args.output or args.root / "aligned"
    return args


def match_object(
    object_directory: Path,
    args: argparse.Namespace,
    manifest: csv.writer,
) -> int:
    before_paths = list_images(object_directory / "train")
    after_paths = list_images(object_directory / "test")
    if args.limit is not None:
        after_paths = after_paths[: args.limit]
    if not before_paths or not after_paths:
        return 0

    relative = object_directory.relative_to(args.root)
    pair_directory = args.output / "scene" / relative
    pair_directory.mkdir(parents=True, exist_ok=True)
    print(f"{relative}: {len(before_paths)} before, {len(after_paths)} after")

    sift = cv2.SIFT_create(nfeatures=args.sift_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    before_features = {}
    unusable = 0
    for path in before_paths:
        image = load_rgb_or_warn(path)
        image = valid_content_crop(image) if image is not None else None
        if image is None:
            unusable += 1
            continue
        before_features[path] = extract_features(
            sift, image, args.alignment_max_size
        )
    before_paths = [path for path in before_paths if path in before_features]

    used_before: set[Path] = set()
    accepted = 0
    for after_path in after_paths:
        after_full = load_rgb_or_warn(after_path)
        after_full = valid_content_crop(after_full) if after_full is not None else None
        if after_full is None:
            unusable += 1
            continue
        after = extract_features(sift, after_full, args.alignment_max_size)

        # Rank candidates by ratio-test match count, then verify the top few.
        scored = []
        for before_path in before_paths:
            if before_path in used_before:
                continue
            matches = good_matches(
                matcher, before_features[before_path], after
            )
            if len(matches) >= args.min_inliers:
                scored.append((len(matches), before_path, matches))
        scored.sort(key=lambda item: item[0], reverse=True)

        best: Match | None = None
        for _, before_path, matches in scored[: args.shortlist]:
            evaluation = evaluate_candidate(
                before_features[before_path], after, matches, args
            )
            if evaluation is None:
                continue
            homography, overlap, inliers, similarity = evaluation
            if best is None or overlap > best.overlap:
                best = Match(before_path, homography, overlap, inliers, similarity)

        if best is None:
            continue

        before_full = valid_content_crop(load_rgb(best.before_path))
        if before_full is None:
            continue
        before_small_shape = before_features[best.before_path].gray.shape
        full_homography = scale_homography_to_full(
            best.homography,
            before_small_shape,
            before_full.shape[:2],
            after.gray.shape,
            after_full.shape[:2],
        )
        before_warped = cv2.warpPerspective(
            before_full,
            full_homography,
            (after_full.shape[1], after_full.shape[0]),
            flags=cv2.INTER_LINEAR,
        )

        # Find the shared valid rectangle on the small mask, then scale it up
        # with a one-pixel inward margin against rounding artifacts.
        small_mask = cv2.warpPerspective(
            np.full(before_small_shape, 255, dtype=np.uint8),
            best.homography,
            (after.gray.shape[1], after.gray.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        left, top, right, bottom = largest_valid_rectangle(small_mask)
        scale_x = after_full.shape[1] / after.gray.shape[1]
        scale_y = after_full.shape[0] / after.gray.shape[0]
        left = min(after_full.shape[1] - 1, round(left * scale_x) + 1)
        top = min(after_full.shape[0] - 1, round(top * scale_y) + 1)
        right = max(left + 1, round(right * scale_x) - 1)
        bottom = max(top + 1, round(bottom * scale_y) - 1)

        save_jpeg(
            before_warped[top:bottom, left:right],
            pair_directory / f"before_{accepted}.jpg",
        )
        save_jpeg(
            after_full[top:bottom, left:right],
            pair_directory / f"after_{accepted}.jpg",
        )
        manifest.writerow(
            [
                accepted,
                str(relative),
                best.before_path.name,
                after_path.name,
                f"{best.overlap:.4f}",
                best.inliers,
                f"{best.similarity:.4f}",
            ]
        )
        used_before.add(best.before_path)
        accepted += 1

    if unusable:
        print(f"{relative}: {unusable} images skipped as mostly truncated")
    print(f"{relative}: accepted {accepted}/{len(after_paths)} pairs")
    return accepted


def main() -> None:
    args = parse_args()
    object_directories = find_object_directories(args.root)
    if args.objects is not None:
        wanted = {name.strip().strip("/") for name in args.objects.split(",")}
        object_directories = [
            directory
            for directory in object_directories
            if str(directory.relative_to(args.root)) in wanted
        ]
        missing = wanted - {
            str(directory.relative_to(args.root))
            for directory in object_directories
        }
        if missing:
            raise SystemExit(f"error: objects not found: {', '.join(sorted(missing))}")
    if not object_directories:
        raise SystemExit(f"error: no <object>/train + test folders under {args.root}")

    args.output.mkdir(parents=True, exist_ok=True)
    selected = {
        str(directory.relative_to(args.root)) for directory in object_directories
    }
    manifest_path = args.output / "pairs.csv"

    # A filtered run keeps other objects' pairs and manifest rows; a full run
    # starts clean.
    kept_rows = []
    if args.objects is not None and manifest_path.is_file():
        with open(manifest_path, newline="") as manifest_file:
            kept_rows = [
                row
                for row in list(csv.reader(manifest_file))[1:]
                if row and row[1] not in selected
            ]
    if args.objects is None:
        for pattern in ("before_*.jpg", "after_*.jpg"):
            for stale in args.output.rglob(pattern):
                stale.unlink()
    else:
        for relative in selected:
            for pattern in ("before_*.jpg", "after_*.jpg"):
                for stale in (args.output / "scene" / relative).glob(pattern):
                    stale.unlink()

    total = 0
    with open(manifest_path, "w", newline="") as manifest_file:
        manifest = csv.writer(manifest_file)
        manifest.writerow(
            [
                "index",
                "object",
                "before_file",
                "after_file",
                "overlap",
                "inliers",
                "similarity",
            ]
        )
        manifest.writerows(kept_rows)
        for object_directory in object_directories:
            total += match_object(object_directory, args, manifest)
            manifest_file.flush()

    print(f"\nTotal pairs saved: {total} (plus {len(kept_rows)} kept)")
    print(f"Output: {args.output} (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
