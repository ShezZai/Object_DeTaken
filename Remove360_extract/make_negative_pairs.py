#!/usr/bin/env python3
"""Generate no-change negative pairs from Remove360 camera walks.

Two frames of the same walk (train/ = object present in both, test/ = object
absent in both) form a genuine "nothing went missing" pair with realistic
viewpoint and lighting variation. Candidates are verified with the same
SIFT + RANSAC homography gates as the positive pairs, and their mutual frame
coverage is kept inside the same band as the positives so overlap statistics
cannot separate the classes.

Pairs are written into an existing dataset tree as
<output>/<scene>/pair_<n>/{before.jpg, after.jpg, label.json} with
{"missing": false, "items": []}, continuing each scene's numbering. Per
scene, as many negatives are generated as there are existing pairs (or use
--per-scene to override).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_aligned_pairs import (
    extract_features,
    find_object_directories,
    frame_coverage,
    good_matches,
    largest_valid_rectangle,
    list_images,
    load_rgb,
    masked_similarity,
    save_jpeg,
    scale_homography_to_full,
    valid_content_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Remove360 root containing <scene>/<object>/train|test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="dataset tree to extend, e.g. ../somethings_missing_here/Remove360_based",
    )
    parser.add_argument("--per-scene", type=int, default=None,
                        help="negatives per scene (default: match existing pair count)")
    parser.add_argument("--overlap-min", type=float, default=0.85)
    parser.add_argument("--overlap-max", type=float, default=0.96)
    parser.add_argument("--min-inliers", type=int, default=40)
    parser.add_argument("--min-similarity", type=float, default=0.475)
    parser.add_argument("--frame-gap-min", type=int, default=8)
    parser.add_argument("--frame-gap-max", type=int, default=45)
    parser.add_argument("--alignment-max-size", type=int, default=800)
    parser.add_argument("--sift-features", type=int, default=800)
    parser.add_argument("--attempts", type=int, default=60,
                        help="sampling attempts per requested negative")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def existing_pair_numbers(scene_directory: Path) -> list[int]:
    numbers = []
    for pair in scene_directory.glob("pair_*"):
        match = re.fullmatch(r"pair_(\d+)", pair.name)
        if match:
            numbers.append(int(match.group(1)))
    return numbers


class WalkImages:
    """Lazily loaded, valid-region-cropped images and features of one walk."""

    def __init__(self, paths, sift, max_size):
        self.paths = paths
        self.sift = sift
        self.max_size = max_size
        self.images = {}
        self.features = {}

    def image(self, index):
        if index not in self.images:
            image = valid_content_crop(load_rgb(self.paths[index]))
            self.images[index] = image
        return self.images[index]

    def feature(self, index):
        if index not in self.features:
            image = self.image(index)
            self.features[index] = (
                extract_features(self.sift, image, self.max_size)
                if image is not None
                else None
            )
        return self.features[index]


def verify(walk, i, j, matcher, args):
    """Return (homography, overlap, inliers, similarity) or None."""
    first, second = walk.feature(i), walk.feature(j)
    if first is None or second is None:
        return None
    matches = good_matches(matcher, first, second)
    if len(matches) < args.min_inliers:
        return None
    source = np.float32([first.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([second.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    if homography is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.sum())
    if inliers < args.min_inliers:
        return None
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    overlap = min(
        frame_coverage(homography, first.gray.shape, second.gray.shape),
        frame_coverage(inverse, second.gray.shape, first.gray.shape),
    )
    if not args.overlap_min <= overlap <= args.overlap_max:
        return None
    warped = cv2.warpPerspective(
        first.gray, homography, (second.gray.shape[1], second.gray.shape[0])
    )
    valid_mask = cv2.warpPerspective(
        np.full(first.gray.shape, 255, dtype=np.uint8),
        homography,
        (second.gray.shape[1], second.gray.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    similarity = masked_similarity(warped, second.gray, valid_mask)
    if similarity < args.min_similarity:
        return None
    return homography, overlap, inliers, similarity


def save_pair(walk, i, j, homography, pair_directory, source_note):
    first_full, second_full = walk.image(i), walk.image(j)
    first_small = walk.feature(i).gray.shape
    second_small = walk.feature(j).gray.shape
    full_homography = scale_homography_to_full(
        homography, first_small, first_full.shape[:2],
        second_small, second_full.shape[:2],
    )
    warped = cv2.warpPerspective(
        first_full, full_homography,
        (second_full.shape[1], second_full.shape[0]), flags=cv2.INTER_LINEAR,
    )
    small_mask = cv2.warpPerspective(
        np.full(first_small, 255, dtype=np.uint8), homography,
        (second_small[1], second_small[0]), flags=cv2.INTER_NEAREST,
    )
    left, top, right, bottom = largest_valid_rectangle(small_mask)
    scale_x = second_full.shape[1] / second_small[1]
    scale_y = second_full.shape[0] / second_small[0]
    left = min(second_full.shape[1] - 1, round(left * scale_x) + 1)
    top = min(second_full.shape[0] - 1, round(top * scale_y) + 1)
    right = max(left + 1, round(right * scale_x) - 1)
    bottom = max(top + 1, round(bottom * scale_y) - 1)

    pair_directory.mkdir(parents=True)
    save_jpeg(warped[top:bottom, left:right], pair_directory / "before.jpg")
    save_jpeg(second_full[top:bottom, left:right], pair_directory / "after.jpg")
    (pair_directory / "label.json").write_text(
        json.dumps({"missing": False, "items": [], "source": source_note}, indent=2)
        + "\n"
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    sift = cv2.SIFT_create(nfeatures=args.sift_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    scene_objects = defaultdict(list)
    for object_directory in find_object_directories(args.root):
        relative = object_directory.relative_to(args.root)
        scene_objects[relative.parts[0]].append(object_directory)

    total = 0
    for scene, objects in sorted(scene_objects.items()):
        scene_directory = args.output / scene
        if not scene_directory.is_dir():
            print(f"{scene}: not in output dataset, skipping")
            continue
        numbers = existing_pair_numbers(scene_directory)
        quota = args.per_scene if args.per_scene is not None else len(numbers)
        next_number = max(numbers, default=0) + 1

        walks = []
        for object_directory in objects:
            for walk_name in ("train", "test"):
                paths = list_images(object_directory / walk_name)
                if len(paths) > args.frame_gap_min:
                    walks.append((
                        f"{object_directory.relative_to(args.root)}/{walk_name}",
                        WalkImages(paths, sift, args.alignment_max_size),
                    ))
        if not walks:
            print(f"{scene}: no usable walks, skipping")
            continue

        used = defaultdict(set)
        made = 0
        exhausted = 0
        while made < quota and exhausted < len(walks) * 2:
            walk_note, walk = walks[(made + exhausted) % len(walks)]
            success = False
            for _ in range(args.attempts):
                i = rng.randrange(len(walk.paths))
                gap = rng.randint(args.frame_gap_min, args.frame_gap_max)
                j = i + gap if rng.random() < 0.5 else i - gap
                if not 0 <= j < len(walk.paths):
                    continue
                if i in used[walk_note] or j in used[walk_note]:
                    continue
                result = verify(walk, i, j, matcher, args)
                if result is None:
                    continue
                homography, overlap, inliers, similarity = result
                pair_directory = scene_directory / f"pair_{next_number:02d}"
                note = (
                    f"{walk_note}: {walk.paths[i].name} -> {walk.paths[j].name} "
                    f"(overlap {overlap:.3f}, similarity {similarity:.3f})"
                )
                save_pair(walk, i, j, homography, pair_directory, note)
                used[walk_note].update((i, j))
                next_number += 1
                made += 1
                total += 1
                success = True
                break
            if not success:
                exhausted += 1
        print(f"{scene}: generated {made}/{quota} negatives")

    print(f"\nTotal negatives: {total}")


if __name__ == "__main__":
    main()
