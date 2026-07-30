#!/usr/bin/env python3
"""Generate no-change negative pairs from an already-distilled pair dataset.

Needs no raw Remove360 data: within one scene, the before.jpg images of two
different pairs that share the same removed-item label come from the same
pre-removal camera walk (the item is present in both), and the after.jpg
images likewise from the same post-removal walk (absent in both). Aligning
two such same-side images yields a genuine "nothing went missing" pair.

Candidates are verified with the same SIFT + RANSAC homography gates as the
positives, with mutual frame coverage kept in the same band so overlap
statistics cannot separate the classes. Results are appended to the dataset
as <scene>/pair_<n>/{before.jpg, after.jpg, label.json} with
{"missing": false, "items": []}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_aligned_pairs import (
    extract_features,
    frame_coverage,
    good_matches,
    largest_valid_rectangle,
    load_rgb,
    masked_similarity,
    save_jpeg,
    scale_homography_to_full,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="distilled dataset root, e.g. ../somethings_missing_here/Remove360_based",
    )
    parser.add_argument("--overlap-min", type=float, default=0.85)
    parser.add_argument("--overlap-max", type=float, default=0.96)
    parser.add_argument("--min-inliers", type=int, default=40)
    parser.add_argument("--min-similarity", type=float, default=0.475)
    parser.add_argument("--alignment-max-size", type=int, default=800)
    parser.add_argument("--sift-features", type=int, default=800)
    parser.add_argument(
        "--max-per-scene",
        type=int,
        default=None,
        help="cap negatives per scene (default: number of existing pairs)",
    )
    return parser.parse_args()


def verify(first, second, matcher, args):
    matches = good_matches(matcher, first, second)
    if len(matches) < args.min_inliers:
        return None
    source = np.float32([first.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([second.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    if homography is None or inlier_mask is None:
        return None
    if int(inlier_mask.sum()) < args.min_inliers:
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
    return homography, overlap, similarity


def save_pair(first_path, second_path, features, homography, pair_directory, note):
    first_full = load_rgb(first_path)
    second_full = load_rgb(second_path)
    first_small = features[first_path].gray.shape
    second_small = features[second_path].gray.shape
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
        json.dumps({"missing": False, "items": [], "source": note}, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    sift = cv2.SIFT_create(nfeatures=args.sift_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    total = 0
    for scene_directory in sorted(p for p in args.dataset.iterdir() if p.is_dir()):
        numbers = []
        groups = defaultdict(list)  # (item, side) -> [(pair_name, image_path)]
        for pair_directory in sorted(scene_directory.glob("pair_*")):
            match = re.fullmatch(r"pair_(\d+)", pair_directory.name)
            if not match:
                continue
            numbers.append(int(match.group(1)))
            meta = json.loads((pair_directory / "label.json").read_text())
            if not meta.get("missing"):
                continue  # only positives contribute walk images
            item = (meta.get("items") or ["unknown"])[0]
            for side in ("before", "after"):
                image = pair_directory / f"{side}.jpg"
                if image.is_file():
                    groups[(item, side)].append((pair_directory.name, image))

        quota = args.max_per_scene or len(numbers)
        next_number = max(numbers, default=0) + 1
        features = {}
        used = set()
        made = 0

        candidates = []
        for (item, side), members in groups.items():
            for (name_a, path_a), (name_b, path_b) in itertools.combinations(members, 2):
                candidates.append((item, side, name_a, path_a, name_b, path_b))

        for item, side, name_a, path_a, name_b, path_b in candidates:
            if made >= quota:
                break
            if path_a in used or path_b in used:
                continue
            for path in (path_a, path_b):
                if path not in features:
                    features[path] = extract_features(
                        sift, load_rgb(path), args.alignment_max_size
                    )
            result = verify(features[path_a], features[path_b], matcher, args)
            if result is None:
                continue
            homography, overlap, similarity = result
            note = (
                f"{scene_directory.name}/{item}/{side}: {name_a} -> {name_b} "
                f"(overlap {overlap:.3f}, similarity {similarity:.3f})"
            )
            save_pair(
                path_a, path_b, features, homography,
                scene_directory / f"pair_{next_number:02d}", note,
            )
            used.update((path_a, path_b))
            next_number += 1
            made += 1
            total += 1
        print(f"{scene_directory.name}: generated {made} negatives "
              f"(quota {quota})")

    print(f"\nTotal negatives: {total}")


if __name__ == "__main__":
    main()
