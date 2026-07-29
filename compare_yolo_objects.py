#!/usr/bin/env python3
"""Mark objects detected in picture 1 but not in picture 2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


class Detection(NamedTuple):
    category: str
    confidence: float
    # Normalized (left, top, right, bottom) coordinates.
    box: tuple[float, float, float, float]


class MarkedDetection(NamedTuple):
    detection: Detection
    overlap: float
    recognized_category: str
    recognition_confidence: float
    pattern_similarity: float


def intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Calculate intersection over union (IoU) for two bounding boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)

    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def extract_detections(result) -> list[Detection]:
    """Convert an Ultralytics result to normalized detections."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    height, width = result.orig_shape
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    detections = []
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        left, top, right, bottom = box
        detections.append(
            Detection(
                category=result.names[class_id],
                confidence=float(confidence),
                box=(left / width, top / height, right / width, bottom / height),
            )
        )
    return detections


def difference_region_candidates(
    first_image: Image.Image,
    second_image: Image.Image,
    minimum_area_ratio: float,
) -> list[Detection]:
    """Create label-independent candidate boxes from aligned image differences."""
    first = np.asarray(first_image.convert("RGB"))
    second = np.asarray(second_image.convert("RGB"))
    height, width = first.shape[:2]
    if second.shape[:2] != (height, width):
        second = cv2.resize(second, (width, height), interpolation=cv2.INTER_LINEAR)

    first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    difference = cv2.absdiff(first_gray, second_gray)
    difference = cv2.GaussianBlur(difference, (9, 9), 0)

    otsu_threshold, _ = cv2.threshold(
        difference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    threshold = max(25.0, otsu_threshold)
    binary = np.where(difference >= threshold, 255, 0).astype(np.uint8)

    kernel_size = max(3, round(min(width, height) / 300))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    minimum_area = width * height * minimum_area_ratio
    padding = max(4, round(min(width, height) / 150))
    candidates = []

    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(width, x + box_width + padding)
        bottom = min(height, y + box_height + padding)
        confidence = float(np.mean(difference[top:bottom, left:right]) / 255.0)
        candidates.append(
            Detection(
                "changed region",
                confidence,
                (left / width, top / height, right / width, bottom / height),
            )
        )
    # Separate contours from one changed object (for example, an object's body
    # and handle) can produce overlapping boxes. Merge boxes when their
    # intersection covers a substantial part of either candidate.
    merged = candidates
    changed = True
    while changed:
        changed = False
        result = []
        while merged:
            current = merged.pop()
            for index, other in enumerate(merged):
                left = max(current.box[0], other.box[0])
                top = max(current.box[1], other.box[1])
                right = min(current.box[2], other.box[2])
                bottom = min(current.box[3], other.box[3])
                intersection = max(0.0, right - left) * max(0.0, bottom - top)
                current_area = (current.box[2] - current.box[0]) * (
                    current.box[3] - current.box[1]
                )
                other_area = (other.box[2] - other.box[0]) * (
                    other.box[3] - other.box[1]
                )
                smaller_area = min(current_area, other_area)
                if smaller_area > 0 and intersection / smaller_area >= 0.20:
                    merged.pop(index)
                    current = Detection(
                        "changed region",
                        max(current.confidence, other.confidence),
                        (
                            min(current.box[0], other.box[0]),
                            min(current.box[1], other.box[1]),
                            max(current.box[2], other.box[2]),
                            max(current.box[3], other.box[3]),
                        ),
                    )
                    changed = True
                    break
            result.append(current)
        merged = result
    return merged


def unique_to_first(
    first: list[Detection], second: list[Detection], overlap_threshold: float
) -> list[tuple[Detection, float]]:
    """Return first-image detections lacking an overlapping box of any category."""
    unique = []
    for detection in first:
        overlaps = [
            intersection_over_union(detection.box, candidate.box)
            for candidate in second
        ]
        best_overlap = max(overlaps, default=0.0)
        if best_overlap < overlap_threshold:
            unique.append((detection, best_overlap))
    return unique


def filter_by_pattern_match(
    first_image: Image.Image,
    second_image: Image.Image,
    candidates: list[tuple[Detection, float]],
    pattern_threshold: float,
) -> list[tuple[Detection, float, float]]:
    """Keep candidates whose crop cannot be found in the second image."""
    first_array = np.asarray(first_image.convert("RGB"))
    second_gray = cv2.cvtColor(
        np.asarray(second_image.convert("RGB")), cv2.COLOR_RGB2GRAY
    )
    height, width = first_array.shape[:2]
    unmatched = []

    for detection, overlap in candidates:
        left, top, right, bottom = detection.box
        x1 = max(0, round(left * width))
        y1 = max(0, round(top * height))
        x2 = min(width, round(right * width))
        y2 = min(height, round(bottom * height))
        crop = first_array[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] > second_gray.shape[0] or crop.shape[1] > second_gray.shape[1]:
            best_similarity = 0.0
        else:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            matches = cv2.matchTemplate(second_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
            finite_matches = matches[np.isfinite(matches)]
            raw_similarity = (
                float(np.max(finite_matches)) if finite_matches.size else 0.0
            )
            # Raw normalized correlation can be misleadingly high when a crop's
            # background exists in picture 2 but its foreground object does not.
            # Cubic calibration keeps near-exact matches high while penalizing
            # these partial, background-dominated matches.
            best_similarity = max(0.0, raw_similarity) ** 3

        if best_similarity < pattern_threshold:
            unmatched.append((detection, overlap, best_similarity))
    return unmatched


def draw_detections(
    image: Image.Image, detections: list[MarkedDetection]
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    line_width = max(2, round(min(width, height) / 250))
    font_size = max(14, round(min(width, height) / 55))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for marked in detections:
        detection = marked.detection
        overlap = marked.overlap
        left, top, right, bottom = detection.box
        pixel_box = (
            round(left * width),
            round(top * height),
            round(right * width),
            round(bottom * height),
        )
        label = (
            f"{marked.recognized_category} {marked.recognition_confidence:.0%} "
            f"overlap {overlap:.0%} pattern {marked.pattern_similarity:.0%}"
        )
        draw.rectangle(pixel_box, outline=(255, 0, 0), width=line_width)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_top = max(0, pixel_box[1] - text_height - 4)
        draw.rectangle(
            (pixel_box[0], text_top, pixel_box[0] + text_width + 4, pixel_box[1]),
            fill=(255, 0, 0),
        )
        draw.text(
            (pixel_box[0] + 2, text_top + 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )
    return output


def recognize_crops(
    image: Image.Image,
    candidates: list[tuple[Detection, float, float]],
    recognition_model,
) -> list[MarkedDetection]:
    """Classify candidate crops and retain their original-image coordinates."""
    if not candidates:
        return []

    width, height = image.size
    crops = []
    for detection, _, _ in candidates:
        left, top, right, bottom = detection.box
        pixel_box = (
            max(0, round(left * width)),
            max(0, round(top * height)),
            min(width, round(right * width)),
            min(height, round(bottom * height)),
        )
        crops.append(image.crop(pixel_box))

    results = recognition_model.predict(source=crops, verbose=False)
    marked = []
    for (detection, overlap, pattern_similarity), result in zip(candidates, results):
        if result.probs is None:
            category = detection.category
            confidence = detection.confidence
        else:
            class_id = int(result.probs.top1)
            category = result.names[class_id]
            confidence = float(result.probs.top1conf.cpu().item())
        marked.append(
            MarkedDetection(
                detection, overlap, category, confidence, pattern_similarity
            )
        )
    return marked


def save_image(image: Image.Image, path: Path) -> None:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=95, subsampling=0)
    else:
        image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect objects in two pictures and mark picture-1 objects whose "
            "IoU overlap with every picture-2 object is below the threshold."
        )
    )
    parser.add_argument("pic1", type=Path)
    parser.add_argument("pic2", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("unique_objects.jpg"))
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="minimum IoU overlap with any detected object (default: 0.15)",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--difference-min-area",
        type=float,
        default=0.001,
        help="minimum changed-region area as a fraction of the image (default: 0.001)",
    )
    parser.add_argument(
        "--pattern-threshold",
        type=float,
        default=0.50,
        help="minimum template-match similarity considered present (default: 0.50)",
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument(
        "--recognition-model",
        default="yolo11n-cls.pt",
        help="classification weights used on candidate crops",
    )
    args = parser.parse_args()

    for path in (args.pic1, args.pic2):
        if not path.is_file():
            parser.error(f"image does not exist or is not a file: {path}")
    if not 0.0 <= args.overlap <= 1.0:
        parser.error("--overlap must be between 0 and 1")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= args.pattern_threshold <= 1.0:
        parser.error("--pattern-threshold must be between 0 and 1")
    if not 0.0 < args.difference_min_area <= 1.0:
        parser.error("--difference-min-area must be greater than 0 and at most 1")
    return args


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "error: ultralytics is not installed; install requirements.txt first"
        ) from error

    try:
        with Image.open(args.pic1) as opened_first, Image.open(args.pic2) as opened_second:
            first_image = ImageOps.exif_transpose(opened_first).convert("RGB")
            second_image = ImageOps.exif_transpose(opened_second).convert("RGB")

        model = YOLO(args.model)
        results = model.predict(
            source=[first_image, second_image],
            conf=args.confidence,
            verbose=False,
        )
        first_detections = extract_detections(results[0])
        second_detections = extract_detections(results[1])

        difference_candidates = difference_region_candidates(
            first_image, second_image, args.difference_min_area
        )
        first_detections.extend(difference_candidates)

        unique = unique_to_first(first_detections, second_detections, args.overlap)
        pattern_unmatched = filter_by_pattern_match(
            first_image, second_image, unique, args.pattern_threshold
        )
        if pattern_unmatched:
            recognition_model = YOLO(args.recognition_model)
            marked = recognize_crops(first_image, pattern_unmatched, recognition_model)
        else:
            marked = []
        save_image(draw_detections(first_image, marked), args.output)
    except Exception as error:
        raise SystemExit(f"error: object comparison failed: {error}") from error

    print(f"Picture 1 detections: {len(first_detections)}")
    print(f"Picture 2 detections: {len(second_detections)}")
    print(f"Image-difference candidates: {len(difference_candidates)}")
    print(f"Objects below {args.overlap:.0%} overlap: {len(unique)}")
    print(
        f"Objects below {args.pattern_threshold:.0%} pattern similarity: "
        f"{len(marked)}"
    )
    for item in marked:
        print(
            f"- {item.recognized_category}: recognition confidence "
            f"{item.recognition_confidence:.1%}, best overlap {item.overlap:.1%} "
            f"pattern similarity {item.pattern_similarity:.1%} "
            f"(detector category: {item.detection.category})"
        )
    print(f"Saved marked image to {args.output}")


if __name__ == "__main__":
    main()
