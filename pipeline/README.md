# Detection pipeline

Classical (non-trained) pipeline that aligns a before/after photo pair and
marks every object present in the first picture but missing from the second.
Used standalone via `align_and_compare.py`, and `align_images.py` also serves
the siamese trainer's `--align` preprocessing
(`siamese_missing/train_missing.py`).

## Usage

Full pipeline — align the two pictures, then detect and mark objects unique
to picture 1 (run from the project root so YOLO weights land there):

```bash
python pipeline/align_and_compare.py with_chair.jpg without_chair.jpg -o unique_objects.jpg
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-o`, `--output` | `unique_objects.jpg` | marked output image |
| `--overlap` | `0.15` | IoU below which a picture-1 object counts as missing from picture 2 |
| `--confidence` | `0.25` | minimum YOLO detection confidence |
| `--pattern-threshold` | `0.50` | template-match similarity below which a crop counts as absent |
| `--difference-min-area` | `0.001` | minimum changed-region size as a fraction of the image |
| `--alignment-max-size` | `800` | thumbnail size used to estimate alignment |
| `--keep-aligned` | off | keep the intermediate `<stem>_aligned` images |

Individual steps:

```bash
# align picture 1 to picture 2, crop both to their shared valid area
python pipeline/align_images.py with_chair.jpg without_chair.jpg

# compare two already-aligned pictures
python pipeline/compare_yolo_objects.py with_chair_aligned.jpg without_chair_aligned.jpg
```

## How it works

1. **Alignment** (`align_images.py`) — estimates an affine transform with
   OpenCV ECC on downscaled copies, warps picture 1 into picture 2's
   coordinate system, and crops both to the largest rectangle of shared valid
   pixels.
2. **Detection** (`compare_yolo_objects.py`) — runs a YOLO detector
   (`yolo11n.pt`, COCO classes) on both pictures, and adds label-independent
   candidate boxes from thresholded image differences so objects YOLO misses
   can still be flagged.
3. **Filtering** — a picture-1 box is kept only if it has no overlapping box
   in picture 2 (IoU below `--overlap`) *and* its crop cannot be found
   anywhere in picture 2 by template matching (below `--pattern-threshold`).
4. **Recognition** — surviving crops are classified with a YOLO
   classification model (`yolo11n-cls.pt`, ImageNet classes), and the result
   is drawn on picture 1 with red boxes and labels showing the recognized
   category, confidence, overlap, and pattern similarity.

Both YOLO weight files are downloaded automatically by Ultralytics into the
working directory on first run. Note the naming caveat: boxes come from COCO
detection plus image differencing, but names on changed-region boxes come
from the ImageNet-1000 classifier, whose vocabulary may not contain the
object (e.g. a sofa is named `studio_couch`; there is no `stroller` class).
Swap in larger models with `--model` / `--recognition-model`
(e.g. `yolo11s.pt`, `yolo11s-cls.pt`) for better boxes and names.

## Files

- `align_and_compare.py` — one-command pipeline (align, then compare)
- `align_images.py` — image alignment step; also imported by
  `train_missing.py --align`
- `compare_yolo_objects.py` — detection, comparison, and marking step
