# Object DeTaken

Find objects that appear in one photo but are missing from another photo of the
same scene — for example, spotting which item was taken from a room by
comparing a "before" and "after" picture.

The pipeline aligns the two photos, detects objects with YOLO, finds changed
regions by image differencing, and marks every object from picture 1 that has
no counterpart in picture 2.

## Setup

Requires Python 3.10+.

```bash
./setup_venv.sh
source .venv/bin/activate
```

The script creates a `.venv` virtual environment and installs the dependencies
from `requirements.txt` (NumPy, OpenCV, Pillow, Ultralytics). YOLO model
weights (`yolo11n.pt`, `yolo11n-cls.pt`) are downloaded automatically by
Ultralytics on first run.

## Usage

### Full pipeline (recommended)

Align the two pictures, then detect and mark objects unique to picture 1:

```bash
python align_and_compare.py with_chair.jpg without_chair.jpg -o unique_objects.jpg
```

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `-o`, `--output` | `unique_objects.jpg` | marked output image |
| `--overlap` | `0.15` | IoU below which a picture-1 object counts as missing from picture 2 |
| `--confidence` | `0.25` | minimum YOLO detection confidence |
| `--pattern-threshold` | `0.50` | template-match similarity below which a crop counts as absent |
| `--difference-min-area` | `0.001` | minimum changed-region size as a fraction of the image |
| `--alignment-max-size` | `800` | thumbnail size used to estimate alignment |
| `--keep-aligned` | off | keep the intermediate `<stem>_aligned` images |

### Individual steps

Align picture 1 to picture 2 and save both cropped to their shared valid area:

```bash
python align_images.py with_chair.jpg without_chair.jpg
```

Compare two already-aligned pictures:

```bash
python compare_yolo_objects.py with_chair_aligned.jpg without_chair_aligned.jpg
```

## How it works

1. **Alignment** (`align_images.py`) — estimates an affine transform with
   OpenCV ECC on downscaled copies, warps picture 1 into picture 2's
   coordinate system, and crops both to the largest rectangle of shared valid
   pixels.
2. **Detection** (`compare_yolo_objects.py`) — runs a YOLO detector on both
   pictures, and adds label-independent candidate boxes from thresholded image
   differences so objects YOLO misses can still be flagged.
3. **Filtering** — a picture-1 box is kept only if it has no overlapping box
   in picture 2 (IoU below `--overlap`) *and* its crop cannot be found
   anywhere in picture 2 by template matching (below `--pattern-threshold`).
4. **Recognition** — surviving crops are classified with a YOLO
   classification model, and the result is drawn on picture 1 with red boxes
   and labels showing the recognized category, confidence, overlap, and
   pattern similarity.

## Files

- `align_and_compare.py` — one-command pipeline (align, then compare)
- `align_images.py` — image alignment step
- `compare_yolo_objects.py` — detection, comparison, and marking step
- `setup_venv.sh` — creates `.venv` and installs `requirements.txt`
- `with_*.jpg` / `without_*.jpg` — paired sample photos (chair, bag, bin,
  iron, stroller, and all items together)
