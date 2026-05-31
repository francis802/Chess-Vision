# Chess-Vision

A computer-vision pipeline that turns a single **photograph of a physical chessboard**
into a **digital twin** of the position: it locates the board, detects and classifies
every piece, maps each piece to its square, and renders a clean 2-D board diagram.

The project also explores the auxiliary task of **counting the pieces** on a board,
both as a multiclass-classification problem and as a regression problem.

> Built on the [ChessReD2k](https://data.4tu.nl/datasets) real-world chess image
> dataset (top-down and angled photographs with annotated corners and pieces).

---

## What it does

Given an input image, the full pipeline (`notebooks/board_and_pieces_pipeline.ipynb`) runs:

```
photo ──► board segmentation (U-Net) ──► homography / perspective warp
       └► piece detection (YOLO)      ──► map detections to 8×8 grid
                                       └► board matrix ──► 2-D diagram
```

1. **Board detection** — segment the board surface and recover its four corners so
   the image can be rectified to a top-down view.
2. **Piece detection & classification** — locate every piece and assign it one of
   13 classes (`{white,black} × {pawn,rook,knight,bishop,queen,king}` + `empty`).
3. **Grid mapping** — split the rectified board into an 8×8 grid and assign each
   detected piece to its square.
4. **Digital twin** — build the board matrix and render a 2-D visualization.

---

## Approaches explored

The project deliberately compares several methods per sub-task and keeps the winners.

| Sub-task | Approaches tried | Final choice |
|---|---|---|
| **Board localization** | Corner-regression CNN, **Mask R-CNN**, **U-Net** segmentation | **U-Net** (best Dice, lightest) |
| **Corner / grid recovery** | Corner regression, direct grid regression | Corner detection + homography |
| **Piece detection** | **YOLOv8 / YOLO11** | YOLO fine-tuned on the chess dataset |
| **Piece counting** | Multiclass classification, regression (ResNet50 → EfficientNet-B0) | EfficientNet-B0 regression head |

Mask R-CNN was abandoned for board localization (it failed to converge on this
data — see metrics below); U-Net replaced it as the final, more accurate and
cheaper segmenter.

---

## Results

Metrics are stored as NumPy arrays under [`metrics/`](metrics/) and were produced
on the validation split.

| Model | Metric | Best value |
|---|---|---|
| Board segmentation — **U-Net** | Validation Dice | **0.996** |
| Board localization — Mask R-CNN | Validation IoU | ~0.00 (did not converge — discarded) |
| Grid regression | Validation MAE (normalized coords) | **0.023** |
| Piece counting — regression | Validation MAE (normalized, scale 32) | **0.025** (≈ <1 piece) |

YOLO training curves, PR/F1 curves and confusion matrices are generated under
`runs/` by Ultralytics (regenerable, not versioned). The two fine-tuned YOLO
detectors are shipped in [`models/`](models/).

---

## Tech stack

- **Python**, **PyTorch**, **torchvision**
- **Ultralytics** — YOLOv8 / YOLO11 object detection
- **U-Net** (custom) and **Mask R-CNN** for board segmentation
- **EfficientNet-B0 / ResNet50** for piece-count regression & classification
- **OpenCV** — homography, perspective warp, grid splitting
- **NumPy**, **Matplotlib**, **tqdm**, **scikit-learn**, **torchmetrics**

---

## Repository structure

```
Chess-Vision/
├── notebooks/
│   ├── board_and_pieces_pipeline.ipynb   # MAIN: end-to-end board+piece pipeline
│   │                                     #   (YOLO + corner detection + U-Net + grid + 2-D twin)
│   ├── piece_counting_classification.ipynb  # piece counting as multiclass classification
│   ├── piece_counting_regression.ipynb      # piece counting as regression (final, full dataset)
│   ├── grid_generation.ipynb                # generate 9×9 grid-point pseudo-labels
│   └── experiments/                         # earlier / alternative attempts
│       ├── piece_counting_regression_resnet.ipynb   # first regression attempt (ResNet50)
│       └── piece_counting_regression_v2.ipynb       # refined regression (EfficientNet + ScaledSigmoid)
├── src/
│   ├── detect_pieces_yolo.py        # YOLO inference → count pieces per image (input.json → output.json)
│   ├── count_pieces_regression.py   # EfficientNet-B0 regression inference → count pieces
│   └── train_board_maskrcnn.py      # Mask R-CNN board-segmentation training (multi-worker)
├── models/
│   ├── yolo_piece_detection.pt      # fine-tuned YOLO for chess pieces
│   └── yolo_board_detection.pt      # fine-tuned YOLO for board corners
├── data/
│   ├── pieces.yaml                  # YOLO dataset config (13 classes)
│   ├── piece_dataset/               # YOLO labels (train/val/test)
│   ├── pieces_png/                  # piece sprites for the 2-D board rendering
│   ├── grid_labels/                 # generated grid-point label arrays (.npy)
│   └── io/                          # input.json / output.json inference I/O examples
├── metrics/
│   ├── corner_detection/            # corner-regression CNN training metrics
│   ├── board_unet/                  # U-Net segmentation metrics (Dice, loss)
│   ├── board_mask_maskrcnn/         # Mask R-CNN metrics (discarded approach)
│   └── grid_regression/             # direct grid-regression metrics
├── docs/
│   └── PIPELINE.md                  # detailed pipeline walkthrough
├── .gitignore
└── README.md
```

> **Note on the dataset:** the raw images and COCO-style `annotations.json` from
> ChessReD2k are **not** redistributed here (they are large and externally licensed).
> Place them at the repository root (so `annotations.json` and `images/` resolve)
> before running the notebooks.

---

## How to run

### 1. Install dependencies

```bash
pip install torch torchvision ultralytics opencv-python numpy matplotlib \
            tqdm scikit-learn torchmetrics pillow pycocotools
```

### 2. Count pieces with the YOLO detector

`data/io/input.json` lists the images to score; results are written to
`data/io/output.json`.

```bash
python src/detect_pieces_yolo.py
```

### 3. Count pieces with the regression model

Requires a trained checkpoint at `models/best_model.pth`
(produced by `notebooks/piece_counting_regression.ipynb`).

```bash
python src/count_pieces_regression.py
```

### 4. Run the full board pipeline

Open `notebooks/board_and_pieces_pipeline.ipynb` and run it top-to-bottom with the
ChessReD2k dataset placed at the repo root. It uses the shipped detectors in
`models/` and the metrics/labels under `metrics/` and `data/`.

> Run the notebooks and scripts **from the repository root** so the relative paths
> (`models/`, `data/`, `metrics/`) resolve correctly.

---

## License

Released for academic / portfolio purposes. The ChessReD2k dataset retains its
own license.
