# PCB Defect Detection with YOLOv8

This project implements a computer vision pipeline for detecting and localizing defects on printed circuit boards (PCBs) using YOLOv8.

The goal of the project is to build an end-to-end object detection workflow: dataset validation, annotation analysis, model training, evaluation on validation and test sets, and visual inference examples.

## Problem

Manual PCB inspection can be slow, repetitive, and error-prone. A computer vision model can help automate visual quality control by detecting and localizing common PCB defects on images.

This project focuses on six PCB defect classes:

- `mouse_bite`
- `spur`
- `missing_hole`
- `short`
- `open_circuit`
- `spurious_copper`

## Dataset

The project uses the **PCB Defect Dataset** from Kaggle:

`norbertelter/pcb-defect-dataset`

The dataset is already organized in YOLO format and contains separate training, validation, and test splits:

```text
pcb-defect-dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Each label file contains bounding box annotations in YOLO format:

```text
class_id x_center y_center bbox_width bbox_height
```

All bounding box coordinates are normalized to the range `[0, 1]`.

## Project Workflow

The project follows a practical computer vision pipeline:

1. Download the dataset with `kagglehub`
2. Inspect the dataset structure
3. Parse YOLO label files into a Pandas DataFrame
4. Validate image-label matching
5. Check bounding box coordinate correctness
6. Analyze class distribution
7. Analyze bounding box sizes
8. Visualize ground truth annotations
9. Train a YOLOv8n smoke test model
10. Train a longer YOLOv8n baseline model
11. Evaluate the model on validation and test sets
12. Run inference on unseen test images

## Dataset Validation

During dataset validation, I found that some label files used the `_256` suffix while the corresponding images used the `_600` suffix. Since YOLO annotations are normalized, I implemented a filename matching fallback that maps `_256` labels to `_600` images.

After validation:

```text
Unmatched labels: 0
Total annotated objects: 21,664
```

The dataset is well balanced across the six defect classes:

```text
mouse_bite         3684
spurious_copper    3676
spur               3636
missing_hole       3612
open_circuit       3548
short              3508
```

## Bounding Box Analysis

The average bounding box size is small:

```text
Mean bbox width:  0.0505
Mean bbox height: 0.0505
Mean bbox area:   0.00261
```

This means that an average defect occupies only about **0.26%** of the image area. This makes the task more challenging because the model has to detect small localized defects.

Average bounding box area by class:

```text
missing_hole       0.002513
mouse_bite         0.002068
open_circuit       0.001420
short              0.003966
spur               0.002798
spurious_copper    0.002918
```

The `open_circuit` class has the smallest average bounding box area, which may make it harder to localize precisely.

## Model

The model used in this project is **YOLOv8n** from the Ultralytics package.

YOLOv8n was selected because it is lightweight, fast to train, and suitable as a first object detection baseline.

Training configuration:

```text
Model: YOLOv8n
Image size: 416
Batch size: 16
Device: Tesla T4 GPU
```

## Results

### Validation and Test Performance

| Model | Split | Epochs | Image Size | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n smoke test | val | 3 | 416 | 0.680 | 0.736 | 0.726 | 0.332 |
| YOLOv8n baseline | val | 15 | 416 | 0.960 | 0.935 | 0.969 | 0.504 |
| YOLOv8n baseline | test | 15 | 416 | 0.970 | 0.930 | 0.971 | 0.497 |

The 15-epoch YOLOv8n baseline achieved strong results on the test set:

```text
Precision: 0.970
Recall:    0.930
mAP50:     0.971
mAP50-95:  0.497
```

The high `mAP50` shows that the model detects PCB defects well. The lower `mAP50-95` indicates that strict bounding box localization remains more challenging, especially for small defects.

### Test Set Results by Class

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| mouse_bite | 0.971 | 0.907 | 0.955 | 0.482 |
| spur | 0.969 | 0.889 | 0.969 | 0.492 |
| missing_hole | 0.983 | 0.986 | 0.991 | 0.547 |
| short | 0.978 | 0.972 | 0.981 | 0.518 |
| open_circuit | 0.984 | 0.911 | 0.976 | 0.455 |
| spurious_copper | 0.933 | 0.916 | 0.955 | 0.487 |

The best-performing class was `missing_hole`. The weakest class by strict localization score was `open_circuit`, which is likely related to its smaller average bounding box size.

## Inference

The trained model was tested on unseen images from the test set. It was able to detect and localize multiple PCB defect types, including:

- `spurious_copper`
- `short`
- `missing_hole`
- `mouse_bite`
- `spur`
- `open_circuit`

Inference speed on a Tesla T4 GPU was approximately:

```text
6–9 ms per image
```

## Key Findings

- YOLOv8n provides a strong lightweight baseline for PCB defect detection.
- The dataset is well balanced across defect classes.
- Most defects are small, with an average bounding box area of about 0.26% of the image.
- The model performs best on `missing_hole` and `short` defects.
- Stricter localization remains more difficult for small defects, especially `open_circuit`.
- The model is fast enough for near real-time inference under GPU acceleration.

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Pandas
- NumPy
- Matplotlib
- KaggleHub
- Google Colab

## Repository Structure

Suggested repository structure:

```text
pcb-defect-detection-yolov8/
├── PCB_Defect_Detection_YOLOv8.ipynb
├── README.md
├── requirements.txt
├── models/
│   └── yolov8n_pcb_defect_best.pt
└── results/
    ├── class_distribution.png
    ├── bbox_area_by_class.png
    └── sample_predictions/
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

The notebook can be run in Google Colab. It downloads the dataset using `kagglehub`, validates the annotations, trains YOLOv8n, evaluates the model, and runs inference on test images.

## Future Improvements

Possible next steps:

- Train with a larger image size, such as 512 or 640
- Compare YOLOv8n with YOLOv8s
- Tune confidence and IoU thresholds
- Add more systematic error analysis
- Export the model to ONNX
- Build a small Gradio or FastAPI demo for inference

## Summary

This project demonstrates an end-to-end computer vision workflow for industrial defect detection. It includes dataset validation, exploratory analysis, YOLOv8 training, test evaluation, and inference visualization.

The final YOLOv8n baseline achieved strong test performance with `mAP50 = 0.971` and `Precision = 0.970`, making it a solid junior-level computer vision portfolio project.
