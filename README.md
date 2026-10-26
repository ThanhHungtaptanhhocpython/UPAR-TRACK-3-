# UPAR Challenge 2027 — Track 3: 2D Human Pose Estimation
## Team: 6em1n1 (RL4FUN) | Official Submission Code & Model Weights

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Codabench](https://img.shields.io/badge/Codabench-Challenge%2018273-blue.svg)](https://www.codabench.org/competitions/18273/)

This repository contains the complete, reproducible source code, training pipelines, and trained weights for **Team 6em1n1 (RL4FUN)** in the **UPAR Challenge 2027 (Track 3: 2D Human Pose Estimation on Surveillance Crops)**, organized in conjunction with the Real-World Surveillance (RWS) Workshop.

---

## 1. Official Leaderboard Results

Our final evaluated submission appearing on the official Codabench Leaderboard is **Submission 18 (ID: `1007061`)**:

| Submission | Codabench ID | Overall AP | $\text{AP}_{50}$ | $\text{AP}_{\text{medium}}$ | $\text{AP}_{\text{large}}$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Submission 18 (Official)** | **`1007061`** | **0.9050** | **0.9731** | **0.8754** | **0.6952** | **Rank #5 on Leaderboard** |
| Submission 19 (Tri-Ensemble) | *Packaged* | *Est. 0.9075+* | *0.9731* | *0.8785+* | *0.6955* | 1.25B Parameters (Huge + Large 384 + Large 256) |

> [!NOTE]
> On $\text{AP}_{50}$ (**0.9731**), our method outperforms Rank #2 (`plpo`: 0.9708), Rank #3 (`huethuong`: 0.9708), and Rank #4 (`DBBui`: 0.9706), confirming exceptionally high recall and localization accuracy across all pedestrian bounding boxes.

---

## 2. Technical Methodology Overview

```
                                    +---> ViTPose-Large (256x192, w=0.3) ---+
                                    |                                       |
Input Surveillance Crop             |                                       v
[Multi-Padding TTA: 0.88, 0.94, 1.0, 1.06]                            Heatmap Fusion
         + Horizontal Flip          |                                       |
                                    |                                       v
                                    +---> ViTPose-Large (384x288, w=0.7) ---> GPU DARK Decoding
                                                                            (Closed-form Hessian 2x2)
                                                                                    |
                                                                                    v
                                                                        Final Sub-pixel Keypoints
                                                                        + 'mean_scored' Confidence
```

### 2.1 Model Architecture
- **ViTPose Backbone:** Plain Vision Transformer with global self-attention across non-overlapping $16 \times 16$ image patches.
- **PoseTrack18 Head Adaptation:** 17 keypoint heatmap deconvolution head matching the challenge specification.
- **Strict Pre-training Compliance:**
  - Backbones initialized from ImageNet-1k MAE.
  - Pose pre-trained strictly on **COCO train2017 only** (official MMPose checkpoints: `td-hm_ViTPose-{large,huge}_8xb64-210e_coco`).
  - **No multi-dataset pre-training** (no ViTPose+, no AIC, no CrowdPose, no MPII), in full accordance with Track 3 regulations.

### 2.2 Surveillance-Targeted Fine-Tuning
- **Head Joint Loss Weighting:** High camera pitch angles in surveillance cause severe foreshortening of the head and shoulders. We applied a $1.5\times$ loss weight on `nose`, `head_bottom`, and `head_top`.
- **UDP (Unbiased Data Processing):** Sub-pixel Gaussian heatmaps ($\sigma=2.0$ for 256x192, $\sigma=3.0$ for 384x288) and unbiased coordinate transforms.
- **Surveillance Data Augmentations:** Heavy coarse dropout (up to 40% bounding box occlusion), random Gaussian/median blur, photometric distortion, grayscale conversion, and scale/rotation jittering.
- **Label Cleaning:** 176 inverted/corrupted annotations identified in the raw training set were removed from training via `exclude_ann_ids.json`.

### 2.3 Fast Vectorized Test-Time Augmentation (TTA) & GPU DARK
- **Multi-Padding TTA:** 4 zoom levels $(0.88, 0.94, 1.0, 1.06) \times \text{padding\_factor}$ with horizontal flip averaging.
- **GPU DARK Decoder:** Standard CPU post-processing creates severe bottlenecks. We implemented closed-form vectorized $2\times 2$ Hessian inversion directly in PyTorch CUDA (`fast_infer.py`), smoothing heatmaps with separable Gaussian blur ($\sigma=1.25$, kernel 11) and computing sub-pixel Taylor offsets in a single GPU kernel.
- **Inference Runtime:** Fully processes all 30,245 test crops in ~60 minutes on Codabench GPU worker (~8.4 ms per person per scale in FP16), well within the 120-minute server limit.

---

## 3. Repository Structure

```
UPAR2027_Track3_Submission/
├── README.md                          # Complete documentation (this file)
├── FACT_SHEET_RL4FUN_6em1n1.md        # Filled Fact Sheet for organizers
├── requirements.txt                   # Minimal dependencies
├── docker/
│   ├── ingestion.py                   # Codabench ingestion program
│   ├── run_docker_submission18.sh     # One-click Docker reproduction for Submission 18
│   └── run_docker_submission19.sh     # One-click Docker reproduction for Submission 19
├── inference/
│   ├── submission_18/                 # Standalone folder for Submission 18 (Leaderboard: 0.9050)
│   │   ├── run.py                     # Entry point called by ingestion program
│   │   ├── metadata.yaml              # Track 3 metadata
│   │   ├── fast_infer.py              # Vectorized GPU DARK inference engine
│   │   ├── pose_infer.py              # Pose normalization and utilities
│   │   ├── vitpose_model.py           # Pure PyTorch ViTPose implementation
│   │   └── assets/                    # Model configs, ensemble definition & weights
│   └── submission_19/                 # Standalone folder for Submission 19 (Tri-Ensemble)
│       ├── run.py
│       ├── metadata.yaml
│       ├── fast_infer.py
│       ├── pose_infer.py
│       ├── vitpose_model.py
│       └── assets/
├── training/
│   ├── train.py                       # Pure PyTorch fine-tuning script
│   ├── data.py                        # Dataset loader, UDP heatmaps & augmentations
│   ├── vitpose_model.py               # Pure PyTorch ViTPose backbone
│   ├── exclude_ann_ids.json           # Filtered noisy annotations
│   └── colab_notebooks/               # Self-contained training notebooks
│       ├── 02_train_l256.ipynb        # Fine-tune ViTPose-Large 256x192 (r03)
│       ├── 04_train_l384.ipynb        # Fine-tune ViTPose-Large 384x288 (r05)
│       ├── 06_train_h256.ipynb        # Fine-tune ViTPose-Huge 256x192 (r06)
│       └── 12_train_h384.ipynb        # Fine-tune ViTPose-Huge 384x288 (r09)
└── weights/
    ├── README.md                      # Weight descriptions & download links
    ├── r03_vitl256_weights_fp16.pt    # Large 256 weights (FP16, 617 MB)
    ├── r05_vitl384_weights_fp16.pt    # Large 384 weights (FP16, 617 MB)
    └── r06_vith256_weights_fp16.pt    # Huge 256 weights (FP16, 1.27 GB)
```

---

## 4. Reproducing Inference with Docker

The submission requires zero manual installation and runs natively inside the official Codabench Docker image (`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`).

### To run Submission 18 (Official Leaderboard 0.9050 AP):
```bash
bash docker/run_docker_submission18.sh /path/to/test_input_data /path/to/output_dir
```

### To run Submission 19 (Tri-Ensemble 1.25B parameters):
```bash
bash docker/run_docker_submission19.sh /path/to/test_input_data /path/to/output_dir
```

---

## 5. Reproducing Model Training

All models can be trained either locally or on Google Colab / cloud instances using our standalone notebooks in `training/colab_notebooks/`:

1. **ViTPose-Large 256x192 (`r03`):** Run `training/colab_notebooks/02_train_l256.ipynb` (~34 mins on A100 / RTX 6000).
2. **ViTPose-Large 384x288 (`r05`):** Run `training/colab_notebooks/04_train_l384.ipynb` (~70 mins on A100 / RTX 6000).
3. **ViTPose-Huge 256x192 (`r06`):** Run `training/colab_notebooks/06_train_h256.ipynb` (~60 mins on A100 / RTX 6000).

Or run directly via command line:
```bash
python training/train.py \
    --preset large \
    --init mmpose: \
    --input_size 384 288 \
    --heatmap_sigma 3.0 \
    --epochs 20 \
    --batch_size 64 \
    --lr 2e-4
```

---

## 6. Contact & Authors

- **Team Name:** `6em1n1` (RL4FUN)
- **Challenge:** UPAR Challenge 2027 (RWS Workshop, ECCV / CVPR)
- **Email:** Contact via challenge registration email.
