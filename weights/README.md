# Trained Weights for UPAR 2027 Track 3 Submissions

This folder contains the trained weights in pure PyTorch FP16 format for the ViTPose models fine-tuned on UPAR-Pose.

## Models Included

1. **`r03_vitl256_weights_fp16.pt`** (~617 MB):
   - Architecture: ViTPose-Large ($256 \times 192$, 307M parameters)
   - Pre-training: ImageNet-1k (MAE) $\rightarrow$ COCO train2017
   - Fine-tuned on UPAR-Pose (PoseTrack18 slots, filtered noisy annotations)

2. **`r05_vitl384_weights_fp16.pt`** (~617 MB):
   - Architecture: ViTPose-Large ($384 \times 288$, 307M parameters)
   - Pre-training: ImageNet-1k (MAE) $\rightarrow$ COCO train2017
   - Fine-tuned on UPAR-Pose ($384 \times 288$, heatmap $\sigma=3.0$, $1.5\times$ head joint loss weight)

3. **`r06_vith256_weights_fp16.pt`** (~1.27 GB):
   - Architecture: ViTPose-Huge ($256 \times 192$, 637M parameters)
   - Pre-training: ImageNet-1k (MAE) $\rightarrow$ COCO train2017
   - Fine-tuned on UPAR-Pose ($256 \times 192$, layer decay 0.85, drop path 0.55)

## Weight Mapping to Submissions

- **Submission 18 (Official Leaderboard ID: 1007061, AP: 0.9050):**
  - `inference/submission_18/assets/m0/weights_fp16.pt` $\leftarrow$ `r03_vitl256_weights_fp16.pt` (Weight: 0.3)
  - `inference/submission_18/assets/m1/weights_fp16.pt` $\leftarrow$ `r05_vitl384_weights_fp16.pt` (Weight: 0.7)

- **Submission 19 (Tri-Ensemble, 1.25B parameters):**
  - `inference/submission_19/assets/m0/weights_fp16.pt` $\leftarrow$ `r03_vitl256_weights_fp16.pt` (Weight: 0.2)
  - `inference/submission_19/assets/m1/weights_fp16.pt` $\leftarrow$ `r05_vitl384_weights_fp16.pt` (Weight: 0.5)
  - `inference/submission_19/assets/m2/weights_fp16.pt` $\leftarrow$ `r06_vith256_weights_fp16.pt` (Weight: 0.3)

## Google Drive Download Link

If cloning from GitHub without large files, download all trained weights and submission zips from:
[Google Drive Link - To be inserted by team upon upload]

