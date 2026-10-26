# UPAR Challenge 2027 — Fact Sheet
## Track 3: 2D Human Pose Estimation on Surveillance Crops

---

### 1. Team & Submission Identification
- **Team Name on Codabench:** `6em1n1` (also known as `RL4FUN`)
- **Submission ID:** `1007061` (Submission 18 on Codabench Leaderboard)
- **Primary Metric on Test Leaderboard:**
  - **AP:** **0.9050**
  - **$\text{AP}_{50}$:** **0.9731**
  - **$\text{AP}_{\text{medium}}$:** **0.8754**
  - **$\text{AP}_{\text{large}}$:** **0.6952**
- **Contact Email:** upar.challenge@gmail.com (Submitted from participant email)
- **Workshop Paper Submitted (RWS Workshop):** Yes / Under preparation (Please cite team `6em1n1` in the challenge summary paper).
- **Code Repository:** [GitHub / GitLab Repository URL provided in email]
- **Trained Weights Storage:** [Google Drive Link provided in email]

---

### 2. Method Description
- **Method Name / Acronym:** ViTPose-MRDARK (Multi-Resolution ViTPose Ensemble with Vectorized GPU DARK and Surveillance-Aware Fine-Tuning)
- **General Description:**
  Our method addresses cross-domain 2D pedestrian pose estimation from surveillance cameras by combining multi-resolution Vision Transformers (ViTPose), surveillance-specific data augmentations, and ultra-fast sub-pixel coordinate decoding. Specifically, we fine-tune ViTPose backbones pre-trained strictly on COCO train2017 using UDP coordinate representations and adaptive head-joint loss weighting. During inference, we employ a multi-resolution heatmap ensemble (ViTPose-Large 256x192 and ViTPose-Large 384x288, extended to ViTPose-Huge in Submission 19) coupled with a 4-scale multi-padding Test-Time Augmentation (TTA) and vectorized closed-form GPU DARK (Distribution-Aware Coordinate Representation) post-processing.

- **Rule Compliance Verification:**
  - **Backbone initialization:** Pretrained on ImageNet-1k (MAE).
  - **Pose pre-training:** Strictly COCO train2017 only (no multi-dataset, no AIC, no CrowdPose, no MPII).
  - **Fine-tuning data:** Strictly UPAR-Pose training set (Market-1501, PA-100K, PETA). No external pose data was utilized.

---

### 3. Model Architecture & Ensembling
- **Network Architecture:**
  - Non-hierarchical Vision Transformer (ViT) with patch size $16 \times 16$.
  - Simple 2-layer Deconvolution Head upsampling features $4\times$ to generate Gaussian heatmaps for 17 PoseTrack18 keypoints.
- **Ensemble Composition (Submission 18 - ID `1007061`):**
  - **Model 1:** ViTPose-Large ($256 \times 192$ input, $64 \times 48$ heatmaps, 307M params) — Weight: **0.30**
  - **Model 2:** ViTPose-Large ($384 \times 288$ input, $96 \times 72$ heatmaps, 307M params) — Weight: **0.70**
  - *Total parameters in Submission 18:* **614 Million parameters**.
- **Alternative / Enhanced Ensemble (Submission 19):**
  - ViTPose-Large 256 ($w=0.20$) + ViTPose-Large 384 ($w=0.50$) + ViTPose-Huge 256 ($w=0.30$).
  - *Total parameters:* **1.25 Billion parameters**.

---

### 4. Training Procedure & Hyperparameters
- **Training Dataset:** UPAR-Pose Train (Market-1501, PA-100K, PETA).
- **Data Cleaning:** 176 suspicious/inverted annotations (severe left/right limb misalignments) were identified and excluded from the training split.
- **Data Augmentations:**
  - UDP affine transformation with 1.25 bounding box padding factor.
  - Horizontal flip ($p=0.5$).
  - Half-body augmentation ($p=0.3$).
  - Scale jittering $[0.75, 1.25]$ and random translation $[-0.16, +0.16]$.
  - Rotation $[-40^\circ, +40^\circ]$ ($p=0.6$).
  - Photometric distortion & channel swap ($p=0.5$).
  - Coarse dropout (simulating surveillance occlusion up to 40% height/width, $p=1.0$).
  - Random Gaussian blur and median blur ($p=0.1$).
  - Grayscale conversion ($p=0.1$) and low-resolution downsampling ($p=0.2$).
- **Loss Function:**
  - Mean Squared Error (MSE) on UDP sub-pixel Gaussian target heatmaps ($\sigma=2.0$ for 256x192; $\sigma=3.0$ for 384x288).
  - Head-joint weighting: $1.5\times$ weight on `nose`, `head_bottom`, and `head_top` to penalize foreshortening errors caused by high surveillance camera pitch angles. Ears (slots 3, 4) masked out (unannotated).
- **Optimization:**
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.1).
  - Learning Rate: Peak $lr = 2 \times 10^{-4}$ with 500-iteration linear warmup and cosine decay.
  - Layer-wise LR Decay: 0.80 for ViTPose-Large; 0.85 for ViTPose-Huge.
  - Drop path rate: 0.50 for Large; 0.55 for Huge.
  - Epochs: 20 epochs.
  - Batch size: 64.
  - Mixed Precision: `torch.bfloat16` autocast.
- **Hardware & Training Time:**
  - Trained on Google Colab GPU (RTX PRO 6000 Blackwell / NVIDIA A100).
  - Training time: ~34 minutes for ViTPose-Large 256; ~71 minutes for ViTPose-Large 384; ~60 minutes for ViTPose-Huge 256.

---

### 5. Inference, Post-Processing & Runtime
- **Test-Time Augmentation (TTA):**
  - Multi-Padding Zoom: 4 scale factors $[0.88, 0.94, 1.0, 1.06] \times \text{padding\_factor}$ to capture pedestrians at varying surveillance distances.
  - Horizontal flip test with flip-pair slot inversion and keypoint coordinate realignment.
- **DARK (Distribution-Aware Coordinate Representation):**
  - Fully vectorized on GPU using pure PyTorch tensors (`fast_infer.py`).
  - Separable Gaussian blur with kernel size 11 and $\sigma=1.25$.
  - Closed-form $2 \times 2$ Taylor expansion and Hessian matrix inversion:
    $$\Delta \mu = - \mathcal{H}^{-1} \mathcal{D}$$
  - Eliminates quantization artifacts without CPU transfer overhead.
- **Person Confidence Scoring:**
  - `mean_scored`: Computes the average confidence of the top detected keypoint heatmaps, ensuring robust ranking without penalizing occluded individuals.
- **Inference Speed:**
  - Total time on Codabench GPU worker for all 30,245 test crops: **~60 minutes** (Submission 18) and **~50.4 minutes** (Submission 19).
  - Average latency per person crop per scale: **~8.4 ms** on GPU.
- **Docker Environment:**
  - Runs inside the official competition image: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`.
  - Zero external C++ or custom compilation dependencies. Pure PyTorch and NumPy.

---

### 6. Summary of Key Innovations
1. **Surveillance Foreshortening Mitigation:** High overhead camera pitch angles collapse head and shoulder keypoints. By reweighting head keypoint losses $\times 1.5$ and applying aggressive coarse dropout, our models exhibit robust structural localization even under severe occlusion.
2. **Multi-Padding Scale Invariant TTA:** Combining zoom-in scales (0.88, 0.94) to enhance small/medium pedestrian details with scale 1.06 for large pedestrians yields uniform AP gains across all person scales.
3. **Pure GPU DARK Post-Processing:** Fully vectorizing the DARK sub-pixel Hessian inversion eliminated 90,700+ CPU-GPU memory copies, enabling 4-scale multi-padding TTA within the competition time limit.
