"""UPAR 2027 Track 3 submission: heatmap ensemble of ViTPose models fine-tuned on UPAR-Pose.

Members: r03_vitl_ft (large 256x192, w=0.2), r05_vitl384_ft (large 384x288, w=0.5), r06_vith_ft (huge 256x192, w=0.3). Pose pre-training: COCO only; backbones initialised from ImageNet-1k MAE.
vitpose_model.py / pose_infer.py are copies of solution/. The ingestion program
imports this file and calls predict_image(sample) once per image.
"""
from __future__ import annotations

import json
import os
from typing import Any

import sys

import numpy as np
import torch
from PIL import Image

# The ingestion program only sets the working directory; make the modules shipped
# next to run.py importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fast_infer as fi  # noqa: E402
import pose_infer as pi  # noqa: E402
import vitpose_model as vm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

FLIP_TEST = True
HALF = True  # fp16 autocast for the models (CUDA only)
SCORE_MODE = "mean_scored"
BATCH_SIZE = 32
PADDING_SCALES = (0.88, 0.94, 1.0, 1.06)  # multi-padding TTA zoom levels (x cfg padding_factor)
DARK_KERNEL = 11  # DARK Gaussian blur kernel
DARK_SIGMA = 1.25  # DARK Gaussian blur sigma

_MEMBERS: list = []  # (model, cfg, weight)
_DEVICE = None
_PRIOR: np.ndarray | None = None


def load_model() -> None:
    global _DEVICE, _PRIOR
    if _MEMBERS:
        return
    with open(os.path.join(ASSETS, "ensemble.json"), "r", encoding="utf-8") as fh:
        ensemble = json.load(fh)
    with open(os.path.join(ASSETS, "mean_pose_pt18.json"), "r", encoding="utf-8") as fh:
        _PRIOR = np.asarray(json.load(fh)["mean_pose_normalized"], dtype=np.float64)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    for member in ensemble["members"]:
        with open(os.path.join(ASSETS, member["dir"], "model_config.json"), "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        model = vm.build_for_config(cfg)
        state = torch.load(os.path.join(ASSETS, member["dir"], "weights_fp16.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.float() for k, v in state.items()})
        _MEMBERS.append((model.to(_DEVICE).eval(), cfg, float(member["weight"])))
    print(f"[submission] ensemble {[(m['dir'], m['preset'], m['input']) for m in ensemble['members']]} "
          f"weights={[w for _, _, w in _MEMBERS]} on {_DEVICE}, flip={FLIP_TEST}, half={HALF}", flush=True)


def _prior_pose(box) -> list[list[float]]:
    bx, by, bw, bh = [float(v) for v in box[:4]]
    return [[bx + u * bw, by + v * bh, 0.01] for u, v in _PRIOR]


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    load_model()
    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}
    image = np.asarray(Image.open(sample["image_path"]).convert("RGB"), dtype=np.float32)
    kpts, joint_scores, usable = fi.predict_boxes_ensemble(image, boxes, _MEMBERS, flip=FLIP_TEST, half=HALF,
                                                           flip_pairs=pi.PT18_FLIP_PAIRS, batch_size=BATCH_SIZE,
                                                           padding_scales=PADDING_SCALES, dark_kernel=DARK_KERNEL,
                                                           dark_sigma=DARK_SIGMA)
    keypoints, scores = [], []
    for i, box in enumerate(boxes):
        if not usable[i]:
            keypoints.append(_prior_pose(box))
            scores.append(0.01)
            continue
        keypoints.append([[float(x), float(y), float(s)] for (x, y), s in zip(kpts[i], joint_scores[i])])
        scores.append(pi.person_score(joint_scores[i], SCORE_MODE))
    return {"keypoints": keypoints, "scores": scores}
