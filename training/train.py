"""Fine-tune ViTPose on UPAR-Pose with PoseTrack18 slots (Colab).

Recipe from MMPose's ViTPose COCO configs: AdamW, weight decay 0.1 (none on
bias/norm/pos_embed), layer-wise lr decay, drop path, grad clip 1.0, linear
warm-up; cosine decay as in the organizers' UPAR-Pose runs. bf16 autocast.

Resumable: the full training state is saved to the run directory (Google Drive)
every `ckpt_every` epochs, the best weights (by validation OKS-AP) whenever they
improve, and one JSON line per epoch is appended to log.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import coco_kpt_eval as ce
import pose_infer as pi
import vitpose_model as vm
from data import PT18_FLIP_PAIRS, PoseDataset, collate

DEFAULTS = dict(
    run_name="vitb_random5",
    preset="base",
    init="hf:usyd-community/vitpose-base",  # or "mmpose:<url or path>"
    split="random5",                          # random5 | lodo_market | lodo_pa100k | lodo_peta | all
    epochs=30,
    batch_size=64,
    lr=5e-4,
    layer_decay=0.75,
    weight_decay=0.1,
    warmup_iters=500,
    min_lr_ratio=0.01,
    drop_path=0.3,
    clip_grad=1.0,
    eval_every=2,
    eval_flip=True,
    ckpt_every=5,
    workers=None,
    aug={},
    seed=0,
    input_size=(256, 192),     # (height, width); 384x288 starts from the 256x192 COCO checkpoint (pos_embed resized)
    heatmap_sigma=2.0,         # MMPose: 2 at 64x48 heatmaps, 3 at 96x72
    joint_weights=None,        # 17 per-slot loss weights, e.g. 1.5 on nose/head_bottom/head_top (slots 0-2)
    grad_checkpoint=False,     # recompute activations in backward: ~30% slower, much less memory
    max_iters_per_epoch=None,  # for smoke tests
    max_val=None,              # cap validation boxes (smoke tests)
    exclude_file="/content/upar/exclude_ann_ids.json",  # annotations removed from TRAINING only (decision 08/10)
    data_dir="/content/data",
    out_root="/content/drive/MyDrive/upar_track3/runs",
)

MMPOSE_URLS = {
    "base": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth",
    "large": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth",
    "huge": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth",
}
INPUT_CFG = {"input_height": 256, "input_width": 192, "image_mean": [0.485, 0.456, 0.406],
             "image_std": [0.229, 0.224, 0.225], "rescale_factor": 1 / 255.0, "normalize_factor": 200.0,
             "padding_factor": 1.25, "num_keypoints": 17}


def input_cfg(cfg: dict) -> dict:
    return {**INPUT_CFG, "input_height": int(cfg["input_size"][0]), "input_width": int(cfg["input_size"][1])}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- data

def load_split(cfg: dict):
    data_dir = cfg["data_dir"]
    ann = json.load(open(f"{data_dir}/annotations/train_upar_all_fixed.json"))
    if cfg["split"] == "all":
        train_ids, val_ids = {im["id"] for im in ann["images"]}, set()
    else:
        split = json.load(open(f"{data_dir}/annotations/splits.json"))[cfg["split"]]
        train_ids, val_ids = set(split["train"]), set(split["val"])
    needed = [im for im in ann["images"] if im["id"] in train_ids | val_ids]
    index = {im["id"]: i for i, im in enumerate(needed)}

    def read(im):
        return np.asarray(Image.open(f"{data_dir}/images/{im['file_name']}").convert("RGB"), dtype=np.uint8)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        images = list(ex.map(read, needed))
    log(f"loaded {len(images)} images into RAM ({sum(i.nbytes for i in images) / 1e9:.2f} GB, {time.time() - t0:.0f}s)")

    def usable(a):
        return a["bbox"][2] > 1 and a["bbox"][3] > 1 and a.get("num_keypoints", 1) > 0

    def entries(ids):
        return [{"image_index": index[a["image_id"]], "bbox": a["bbox"], "keypoints": a["keypoints"], "ann": a}
                for a in ann["annotations"] if a["image_id"] in ids and usable(a)]

    excluded = set()
    if cfg.get("exclude_file"):
        if not os.path.isfile(cfg["exclude_file"]):
            raise FileNotFoundError(f"exclude_file {cfg['exclude_file']} not found (set exclude_file=None to train on all labels)")
        excluded = set(json.load(open(cfg["exclude_file"]))["ann_ids"])
    train_entries = [e for e in entries(train_ids) if e["ann"]["id"] not in excluded]
    log(f"excluded {len(entries(train_ids)) - len(train_entries)} suspicious annotations from training "
        f"(list has {len(excluded)}; validation keeps all labels)")
    val_gt = {"images": [im for im in needed if im["id"] in val_ids],
              "annotations": [a for a in ann["annotations"] if a["image_id"] in val_ids],
              "categories": ann["categories"]}
    return images, train_entries, entries(val_ids), val_gt


# ---------------------------------------------------------------- model / optimisation

def build_model(cfg: dict) -> torch.nn.Module:
    model = vm.build(cfg["preset"], drop_path_rate=cfg["drop_path"], img_size=tuple(int(v) for v in cfg["input_size"]))
    model.grad_checkpoint = bool(cfg["grad_checkpoint"])
    kind, _, ref = cfg["init"].partition(":")
    if kind == "hf":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        state = vm.from_hf(load_file(hf_hub_download(ref, "model.safetensors")), model.depth)
    elif kind == "mmpose":
        url = ref or MMPOSE_URLS[cfg["preset"]]
        path = url
        if url.startswith("http"):
            path = f"/content/ckpt/{os.path.basename(url)}"
            if not os.path.isfile(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                log(f"downloading {url}")
                urllib.request.urlretrieve(url, path + ".part")
                os.replace(path + ".part", path)
        state = vm.from_mmpose(vm.load_checkpoint_file(path), model.depth)
    elif kind in ("pt", "drive", "file"):
        path = ref
        if kind == "drive" and not path.startswith("/"):
            path = f"/content/drive/MyDrive/upar_track3/runs/{ref}"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        if "pos_embed" in state:
            state["pos_embed"] = vm.resize_pos_embed(state["pos_embed"], model.grid)
        model.load_state_dict(state, strict=True)
        return model
    else:
        raise ValueError(f"unknown init {cfg['init']}")
    state["pos_embed"] = vm.resize_pos_embed(state["pos_embed"], model.grid)
    model.load_state_dict(vm.coco_head_to_pt18(state), strict=True)
    return model


def param_groups(model, cfg: dict):
    groups = {}
    num_layers = model.depth + 1
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim == 1 or name == "pos_embed"
        lid = model.layer_id(name)
        key = (lid, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if no_decay else cfg["weight_decay"],
                           "lr_scale": cfg["layer_decay"] ** (num_layers - lid)}
        groups[key]["params"].append(p)
    return list(groups.values())


def lr_factor(it: int, total: int, cfg: dict) -> float:
    if it < cfg["warmup_iters"]:
        return 0.001 + (1 - 0.001) * it / cfg["warmup_iters"]
    t = (it - cfg["warmup_iters"]) / max(1, total - cfg["warmup_iters"])
    return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, loader, entries, val_gt, flip: bool, device) -> dict:
    model.eval()
    preds = []
    for x, _, _, meta in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hm = model(x)
            if flip:
                hm_f = model(torch.flip(x, dims=[3]))
        hm = hm.float().cpu().numpy()
        if flip:
            hm = 0.5 * (hm + pi.flip_back(hm_f.float().cpu().numpy(), PT18_FLIP_PAIRS, shift=False))
        coords, maxvals = pi.decode_heatmaps(hm, meta["center"], meta["size"] / 200.0)
        for j, idx in enumerate(meta["index"]):
            a = entries[idx]["ann"]
            k = np.concatenate([coords[j], maxvals[j][:, None]], axis=1).reshape(-1).tolist()
            preds.append({"image_id": a["image_id"], "category_id": 1, "box_id": a["id"], "keypoints": k,
                          "score": pi.person_score(maxvals[j], "mean_all")})
    model.train()
    metrics = ce.evaluate(val_gt, preds)
    per_joint = ce.per_keypoint_similarity(val_gt, preds)
    metrics["pck"] = {k: round(v["pck50"], 4) for k, v in per_joint.items()}
    metrics["head_oks"] = float(np.mean([per_joint[k]["oks"] for k in ("nose", "head_bottom", "head_top") if k in per_joint]))
    metrics["lr_swap_rate"] = lr_swap_rate(val_gt, preds)
    return metrics


def lr_swap_rate(gt: dict, preds: list[dict], min_joints: int = 4) -> float:
    """Share of people whose limb joints (slots 5-16) agree better with the ground truth after
    swapping left/right: a front/back confusion that costs the whole pose (failure mode #3)."""
    var = (ce.DEFAULT_SIGMAS * 2) ** 2
    anns = {a["id"]: a for a in gt["annotations"]}
    swap = np.arange(17)
    for l, r in PT18_FLIP_PAIRS:
        swap[l], swap[r] = r, l
    limbs = np.arange(5, 17)
    n = swapped = 0
    for d in preds:
        a = anns.get(d["box_id"])
        if a is None:
            continue
        g = np.asarray(a["keypoints"], dtype=np.float64).reshape(17, 3)
        k = np.asarray(d["keypoints"], dtype=np.float64).reshape(17, 3)[:, :2]
        vis = limbs[(g[limbs, 2] > 0) & (g[swap[limbs], 2] > 0)]
        if len(vis) < min_joints:
            continue
        area = a["area"] + np.spacing(1)
        same = np.exp(-((k[vis] - g[vis, :2]) ** 2).sum(1) / var[vis] / area / 2).mean()
        flip = np.exp(-((k[vis] - g[swap[vis], :2]) ** 2).sum(1) / var[vis] / area / 2).mean()
        n += 1
        swapped += flip > same
    return round(swapped / max(n, 1), 4)


# ---------------------------------------------------------------- main loop

def train(user_cfg: dict) -> dict:
    cfg = {**DEFAULTS, **user_cfg}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = f"{cfg['out_root']}/{cfg['run_name']}"
    os.makedirs(run_dir, exist_ok=True)
    json.dump(cfg, open(f"{run_dir}/config.json", "w"), indent=1)
    log(f"run {cfg['run_name']} -> {run_dir}")

    images, train_entries, val_entries, val_gt = load_split(cfg)
    if cfg["max_val"]:
        val_entries = val_entries[:cfg["max_val"]]
        keep = {e["ann"]["image_id"] for e in val_entries}
        val_gt = {**val_gt, "images": [im for im in val_gt["images"] if im["id"] in keep],
                  "annotations": [a for a in val_gt["annotations"] if a["image_id"] in keep]}
    log(f"train {len(train_entries)} boxes | val {len(val_entries)} boxes")
    workers = cfg["workers"] if cfg["workers"] is not None else min(24, os.cpu_count() or 2)
    in_cfg = input_cfg(cfg)
    train_ds = PoseDataset(images, train_entries, in_cfg, train=True, aug=cfg["aug"], sigma=cfg["heatmap_sigma"])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                                           num_workers=workers, collate_fn=collate, pin_memory=True,
                                           persistent_workers=workers > 0)
    val_dl = None
    if val_entries:
        val_ds = PoseDataset(images, val_entries, in_cfg, train=False, sigma=cfg["heatmap_sigma"])
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=workers,
                                             collate_fn=collate, pin_memory=True)

    model = build_model(cfg).to(device)
    joint_w = None
    if cfg["joint_weights"] is not None:
        joint_w = torch.tensor(cfg["joint_weights"], dtype=torch.float32, device=device)
        if joint_w.shape != (17,):
            raise ValueError("joint_weights needs 17 values (PoseTrack18 slots)")
    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg["lr"], betas=(0.9, 0.999))
    iters_per_epoch = len(train_dl) if not cfg["max_iters_per_epoch"] else min(len(train_dl), cfg["max_iters_per_epoch"])
    total_iters = iters_per_epoch * cfg["epochs"]

    start_epoch, it, best_ap = 0, 0, -1.0
    last_path = f"{run_dir}/last.pt"
    if os.path.isfile(last_path):
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        for key in ("preset", "input_size", "split"):
            old, new = state["cfg"].get(key, DEFAULTS[key]), cfg[key]
            if json.dumps(old) != json.dumps(list(new) if isinstance(new, tuple) else new):
                raise ValueError(f"{last_path} was trained with {key}={old}, config says {new}: use a new run_name")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch, it, best_ap = state["epoch"] + 1, state["iter"], state["best_ap"]
        log(f"resumed from epoch {state['epoch']} (best AP {best_ap:.4f})")

    model.train()
    for epoch in range(start_epoch, cfg["epochs"]):
        t0, loss_sum, n = time.time(), 0.0, 0
        for step, (x, target, weight, _) in enumerate(train_dl):
            if step >= iters_per_epoch:
                break
            f = lr_factor(it, total_iters, cfg)
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * f * g["lr_scale"]
            x, target, weight = (t.to(device, non_blocking=True) for t in (x, target, weight))
            if joint_w is not None:
                weight = weight * joint_w
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
            loss = (((out.float() - target) ** 2) * weight[:, :, None, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            opt.step()
            loss_sum += loss.item()
            n += 1
            it += 1
        dt = time.time() - t0
        record = {"epoch": epoch, "iter": it, "loss": loss_sum / max(n, 1), "lr": cfg["lr"] * lr_factor(it, total_iters, cfg),
                  "sec": round(dt, 1), "img_per_s": round(n * cfg["batch_size"] / dt, 1)}

        last_epoch = epoch == cfg["epochs"] - 1
        if val_dl is not None and ((epoch + 1) % cfg["eval_every"] == 0 or last_epoch):
            metrics = evaluate(model, val_dl, val_entries, val_gt, cfg["eval_flip"], device)
            record.update({k: round(metrics[k], 4) for k in ("AP", "AP50", "AP75", "AP_medium", "AP_large", "AR_large")})
            record["lr_swap"] = metrics["lr_swap_rate"]
            record["head_oks"] = round(metrics["head_oks"], 4)
            record["pck"] = {k: metrics["pck"].get(k) for k in ("nose", "head_top", "left_wrist", "right_wrist")}
            if metrics["AP"] > best_ap:
                best_ap = metrics["AP"]
                torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                           f"{run_dir}/best_fp16.pt")
                record["best"] = True
        elif val_dl is None and last_epoch:
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                       f"{run_dir}/best_fp16.pt")
        with open(f"{run_dir}/log.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
        log(json.dumps(record))

        if (epoch + 1) % cfg["ckpt_every"] == 0 or last_epoch:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "iter": it,
                        "best_ap": best_ap, "cfg": cfg}, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)

    export = {"preset": cfg["preset"], "init": cfg["init"], "split": cfg["split"], "slots": "posetrack18",
              "flip_pairs": [list(p) for p in PT18_FLIP_PAIRS], "flip_shift": False, "best_val_ap": best_ap,
              **in_cfg}
    json.dump(export, open(f"{run_dir}/model_config.json", "w"), indent=1)
    log(f"done. best val AP {best_ap:.4f}. weights: {run_dir}/best_fp16.pt")
    return {"run_dir": run_dir, "best_ap": best_ap}
