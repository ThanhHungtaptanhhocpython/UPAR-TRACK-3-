"""Prepare the UPAR Track 3 training data for training on Colab.

Steps:
  1. Fix the PETA image paths in train_upar_all.json. download_datasets.py
     rewrites every "<x>/images/..." path to "pa100k/images/...", which also
     hits the PETA images ("peta/images/*.png"). PA-100K images are all .jpg,
     so "pa100k/images/*.png" is mapped back to "peta/images/*.png".
  2. Verify that every image exists and its size matches the annotation.
  3. Tag each image with its source dataset and write the splits
     (leave-one-dataset-out + a small random in-domain split).
  4. Optionally pack only the used images + annotations into one zip
     (stored, not compressed: the images are already JPEG/PNG).

Usage:
  python -I tools/prepare_data.py --data-dir <starter-kit>/data --out-dir <prepared> [--pack]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

SOURCES = ("market", "pa100k", "peta")
PER_SOURCE_FILES = {s: f"train_{s}.json" for s in SOURCES}
ALL_FILE = "train_upar_all.json"
RANDOM_VAL_FRACTION = 0.05
SEED = 0


def fix_file_name(file_name: str) -> str:
    path = PurePosixPath(file_name)
    if path.parts[0] == "pa100k" and path.suffix.lower() == ".png":
        return str(PurePosixPath("peta", *path.parts[1:]))
    return file_name


def market_person_id(file_name: str) -> str:
    # market/bounding_box_train/1350_c2s3_034732_02.jpg -> "1350"
    return PurePosixPath(file_name).name.split("_")[0]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fix_paths(ann_dir: Path) -> dict:
    data = load_json(ann_dir / ALL_FILE)
    fixed = 0
    for image in data["images"]:
        new_name = fix_file_name(image["file_name"])
        if new_name != image["file_name"]:
            image["file_name"] = new_name
            fixed += 1
        image["source"] = PurePosixPath(new_name).parts[0]
    print(f"[fix] rewrote {fixed} PETA paths")

    # Cross-check against the per-source files, which carry the correct prefixes.
    for source, name in PER_SOURCE_FILES.items():
        expected = {img["file_name"] for img in load_json(ann_dir / name)["images"]}
        got = {img["file_name"] for img in data["images"] if img["source"] == source}
        status = "OK" if expected == got else "MISMATCH"
        print(f"[fix] {source:7s} images={len(got):6d} per-source file={len(expected):6d} {status}")
        if expected != got:
            raise SystemExit(f"{source}: {len(expected - got)} missing, {len(got - expected)} unexpected")
    return data


def verify_images(data: dict, data_dir: Path) -> None:
    missing, size_mismatch = [], []
    for image in data["images"]:
        path = data_dir / image["file_name"]
        if not path.is_file():
            missing.append(image["file_name"])
            continue
        with Image.open(path) as im:
            if im.size != (image["width"], image["height"]):
                size_mismatch.append((image["file_name"], im.size, (image["width"], image["height"])))
    print(f"[verify] images={len(data['images'])} missing={len(missing)} size_mismatch={len(size_mismatch)}")
    for name in missing[:10]:
        print("   missing:", name)
    for item in size_mismatch[:10]:
        print("   size mismatch:", item)
    if missing or size_mismatch:
        raise SystemExit("image verification failed")


def make_splits(data: dict) -> dict:
    by_source = collections.defaultdict(list)
    for image in data["images"]:
        by_source[image["source"]].append(image)

    splits = {}
    for held_out in SOURCES:
        splits[f"lodo_{held_out}"] = {
            "train": sorted(img["id"] for s in SOURCES if s != held_out for img in by_source[s]),
            "val": sorted(img["id"] for img in by_source[held_out]),
        }

    # Random in-domain split, stratified by source. Market images of the same
    # person stay on the same side to avoid identity leakage.
    rng = random.Random(SEED)
    train_ids, val_ids = [], []
    for source in SOURCES:
        groups = collections.defaultdict(list)
        for img in by_source[source]:
            key = market_person_id(img["file_name"]) if source == "market" else img["id"]
            groups[key].append(img["id"])
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = round(len(keys) * RANDOM_VAL_FRACTION)
        for i, key in enumerate(keys):
            (val_ids if i < n_val else train_ids).extend(groups[key])
    splits["random5"] = {"train": sorted(train_ids), "val": sorted(val_ids)}

    for name, split in splits.items():
        print(f"[split] {name:12s} train={len(split['train']):6d} val={len(split['val']):6d}")
    return splits


def pack(data_dir: Path, out_dir: Path, data: dict, ann_path: Path, splits_path: Path) -> Path:
    zip_path = out_dir / "upar_track3_train.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(ann_path, f"annotations/{ann_path.name}")
        zf.write(splits_path, f"annotations/{splits_path.name}")
        for image in data["images"]:
            zf.write(data_dir / image["file_name"], f"images/{image['file_name']}")
    print(f"[pack] wrote {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="starter kit data/ directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pack", action="store_true", help="also write upar_track3_train.zip")
    args = parser.parse_args()

    ann_dir = args.data_dir / "annotations" / "specialization"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = fix_paths(ann_dir)
    verify_images(data, args.data_dir)
    splits = make_splits(data)

    ann_path = args.out_dir / "train_upar_all_fixed.json"
    splits_path = args.out_dir / "splits.json"
    ann_path.write_text(json.dumps(data), encoding="utf-8")
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    print(f"[write] {ann_path}\n[write] {splits_path}")

    if args.pack:
        pack(args.data_dir, args.out_dir, data, ann_path, splits_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
