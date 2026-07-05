
import argparse
import os
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ── Config ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Balanced 4-class dataset
    "data":         "dataset_hazard/data.yaml",

    # FIXED: using yolo11s for better accuracy over yolo11n
    "model":        "yolo11s.pt",

    # FIXED: increased epochs from 40 to 150 for proper convergence
    "epochs":       150,

    "imgsz":        640,
    "batch":        16,          # FIXED: increased from 8 to 16 for stable gradients

    # FIXED: patience aligned with epochs — was 40 with 40 epochs (useless)
    "patience":     30,

    # FIXED: lr0 lowered from 0.003 to 0.001 — 0.003 causes unstable training
    "lr0":          0.001,
    "lrf":          0.01,        # FIXED: raised from 0.001 — too aggressive decay before

    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,          # FIXED: increased from 3 to 5 for better warmup

    "device":       "",          # "" = auto (GPU if available, else CPU)
    "workers":      4,
    "project":      "runs/detect",
    "name":         "hazard_v1",
    "exist_ok":     True,        # FIXED: True so reruns don't crash
    "pretrained":   True,
    "optimizer":    "AdamW",
    "seed":         42,
    "cos_lr":       True,
    "amp":          False,
    "fraction":     1.0,
    "val":          True,
    "save":         True,
    "save_period":  10,
    "cache":        False,

    # ── Augmentation ─────────────────────────────────────
    # FIXED: stronger augmentation for small/imbalanced dataset
    "hsv_h":        0.015,
    "hsv_s":        0.7,
    "hsv_v":        0.4,
    "degrees":      15.0,        # FIXED: increased from 5.0
    "translate":    0.1,
    "scale":        0.5,
    "shear":        2.0,
    "perspective":  0.0,
    "flipud":       0.1,         # FIXED: was 0.0 — small flipud helps
    "fliplr":       0.5,
    "mosaic":       1.0,
    "mixup":        0.1,         # FIXED: increased from 0.05
    "copy_paste":   0.1,         # FIXED: was 0.0 — helps minority classes
    "close_mosaic": 10,          # FIXED: was 15 — stop mosaic earlier

    # ── Loss weights ─────────────────────────────────────
    # FIXED: increased box weight to improve localization
    "box":          7.5,
    "cls":          0.5,
    "dfl":          1.5,
}

CLASS_NAMES = ["light_pole", "pothole-ditch", "roadside_stall", "vehicle"]


def validate_dataset(data_yaml: str) -> dict:
    """Check data.yaml exists and verify expected classes."""
    path = Path(data_yaml)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            f"Expected at: {path.resolve()}"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    print("\n── Dataset Info ────────────────────────────────")
    print(f"  Config : {path.resolve()}")
    nc    = config.get("nc", 0)
    names = config.get("names", [])
    print(f"  Classes: {nc} → {names}")

    if nc != 4:
        print(f"  WARNING: Expected 4 classes, found {nc}. Check data.yaml.")
    else:
        print(f"  OK: Class count correct (4)")

    for split in ("train", "val", "test"):
        split_path = config.get(split, "")
        if split_path:
            print(f"  {split:6s} : {split_path}")
    print("────────────────────────────────────────────────\n")
    return config


def print_class_distribution(data_yaml: str):
    """Count label files and objects per class."""
    from collections import defaultdict

    path = Path(data_yaml)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    base = path.parent

    def resolve_label_path(split_img_path: str) -> Path:
        label_path = Path(
            split_img_path
            .replace("/images", "/labels")
            .replace("\\images", "\\labels")
        )
        if label_path.is_absolute():
            return label_path

        stripped_parts = [
            part for part in label_path.parts
            if part not in (".", "..")
        ]
        candidates = []
        if stripped_parts:
            candidates.append(base.joinpath(*stripped_parts))
        candidates.extend([base / label_path, Path.cwd() / label_path])

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1] if candidates else label_path

    print("── Class Distribution Per Split ────────────────")
    for split in ("train", "val", "test"):
        split_img_path = config.get(split, "")
        if not split_img_path:
            continue
        label_path = resolve_label_path(split_img_path)

        if not label_path.exists():
            print(f"  {split:6s}: label folder not found at {label_path}")
            continue

        label_files   = list(label_path.glob("*.txt"))
        class_counts  = defaultdict(int)
        empty         = 0

        for lf in label_files:
            lines = [l.strip() for l in lf.read_text().splitlines() if l.strip()]
            if not lines:
                empty += 1
                continue
            for line in lines:
                class_counts[int(line.split()[0])] += 1

        total = sum(class_counts.values())
        print(f"\n  {split.upper()} ({len(label_files)} files | {empty} empty):")
        for cid in sorted(class_counts):
            name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
            count = class_counts[cid]
            pct   = count / total * 100 if total else 0
            print(f"    {name:20s}: {count:5d}  ({pct:.1f}%)")

    print("────────────────────────────────────────────────\n")


def train(cfg: dict):
    print("=" * 60)
    print("  HAZARD DETECTION MODEL — TRAINING")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Classes : {CLASS_NAMES}")
    print(f"  Dataset : {cfg['data']}")
    print(f"  Model   : {cfg['model']}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  Batch   : {cfg['batch']}")
    print(f"  ImgSz   : {cfg['imgsz']}")
    print("=" * 60)

    # Validate dataset before starting
    validate_dataset(cfg["data"])

    try:
        print_class_distribution(cfg["data"])
    except Exception as e:
        print(f"  Note: Could not print distribution — {e}")

    # Load pretrained YOLOv11 model
    print(f"Loading base model: {cfg['model']}")
    model = YOLO(cfg["model"])

    # ── Training ──────────────────────────────────────────
    results = model.train(
        data          = cfg["data"],
        epochs        = cfg["epochs"],
        imgsz         = cfg["imgsz"],
        batch         = cfg["batch"],
        patience      = cfg["patience"],
        lr0           = cfg["lr0"],
        lrf           = cfg["lrf"],
        momentum      = cfg["momentum"],
        weight_decay  = cfg["weight_decay"],
        warmup_epochs = cfg["warmup_epochs"],
        device        = cfg["device"],
        workers       = cfg["workers"],
        project       = cfg["project"],
        name          = cfg["name"],
        exist_ok      = cfg["exist_ok"],
        pretrained    = cfg["pretrained"],
        optimizer     = cfg["optimizer"],
        seed          = cfg["seed"],
        cos_lr        = cfg["cos_lr"],
        amp           = cfg["amp"],
        fraction      = cfg["fraction"],
        val           = cfg["val"],
        save          = cfg["save"],
        save_period   = cfg["save_period"],
        cache         = cfg["cache"],
        # Augmentation
        hsv_h         = cfg["hsv_h"],
        hsv_s         = cfg["hsv_s"],
        hsv_v         = cfg["hsv_v"],
        degrees       = cfg["degrees"],
        translate     = cfg["translate"],
        scale         = cfg["scale"],
        shear         = cfg["shear"],
        perspective   = cfg["perspective"],
        flipud        = cfg["flipud"],
        fliplr        = cfg["fliplr"],
        mosaic        = cfg["mosaic"],
        mixup         = cfg["mixup"],
        copy_paste    = cfg["copy_paste"],
        close_mosaic  = cfg["close_mosaic"],
        # Loss weights
        box           = cfg["box"],
        cls           = cfg["cls"],
        dfl           = cfg["dfl"],
        verbose       = True,
    )

    # ── Post-training summary ─────────────────────────────
    save_dir     = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Finished     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results dir  : {save_dir.resolve()}")
    print(f"  Best weights : {best_weights.resolve()}")
    print("=" * 60)

    if best_weights.exists():
        print("\n── Final Validation on Best Weights ────────────")
        val_model    = YOLO(str(best_weights))
        val_results  = val_model.val(
            data  = cfg["data"],
            imgsz = cfg["imgsz"],
            split = "test",          # evaluate on held-out test set
        )

        map50    = val_results.box.map50
        map5095  = val_results.box.map

        print(f"\n  mAP50      : {map50:.4f}")
        print(f"  mAP50-95   : {map5095:.4f}")

        # Rating
        if map50 >= 0.70:
            rating = "EXCELLENT"
        elif map50 >= 0.60:
            rating = "GOOD — acceptable for MSc project"
        elif map50 >= 0.50:
            rating = "MODERATE — consider more epochs or data"
        else:
            rating = "LOW — check dataset and rerun"

        print(f"\n  Result     : {rating}")

        print("\n── Per-class AP (mAP50-95) ─────────────────────")
        maps = val_results.box.maps
        for i, cls in enumerate(CLASS_NAMES):
            if i < len(maps):
                bar = "█" * int(maps[i] * 20)
                print(f"  {cls:<20s}: {maps[i]:.4f}  {bar}")
        print("────────────────────────────────────────────────")

        print(f"\n  Set HAZARD_MODEL_PATH in dashboard.py to:")
        print(f"  {best_weights.resolve()}\n")
    else:
        print(f"\n  WARNING: Weights not found at {best_weights}")
        print("  Check runs/detect/ folder manually.\n")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 hazard detection model (Stage 2)"
    )
    parser.add_argument("--data",     default=DEFAULT_CONFIG["data"])
    parser.add_argument("--model",    default=DEFAULT_CONFIG["model"])
    parser.add_argument("--epochs",   type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--imgsz",    type=int, default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--batch",    type=int, default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--device",   default=DEFAULT_CONFIG["device"])
    parser.add_argument("--project",  default=DEFAULT_CONFIG["project"])
    parser.add_argument("--name",     default=DEFAULT_CONFIG["name"])
    parser.add_argument("--workers",  type=int, default=DEFAULT_CONFIG["workers"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["amp"],
        help="Enable AMP (mixed precision). Keep disabled on GTX 1650.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "data":     args.data,
        "model":    args.model,
        "epochs":   args.epochs,
        "imgsz":    args.imgsz,
        "batch":    args.batch,
        "device":   args.device,
        "project":  args.project,
        "name":     args.name,
        "workers":  args.workers,
        "patience": args.patience,
        "amp":      args.amp,
    })

    train(cfg)
