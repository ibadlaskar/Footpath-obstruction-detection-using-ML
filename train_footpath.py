

import argparse
import os
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ── Config ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data":        "dataset_footpath/data.yaml",
    "model":       "yolo11n-seg.pt",        # pretrained YOLOv11 small-seg checkpoint
    "epochs":      50,
    "imgsz":       480,
    "batch":       4,
    "patience":    20,                       # early stopping
    "lr0":         0.01,
    "lrf":         0.001,
    "momentum":    0.937,
    "weight_decay":0.0005,
    "warmup_epochs":3,
    "device":      0,                        # 0 = first GPU, "cpu" = CPU, "" = auto
    "workers":     2,
    "project":     "model",
    "name":        "footpath_segment",
    "exist_ok":    True,
    "pretrained":  True,
    "optimizer":   "AdamW",
    "seed":        42,
    "cos_lr":      True,
    "amp":         True,                     # Automatic Mixed Precision
    "fraction":    1.0,
    "val":         True,
    "save":        True,
    "save_period": 10,                       # save checkpoint every N epochs
    "cache":       False,
    "overlap_mask": True,
    "mask_ratio":   1,
    "retina_masks": True,
    "degrees":     10,
    "translate":   0.1,
    "scale":       0.5,
    "fliplr":      0.5,
    "mosaic":      1.0,
}


def validate_dataset(data_yaml: str) -> dict:
    """Check data.yaml exists and print class info."""
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
    print(f"  Classes: {config.get('nc', '?')} → {config.get('names', [])}")
    for split in ("train", "val", "test"):
        split_path = config.get(split, "")
        if split_path:
            resolved_split = Path(split_path)
            if not resolved_split.is_absolute():
                resolved_split = path.parent / resolved_split
            if not resolved_split.exists():
                raise FileNotFoundError(
                    f"{split} image folder not found: {resolved_split.resolve()}"
                )
            print(f"  {split:6s} : {split_path}")
    print("────────────────────────────────────────────────\n")
    return config


def train(cfg: dict):
    print("=" * 60)
    print("  FOOTPATH SEGMENTATION MODEL — TRAINING")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Validate dataset before starting
    validate_dataset(cfg["data"])

    # Load pretrained YOLOv11 segmentation model
    print(f"Loading base model : {cfg['model']}")
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
        overlap_mask  = cfg["overlap_mask"],
        mask_ratio    = cfg["mask_ratio"],
        retina_masks  = cfg["retina_masks"],
        degrees       = cfg["degrees"],
        translate     = cfg["translate"],
        scale         = cfg["scale"],
        fliplr        = cfg["fliplr"],
        mosaic        = cfg["mosaic"],
        verbose       = True,
    )

    # ── Post-training summary ─────────────────────────────
    best_weights = Path("runs/segment") / cfg["project"] / cfg["name"] / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Best weights : {best_weights.resolve()}")
    print("=" * 60)

    if best_weights.exists():
        print("\n── Validation on best weights ──────────────────")
        val_model = YOLO(str(best_weights))
        val_results = val_model.val(data=cfg["data"], imgsz=cfg["imgsz"])
        print(f"  mAP50     : {val_results.seg.map50:.4f}")
        print(f"  mAP50-95  : {val_results.seg.map:.4f}")
        print("────────────────────────────────────────────────")
        print(f"\n✅ Stage 1 model ready. Set FOOTPATH_MODEL_PATH in dashboard.py to:")
        print(f"   {best_weights.resolve()}\n")
    else:
        print(f"⚠  Weights not found at expected path: {best_weights}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 footpath segmentation model"
    )
    parser.add_argument("--data",    default=DEFAULT_CONFIG["data"],    help="Path to data.yaml")
    parser.add_argument("--model",   default=DEFAULT_CONFIG["model"],   help="Base model checkpoint")
    parser.add_argument("--epochs",  type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--imgsz",   type=int,   default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--batch",   type=int,   default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--device",  default=DEFAULT_CONFIG["device"],  help="cuda device or 'cpu'")
    parser.add_argument("--degrees", type=float, default=DEFAULT_CONFIG["degrees"])
    parser.add_argument("--translate", type=float, default=DEFAULT_CONFIG["translate"])
    parser.add_argument("--scale",   type=float, default=DEFAULT_CONFIG["scale"])
    parser.add_argument("--fliplr",  type=float, default=DEFAULT_CONFIG["fliplr"])
    parser.add_argument("--mosaic",  type=float, default=DEFAULT_CONFIG["mosaic"])
    parser.add_argument("--project", default=DEFAULT_CONFIG["project"])
    parser.add_argument("--name",    default=DEFAULT_CONFIG["name"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "data":    args.data,
        "model":   args.model,
        "epochs":  args.epochs,
        "imgsz":   args.imgsz,
        "batch":   args.batch,
        "device":  args.device,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale":   args.scale,
        "fliplr":  args.fliplr,
        "mosaic":  args.mosaic,
        "project": args.project,
        "name":    args.name,
    })

    train(cfg)
