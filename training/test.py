"""
DeepFakeBench test.py (pretrained evaluation) + optional feature dump (.pkl) for t-SNE

✅ What this script does:
- Load detector yaml + (optional) training/test config overrides
- Build DeepFakeBench test dataloader(s)
- Load pretrained weights (.pth)
- Run inference: collect prob, label(binary), feat
- Compute metrics via get_test_metrics
- ✅ ONLY IF you pass --save_feat:
    Save feature pickle per dataset:
      tsne_dict_<model>_<dataset>.pkl
    containing:
      {'feat': (N,D), 'label_spe': (N,), 'pred': (N,), 'label': (N,), 'img_names': ...}

Run example:
cd /kaggle/working/DeepfakeBench
PYTHONPATH=. python -u training/test.py \
  --detector_path training/config/detector/BiasLoraAsy.yaml \
  --test_dataset FaceForensics++ \
  --weights_path /kaggle/input/datasets/xuanhuydinh/deepfakebench/Weight/effort_clip_L14_trainOn_FaceForensic.pth \
  --save_feat \
  --feat_out_dir /kaggle/working/tsne_pkls
"""

import os
import sys
import yaml
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

# ---------------- PATH FIX (repo root) ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /kaggle/working/DeepfakeBench
sys.path.insert(0, ROOT)

# ---------------- Imports: metrics/detectors/datasets ----------------
from metrics.utils import get_test_metrics
from detectors import DETECTOR

try:
    from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
except Exception:
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="DeepFakeBench test + optional feature dump for TSNE")
parser.add_argument(
    "--detector_path",
    type=str,
    default="training/config/detector/BiasLoraAsy.yaml",
    help="path to detector YAML file",
)
parser.add_argument("--test_dataset", nargs="+", default=None, help="list of test dataset names")
parser.add_argument("--weights_path", type=str, default=None, help="path to pretrained weights .pth")

# ✅ Only save feature when user passes this flag
parser.add_argument("--save_feat", action="store_true", default=False, help="save feature pkl for TSNE")
parser.add_argument("--feat_out_dir", type=str, default="tsne_pkls", help="output directory for tsne_dict_*.pkl")
parser.add_argument("--max_samples", type=int, default=None, help="optional cap on number of samples per dataset")
parser.add_argument(
    "--allow_partial_load",
    action="store_true",
    default=False,
    help="allow missing/unexpected checkpoint keys instead of failing",
)

args = parser.parse_args()


def init_seed(config):
    if config.get("manualSeed", None) is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    np.random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if config.get("cuda", True):
        torch.cuda.manual_seed_all(config["manualSeed"])


def prepare_testing_data(config):
    """
    Mirrors DeepFakeBench style:
    - DeepfakeAbstractBaseDataset(config, mode='test') for most cases
    """
    def get_test_data_loader(cfg, test_name):
        cfg = cfg.copy()
        cfg["test_dataset"] = test_name

        test_set = DeepfakeAbstractBaseDataset(config=cfg, mode="test")

        dl = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=int(cfg["test_batchSize"]),
            shuffle=False,
            num_workers=int(cfg["workers"]),
            collate_fn=test_set.collate_fn,
            drop_last=False,
        )
        return dl

    test_data_loaders = {}
    for one_test_name in config["test_dataset"]:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


@torch.no_grad()
def inference(model, data_dict):
    return model(data_dict, inference=True)


def _move_batch_to_device(data_dict):
    # trainer style: move all tensor fields except 'name'
    for k in list(data_dict.keys()):
        if data_dict[k] is None or k == "name":
            continue
        if torch.is_tensor(data_dict[k]):
            data_dict[k] = data_dict[k].to(device)

    # Handle video frames: [B,T,3,H,W] -> take first frame
    if "image" in data_dict and torch.is_tensor(data_dict["image"]) and data_dict["image"].ndim == 5:
        data_dict["image"] = data_dict["image"][:, 0]


def test_one_dataset(model, data_loader, max_samples=None):
    """
    Returns:
      pred_prob: (N,) float
      label_bin: (N,) int (0/1)
      feat:      (N,D) float
      label_spe: (N,) int  (0..4 if exists; else fallback to binary label)
      img_names: list[str]
    """
    pred_list = []
    label_list = []
    feat_list = []
    label_spe_list = []
    img_names = []

    n = 0
    for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Keep original label_spe if present BEFORE overwriting label
        batch_label_spe = None
        if "label_spe" in data_dict and data_dict["label_spe"] is not None:
            batch_label_spe = data_dict["label_spe"]

        # Convert label to binary for metrics (same as Trainer.test_one_dataset does)
        # In many configs, data_dict['label'] is multi-class or real/fake id, we binarize:
        label_bin = torch.where(data_dict["label"] != 0, 1, 0)
        data_dict["label"] = label_bin

        _move_batch_to_device(data_dict)

        preds = inference(model, data_dict)  # expects {'prob','feat'}

        pred_list.append(preds["prob"].detach().cpu().numpy())
        label_list.append(data_dict["label"].detach().cpu().numpy())
        feat_list.append(preds["feat"].detach().cpu().numpy())

        if batch_label_spe is not None:
            label_spe_list.append(batch_label_spe.detach().cpu().numpy())

        # image names (if available) for debugging/traceability
        if hasattr(data_loader.dataset, "data_dict") and isinstance(data_loader.dataset.data_dict, dict):
            # dataset.data_dict['image'] is global list, but per-batch names may not be provided
            pass
        if "name" in data_dict and data_dict["name"] is not None:
            # some datasets provide per-item names
            try:
                img_names.extend(list(data_dict["name"]))
            except Exception:
                pass

        n += preds["feat"].shape[0]
        if max_samples is not None and n >= max_samples:
            break

    pred_prob = np.concatenate(pred_list, axis=0)
    label_bin = np.concatenate(label_list, axis=0).astype(int)
    feat = np.concatenate(feat_list, axis=0)

    if len(label_spe_list) > 0:
        label_spe = np.concatenate(label_spe_list, axis=0).astype(int)
    else:
        # fallback if dataset has no label_spe
        label_spe = label_bin.astype(int)

    return pred_prob, label_bin, feat, label_spe, img_names


def _strip_module_prefix(key):
    return key[len("module."):] if key.startswith("module.") else key


def _slice_like_predictions(values, n):
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values[:n]
    if isinstance(values, tuple):
        return list(values[:n])
    if isinstance(values, list):
        return values[:n]
    return values


def _class_accuracy(prob, label):
    pred = (prob > 0.5).astype(int)
    real_idx = label == 0
    fake_idx = label == 1
    acc_real = float((pred[real_idx] == label[real_idx]).mean()) if real_idx.any() else float("nan")
    acc_fake = float((pred[fake_idx] == label[fake_idx]).mean()) if fake_idx.any() else float("nan")
    return acc_real, acc_fake


def save_tsne_pkl(out_dir, model_name, dataset_name, weights_path, pred_prob, label_bin, feat, label_spe, img_names):
    os.makedirs(out_dir, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    safe_data = dataset_name.replace("/", "_")
    out_path = os.path.join(out_dir, f"tsne_dict_{safe_model}_{safe_data}.pkl")

    tsne_dict = {
        "feat": feat,                 # (N,D)
        "label_spe": label_spe,       # (N,)
        "pred": pred_prob,            # (N,)
        "label": label_bin,           # (N,)
        "img_names": img_names,       # optional
        "dataset": dataset_name,
        "model_name": model_name,
        "weights_path": weights_path,
    }

    with open(out_path, "wb") as f:
        pickle.dump(tsne_dict, f)

    print(f"[SAVE_FEAT] {out_path}")
    print(f"           feat={feat.shape}, label_spe_unique={np.unique(label_spe)}")


def main():
    # ---- load yaml configs ----
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)

    # optional override configs (like your code)
    test_cfg_path = os.path.join(ROOT, "training", "config", "test_config.yaml")
    if os.path.exists(test_cfg_path):
        with open(test_cfg_path, "r") as f:
            config2 = yaml.safe_load(f)
        if isinstance(config2, dict):
            config.update(config2)

    # ensure required defaults
    config.setdefault("cuda", True)
    config.setdefault("cudnn", True)
    config.setdefault("workers", 4)
    config.setdefault("test_batchSize", 64)
    config.setdefault("metric_scoring", "auc")

    # override datasets from CLI
    if args.test_dataset is not None:
        config["test_dataset"] = args.test_dataset

    if "test_dataset" not in config or not config["test_dataset"]:
        raise ValueError("You must provide --test_dataset (e.g., FaceForensics++) or set it in YAML.")

    # seed + cudnn
    init_seed(config)
    if config.get("cudnn", True):
        cudnn.benchmark = True

    # ---- build dataloaders ----
    test_data_loaders = prepare_testing_data(config)

    # ---- build model ----
    if "model_name" not in config:
        raise ValueError("YAML must contain config['model_name'] (e.g., gend_effort).")

    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)

    # ---- load weights ----
    if args.weights_path is None:
        raise ValueError("--weights_path is required to test a pretrained model.")

    print(f"===> Loading weights from: {args.weights_path}")
    ckpt = torch.load(args.weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[_strip_module_prefix(k)] = v

    msg = model.load_state_dict(new_ckpt, strict=False)
    print("===> Load checkpoint done!")
    print(f"     Missing keys: {len(msg.missing_keys)}")
    print(f"     Unexpected keys: {len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print(f"     Example missing: {msg.missing_keys[:3]} ...")
    if len(msg.unexpected_keys) > 0:
        print(f"     Example unexpected: {msg.unexpected_keys[:3]} ...")

    if not args.allow_partial_load and (msg.missing_keys or msg.unexpected_keys):
        raise RuntimeError(
            "Checkpoint does not exactly match the model. "
            "This often means --detector_path, LoRA config, or --weights_path is wrong. "
            "Pass --allow_partial_load only if you intentionally want partial loading."
        )

    # ---- test ----
    model.eval()
    all_metrics = {}

    for dataset_name, loader in test_data_loaders.items():
        print(f"\n===== Testing on: {dataset_name} =====")
        data_dict_global = loader.dataset.data_dict if hasattr(loader.dataset, "data_dict") else {}
        img_names_global = data_dict_global.get("image", None)

        pred_prob, label_bin, feat, label_spe, img_names_batch = test_one_dataset(
            model, loader, max_samples=args.max_samples
        )

        # metrics use binary labels + prob
        # DeepFakeBench get_test_metrics expects img_names list; prefer dataset.data_dict['image'] when available
        img_names_for_metric = img_names_global if img_names_global is not None else img_names_batch
        img_names_for_metric = _slice_like_predictions(img_names_for_metric, len(pred_prob))
        if img_names_for_metric is not None and len(img_names_for_metric) != len(pred_prob):
            raise ValueError(
                f"Metric input length mismatch for {dataset_name}: "
                f"{len(pred_prob)} predictions but {len(img_names_for_metric)} image names."
            )
        metric_one_dataset = get_test_metrics(y_pred=pred_prob, y_true=label_bin, img_names=img_names_for_metric)
        all_metrics[dataset_name] = metric_one_dataset

        # print metrics
        for k, v in metric_one_dataset.items():
            if k in ["pred", "label", "dataset_dict"]:
                continue
            print(f"{k}: {v}")
        acc_real, acc_fake = _class_accuracy(pred_prob, label_bin)
        print(f"acc_real: {acc_real}; acc_fake: {acc_fake}")
        print(
            "pred_prob: "
            f"min={float(pred_prob.min()):.6f}, "
            f"max={float(pred_prob.max()):.6f}, "
            f"mean={float(pred_prob.mean()):.6f}, "
            f">0.5={float((pred_prob > 0.5).mean()):.6f}"
        )

        # ✅ Save feature ONLY when --save_feat is provided
        if args.save_feat:
            save_tsne_pkl(
                out_dir=args.feat_out_dir,
                model_name=config["model_name"],
                dataset_name=dataset_name,
                weights_path=args.weights_path,
                pred_prob=pred_prob,
                label_bin=label_bin,
                feat=feat,
                label_spe=label_spe,
                img_names=(img_names_for_metric if isinstance(img_names_for_metric, list) else []),
            )

    print("\n===> Test Done!")
    if args.save_feat:
        print(f"===> Feature pkls saved under: {os.path.abspath(args.feat_out_dir)}")


if __name__ == "__main__":
    main()
