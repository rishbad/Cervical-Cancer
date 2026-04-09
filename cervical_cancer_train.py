"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CERVICAL CANCER DETECTION — Advanced Training Pipeline                    ║
║   SIPaKMeD Dataset | EfficientNet-B4 + ResNet50 + DenseNet + Ensemble       ║
║                                                                              ║
║   FEATURES:                                                                  ║
║   ✔ 3 Models + optional ConvNeXt + Ensemble + Stacked (OOF)                 ║
║   ✔ Focal Loss, EMA, RandAugment, MLP meta-learner                          ║
║   ✔ Hybrid with trained backbone (improved accuracy)                        ║
║   ✔ Checkpoints every epoch + resume from checkpoint                        ║
║   ✔ Full training history saved to JSON for Streamlit dashboard             ║
║   ✔ Grad-CAM per-class gallery saved as images                              ║
║   ✔ Confusion matrix, ROC, F1, AUC all persisted                           ║
║   ✔ Label smoothing + CutMix + MixUp augmentation                          ║
║   ✔ Mixed precision (AMP) training                                          ║
║   ✔ Cosine annealing with warm restarts                                     ║
║   ✔ Test-Time Augmentation (TTA) for final predictions                      ║
║                                                                              ║
║   HOW TO RUN:                                                                ║
║   1. pip install -r requirements.txt                                         ║
║   2. Place SIPaKMeD folder next to this script                              ║
║   3. python cervical_cancer_train.py                                         ║
║   4. python cervical_cancer_train.py --resume  (resume from checkpoint)     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, copy, time, random, warnings, argparse, shutil
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import torchvision.transforms as T
from torchvision import models

try:
    import timm; TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    DATA_DIR       = "./SIPaKMeD"
    OUTPUT_DIR     = "./cervical_output"
    CHECKPOINT_DIR = "./cervical_output/checkpoints"
    RESULTS_DIR    = "./cervical_output/results"
    PLOTS_DIR      = "./cervical_output/plots"
    GRADCAM_DIR    = "./cervical_output/gradcam"

    CLASSES = [
        "Dyskeratotic",
        "Koilocytotic",
        "Metaplastic",
        "Parabasal",
        "Superficial_Intermediate"
    ]
    CLASS_LABELS = {
        "im_dyskeratotic"            : 0,
        "im_koilocytotic"            : 1,
        "im_metaplastic"             : 2,
        "im_parabasal"               : 3,
        "im_superficial-intermediate": 4,
    }
    NUM_CLASSES = 5

    IMG_SIZE   = 384          # ↑ from 224 — captures finer cervical cell detail
    MEAN       = [0.485, 0.456, 0.406]
    STD        = [0.229, 0.224, 0.225]

    BATCH_SIZE    = 16        # reduced from 32 to fit 384px images in GPU memory
    NUM_EPOCHS    = 60
    LR            = 3e-4
    WEIGHT_DECAY  = 1e-4
    PATIENCE      = 15
    GRAD_CLIP     = 1.0
    LABEL_SMOOTH  = 0.1
    MIXUP_ALPHA   = 0.4
    CUTMIX_ALPHA  = 1.0
    TTA_STEPS     = 12        # ↑ from 5 — more TTA passes = better inference accuracy

    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15

    SEED        = 42
    NUM_WORKERS = 0
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP     = torch.cuda.is_available()

    # ↑ Added ConvNeXt_Tiny + Swin_Tiny for architectural diversity & better ensemble
    MODELS_TO_TRAIN = ["EfficientNet_B4", "ResNet50", "DenseNet201", "ConvNeXt_Tiny", "Swin_Tiny"]

    # ── Improvements ──
    USE_FOCAL_LOSS        = True                # Focal loss for hard examples (vs LabelSmoothCE)
    FOCAL_GAMMA           = 2.0                 # higher = more focus on hard examples
    USE_CLASS_WEIGHTED_FL = True                # ↑ per-class weights inside Focal Loss
    USE_EMA               = True                # Exponential Moving Average of weights
    EMA_DECAY             = 0.9999              # EMA decay per step
    USE_RAND_AUGMENT      = True                # RandAugment for stronger augmentation
    USE_SAM               = False                # ↑ Sharpness-Aware Minimization

    # Hybrid (classical ML) options
    TRAIN_HYBRID = True
    HYBRID_BACKBONE = "EfficientNet_B4"         # deep embedding source for fusion
    HYBRID_USE_TRAINED = True                   # use trained backbone (vs frozen ImageNet)
    HYBRID_IMAGE_SIZE = 384                    # match new IMG_SIZE
    HYBRID_COLOR_BINS = 16

    # Stacked ensemble
    TRAIN_STACKED = True
    STACKED_META = "mlp"                        # 'logreg' or 'mlp' (MLP often better)
    STACKED_OOF = True
    STACKED_OOF_FOLDS = 5
    STACKED_OOF_EPOCHS = 20                   # ↑ from 20 — meta-learner trains more thoroughly

    @classmethod
    def make_dirs(cls):
        for d in [cls.OUTPUT_DIR, cls.CHECKPOINT_DIR, cls.RESULTS_DIR,
                  cls.PLOTS_DIR, cls.GRADCAM_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True

set_seed(Config.SEED)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def discover_images(data_dir):
    data_path = Path(data_dir)
    samples   = []
    exts      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    found_dirs = set()

    for img_path in data_path.rglob("*"):
        if img_path.suffix.lower() not in exts:
            continue
        for part in img_path.parts:
            key = part.lower().replace(" ","_").replace("-","_")
            matched = False
            for label_key, idx in Config.CLASS_LABELS.items():
                lk = label_key.lower().replace("-","_")
                if lk in key or key in lk or \
                   lk.replace("im_","") in key or key.replace("im_","") in lk:
                    samples.append((str(img_path), idx))
                    found_dirs.add(part)
                    matched = True
                    break
            if matched:
                break

    samples = list(set(samples))
    if not samples:
        print(f"\n[ERROR] No images found in {data_path.resolve()}")
        print("Expected subfolders: im_Dyskeratotic, im_Koilocytotic,")
        print("im_Metaplastic, im_Parabasal, im_Superficial-Intermediate")
        sys.exit(1)

    cnt = Counter(s[1] for s in samples)
    print(f"\n[DATA] Found {len(samples)} images in {len(found_dirs)} folders")
    for i, cls in enumerate(Config.CLASSES):
        print(f"       [{i}] {cls:<35} {cnt.get(i,0):>4} images")
    return samples


def stratified_split(samples):
    buckets = defaultdict(list)
    for p, l in samples:
        buckets[l].append((p, l))
    train, val, test = [], [], []
    for l, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        nt = int(n * Config.TRAIN_RATIO)
        nv = int(n * Config.VAL_RATIO)
        train += items[:nt]; val += items[nt:nt+nv]; test += items[nt+nv:]
    random.shuffle(train)
    print(f"[DATA] Split → Train:{len(train)} Val:{len(val)} Test:{len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def get_transform(phase):
    if phase == "train":
        augs = [
            T.Resize((Config.IMG_SIZE + 40, Config.IMG_SIZE + 40)),
            T.RandomCrop(Config.IMG_SIZE),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.3),
            T.RandomRotation(25),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
            T.RandomAffine(degrees=0, translate=(0.1,0.1), shear=12, scale=(0.85,1.15)),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ]
        if getattr(Config, "USE_RAND_AUGMENT", False):
            try:
                augs.insert(-1, T.RandAugment(num_ops=2, magnitude=9))
            except AttributeError:
                pass  # older torchvision
        augs += [
            T.ToTensor(),
            T.Normalize(Config.MEAN, Config.STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
        return T.Compose(augs)
    elif phase == "tta":
        return T.Compose([
            T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(Config.MEAN, Config.STD),
        ])
    else:
        return T.Compose([
            T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(Config.MEAN, Config.STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class CervicalDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class CervicalPathDataset(Dataset):
    """Returns (image_tensor, label, path) for feature extraction."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = img
        return img_t, label, path


def make_weighted_sampler(samples):
    labels  = [s[1] for s in samples]
    counts  = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP / CUTMIX
# ─────────────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    B, C, H, W = x.shape
    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    x_new       = x.clone()
    x_new[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_new, y, y[idx], lam_adj


def mixed_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothCE(nn.Module):
    def __init__(self, n_cls, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.n_cls     = n_cls
        self.weight    = weight

    def forward(self, pred, target):
        conf = 1 - self.smoothing
        sv   = self.smoothing / (self.n_cls - 1)
        oh   = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        sm   = oh * conf + (1 - oh) * sv
        lp   = F.log_softmax(pred, dim=1)
        if self.weight is not None:
            w = self.weight.to(pred.device)[target].unsqueeze(1)
            return -(sm * lp * w).sum(dim=1).mean()
        return -(sm * lp).sum(dim=1).mean()


class FocalLoss(nn.Module):
    """Focuses on hard examples; reduces loss contribution from easy ones."""
    def __init__(self, gamma=2.0, alpha=None, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights (tensor or None)
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction="none", weight=self.weight)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(pred.device)[target]
            focal = alpha_t * focal
        return focal.mean() if self.reduction == "mean" else focal.sum()


def class_weights(samples):
    labels  = [s[1] for s in samples]
    counts  = Counter(labels)
    total   = sum(counts.values())
    w = [total / (Config.NUM_CLASSES * counts.get(i, 1))
         for i in range(Config.NUM_CLASSES)]
    return torch.FloatTensor(w).to(Config.DEVICE)


class ModelEMA:
    """Exponential Moving Average of model parameters. Improves generalization."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def store(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if n in self.shadow}

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        del self.backup


# ─────────────────────────────────────────────────────────────────────────────
# SAM OPTIMIZER — Sharpness-Aware Minimization
# Finds flatter loss minima → better generalization on small medical datasets
# ─────────────────────────────────────────────────────────────────────────────
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021). Wraps a base optimizer."""
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM requires two forward passes — use first_step/second_step.")

    def _grad_norm(self):
        norms = [p.grad.norm(p=2).to(self.param_groups[0]["params"][0])
                 for group in self.param_groups
                 for p in group["params"] if p.grad is not None]
        return torch.stack(norms).norm(p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name, num_classes, pretrained=True):
    if name == "EfficientNet_B4":
        if TIMM_AVAILABLE:
            m = timm.create_model("efficientnet_b4", pretrained=pretrained,
                                  num_classes=num_classes, drop_rate=0.4,
                                  drop_path_rate=0.2)
        else:
            m = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
            m.classifier = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, num_classes))
        print(f"[MODEL] EfficientNet-B4 {'(timm)' if TIMM_AVAILABLE else '(torchvision)'}")

    elif name == "ResNet50":
        m = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes))
        print("[MODEL] ResNet50 with custom head")

    elif name == "DenseNet201":
        m = models.densenet201(
            weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes))
        print("[MODEL] DenseNet-201 with custom head")

    elif name == "ConvNeXt_Tiny":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for ConvNeXt_Tiny. pip install timm")
        m = timm.create_model("convnext_tiny", pretrained=pretrained,
                              num_classes=num_classes, drop_path_rate=0.2)
        print("[MODEL] ConvNeXt-Tiny (timm)")

    elif name == "Swin_Tiny":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for Swin_Tiny. pip install timm")
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained,
                              num_classes=num_classes, drop_path_rate=0.2,
                              img_size=Config.IMG_SIZE)
        print("[MODEL] Swin-Tiny Transformer (timm)")

    else:
        raise ValueError(f"Unknown model: {name}")

    return m


def build_embedding_model(model_name):
    """
    Returns a model that outputs a 1D embedding per image (no classifier head).
    We keep it simple and robust across torchvision/timm variants.
    """
    if model_name == "EfficientNet_B4":
        if TIMM_AVAILABLE:
            m = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)
        else:
            m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            m.classifier = nn.Identity()
        return m

    if model_name == "ResNet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        return m

    if model_name == "DenseNet201":
        m = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
        return m

    if model_name == "ConvNeXt_Tiny" and TIMM_AVAILABLE:
        m = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        return m

    if model_name == "Swin_Tiny" and TIMM_AVAILABLE:
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True,
                              num_classes=0, img_size=Config.IMG_SIZE)
        return m

    raise ValueError(f"Unknown embedding backbone: {model_name}")


def _color_histogram(img_rgb_u8, bins=16):
    # img: HxWx3 uint8
    feats = []
    for c in range(3):
        h, _ = np.histogram(img_rgb_u8[..., c], bins=bins, range=(0, 255), density=True)
        feats.append(h.astype(np.float32))
    return np.concatenate(feats, axis=0)


def _color_moments(img_rgb_u8):
    x = img_rgb_u8.astype(np.float32) / 255.0
    # mean, std, skew per channel
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1)) + 1e-8
    skew = (((x - mean) / std) ** 3).mean(axis=(0, 1))
    return np.concatenate([mean, std, skew], axis=0).astype(np.float32)


def _edge_density(img_rgb_u8):
    # simple gradient magnitude summary on grayscale
    g = (0.2989 * img_rgb_u8[..., 0] + 0.5870 * img_rgb_u8[..., 1] + 0.1140 * img_rgb_u8[..., 2]).astype(np.float32)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    mag = gx + gy
    # a few robust stats
    return np.array([mag.mean(), mag.std(), np.percentile(mag, 90)], dtype=np.float32)


def handcrafted_features_from_path(img_path):
    img = Image.open(img_path).convert("RGB").resize((Config.HYBRID_IMAGE_SIZE, Config.HYBRID_IMAGE_SIZE))
    arr = np.array(img, dtype=np.uint8)
    return np.concatenate([
        _color_histogram(arr, bins=Config.HYBRID_COLOR_BINS),
        _color_moments(arr),
        _edge_density(arr),
    ], axis=0)


@torch.no_grad()
def extract_deep_embeddings(backbone_name, samples, batch_size=None, use_trained=False):
    """
    use_trained: if True, load backbone from best checkpoint (fine-tuned on dataset)
    instead of ImageNet pretrained. Improves Hybrid accuracy.
    """
    batch_size = batch_size or Config.BATCH_SIZE
    tf = get_transform("val")
    ds = CervicalPathDataset(samples, tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS)

    emb_model = build_embedding_model(backbone_name).to(Config.DEVICE)
    if use_trained:
        ckpt = Path(Config.CHECKPOINT_DIR) / backbone_name / "best.pth"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=Config.DEVICE)
            emb_model.load_state_dict(state["model"], strict=False)
            print(f"  [Hybrid] Using trained backbone from {ckpt.name}")
    emb_model.eval()

    embs = []
    labels = []
    paths = []
    for imgs, y, p in tqdm(loader, desc=f"  Embeddings ({backbone_name})", leave=False, ncols=80):
        out = emb_model(imgs.to(Config.DEVICE))
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.view(out.size(0), -1).detach().cpu().numpy().astype(np.float32)
        embs.append(out)
        labels.extend(list(y))
        paths.extend(list(p))
    return np.vstack(embs), np.array(labels, dtype=np.int64), paths


def extract_handcrafted_features(paths):
    feats = []
    for p in tqdm(paths, desc="  Handcrafted features", leave=False, ncols=80):
        feats.append(handcrafted_features_from_path(p))
    return np.vstack(feats).astype(np.float32)


def train_hybrid_classifier(X_train, y_train):
    """
    Hybrid classical model: StandardScaler + calibrated SVM.
    Calibration gives reliable class probabilities for ROC/AUC + soft-voting later.
    """
    # Sanitize NaN/Inf in feature matrix before fitting
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    base = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", SVC(C=5.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=False)),
    ])
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)
    return clf


def train_stacked_meta_classifier(X_meta_train, y_meta_train):
    """
    Meta-classifier over base-model probabilities.
    Input shape: (N, NUM_MODELS * NUM_CLASSES)
    """
    meta = getattr(Config, "STACKED_META", "logreg")
    if meta == "logreg":
        clf = LogisticRegression(
            max_iter=5000, solver="lbfgs", class_weight="balanced", C=2.0,
        )
        clf.fit(X_meta_train, y_meta_train)
        return clf
    if meta == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=Config.SEED,
        )
        sw = compute_sample_weight("balanced", y_meta_train)
        clf.fit(X_meta_train, y_meta_train, sample_weight=sw)
        return clf
    raise ValueError(f"Unknown STACKED_META: {meta}")


@torch.no_grad()
def predict_probs_tta(model, samples, n_tta=5):
    """
    Returns softmax probabilities (N, NUM_CLASSES) for given samples using TTA.
    """
    model.eval()
    tta_tf = get_transform("tta")
    val_tf = get_transform("val")
    ds_base = CervicalDataset(samples, val_tf)
    loader = DataLoader(ds_base, batch_size=Config.BATCH_SIZE,
                        shuffle=False, num_workers=Config.NUM_WORKERS)

    all_probs = []
    for imgs, _ in tqdm(loader, desc="  Predict base", leave=False, ncols=70):
        p = F.softmax(model(imgs.to(Config.DEVICE)), dim=1).cpu().numpy()
        all_probs.append(p)
    base_probs = np.vstack(all_probs)

    # TTA passes
    tta_accum = np.zeros_like(base_probs)
    ds_tta = CervicalDataset(samples, tta_tf)
    for _ in range(n_tta):
        loader_t = DataLoader(ds_tta, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS)
        step_probs = []
        for imgs, _ in loader_t:
            p = F.softmax(model(imgs.to(Config.DEVICE)), dim=1).cpu().numpy()
            step_probs.append(p)
        tta_accum += np.vstack(step_probs)

    probs = (base_probs + tta_accum) / (1 + n_tta)
    # Sanitize NaN/Inf
    if not np.isfinite(probs).all():
        bad = ~np.isfinite(probs).all(axis=1)
        probs[bad] = 1.0 / Config.NUM_CLASSES
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _labels_from_samples(samples):
    return np.array([l for _, l in samples], dtype=np.int64)


def run_oof_stacking(train_samples, base_model_names, resume=False):
    """
    Train fold-specific base models and collect out-of-fold probabilities.
    Then fit the meta-classifier on the OOF probability features.
    """
    y = _labels_from_samples(train_samples)
    idx_all = np.arange(len(train_samples))
    skf = StratifiedKFold(n_splits=Config.STACKED_OOF_FOLDS, shuffle=True, random_state=Config.SEED)

    oof_by_model = {
        name: np.zeros((len(train_samples), Config.NUM_CLASSES), dtype=np.float32)
        for name in base_model_names
    }

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(idx_all, y), start=1):
        fold_train = [train_samples[i] for i in tr_idx]
        fold_val = [train_samples[i] for i in va_idx]

        sampler = make_weighted_sampler(fold_train)
        train_ds = CervicalDataset(fold_train, get_transform("train"))
        val_ds = CervicalDataset(fold_val, get_transform("val"))
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                                  sampler=sampler, num_workers=Config.NUM_WORKERS,
                                  pin_memory=(Config.DEVICE == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                                shuffle=False, num_workers=Config.NUM_WORKERS)

        print(f"\n{'─'*65}")
        print(f"  OOF Fold {fold_idx}/{Config.STACKED_OOF_FOLDS}  (train={len(fold_train)} val={len(fold_val)})")
        print(f"{'─'*65}")

        for name in base_model_names:
            run_name = f"{name}_OOF_F{fold_idx}"
            model, _ = train_one_model(
                name, train_loader, val_loader, fold_train,
                resume=False, run_name=run_name, num_epochs=Config.STACKED_OOF_EPOCHS
            )
            pv = predict_probs_tta(model, fold_val, n_tta=Config.TTA_STEPS)
            oof_by_model[name][va_idx] = pv.astype(np.float32)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    oof_X = np.concatenate([oof_by_model[n] for n in base_model_names], axis=1)
    meta = train_stacked_meta_classifier(oof_X, y)
    return meta


class EnsembleModel(nn.Module):
    def __init__(self, model_dict, weights=None):
        super().__init__()
        self.models  = nn.ModuleList(list(model_dict.values()))
        self.names   = list(model_dict.keys())
        self.weights = weights or [1/len(self.models)] * len(self.models)

    def forward(self, x):
        probs = [w * F.softmax(m(x), dim=1)
                 for m, w in zip(self.models, self.weights)]
        return sum(probs)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, model_name, ckpt_dir):
        self.name    = model_name
        self.dir     = Path(ckpt_dir) / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.dir / "checkpoint_log.json"
        self.log      = self._load_log()

    def _load_log(self):
        if self.log_file.exists():
            with open(self.log_file) as f:
                return json.load(f)
        return {"checkpoints": [], "best_epoch": -1, "best_val_acc": 0.0}

    def _save_log(self):
        with open(self.log_file, "w") as f:
            json.dump(self.log, f, indent=2)

    def save(self, model, optimizer, scheduler, scaler, epoch,
             val_acc, val_loss, history, is_best=False):
        state = {
            "epoch"    : epoch,
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler"   : scaler.state_dict(),
            "val_acc"  : val_acc,
            "val_loss" : val_loss,
            "history"  : history,
            "timestamp": datetime.now().isoformat(),
        }
        # Always save latest
        latest = self.dir / "latest.pth"
        torch.save(state, latest)

        # Save periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            ep_ckpt = self.dir / f"epoch_{epoch:03d}.pth"
            torch.save(state, ep_ckpt)
            self.log["checkpoints"].append({
                "epoch": epoch, "val_acc": val_acc,
                "val_loss": val_loss, "file": str(ep_ckpt)
            })

        if is_best:
            best_path = self.dir / "best.pth"
            torch.save(state, best_path)
            self.log["best_epoch"]   = epoch
            self.log["best_val_acc"] = val_acc
            print(f"  [Checkpoint] ★ Best saved  →  epoch {epoch}  acc {val_acc:.4f}")

        self._save_log()

    def load_latest(self, model, optimizer=None, scheduler=None, scaler=None):
        latest = self.dir / "latest.pth"
        if not latest.exists():
            return 0, {}
        state = torch.load(latest, map_location=Config.DEVICE)
        model.load_state_dict(state["model"])
        if optimizer  and "optimizer"  in state: optimizer.load_state_dict(state["optimizer"])
        if scheduler  and "scheduler"  in state: scheduler.load_state_dict(state["scheduler"])
        if scaler     and "scaler"     in state: scaler.load_state_dict(state["scaler"])
        print(f"  [Checkpoint] Resumed from epoch {state['epoch']}  val_acc {state['val_acc']:.4f}")
        return state["epoch"], state.get("history", {})

    def load_best(self, model):
        best = self.dir / "best.pth"
        if best.exists():
            state = torch.load(best, map_location=Config.DEVICE)
            model.load_state_dict(state["model"])
            print(f"  [Checkpoint] Loaded best  →  epoch {state['epoch']}  acc {state['val_acc']:.4f}")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._acts = self._grads = None
        self._h    = [
            target_layer.register_forward_hook(
                lambda _, __, o: setattr(self, '_acts', o.detach())),
            target_layer.register_full_backward_hook(
                lambda _, __, g: setattr(self, '_grads', g[0].detach())),
        ]

    def generate(self, img_t, cls=None):
        self.model.eval()
        out = self.model(img_t.unsqueeze(0).to(Config.DEVICE))
        if cls is None: cls = out.argmax(1).item()
        self.model.zero_grad()
        out[0, cls].backward()
        w   = self._grads.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, (Config.IMG_SIZE, Config.IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls

    def remove(self):
        for h in self._h: h.remove()


def get_last_conv(model, name):
    if "EfficientNet" in name:
        return model.conv_head if TIMM_AVAILABLE else model.features[-1][0]
    elif "ResNet" in name:
        l = model.layer4[-1]
        return getattr(l, "conv3", l.conv2)
    elif "DenseNet" in name:
        return model.features.denseblock4.denselayer32.conv2 \
               if hasattr(model.features.denseblock4, "denselayer32") \
               else list(model.features.denseblock4.children())[-1].conv2
    elif "ConvNeXt" in name:
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv
    elif "Swin" in name:
        last_ln = None
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                last_ln = m
        return last_ln
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TTA EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_tta(model, samples, criterion, n_tta=5):
    """Test-Time Augmentation for improved final evaluation."""
    model.eval()
    tta_tf  = get_transform("tta")
    val_tf  = get_transform("val")
    ds_base = CervicalDataset(samples, val_tf)
    loader  = DataLoader(ds_base, batch_size=Config.BATCH_SIZE,
                         shuffle=False, num_workers=Config.NUM_WORKERS)

    all_probs  = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="  TTA eval", leave=False, ncols=70):
        batch_probs = F.softmax(
            model(imgs.to(Config.DEVICE)), dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.append(batch_probs)

    base_probs = np.vstack(all_probs)

    # TTA augmented passes
    tta_accum = np.zeros_like(base_probs)
    ds_tta    = CervicalDataset(samples, tta_tf)
    for _ in range(n_tta):
        loader_t   = DataLoader(ds_tta, batch_size=Config.BATCH_SIZE,
                                shuffle=False, num_workers=Config.NUM_WORKERS)
        step_probs = []
        for imgs, _ in loader_t:
            p = F.softmax(model(imgs.to(Config.DEVICE)), dim=1).cpu().numpy()
            step_probs.append(p)
        tta_accum += np.vstack(step_probs)

    final_probs = (base_probs + tta_accum) / (1 + n_tta)
    # Sanitize: replace NaN/Inf (can occur with SAM+AMP) with uniform probs
    if not np.isfinite(final_probs).all():
        print("  [WARN] NaN/Inf detected in TTA probs — replacing with uniform distribution")
        bad = ~np.isfinite(final_probs).all(axis=1)
        final_probs[bad] = 1.0 / Config.NUM_CLASSES
    final_probs = np.clip(final_probs, 1e-7, 1.0)
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
    y_pred      = final_probs.argmax(axis=1)
    y_true      = np.array(all_labels)
    acc         = accuracy_score(y_true, y_pred)

    dummy_loader = DataLoader(ds_base, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS)
    total_loss = 0
    for imgs, labels in dummy_loader:
        with autocast(enabled=Config.USE_AMP):
            out  = model(imgs.to(Config.DEVICE))
            loss = criterion(out, labels.to(Config.DEVICE))
        total_loss += loss.item() * labels.size(0)
    avg_loss = total_loss / len(samples)

    return avg_loss, acc, y_pred, y_true, final_probs


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    preds_all, labels_all, probs_all = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
        with autocast(enabled=Config.USE_AMP):
            out  = model(imgs)
            loss = criterion(out, labels)
        probs      = F.softmax(out, 1)
        p          = probs.argmax(1)
        correct   += (p == labels).sum().item()
        total     += labels.size(0)
        loss_sum  += loss.item() * labels.size(0)
        preds_all .extend(p.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        probs_all .extend(probs.cpu().numpy())
    return (loss_sum/total, correct/total,
            np.array(preds_all), np.array(labels_all), np.array(probs_all))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_one_model(model_name, train_loader, val_loader, train_samples,
                    resume=False, run_name=None, num_epochs=None):
    Config.make_dirs()
    run_name = run_name or model_name
    num_epochs = int(num_epochs or Config.NUM_EPOCHS)
    model    = build_model(model_name, Config.NUM_CLASSES).to(Config.DEVICE)
    cw       = class_weights(train_samples)
    use_focal = getattr(Config, "USE_FOCAL_LOSS", False)
    use_cw_fl = getattr(Config, "USE_CLASS_WEIGHTED_FL", False)
    cw_alpha  = cw if use_cw_fl else None
    crit     = (FocalLoss(gamma=getattr(Config, "FOCAL_GAMMA", 2.0),
                          alpha=cw_alpha, weight=cw if use_focal else None)
                if use_focal else LabelSmoothCE(Config.NUM_CLASSES, Config.LABEL_SMOOTH, cw))

    use_sam = getattr(Config, "USE_SAM", False)
    if use_sam:
        opt = SAM(model.parameters(), optim.AdamW,
                  rho=0.05, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        print(f"  Optimizer: SAM (AdamW base, rho=0.05)")
    else:
        opt = optim.AdamW(model.parameters(), lr=Config.LR,
                          weight_decay=Config.WEIGHT_DECAY)
        print(f"  Optimizer: AdamW")
    sched    = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6)
    scaler   = GradScaler(enabled=Config.USE_AMP)
    ckpt_mgr = CheckpointManager(run_name, Config.CHECKPOINT_DIR)

    use_ema = getattr(Config, "USE_EMA", False)
    ema     = ModelEMA(model, decay=getattr(Config, "EMA_DECAY", 0.9999)) if use_ema else None

    start_epoch = 0
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "lr":[]}

    if resume:
        start_epoch, saved_hist = ckpt_mgr.load_latest(model, opt, sched, scaler)
        if saved_hist:
            history = saved_hist
        if ema is not None:
            ema = ModelEMA(model, decay=getattr(Config, "EMA_DECAY", 0.9999))

    best_val_acc   = max(history["val_acc"], default=0.0)
    patience_count = 0

    print(f"\n{'═'*65}")
    print(f"  Training: {model_name}  |  Run: {run_name}  |  Device: {Config.DEVICE}  |  AMP: {Config.USE_AMP}")
    print(f"  Epochs: {num_epochs}  |  Batch: {Config.BATCH_SIZE}  |  LR: {Config.LR}")
    if use_focal:
        print(f"  Loss: Focal (γ={getattr(Config,'FOCAL_GAMMA',2.0)})")
    if use_ema:
        print(f"  EMA: decay={getattr(Config,'EMA_DECAY',0.9999)}")
    if resume and start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch}")
    print(f"{'═'*65}")

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss, correct, total = 0, 0, 0
        bar = tqdm(train_loader, desc=f"  Ep {epoch:02d} Train",
                   leave=False, ncols=80)

        for imgs, labels in bar:
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            # Random choice: MixUp or CutMix or plain
            r = random.random()
            if r < 0.33:
                imgs, ya, yb, lam = mixup_data(imgs, labels, Config.MIXUP_ALPHA)
            elif r < 0.66:
                imgs, ya, yb, lam = cutmix_data(imgs, labels, Config.CUTMIX_ALPHA)
            else:
                ya, yb, lam = labels, labels, 1.0

            if use_sam:
                # SAM requires two forward passes
                with autocast(enabled=Config.USE_AMP):
                    out  = model(imgs)
                    loss = mixed_criterion(crit, out, ya, yb, lam)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                opt.first_step(zero_grad=True)
                scaler.update()

                with autocast(enabled=Config.USE_AMP):
                    out2  = model(imgs)
                    loss2 = mixed_criterion(crit, out2, ya, yb, lam)
                scaler.scale(loss2).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                opt.second_step(zero_grad=True)
                scaler.update()
            else:
                opt.zero_grad()
                with autocast(enabled=Config.USE_AMP):
                    out  = model(imgs)
                    loss = mixed_criterion(crit, out, ya, yb, lam)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                scaler.step(opt); scaler.update()

            if ema is not None:
                ema.update(model)

            preds    = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            tr_loss += loss.item() * labels.size(0)
            bar.set_postfix(loss=f"{loss.item():.3f}",
                            acc=f"{correct/total:.3f}")

        sched.step()
        tr_acc   = correct / total
        tr_loss /= total

        eval_model = model
        if ema is not None:
            ema.store(model)
            ema.apply_shadow(model)
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, crit)
        if ema is not None:
            ema.restore(model)
        elapsed  = time.time() - t0
        cur_lr   = opt.param_groups[0]["lr"]

        history["train_loss"].append(float(tr_loss))
        history["val_loss"]  .append(float(vl_loss))
        history["train_acc"] .append(float(tr_acc))
        history["val_acc"]   .append(float(vl_acc))
        history["lr"]        .append(float(cur_lr))

        is_best = vl_acc > best_val_acc
        if is_best:
            best_val_acc   = vl_acc
            patience_count = 0
        else:
            patience_count += 1

        save_model = model
        if is_best and ema is not None:
            ema.store(model)
            ema.apply_shadow(model)
            save_model = model
        ckpt_mgr.save(save_model, opt, sched, scaler, epoch,
                      vl_acc, vl_loss, history, is_best)
        if is_best and ema is not None:
            ema.restore(model)

        flag = "  ★" if is_best else ""
        print(f"  [{epoch:02d}/{num_epochs}]"
              f"  TrL:{tr_loss:.4f} TrA:{tr_acc:.4f}"
              f"  |  VlL:{vl_loss:.4f} VlA:{vl_acc:.4f}"
              f"  |  LR:{cur_lr:.2e}  {elapsed:.0f}s{flag}")

        if patience_count >= Config.PATIENCE:
            print(f"\n  [Early Stop] Patience {Config.PATIENCE} reached.")
            break

    # Load best weights for final evaluation
    model = ckpt_mgr.load_best(model)
    print(f"  Best Val Acc: {best_val_acc*100:.2f}%")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ALL RESULTS TO DISK (for Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def save_results(model_name, history, y_true, y_pred, y_probs, acc,
                 extra_metrics, output_dir):
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cm      = confusion_matrix(y_true, y_pred)
    report  = classification_report(y_true, y_pred,
                                    target_names=Config.CLASSES,
                                    output_dict=True)
    y_bin   = label_binarize(y_true, classes=list(range(Config.NUM_CLASSES)))

    # Per-class ROC — safe against empty classes and bad probabilities
    roc_data = {}
    for i, cls in enumerate(Config.CLASSES):
        if y_bin[:, i].sum() == 0:
            roc_data[cls] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_data[cls] = {
                "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "auc": float(auc(fpr, tpr))
            }
        except Exception:
            roc_data[cls] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}

    payload = {
        "model_name"     : model_name,
        "timestamp"      : datetime.now().isoformat(),
        "accuracy"       : float(acc),
        "f1_macro"       : float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro"   : float(recall_score(y_true, y_pred, average="macro")),
        "auc_macro"      : float(np.mean([v["auc"] for v in roc_data.values()])),
        "history"        : history,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_data"       : roc_data,
        "classes"        : Config.CLASSES,
        **extra_metrics,
    }

    out_path = results_dir / f"{model_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [Saved] Results → {out_path}")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_history(history, model_name, plots_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight="bold")
    cfg = [
        (("train_loss","val_loss"),  "Loss",     None),
        (("train_acc", "val_acc"),   "Accuracy", (0, 1.02)),
        (("lr",),                    "Learning Rate", None),
    ]
    for ax, (keys, title, ylim) in zip(axes, cfg):
        for k in keys:
            if k in history and history[k]:
                ax.plot(history[k], label=k.replace("_"," ").title(), lw=2)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
        if ylim: ax.set_ylim(ylim)
    plt.tight_layout()
    out = Path(plots_dir) / f"{model_name}_history.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Plot] {out.name}")


def plot_confusion_matrix(cm, model_name, plots_dir):
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    short = ["Dysk.", "Koilo.", "Meta.", "Para.", "Sup-Int."]
    for ax, data, fmt, title in zip(
        axes, [cm, cm_n], ["d", ".2f"], ["Counts", "Normalized"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=short, yticklabels=short,
                    ax=ax, linewidths=0.5, cbar_kws={"shrink":0.8})
        ax.set_title(f"Confusion Matrix ({title})", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle(f"{model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(plots_dir) / f"{model_name}_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Plot] {out.name}")


def plot_roc(roc_data, model_name, plots_dir):
    colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"]
    plt.figure(figsize=(9, 7))
    for (cls, d), c in zip(roc_data.items(), colors):
        plt.plot(d["fpr"], d["tpr"], color=c, lw=2,
                 label=f"{cls[:12]} (AUC={d['auc']:.3f})")
    plt.plot([0,1],[0,1],"k--", lw=1)
    macro = np.mean([d["auc"] for d in roc_data.values()])
    plt.title(f"{model_name} — ROC  |  Macro-AUC: {macro:.4f}", fontweight="bold")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(plots_dir) / f"{model_name}_roc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Plot] {out.name}")


def plot_comparison(all_results, plots_dir):
    if len(all_results) < 2:
        return
    metrics = ["accuracy","f1_macro","precision_macro","recall_macro","auc_macro"]
    labels  = ["Accuracy","F1","Precision","Recall","AUC-ROC"]
    models  = [r["model_name"] for r in all_results]
    x       = np.arange(len(metrics))
    w       = 0.8 / len(models)
    colors  = ["#4C72B0","#DD8452","#55A868","#C44E52"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (r, c) in enumerate(zip(all_results, colors)):
        vals = [r.get(m, 0) for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=r["model_name"], color=c,
                      alpha=0.88, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w*(len(models)-1)/2)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.75, 1.02])
    ax.set_title("Model Comparison — Test Set", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = Path(plots_dir) / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Plot] {out.name}")


def save_gradcam_gallery(model, model_name, test_samples, gradcam_dir, n=20):
    layer = get_last_conv(model, model_name)
    if layer is None:
        print(f"  [Grad-CAM] Layer not found for {model_name}, skipping.")
        return

    gc   = GradCAM(model, layer)
    inv  = T.Normalize([-m/s for m,s in zip(Config.MEAN,Config.STD)],
                        [1/s for s in Config.STD])
    tf   = get_transform("val")

    # Sample equally from each class
    by_class = defaultdict(list)
    for p, l in test_samples:
        by_class[l].append(p)
    picks = []
    per_class = max(1, n // Config.NUM_CLASSES)
    for cls_idx in range(Config.NUM_CLASSES):
        items = by_class[cls_idx]
        random.shuffle(items)
        picks += [(p, cls_idx) for p in items[:per_class]]

    if not picks:
        print(f"  [Grad-CAM] No samples to visualize for {model_name}, skipping.")
        gc.remove()
        return
    n_actual = len(picks)
    cols = 5;
    rows = (n_actual + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    model.eval()
    for ax_idx, (path, true_label) in enumerate(picks):
        img  = Image.open(path).convert("RGB")
        t    = tf(img)
        cam, pred = gc.generate(t)
        raw  = inv(t).permute(1,2,0).numpy()
        raw  = np.clip(raw, 0, 1)
        ax   = axes[ax_idx]
        ax.imshow(raw)
        ax.imshow(cam, cmap="jet", alpha=0.42)
        color = "green" if pred == true_label else "red"
        tc, pc = Config.CLASSES[true_label], Config.CLASSES[pred]
        ax.set_title(f"T: {tc[:10]}\nP: {pc[:10]}", fontsize=8,
                     color=color, fontweight="bold")
        ax.axis("off")

    for ax in axes[len(picks):]:
        ax.axis("off")

    plt.suptitle(f"{model_name} — Grad-CAM Gallery\n"
                 "Green=Correct  |  Red=Wrong",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(gradcam_dir) / f"{model_name}_gradcam_gallery.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    gc.remove()
    print(f"  [Grad-CAM] Saved gallery → {out.name}")

    # Also save individual per-class Grad-CAM images
    per_cls_dir = Path(gradcam_dir) / model_name
    per_cls_dir.mkdir(parents=True, exist_ok=True)
    for path, true_label in picks:
        img  = Image.open(path).convert("RGB")
        t    = tf(img)
        gc2  = GradCAM(model, layer)
        cam, pred = gc2.generate(t)
        raw  = inv(t).permute(1,2,0).numpy()
        raw  = np.clip(raw, 0, 1)
        fig2, ax2 = plt.subplots(1, 2, figsize=(7, 3.5))
        ax2[0].imshow(raw); ax2[0].set_title("Original"); ax2[0].axis("off")
        ax2[1].imshow(raw); ax2[1].imshow(cam, cmap="jet", alpha=0.45)
        ax2[1].set_title("Grad-CAM Overlay"); ax2[1].axis("off")
        tc = Config.CLASSES[true_label]
        pc = Config.CLASSES[pred]
        plt.suptitle(f"True: {tc}  |  Pred: {pc}", fontsize=10,
                     color="green" if pred==true_label else "red")
        stem = Path(path).stem
        fig2.savefig(per_cls_dir / f"{tc}_{stem}.png",
                     dpi=100, bbox_inches="tight")
        plt.close(fig2)
        gc2.remove()

    print(f"  [Grad-CAM] Individual images → {per_cls_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main(resume=False):
    Config.make_dirs()
    print("\n" + "═"*65)
    print("  CERVICAL CANCER DETECTION — Advanced Training Pipeline")
    print(f"  Device: {Config.DEVICE}  |  AMP: {Config.USE_AMP}")
    print(f"  Models: {Config.MODELS_TO_TRAIN}")
    print("═"*65)

    samples          = discover_images(Config.DATA_DIR)
    train_s, val_s, test_s = stratified_split(samples)

    sampler     = make_weighted_sampler(train_s)
    train_ds    = CervicalDataset(train_s, get_transform("train"))
    val_ds      = CervicalDataset(val_s,   get_transform("val"))
    test_ds     = CervicalDataset(test_s,  get_transform("val"))
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              sampler=sampler, num_workers=Config.NUM_WORKERS,
                              pin_memory=(Config.DEVICE=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS)

    cw      = class_weights(train_s)
    crit    = LabelSmoothCE(Config.NUM_CLASSES, Config.LABEL_SMOOTH, cw)

    trained_models = {}
    all_results    = []

    for idx, mname in enumerate(Config.MODELS_TO_TRAIN, 1):
        print(f"\n{'─'*65}")
        print(f"  [{idx}/{len(Config.MODELS_TO_TRAIN)}] {mname}")
        print(f"{'─'*65}")

        results_path = Path(Config.RESULTS_DIR) / f"{mname}_results.json"
        if resume and results_path.exists():
            print(f"  [Skip] {mname} already done, loading best checkpoint.")
            model = build_model(mname, Config.NUM_CLASSES).to(Config.DEVICE)
            ckpt_mgr = CheckpointManager(mname, Config.CHECKPOINT_DIR)
            ckpt_mgr.load_best(model)
            trained_models[mname] = model
            import json as _json
            with open(results_path) as _f:
                all_results.append(_json.load(_f))
            continue

        model, history = train_one_model(mname, train_loader, val_loader,
                                          train_s, resume=resume)
        trained_models[mname] = model

        # TTA evaluation on test set
        print(f"  Running TTA evaluation on test set...")
        te_loss, te_acc, y_pred, y_true, y_probs = evaluate_tta(
            model, test_s, crit, n_tta=Config.TTA_STEPS)

        print(f"\n  ─── {mname} Test Results (TTA={Config.TTA_STEPS}) ───")
        print(f"  Accuracy : {te_acc*100:.2f}%")
        print(f"  F1 Macro : {f1_score(y_true, y_pred, average='macro'):.4f}")
        # Safe AUC: skip classes absent from test set (avoids NaN in roc_curve)
        y_bin = label_binarize(y_true, classes=list(range(Config.NUM_CLASSES)))
        auc_scores = []
        for i in range(Config.NUM_CLASSES):
            if y_bin[:, i].sum() == 0:
                continue  # class not present in test split — skip
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                auc_scores.append(auc(fpr, tpr))
            except Exception:
                pass
        if auc_scores:
            print(f"  AUC-ROC  : {np.mean(auc_scores):.4f}")
        else:
            print(f"  AUC-ROC  : N/A (insufficient class coverage in test set)")
        print(classification_report(y_true, y_pred, target_names=Config.CLASSES, digits=4))

        # Save everything
        results = save_results(
            mname, history, y_true, y_pred, y_probs, te_acc,
            {"test_loss": float(te_loss)}, Config.OUTPUT_DIR)
        all_results.append(results)

        # Plots
        cm = np.array(results["confusion_matrix"])
        plot_history(history, mname, Config.PLOTS_DIR)
        plot_confusion_matrix(cm, mname, Config.PLOTS_DIR)
        plot_roc(results["roc_data"], mname, Config.PLOTS_DIR)
        save_gradcam_gallery(model, mname, test_s, Config.GRADCAM_DIR)

    # ── Stacked ensemble (cascade of model outputs) ───────────────────────────
    if Config.TRAIN_STACKED and len(trained_models) >= 2:
        print(f"\n{'─'*65}")
        if Config.STACKED_OOF:
            print("  Training Stacked Ensemble (OOF K-fold base probs → meta-classifier)")
            print(f"  Meta model: {Config.STACKED_META} | Folds: {Config.STACKED_OOF_FOLDS} | OOF epochs: {Config.STACKED_OOF_EPOCHS}")
        else:
            print("  Training Stacked Ensemble (VAL base probs → meta-classifier)")
            print(f"  Meta model: {Config.STACKED_META}")
        print(f"{'─'*65}")

        base_names = [n for n in Config.MODELS_TO_TRAIN if n in trained_models]
        y_test = _labels_from_samples(test_s)

        if Config.STACKED_OOF:
            meta = run_oof_stacking(train_s, base_names, resume=resume)
        else:
            val_probs = []
            for name in base_names:
                m = trained_models[name]
                print(f"  Base predictions: {name} (VAL)")
                pv = predict_probs_tta(m, val_s, n_tta=Config.TTA_STEPS)
                val_probs.append(pv)
            X_meta_val = np.concatenate(val_probs, axis=1)
            y_meta_val = _labels_from_samples(val_s)
            meta = train_stacked_meta_classifier(X_meta_val, y_meta_val)

        test_probs = []
        for name in base_names:
            m = trained_models[name]
            print(f"  Base predictions: {name} (TEST)")
            pt = predict_probs_tta(m, test_s, n_tta=Config.TTA_STEPS)
            test_probs.append(pt)
        X_meta_test = np.concatenate(test_probs, axis=1)

        y_pred = meta.predict(X_meta_test)
        y_probs = meta.predict_proba(X_meta_test)
        te_acc = accuracy_score(y_test, y_pred)

        print(f"\n  ─── Stacked Ensemble Test Results ───")
        print(f"  Accuracy : {te_acc*100:.2f}%")
        print(f"  F1 Macro : {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(classification_report(y_test, y_pred, target_names=Config.CLASSES, digits=4))

        stacked_history = {"note": "Stacked ensemble meta-model trained on VAL probs"}
        if Config.STACKED_OOF:
            stacked_history = {"note": "Stacked ensemble meta-model trained on OOF probs (K-fold)"}
        stacked_results = save_results(
            "Stacked_Ensemble", stacked_history,
            y_test, y_pred, y_probs, te_acc,
            {"meta_model": Config.STACKED_META, "base_models": Config.MODELS_TO_TRAIN,
             "tta_steps": int(Config.TTA_STEPS),
             "oof_folds": int(Config.STACKED_OOF_FOLDS) if Config.STACKED_OOF else 0,
             "oof_epochs": int(Config.STACKED_OOF_EPOCHS) if Config.STACKED_OOF else 0},
            Config.OUTPUT_DIR
        )
        all_results.append(stacked_results)

        cm = np.array(stacked_results["confusion_matrix"])
        plot_confusion_matrix(cm, "Stacked_Ensemble", Config.PLOTS_DIR)
        plot_roc(stacked_results["roc_data"], "Stacked_Ensemble", Config.PLOTS_DIR)

    # ── Hybrid pipeline (handcrafted + deep embeddings) ───────────────────────
    if Config.TRAIN_HYBRID:
        print(f"\n{'─'*65}")
        print("  Training Hybrid Model (handcrafted + deep embeddings + classical ML)")
        print(f"  Backbone embeddings: {Config.HYBRID_BACKBONE}")
        use_trained = getattr(Config, "HYBRID_USE_TRAINED", False)
        if use_trained:
            print(f"  Using trained backbone (from best checkpoint)")
        print(f"{'─'*65}")

        # Deep embeddings (trained or pretrained) + handcrafted features
        Xtr_deep, ytr, p_tr = extract_deep_embeddings(Config.HYBRID_BACKBONE, train_s, use_trained=use_trained)
        Xva_deep, yva, p_va = extract_deep_embeddings(Config.HYBRID_BACKBONE, val_s, use_trained=use_trained)
        Xte_deep, yte, p_te = extract_deep_embeddings(Config.HYBRID_BACKBONE, test_s, use_trained=use_trained)

        Xtr_hand = extract_handcrafted_features(p_tr)
        Xva_hand = extract_handcrafted_features(p_va)
        Xte_hand = extract_handcrafted_features(p_te)

        # Sanitize all feature matrices before SVM — NaN/Inf from backbone or image stats
        Xtr_deep = np.nan_to_num(Xtr_deep, nan=0.0, posinf=0.0, neginf=0.0)
        Xva_deep = np.nan_to_num(Xva_deep, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_deep = np.nan_to_num(Xte_deep, nan=0.0, posinf=0.0, neginf=0.0)
        Xtr_hand = np.nan_to_num(Xtr_hand, nan=0.0, posinf=0.0, neginf=0.0)
        Xva_hand = np.nan_to_num(Xva_hand, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_hand = np.nan_to_num(Xte_hand, nan=0.0, posinf=0.0, neginf=0.0)

        Xtr = np.concatenate([Xtr_hand, Xtr_deep], axis=1)
        Xva = np.concatenate([Xva_hand, Xva_deep], axis=1)
        Xte = np.concatenate([Xte_hand, Xte_deep], axis=1)

        hybrid = train_hybrid_classifier(Xtr, ytr)

        # Evaluate on test (also report val quickly)
        va_pred = hybrid.predict(Xva)
        va_acc = accuracy_score(yva, va_pred)
        te_pred = hybrid.predict(Xte)
        te_probs = hybrid.predict_proba(Xte)
        te_acc = accuracy_score(yte, te_pred)

        print(f"\n  ─── Hybrid Test Results ───")
        print(f"  Val Accuracy  : {va_acc*100:.2f}%")
        print(f"  Test Accuracy : {te_acc*100:.2f}%")
        print(f"  F1 Macro      : {f1_score(yte, te_pred, average='macro'):.4f}")
        print(classification_report(yte, te_pred, target_names=Config.CLASSES, digits=4))

        hybrid_history = {"note": "Hybrid classical ML; no epoch history"}
        hybrid_results = save_results(
            "Hybrid_SVM_Fusion", hybrid_history,
            yte, te_pred, te_probs, te_acc,
            {"val_accuracy": float(va_acc), "hybrid_backbone": Config.HYBRID_BACKBONE,
             "handcrafted_dim": int(Xtr_hand.shape[1]), "deep_dim": int(Xtr_deep.shape[1])},
            Config.OUTPUT_DIR
        )
        all_results.append(hybrid_results)

        cm = np.array(hybrid_results["confusion_matrix"])
        plot_confusion_matrix(cm, "Hybrid_SVM_Fusion", Config.PLOTS_DIR)
        plot_roc(hybrid_results["roc_data"], "Hybrid_SVM_Fusion", Config.PLOTS_DIR)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    if len(trained_models) >= 2:
        print(f"\n{'─'*65}")
        print("  Evaluating Ensemble (soft-voting, learned weights)")

        ens_names = [n for n in Config.MODELS_TO_TRAIN if n in trained_models]
        base_model_results = [r for r in all_results if r["model_name"] in trained_models]
        accs = {r["model_name"]: r["accuracy"] for r in base_model_results}
        tot = sum(accs.values()) or 1
        weights = [accs[n] / tot for n in ens_names]
        print(f"  Ensemble weights: { {n: round(w, 3) for n, w in zip(ens_names, weights)} }")

        ensemble = EnsembleModel(trained_models, weights).to(Config.DEVICE)
        ensemble.eval()

        te_loss, te_acc, y_pred, y_true, y_probs = evaluate_tta(
            ensemble, test_s, nn.CrossEntropyLoss(), n_tta=Config.TTA_STEPS)

        print(f"  Ensemble Accuracy : {te_acc * 100:.2f}%")
        ens_history = {"note": "Ensemble — no independent training history"}
        ens_results = save_results(
            "Ensemble", ens_history, y_true, y_pred, y_probs, te_acc,
            {"test_loss": float(te_loss), "component_weights": dict(zip(ens_names, [round(w, 4) for w in weights]))},
            Config.OUTPUT_DIR)
        all_results.append(ens_results)

        cm = np.array(ens_results["confusion_matrix"])
        plot_confusion_matrix(cm, "Ensemble", Config.PLOTS_DIR)
        plot_roc(ens_results["roc_data"], "Ensemble", Config.PLOTS_DIR)

    # ── Save master summary ───────────────────────────────────────────────────
    plot_comparison(all_results, Config.PLOTS_DIR)
    summary = {
        "run_timestamp"    : datetime.now().isoformat(),
        "device"           : Config.DEVICE,
        "models_trained"   : Config.MODELS_TO_TRAIN,
        "dataset_total"    : len(samples),
        "train_size"       : len(train_s),
        "val_size"         : len(val_s),
        "test_size"        : len(test_s),
        "results"          : [
            {k: r[k] for k in ["model_name","accuracy","f1_macro",
                                "precision_macro","recall_macro","auc_macro"]}
            for r in all_results
        ]
    }
    summary_path = Path(Config.OUTPUT_DIR) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*65}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'═'*65}")
    for r in all_results:
        print(f"  {r['model_name']:<22}"
              f"  Acc: {r['accuracy']*100:.2f}%"
              f"  F1: {r['f1_macro']:.4f}"
              f"  AUC: {r['auc_macro']:.4f}")
    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"\n  ★ Best: {best['model_name']} ({best['accuracy']*100:.2f}%)")
    print(f"  All outputs → {Path(Config.OUTPUT_DIR).resolve()}")
    print("\n  Launch Streamlit:")
    print("  streamlit run cervical_cancer_app.py")
    print("═"*65)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()
    main(resume=args.resume)