#!/usr/bin/env python3
"""
ReSPT-24 Reliability-Guided Selective Test-Time Adaptation on CIFAR-100-C
============================================================================

Purpose
-------
Evaluate the central research question:
    When should a model adapt under distribution shift, and when should it
    preserve its learned representation?

Methods (same adaptation objective; only the sample gate changes)
------------------------------------------------------------------
1. source                : never adapt
2. always                : adapt on every test sample
3. entropy_gate          : adapt only on low-entropy samples
4. reliability_guided    : adapt only when the sample is BOTH shifted and
                            sufficiently reliable according to a multi-signal gate

The reliability-guided gate combines:
- reliability-head probability
- normalized predictive entropy
- top-1/top-2 probability margin
- prediction consistency under horizontal flip
and uses source-prototype distance as a separate shift signal.

Adaptation is deliberately parameter-efficient:
- only LayerNorm affine parameters in the last N Transformer blocks plus the
  final Transformer LayerNorm are updated;
- classifier and reliability heads remain frozen.

Unsupervised adaptation objective (same for all adapting baselines):
    entropy minimization
  + representation anchor to frozen source model
  + source-prototype alignment using source pseudo-labels
  + optional flip-consistency loss

Important protocol
------------------
- No test labels are used for adaptation or gating.
- Labels are used only AFTER predictions/updates to calculate evaluation metrics.
- The model is reset to the same source checkpoint for every
  corruption x severity x method (episodic TTA).
- Gate thresholds are calibrated only from clean CIFAR-100 training images.

Outputs
-------
summary.csv
batch_history.csv
overall_by_method.csv
calibration_thresholds.json
run_config.json

Metrics
-------
Top-1, Top-5, NLL, ECE, Brier, reliability Brier,
error-detection AUROC, AURC, adaptation rate, adapted-batch rate,
harmful-update rate, beneficial-update rate, representation drift,
and clean-source forgetting.

Place this file beside train_respt24_cifar100.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import train_respt24_cifar100 as base

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# -----------------------------------------------------------------------------
# CIFAR-100-C metadata
# -----------------------------------------------------------------------------

BENCHMARK_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

EXTRA_CORRUPTIONS = [
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]

QUICK_CORRUPTIONS = [
    "gaussian_noise",
    "motion_blur",
    "snow",
    "brightness",
    "jpeg_compression",
]

CIFAR100C_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
CIFAR100C_MD5 = "11f0ed0f1191edbf9fa23466ae6021d3"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_dump(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(destination)


def download_with_progress(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * downloaded / total_size)
            gib = downloaded / (1024 ** 3)
            total_gib = total_size / (1024 ** 3)
            print(f"\rDownloading CIFAR-100-C: {pct:6.2f}% | {gib:.2f}/{total_gib:.2f} GiB", end="", flush=True)
        else:
            mib = downloaded / (1024 ** 2)
            print(f"\rDownloading CIFAR-100-C: {mib:.1f} MiB", end="", flush=True)

    urllib.request.urlretrieve(url, out_path, reporthook=report)
    print()


def ensure_cifar100c(root: Path, download_if_missing: bool) -> Path:
    root = root.expanduser().resolve()
    if (root / "labels.npy").exists():
        return root

    if not download_if_missing:
        raise FileNotFoundError(
            f"CIFAR-100-C was not found at {root}. "
            "Download/extract CIFAR-100-C there, or rerun with --download_if_missing."
        )

    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    tar_path = parent / "CIFAR-100-C.tar"

    if not tar_path.exists():
        print("CIFAR-100-C is missing. Downloading the official archive (~2.9 GB)...")
        download_with_progress(CIFAR100C_URL, tar_path)
    else:
        print(f"Using existing archive: {tar_path}")

    print("Verifying CIFAR-100-C archive MD5...")
    digest = md5sum(tar_path)
    if digest != CIFAR100C_MD5:
        raise RuntimeError(
            f"CIFAR-100-C archive MD5 mismatch. Expected {CIFAR100C_MD5}, got {digest}. "
            "Delete the archive and download it again."
        )

    print("Extracting CIFAR-100-C...")
    with tarfile.open(tar_path, "r") as tar:
        safe_extract_tar(tar, parent)

    if not (root / "labels.npy").exists():
        # Be tolerant if the archive extracted under a slightly different path.
        candidates = list(parent.rglob("CIFAR-100-C/labels.npy"))
        if not candidates:
            raise FileNotFoundError("Extraction finished but CIFAR-100-C/labels.npy was not found.")
        root = candidates[0].parent

    print(f"CIFAR-100-C ready: {root}")
    return root


def normalize_entropy(probs: torch.Tensor) -> torch.Tensor:
    ent = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
    return ent / math.log(probs.shape[1])


def probability_margin(probs: torch.Tensor) -> torch.Tensor:
    top2 = probs.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def consistency_score(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JS is bounded by ln(2) for distributions; map to [0, 1].
    return (1.0 - js_divergence(p, q) / math.log(2.0)).clamp(0.0, 1.0)


def expected_calibration_error(probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    correct = pred.eq(target).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.tensor(0.0)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += mask.float().mean() * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(ece.item() * 100.0)


def multiclass_brier(probs: torch.Tensor, target: torch.Tensor, num_classes: int = 100) -> float:
    y = F.one_hot(target, num_classes=num_classes).float()
    return float(((probs - y) ** 2).sum(dim=1).mean().item())


def aurc_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under risk-coverage curve. Lower is better."""
    if confidence.size == 0:
        return float("nan")
    order = np.argsort(-confidence)
    errors = 1.0 - correct[order].astype(np.float64)
    cum_errors = np.cumsum(errors)
    coverage_count = np.arange(1, len(errors) + 1, dtype=np.float64)
    risk = cum_errors / coverage_count
    return float(risk.mean())


def safe_auroc(correct: np.ndarray, score: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    if np.unique(correct).size < 2:
        return float("nan")
    return float(roc_auc_score(correct.astype(np.int32), score.astype(np.float64)))


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def make_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(base.CIFAR100_MEAN, base.CIFAR100_STD),
    ])


class CIFAR100CDataset(Dataset):
    def __init__(self, root: Path, corruption: str, severity: int, transform):
        if severity not in (1, 2, 3, 4, 5):
            raise ValueError("severity must be in {1,2,3,4,5}")
        file_path = root / f"{corruption}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing corruption file: {file_path}")
        labels_path = root / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        self.images = np.load(file_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.transform = transform
        self.start = (severity - 1) * 10000
        self.end = min(self.start + 10000, len(self.images))
        if self.end <= self.start:
            raise ValueError(
                f"{file_path.name} has {len(self.images)} images; cannot access severity {severity}."
            )
        self.length = self.end - self.start

        if len(self.labels) not in (10000, len(self.images)):
            raise ValueError(
                f"Unexpected labels.npy length {len(self.labels)} for corruption length {len(self.images)}"
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        j = self.start + idx
        arr = np.asarray(self.images[j], dtype=np.uint8)
        image = Image.fromarray(arr)
        if len(self.labels) == len(self.images):
            label = int(self.labels[j])
        else:
            label = int(self.labels[idx])
        return self.transform(image), label


def make_clean_dataset(data_dir: str, image_size: int, train: bool):
    return datasets.CIFAR100(
        root=data_dir,
        train=train,
        download=True,
        transform=make_eval_transform(image_size),
    )


def subset_indices(n_total: int, n_keep: int, seed: int) -> np.ndarray:
    n_keep = min(n_total, n_keep)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_keep, replace=False))


def make_loader(ds, batch_size: int, workers: int, device: torch.device, shuffle: bool = False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


# -----------------------------------------------------------------------------
# Model loading / pruning reconstruction
# -----------------------------------------------------------------------------


def model_from_dense_checkpoint(dense_checkpoint: str, device: torch.device):
    ckpt = torch.load(dense_checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", {})
    model = base.ReSPTImageClassifier(
        image_size=int(cfg.get("image_size", 128)),
        patch_size=int(cfg.get("patch_size", 16)),
        num_classes=int(cfg.get("num_classes", 100)),
        dim=int(cfg.get("dim", 384)),
        depth=int(cfg.get("depth", 24)),
        num_heads=int(cfg.get("heads", 8)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.0)),
        attn_dropout=float(cfg.get("attn_dropout", 0.0)),
        drop_path_rate=float(cfg.get("drop_path", 0.20)),
        layer_scale_init=float(cfg.get("layer_scale_init", 1e-5)),
        ffn_groups=int(cfg.get("ffn_groups", 16)),
    )
    state = ckpt.get("ema", ckpt.get("model"))
    if state is None:
        raise KeyError(f"Checkpoint {dense_checkpoint} contains neither 'ema' nor 'model'.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def load_source_model(
    checkpoint: str,
    dense_checkpoint: str,
    prune_ratio: float,
    device: torch.device,
):
    dense_model, cfg = model_from_dense_checkpoint(dense_checkpoint, device)

    if prune_ratio > 0.0:
        model, _ = base.structured_prune_model(dense_model, prune_ratio)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("ema", ckpt.get("model"))
        if state is None:
            raise KeyError(f"Checkpoint {checkpoint} contains neither 'ema' nor 'model'.")
        model.load_state_dict(state, strict=True)
        del dense_model
    else:
        if Path(checkpoint).resolve() == Path(dense_checkpoint).resolve():
            model = dense_model
        else:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("ema", ckpt.get("model"))
            if state is None:
                raise KeyError(f"Checkpoint {checkpoint} contains neither 'ema' nor 'model'.")
            dense_model.load_state_dict(state, strict=True)
            model = dense_model

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def configure_tta_parameters(model: nn.Module, last_blocks: int) -> List[nn.Parameter]:
    """Update only LayerNorm affine parameters in the last N blocks + final norm."""
    for p in model.parameters():
        p.requires_grad_(False)

    depth = len(model.core.blocks)
    start = max(0, depth - last_blocks)
    modules: List[nn.Module] = []
    for block in model.core.blocks[start:]:
        modules.extend([block.norm1, block.norm2])
    modules.append(model.core.norm)

    params: List[nn.Parameter] = []
    seen = set()
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
            if id(p) not in seen:
                params.append(p)
                seen.add(id(p))

    # Keep eval mode: LayerNorm is unaffected by train/eval, while this ensures
    # dropout and stochastic depth remain disabled during online adaptation.
    model.eval()
    return params


# -----------------------------------------------------------------------------
# Source calibration: prototypes + gate thresholds
# -----------------------------------------------------------------------------


@dataclass
class GateThresholds:
    entropy_max: float
    reliability_min: float
    shift_min: float
    source_entropy_mean: float
    source_reliability_mean: float
    source_shift_mean: float
    source_rel_head_mean: float
    source_margin_mean: float
    source_consistency_mean: float


@torch.inference_mode()
def compute_prototypes(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    dim: int,
    amp_dtype: str,
):
    sums = torch.zeros(num_classes, dim, device=device, dtype=torch.float32)
    counts = torch.zeros(num_classes, device=device, dtype=torch.float32)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(device, amp_dtype):
            _, _, emb = model(images, return_embedding=True)
        emb = F.normalize(emb.float(), dim=1)
        sums.index_add_(0, labels, emb)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    protos = sums / counts.clamp_min(1.0)[:, None]
    return F.normalize(protos, dim=1), counts


@torch.inference_mode()
def source_signals(
    model: nn.Module,
    images: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
):
    with autocast_context(device, amp_dtype):
        logits, rel_logit, emb = model(images, return_embedding=True)
        logits_flip, _, _ = model(torch.flip(images, dims=[3]), return_embedding=True)

    probs = F.softmax(logits.float(), dim=1)
    probs_flip = F.softmax(logits_flip.float(), dim=1)
    rel_prob = torch.sigmoid(rel_logit.float())
    entropy = normalize_entropy(probs)
    margin = probability_margin(probs)
    consist = consistency_score(probs, probs_flip)

    emb_n = F.normalize(emb.float(), dim=1)
    pseudo = probs.argmax(dim=1)
    proto_sim = (emb_n * prototypes[pseudo]).sum(dim=1).clamp(-1.0, 1.0)
    shift = 1.0 - proto_sim

    reliability = 0.25 * (
        rel_prob
        + (1.0 - entropy)
        + margin
        + consist
    )
    reliability = reliability.clamp(0.0, 1.0)

    return {
        "logits": logits.float(),
        "probs": probs,
        "rel_prob": rel_prob,
        "emb": emb.float(),
        "entropy": entropy,
        "margin": margin,
        "consistency": consist,
        "proto_sim": proto_sim,
        "shift": shift,
        "reliability": reliability,
        "pseudo": pseudo,
    }


@torch.inference_mode()
def calibrate_source(
    model: nn.Module,
    loader: DataLoader,
    prototypes: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
    entropy_quantile: float,
    reliability_quantile: float,
    shift_quantile: float,
):
    ent, rel, shift = [], [], []
    relhead, margins, cons = [], [], []
    clean_correct, clean_rel = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        s = source_signals(model, images, prototypes, device, amp_dtype)
        ent.append(s["entropy"].cpu())
        rel.append(s["reliability"].cpu())
        shift.append(s["shift"].cpu())
        relhead.append(s["rel_prob"].cpu())
        margins.append(s["margin"].cpu())
        cons.append(s["consistency"].cpu())
        clean_correct.append(s["probs"].argmax(1).eq(labels).float().cpu())
        clean_rel.append(s["rel_prob"].cpu())

    ent_t = torch.cat(ent)
    rel_t = torch.cat(rel)
    shift_t = torch.cat(shift)
    relhead_t = torch.cat(relhead)
    margin_t = torch.cat(margins)
    cons_t = torch.cat(cons)
    corr_t = torch.cat(clean_correct).numpy()
    relscore_t = torch.cat(clean_rel).numpy()

    th = GateThresholds(
        entropy_max=float(torch.quantile(ent_t, entropy_quantile).item()),
        reliability_min=float(torch.quantile(rel_t, reliability_quantile).item()),
        shift_min=float(torch.quantile(shift_t, shift_quantile).item()),
        source_entropy_mean=float(ent_t.mean().item()),
        source_reliability_mean=float(rel_t.mean().item()),
        source_shift_mean=float(shift_t.mean().item()),
        source_rel_head_mean=float(relhead_t.mean().item()),
        source_margin_mean=float(margin_t.mean().item()),
        source_consistency_mean=float(cons_t.mean().item()),
    )
    calib_rel_auroc = safe_auroc(corr_t, relscore_t)
    return th, calib_rel_auroc


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


@torch.inference_mode()
def clean_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
) -> float:
    correct = 0
    total = 0
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(device, amp_dtype):
            logits, _ = model(images)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(1, total)


@dataclass
class StreamMetrics:
    method: str
    corruption: str
    severity: int
    top1: float
    top5: float
    nll: float
    ece: float
    brier: float
    reliability_brier: float
    error_detection_auroc: float
    aurc: float
    adaptation_rate: float
    adapted_batch_rate: float
    harmful_update_rate: float
    beneficial_update_rate: float
    mean_representation_drift: float
    clean_accuracy_after: float
    source_forgetting: float
    mean_gate_reliability: float
    mean_shift: float
    seconds: float


def final_metrics_from_buffers(
    probs: torch.Tensor,
    labels: torch.Tensor,
    rel_prob: torch.Tensor,
    method: str,
    corruption: str,
    severity: int,
    selected_count: int,
    total_count: int,
    adapted_sample_count: int,
    adapted_batches: int,
    total_batches: int,
    harmful_samples: int,
    beneficial_samples: int,
    drift_sum: float,
    drift_count: int,
    clean_acc_after: float,
    clean_acc_before: float,
    gate_reliability_sum: float,
    shift_sum: float,
    seconds: float,
):
    pred = probs.argmax(dim=1)
    top1 = 100.0 * pred.eq(labels).float().mean().item()
    k = min(5, probs.shape[1])
    top5 = 100.0 * probs.topk(k, dim=1).indices.eq(labels[:, None]).any(dim=1).float().mean().item()
    nll = F.nll_loss(probs.clamp_min(1e-8).log(), labels).item()
    ece = expected_calibration_error(probs, labels)
    brier = multiclass_brier(probs, labels, probs.shape[1])
    correct = pred.eq(labels).float()
    rel_brier = ((rel_prob - correct) ** 2).mean().item()

    corr_np = correct.numpy()
    rel_np = rel_prob.numpy()
    auroc = safe_auroc(corr_np, rel_np)
    aurc = aurc_score(rel_np, corr_np)

    return StreamMetrics(
        method=method,
        corruption=corruption,
        severity=severity,
        top1=top1,
        top5=top5,
        nll=nll,
        ece=ece,
        brier=brier,
        reliability_brier=rel_brier,
        error_detection_auroc=auroc,
        aurc=aurc,
        adaptation_rate=100.0 * adapted_sample_count / max(1, total_count),
        adapted_batch_rate=100.0 * adapted_batches / max(1, total_batches),
        harmful_update_rate=(100.0 * harmful_samples / adapted_sample_count) if adapted_sample_count else 0.0,
        beneficial_update_rate=(100.0 * beneficial_samples / adapted_sample_count) if adapted_sample_count else 0.0,
        mean_representation_drift=drift_sum / max(1, drift_count),
        clean_accuracy_after=clean_acc_after,
        source_forgetting=clean_acc_before - clean_acc_after,
        mean_gate_reliability=gate_reliability_sum / max(1, total_count),
        mean_shift=shift_sum / max(1, total_count),
        seconds=seconds,
    )


# -----------------------------------------------------------------------------
# TTA stream
# -----------------------------------------------------------------------------


def select_mask(method: str, s: Dict[str, torch.Tensor], thresholds: GateThresholds):
    n = s["entropy"].shape[0]
    if method == "source":
        return torch.zeros(n, dtype=torch.bool, device=s["entropy"].device)
    if method == "always":
        return torch.ones(n, dtype=torch.bool, device=s["entropy"].device)
    if method == "entropy_gate":
        return s["entropy"] <= thresholds.entropy_max
    if method == "reliability_guided":
        reliable = s["reliability"] >= thresholds.reliability_min
        shifted = s["shift"] >= thresholds.shift_min
        return reliable & shifted
    raise ValueError(f"Unknown method: {method}")


def adaptation_loss(
    model: nn.Module,
    images: torch.Tensor,
    selected: torch.Tensor,
    ref_signals: Dict[str, torch.Tensor],
    prototypes: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
    lambda_entropy: float,
    lambda_anchor: float,
    lambda_proto: float,
    lambda_consistency: float,
):
    with autocast_context(device, amp_dtype):
        logits, _, emb = model(images, return_embedding=True)
        probs = F.softmax(logits.float(), dim=1)
        ent = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)

        emb_n = F.normalize(emb.float(), dim=1)
        # Materialize normal tensors because source_signals is produced under
        # torch.inference_mode(), while these values participate in a loss whose
        # backward graph depends on the adapted embedding.
        ref_emb = ref_signals["emb"].detach().clone().float()
        ref_emb_n = F.normalize(ref_emb, dim=1)
        anchor = 1.0 - (emb_n * ref_emb_n).sum(dim=1)

        pseudo = ref_signals["pseudo"].detach().clone()
        target_proto = prototypes[pseudo].detach().clone()
        proto_loss = 1.0 - (emb_n * target_proto).sum(dim=1)

        total = (
            lambda_entropy * ent[selected].mean()
            + lambda_anchor * anchor[selected].mean()
            + lambda_proto * proto_loss[selected].mean()
        )

        cons_value = torch.zeros((), device=device)
        if lambda_consistency > 0:
            logits_flip, _ = model(torch.flip(images, dims=[3]))
            probs_flip = F.softmax(logits_flip.float(), dim=1)
            cons = js_divergence(probs, probs_flip)
            cons_value = cons[selected].mean()
            total = total + lambda_consistency * cons_value

    return total


def run_stream(
    source_model: nn.Module,
    loader: DataLoader,
    method: str,
    corruption: str,
    severity: int,
    prototypes: torch.Tensor,
    thresholds: GateThresholds,
    clean_loader: DataLoader,
    clean_acc_before: float,
    device: torch.device,
    args,
    batch_writer,
):
    t0 = time.time()

    # Fresh copy = episodic reset for every method/corruption/severity.
    model = copy.deepcopy(source_model).to(device)
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = None
    if method != "source":
        tta_params = configure_tta_parameters(model, args.adapt_last_blocks)
        optimizer = torch.optim.AdamW(
            tta_params,
            lr=args.tta_lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )

    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_rel: List[torch.Tensor] = []
    selected_count = 0
    total_count = 0
    adapted_batches = 0
    adapted_sample_count = 0
    harmful_samples = 0
    beneficial_samples = 0
    drift_sum = 0.0
    drift_count = 0
    gate_rel_sum = 0.0
    shift_sum = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        # Frozen source signals: no labels are used here.
        with torch.inference_mode():
            ref = source_signals(source_model, images, prototypes, device, args.amp_dtype)
        # Masks originating from inference_mode cannot be saved by autograd during
        # indexed adaptation loss, so materialize a normal tensor here.
        selected = select_mask(method, ref, thresholds).detach().clone()

        bsz = labels.numel()
        selected_n = int(selected.sum().item())
        selected_count += selected_n
        total_count += bsz
        gate_rel_sum += float(ref["reliability"].sum().item())
        shift_sum += float(ref["shift"].sum().item())

        # Pre-update performance is calculated only for the harmful-update audit.
        # Labels are NOT used to decide whether or how to adapt.
        with torch.inference_mode():
            with autocast_context(device, args.amp_dtype):
                pre_logits, _, _ = model(images, return_embedding=True)
            pre_pred = pre_logits.argmax(1)
            pre_acc = pre_pred.eq(labels).float().mean().item()
            pre_nll_each = F.cross_entropy(pre_logits.float(), labels, reduction="none")

        did_adapt = False
        loss_value = float("nan")
        if method != "source" and selected_n >= args.min_selected:
            did_adapt = True
            adapted_batches += 1
            adapted_sample_count += selected_n
            for _tta_step in range(max(1, args.tta_steps)):
                optimizer.zero_grad(set_to_none=True)
                loss = adaptation_loss(
                    model=model,
                    images=images,
                    selected=selected,
                    ref_signals=ref,
                    prototypes=prototypes,
                    device=device,
                    amp_dtype=args.amp_dtype,
                    lambda_entropy=args.lambda_entropy,
                    lambda_anchor=args.lambda_anchor,
                    lambda_proto=args.lambda_proto,
                    lambda_consistency=args.lambda_consistency,
                )
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.grad_clip
                    )
                optimizer.step()
                loss_value = float(loss.detach().item())

        # Evaluate after the optional update.
        with torch.inference_mode():
            with autocast_context(device, args.amp_dtype):
                post_logits, post_rel_logit, post_emb = model(images, return_embedding=True)
            post_probs = F.softmax(post_logits.float(), dim=1)
            post_rel_prob = torch.sigmoid(post_rel_logit.float())
            post_pred = post_probs.argmax(1)
            post_acc = post_pred.eq(labels).float().mean().item()
            post_nll_each = F.cross_entropy(post_logits.float(), labels, reduction="none")

            post_emb_n = F.normalize(post_emb.float(), dim=1)
            ref_emb_n = F.normalize(ref["emb"].float(), dim=1)
            drift = (1.0 - (post_emb_n * ref_emb_n).sum(dim=1)).mean().item()

        if did_adapt:
            # Label-aware audit only; labels never participate in the adaptation decision.
            # A selected sample is harmful if its true-label NLL increases after the update,
            # and beneficial if its true-label NLL decreases. This is more sensitive than
            # batch-level accuracy, which can remain unchanged even when confidence worsens.
            eps = 1e-7
            harmful_samples += int(((post_nll_each > pre_nll_each + eps) & selected).sum().item())
            beneficial_samples += int(((post_nll_each < pre_nll_each - eps) & selected).sum().item())

        drift_sum += drift * bsz
        drift_count += bsz
        all_probs.append(post_probs.cpu())
        all_labels.append(labels.cpu())
        all_rel.append(post_rel_prob.cpu())

        batch_writer.writerow({
            "method": method,
            "corruption": corruption,
            "severity": severity,
            "batch": batch_idx,
            "batch_size": bsz,
            "selected": selected_n,
            "selected_pct": 100.0 * selected_n / bsz,
            "adapted": int(did_adapt),
            "pre_accuracy": 100.0 * pre_acc,
            "post_accuracy": 100.0 * post_acc,
            "harmful_update": int(did_adapt and post_acc < pre_acc - 1e-12),
            "beneficial_update": int(did_adapt and post_acc > pre_acc + 1e-12),
            "loss": loss_value,
            "representation_drift": drift,
            "mean_gate_reliability": float(ref["reliability"].mean().item()),
            "mean_shift": float(ref["shift"].mean().item()),
            "mean_entropy": float(ref["entropy"].mean().item()),
        })

        if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
            print(
                f"  {method:19s} | batch {batch_idx+1:04d}/{len(loader):04d} | "
                f"selected {selected_n:3d}/{bsz:3d} | pre {100*pre_acc:5.2f} | post {100*post_acc:5.2f}"
            )

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    rel_prob = torch.cat(all_rel, dim=0)

    if method == "source":
        clean_acc_after = clean_acc_before
    else:
        clean_acc_after = clean_accuracy(model, clean_loader, device, args.amp_dtype)

    metrics = final_metrics_from_buffers(
        probs=probs,
        labels=labels,
        rel_prob=rel_prob,
        method=method,
        corruption=corruption,
        severity=severity,
        selected_count=selected_count,
        total_count=total_count,
        adapted_sample_count=adapted_sample_count,
        adapted_batches=adapted_batches,
        total_batches=len(loader),
        harmful_samples=harmful_samples,
        beneficial_samples=beneficial_samples,
        drift_sum=drift_sum,
        drift_count=drift_count,
        clean_acc_after=clean_acc_after,
        clean_acc_before=clean_acc_before,
        gate_reliability_sum=gate_rel_sum,
        shift_sum=shift_sum,
        seconds=time.time() - t0,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("ReSPT-24 CIFAR-100-C selective TTA benchmark")

    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--cifar100c_dir", type=str, default="./data/CIFAR-100-C")
    p.add_argument("--download_if_missing", action="store_true")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--checkpoint", type=str, required=True,
                   help="Dense best.pt or recovered pruned best_recovered.pt")
    p.add_argument("--dense_checkpoint", type=str, required=True,
                   help="Original dense best.pt used to reconstruct pruned architecture")
    p.add_argument("--prune_ratio", type=float, default=0.0,
                   help="0 for dense, 0.25 or 0.50 for your recovered pruned models")
    p.add_argument("--model_name", type=str, default="respt24")

    p.add_argument("--methods", nargs="+",
                   default=["source", "always", "entropy_gate", "reliability_guided"],
                   choices=["source", "always", "entropy_gate", "reliability_guided"])
    p.add_argument("--corruptions", nargs="+", default=["quick"],
                   help="Use quick, benchmark, all19, or explicit corruption names")
    p.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_samples_per_stream", type=int, default=0,
                   help="0 = all 10,000 images; use e.g. 1000 for a smoke test")
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=40)

    # Calibration from clean source training data.
    p.add_argument("--calibration_samples", type=int, default=10000)
    p.add_argument("--calibration_batch_size", type=int, default=128)
    p.add_argument("--entropy_quantile", type=float, default=0.75,
                   help="Low-entropy gate: adapt below this source entropy quantile")
    p.add_argument("--reliability_quantile", type=float, default=0.25,
                   help="Reliability gate: adapt above this source reliability quantile")
    p.add_argument("--shift_quantile", type=float, default=0.95,
                   help="Shift gate: adapt above this source prototype-distance quantile")

    # TTA parameters.
    p.add_argument("--adapt_last_blocks", type=int, default=6)
    p.add_argument("--tta_lr", type=float, default=1e-4)
    p.add_argument("--tta_steps", type=int, default=1,
                   help="Gradient update steps per adapted test batch")
    p.add_argument("--min_selected", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_entropy", type=float, default=1.0)
    p.add_argument("--lambda_anchor", type=float, default=0.05)
    p.add_argument("--lambda_proto", type=float, default=0.10)
    p.add_argument("--lambda_consistency", type=float, default=0.05)

    # Source forgetting audit.
    p.add_argument("--forgetting_samples", type=int, default=2000)
    p.add_argument("--forgetting_batch_size", type=int, default=128)

    return p.parse_args()


def expand_corruptions(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        x = item.lower()
        if x == "quick":
            out.extend(QUICK_CORRUPTIONS)
        elif x in ("benchmark", "all15"):
            out.extend(BENCHMARK_CORRUPTIONS)
        elif x == "all19":
            out.extend(BENCHMARK_CORRUPTIONS + EXTRA_CORRUPTIONS)
        else:
            out.append(item)
    # Stable unique order.
    seen = set()
    unique = []
    for x in out:
        if x not in seen:
            unique.append(x)
            seen.add(x)
    return unique


def aggregate_rows(rows: List[dict], methods: Sequence[str]) -> List[dict]:
    numeric = [
        "top1", "top5", "nll", "ece", "brier", "reliability_brier",
        "error_detection_auroc", "aurc", "adaptation_rate", "adapted_batch_rate",
        "harmful_update_rate", "beneficial_update_rate", "mean_representation_drift",
        "clean_accuracy_after", "source_forgetting", "mean_gate_reliability", "mean_shift",
        "seconds",
    ]
    out = []
    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        if not mr:
            continue
        row = {"method": method, "n_streams": len(mr)}
        for k in numeric:
            vals = np.array([float(r[k]) for r in mr], dtype=np.float64)
            row[f"mean_{k}"] = float(np.nanmean(vals))
        out.append(row)
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)

    if not (0.0 <= args.prune_ratio < 1.0):
        raise ValueError("--prune_ratio must be in [0, 1)")
    for s in args.severities:
        if s not in (1, 2, 3, 4, 5):
            raise ValueError("All --severities values must be in {1,2,3,4,5}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dump(vars(args), out_dir / "run_config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    cifar100c_root = ensure_cifar100c(Path(args.cifar100c_dir), args.download_if_missing)
    corruptions = expand_corruptions(args.corruptions)
    missing = [c for c in corruptions if not (cifar100c_root / f"{c}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Requested corruption files are missing: {missing}")

    source_model, cfg = load_source_model(
        checkpoint=args.checkpoint,
        dense_checkpoint=args.dense_checkpoint,
        prune_ratio=args.prune_ratio,
        device=device,
    )
    image_size = int(cfg.get("image_size", 128))
    dim = int(cfg.get("dim", 384))
    print(f"Model: {args.model_name} | prune_ratio={args.prune_ratio:.2f}")
    print(f"Parameters: {count_params(source_model)/1e6:.2f}M | image_size={image_size} | dim={dim}")

    # Clean source calibration subset.
    clean_train = make_clean_dataset(args.data_dir, image_size, train=True)
    cal_idx = subset_indices(len(clean_train), args.calibration_samples, args.seed)
    cal_ds = Subset(clean_train, cal_idx.tolist())
    cal_loader = make_loader(cal_ds, args.calibration_batch_size, args.workers, device)

    print(f"Calibrating source prototypes and gates on {len(cal_ds):,} clean training images...")
    prototypes, proto_counts = compute_prototypes(
        source_model, cal_loader, device, num_classes=100, dim=dim, amp_dtype=args.amp_dtype
    )
    # compute_prototypes runs under inference_mode; materialize a normal tensor
    # because prototypes are later used inside gradient-tracked adaptation loss.
    prototypes = prototypes.detach().clone()
    thresholds, calib_rel_auroc = calibrate_source(
        source_model,
        cal_loader,
        prototypes,
        device,
        args.amp_dtype,
        entropy_quantile=args.entropy_quantile,
        reliability_quantile=args.reliability_quantile,
        shift_quantile=args.shift_quantile,
    )
    th_dict = asdict(thresholds)
    th_dict["calibration_rel_head_auroc"] = calib_rel_auroc
    th_dict["prototype_min_count"] = float(proto_counts.min().item())
    th_dict["prototype_max_count"] = float(proto_counts.max().item())
    json_dump(th_dict, out_dir / "calibration_thresholds.json")

    print("Gate thresholds from CLEAN source data:")
    print(f"  entropy <= {thresholds.entropy_max:.4f}")
    print(f"  reliability >= {thresholds.reliability_min:.4f}")
    print(f"  shift >= {thresholds.shift_min:.4f}")
    print(f"  clean calibration rel-head AUROC: {calib_rel_auroc:.4f}")

    # Clean holdout for forgetting audit.
    clean_test = make_clean_dataset(args.data_dir, image_size, train=False)
    forget_idx = subset_indices(len(clean_test), args.forgetting_samples, args.seed + 1)
    forget_ds = Subset(clean_test, forget_idx.tolist())
    forget_loader = make_loader(forget_ds, args.forgetting_batch_size, args.workers, device)
    clean_acc_before = clean_accuracy(source_model, forget_loader, device, args.amp_dtype)
    print(f"Clean reference accuracy on forgetting subset ({len(forget_ds):,} images): {clean_acc_before:.2f}%")

    summary_path = out_dir / "summary.csv"
    batch_path = out_dir / "batch_history.csv"

    summary_fields = list(StreamMetrics.__dataclass_fields__.keys())
    batch_fields = [
        "method", "corruption", "severity", "batch", "batch_size", "selected",
        "selected_pct", "adapted", "pre_accuracy", "post_accuracy", "harmful_update",
        "beneficial_update", "loss", "representation_drift", "mean_gate_reliability",
        "mean_shift", "mean_entropy",
    ]

    rows: List[dict] = []
    with open(summary_path, "w", newline="", encoding="utf-8") as sf, \
         open(batch_path, "w", newline="", encoding="utf-8") as bf:
        sw = csv.DictWriter(sf, fieldnames=summary_fields)
        bw = csv.DictWriter(bf, fieldnames=batch_fields)
        sw.writeheader()
        bw.writeheader()

        for corruption in corruptions:
            for severity in args.severities:
                ds = CIFAR100CDataset(
                    cifar100c_root,
                    corruption=corruption,
                    severity=severity,
                    transform=make_eval_transform(image_size),
                )
                if args.max_samples_per_stream > 0 and args.max_samples_per_stream < len(ds):
                    idx = subset_indices(
                        len(ds), args.max_samples_per_stream,
                        args.seed + 1000 * severity + sum(ord(ch) for ch in corruption)
                    )
                    ds = Subset(ds, idx.tolist())
                loader = make_loader(ds, args.batch_size, args.workers, device)
                print("\n" + "=" * 105)
                print(f"CORRUPTION: {corruption} | severity {severity} | {len(ds):,} images")
                print("=" * 105)

                for method in args.methods:
                    metrics = run_stream(
                        source_model=source_model,
                        loader=loader,
                        method=method,
                        corruption=corruption,
                        severity=severity,
                        prototypes=prototypes,
                        thresholds=thresholds,
                        clean_loader=forget_loader,
                        clean_acc_before=clean_acc_before,
                        device=device,
                        args=args,
                        batch_writer=bw,
                    )
                    row = asdict(metrics)
                    rows.append(row)
                    sw.writerow(row)
                    sf.flush()
                    bf.flush()
                    print(
                        f"{method:19s} | Top1 {metrics.top1:6.2f} | ECE {metrics.ece:5.2f}% | "
                        f"Adapt {metrics.adaptation_rate:5.1f}% | Harm {metrics.harmful_update_rate:5.1f}% | "
                        f"Benefit {metrics.beneficial_update_rate:5.1f}% | Rel {metrics.mean_gate_reliability:.3f} | "
                        f"Shift {metrics.mean_shift:.3f} | Drift {metrics.mean_representation_drift:.5f} | "
                        f"Forget {metrics.source_forgetting:+.2f} pp"
                    )

    overall = aggregate_rows(rows, args.methods)
    overall_path = out_dir / "overall_by_method.csv"
    if overall:
        fields = list(overall[0].keys())
        with open(overall_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(overall)

    print("\n" + "=" * 105)
    print("TTA BENCHMARK COMPLETE")
    print("=" * 105)
    for row in overall:
        print(
            f"{row['method']:19s} | mean Top1 {row['mean_top1']:.2f} | "
            f"mean ECE {row['mean_ece']:.2f}% | adapt {row['mean_adaptation_rate']:.1f}% | "
            f"harm {row['mean_harmful_update_rate']:.1f}% | forgetting {row['mean_source_forgetting']:+.2f} pp"
        )
    print(f"\nSummary:       {summary_path}")
    print(f"Batch history: {batch_path}")
    print(f"Overall:       {overall_path}")
    print(f"Thresholds:    {out_dir / 'calibration_thresholds.json'}")


if __name__ == "__main__":
    main()
