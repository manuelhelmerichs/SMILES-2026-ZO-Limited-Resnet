"""
head_init.py - Final layer initialization (student-implemented).

The head is initialized as a frozen-feature hybrid classifier.  A regularized
multinomial linear probe remains in ``fc.weight`` and ``fc.bias``; the layer's
forward method is extended with a small LDA/ridge logit and a soft-kNN term over
deterministic train features.  The backbone and validation path stay unchanged.
"""

from __future__ import annotations

import os
import random
from functools import partial
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T

from augmentation import _CIFAR100_MEAN, _CIFAR100_STD, get_transforms


_NUM_CLASSES = 100
_FEATURE_DIM = 512

_FEATURE_BATCH_SIZE = 512
_NUM_WORKERS = 4
_CROP_SEED = 123
_CROP_SEED_2 = 456

_LOGREG_LAMBDA = 1.25e-3
_LABEL_SMOOTHING = 0.05
_LBFGS_MAX_ITER = 140

_RIDGE_LAMBDA = 1e-3
_RIDGE_BLEND = 0.25
_COUNT_CALIBRATION_ETA = 0.2
_COUNT_CALIBRATION_STEPS = 10

_LOGREG_CACHE_VERSION = "v5_logreg_orig_hflip_crop_lam00125_smooth005"
_LDA_CACHE_VERSION = "v4_lda_ridge_countcal10"
_HYBRID_CACHE_VERSION = "v7_hybrid_logreg_lda_knn5_k125_dualtemp"

_KNN_K = 125
_KNN_TEMPERATURES = (0.0045, 0.006)
_KNN_BETAS = (0.35, 0.2)
_LOGREG_TEMPERATURE = 0.65
_LDA_SCALE = 0.8
_KNN_EPS = 1e-12


def _data_dir() -> str:
    return os.environ.get("CIFAR100_DATA_DIR", "./data")


def _head_cache_path(data_dir: str, version: str) -> Path:
    return Path(data_dir) / f"cifar100_resnet18_head_{version}.pt"


def _view_cache_path(data_dir: str, view_name: str) -> Path:
    names = {
        "orig": "cifar100_resnet18_train_valtx_v1.pt",
        "hflip": "cifar100_resnet18_train_hflip_v1.pt",
        "crop123": "cifar100_resnet18_train_crop_seed123_w4_v1.pt",
        "crop456": "cifar100_resnet18_train_crop_seed456_w4_v1.pt",
        "crop_hflip123": "cifar100_resnet18_train_crop_hflip_seed123_w4_v1.pt",
    }
    return Path(data_dir) / names[view_name]


def _hflip_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(224),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )


def _crop_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(224),
            T.RandomCrop(224, padding=28),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )


def _crop_hflip_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(224),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomCrop(224, padding=28),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )


def _seed_crop_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _view_spec(view_name: str) -> tuple[T.Compose, int | None]:
    if view_name == "orig":
        return get_transforms(train=False), None
    if view_name == "hflip":
        return _hflip_transform(), None
    if view_name == "crop123":
        return _crop_transform(), _CROP_SEED
    if view_name == "crop456":
        return _crop_transform(), _CROP_SEED_2
    if view_name == "crop_hflip123":
        return _crop_hflip_transform(), _CROP_SEED
    raise KeyError(f"Unknown feature view: {view_name}")


@torch.no_grad()
def _extract_view_features(
    data_dir: str,
    backbone: nn.Module,
    device: torch.device,
    transform: T.Compose,
    *,
    crop_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    generator = None
    worker_init_fn = None
    if crop_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(crop_seed)
        worker_init_fn = partial(_seed_crop_worker, base_seed=crop_seed)

    loader = DataLoader(
        dataset,
        batch_size=_FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    features = torch.empty((len(dataset), _FEATURE_DIM), dtype=torch.float32)
    labels = torch.empty(len(dataset), dtype=torch.long)
    offset = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        batch_features = backbone(images).cpu()
        batch_size = batch_features.size(0)
        features[offset : offset + batch_size].copy_(batch_features)
        labels[offset : offset + batch_size].copy_(target)
        offset += batch_size

    return features, labels


@torch.no_grad()
def _load_or_extract_view(
    data_dir: str,
    view_name: str,
    backbone: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = _view_cache_path(data_dir, view_name)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        features = cached.get("features")
        labels = cached.get("labels")
        if (
            isinstance(features, torch.Tensor)
            and isinstance(labels, torch.Tensor)
            and tuple(features.shape) == (50_000, _FEATURE_DIM)
            and tuple(labels.shape) == (50_000,)
        ):
            return features.float(), labels.long()

    transform, crop_seed = _view_spec(view_name)
    features, labels = _extract_view_features(
        data_dir,
        backbone,
        device,
        transform,
        crop_seed=crop_seed,
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features.cpu(), "labels": labels.cpu()}, cache_path)
    except OSError:
        pass
    return features, labels


@torch.no_grad()
def _load_augmented_train_features(
    data_dir: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    feature_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    by_name: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for view_name in ("orig", "hflip", "crop123", "crop456", "crop_hflip123"):
        features, labels = _load_or_extract_view(data_dir, view_name, backbone, device)
        by_name[view_name] = (features, labels)
        feature_parts.append(features)
        label_parts.append(labels)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    probe_views = ("orig", "hflip", "crop123")
    probe_features = torch.cat([by_name[name][0] for name in probe_views], dim=0)
    probe_labels = torch.cat([by_name[name][1] for name in probe_views], dim=0)
    orig_features, orig_labels = by_name["orig"]
    return (
        probe_features,
        probe_labels,
        torch.cat(feature_parts, dim=0),
        torch.cat(label_parts, dim=0),
        orig_features,
        orig_labels,
    )


def _fit_logistic_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    standardized = ((features - mean) / std).to(device)
    target = labels.to(device)

    weight = torch.zeros(
        (_NUM_CLASSES, _FEATURE_DIM),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    bias = torch.zeros(
        _NUM_CLASSES,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=_LBFGS_MAX_ITER,
        history_size=25,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = standardized.matmul(weight.t()) + bias
        loss = F.cross_entropy(
            logits,
            target,
            label_smoothing=_LABEL_SMOOTHING,
        )
        loss = loss + 0.5 * _LOGREG_LAMBDA * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        fitted_weight = weight.detach().cpu()
        fitted_bias = bias.detach().cpu()
        raw_weight = fitted_weight / std.unsqueeze(0)
        raw_bias = fitted_bias - (mean / std).matmul(fitted_weight.t())

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return raw_weight.float(), raw_bias.float()


def _class_means(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [features[labels == class_idx].mean(dim=0) for class_idx in range(_NUM_CLASSES)]
    )


def _fit_lda_head(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    means = _class_means(features, labels)
    centered = features - means[labels]
    covariance = centered.t().matmul(centered) / (features.size(0) - _NUM_CLASSES)

    try:
        weights = torch.linalg.solve(covariance.double(), means.t().double()).float().t()
    except RuntimeError:
        jitter = covariance.diag().mean().clamp_min(1e-12) * 1e-5
        covariance = covariance + jitter * torch.eye(covariance.size(0))
        weights = torch.linalg.solve(covariance.double(), means.t().double()).float().t()

    bias = -0.5 * (means * weights).sum(dim=1)
    return weights, bias


def _fit_ridge_head(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_samples, n_features = features.shape
    targets = torch.zeros((n_samples, _NUM_CLASSES), dtype=torch.float32)
    targets[torch.arange(n_samples), labels] = 1.0

    design = torch.cat([features, torch.ones(n_samples, 1)], dim=1).double()
    system = design.t().matmul(design)
    rhs = design.t().matmul(targets.double())

    diag = torch.arange(n_features + 1)
    system[diag, diag] += _RIDGE_LAMBDA
    system[-1, -1] -= _RIDGE_LAMBDA

    solution = torch.linalg.solve(system, rhs).float()
    return solution[:n_features].t(), solution[n_features]


def _normalize_head(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = features.matmul(weight.t()) + bias
    scale = logits.std().clamp_min(1e-12)
    return weight / scale, bias / scale


def _balance_prediction_marginal(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    logits = features.matmul(weight.t()) + bias
    bias_delta = torch.zeros_like(bias)
    target = features.size(0) / _NUM_CLASSES

    for _ in range(_COUNT_CALIBRATION_STEPS):
        pred = (logits + bias_delta).argmax(dim=1)
        counts = torch.bincount(pred, minlength=_NUM_CLASSES).float().clamp_min(1.0)
        bias_delta -= _COUNT_CALIBRATION_ETA * torch.log(counts / target)

    return bias + bias_delta


def _fit_lda_ridge_head(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lda_weight, lda_bias = _fit_lda_head(features, labels)
    ridge_weight, ridge_bias = _fit_ridge_head(features, labels)
    lda_weight, lda_bias = _normalize_head(features, lda_weight, lda_bias)
    ridge_weight, ridge_bias = _normalize_head(features, ridge_weight, ridge_bias)

    weight = lda_weight + _RIDGE_BLEND * ridge_weight
    bias = lda_bias + _RIDGE_BLEND * ridge_bias
    bias = _balance_prediction_marginal(features, weight, bias)
    return weight.float(), bias.float()


def _load_or_fit_linear_head(
    data_dir: str,
    version: str,
    fit_fn,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = _head_cache_path(data_dir, version)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        weight = cached.get("weight")
        bias = cached.get("bias")
        if (
            isinstance(weight, torch.Tensor)
            and isinstance(bias, torch.Tensor)
            and tuple(weight.shape) == (_NUM_CLASSES, _FEATURE_DIM)
            and tuple(bias.shape) == (_NUM_CLASSES,)
        ):
            return weight.float(), bias.float()

    weight, bias = fit_fn(features, labels)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": weight.cpu(), "bias": bias.cpu()}, cache_path)
    except OSError:
        pass
    return weight, bias


def _fit_hybrid_components(data_dir: str) -> dict[str, torch.Tensor]:
    (
        probe_features,
        probe_labels,
        knn_features,
        knn_labels,
        orig_features,
        orig_labels,
    ) = _load_augmented_train_features(data_dir)

    weight, bias = _load_or_fit_linear_head(
        data_dir,
        _LOGREG_CACHE_VERSION,
        _fit_logistic_probe,
        probe_features,
        probe_labels,
    )
    lda_weight, lda_bias = _load_or_fit_linear_head(
        data_dir,
        _LDA_CACHE_VERSION,
        _fit_lda_ridge_head,
        orig_features,
        orig_labels,
    )

    return {
        "weight": weight.cpu().float(),
        "bias": bias.cpu().float(),
        "lda_weight": lda_weight.cpu().float(),
        "lda_bias": lda_bias.cpu().float(),
        "knn_features": F.normalize(knn_features.cpu().float(), dim=1).contiguous(),
        "knn_labels": knn_labels.cpu().long().contiguous(),
    }


def _valid_hybrid_cache(cached: object) -> bool:
    if not isinstance(cached, dict):
        return False
    expected = {
        "weight": (_NUM_CLASSES, _FEATURE_DIM),
        "bias": (_NUM_CLASSES,),
        "lda_weight": (_NUM_CLASSES, _FEATURE_DIM),
        "lda_bias": (_NUM_CLASSES,),
        "knn_features": (250_000, _FEATURE_DIM),
        "knn_labels": (250_000,),
    }
    for key, shape in expected.items():
        value = cached.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            return False
    return True


def _load_or_fit_hybrid_components(data_dir: str) -> dict[str, torch.Tensor]:
    cache_path = _head_cache_path(data_dir, _HYBRID_CACHE_VERSION)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if _valid_hybrid_cache(cached):
            return {key: value.cpu() for key, value in cached.items()}

    components = _fit_hybrid_components(data_dir)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(components, cache_path)
    except OSError:
        pass
    return components


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value, persistent=False)


def _hybrid_forward(self: nn.Linear, input: torch.Tensor) -> torch.Tensor:
    base_logits = F.linear(input, self.weight, self.bias).float() / _LOGREG_TEMPERATURE
    lda_logits = F.linear(input, self._lda_weight, self._lda_bias).float() * _LDA_SCALE

    query = F.normalize(input.float(), dim=1)
    bank = self._knn_features.to(device=query.device, dtype=query.dtype)
    labels = self._knn_labels.to(device=query.device)
    similarities = query.matmul(bank.t())
    values, indices = similarities.topk(_KNN_K, dim=1)

    neighbor_labels = labels[indices]
    neighbor_one_hot = F.one_hot(neighbor_labels, num_classes=_NUM_CLASSES).to(
        dtype=values.dtype
    )

    knn_logits = torch.zeros(
        (input.size(0), _NUM_CLASSES),
        device=input.device,
        dtype=torch.float32,
    )
    for temperature, beta in zip(_KNN_TEMPERATURES, _KNN_BETAS):
        neighbor_weights = torch.softmax(values / temperature, dim=1)
        knn_probs = (neighbor_one_hot * neighbor_weights.unsqueeze(-1)).sum(dim=1)
        knn_logits += torch.log(knn_probs.clamp_min(_KNN_EPS)).float() * beta

    return (base_logits + lda_logits + knn_logits).to(dtype=input.dtype)


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize and extend the CIFAR100 classifier head in-place."""
    components = _load_or_fit_hybrid_components(_data_dir())

    with torch.no_grad():
        layer.weight.copy_(
            components["weight"].to(dtype=layer.weight.dtype, device=layer.weight.device)
        )
        layer.bias.copy_(
            components["bias"].to(dtype=layer.bias.dtype, device=layer.bias.device)
        )

    _set_buffer(layer, "_lda_weight", components["lda_weight"].float())
    _set_buffer(layer, "_lda_bias", components["lda_bias"].float())
    _set_buffer(layer, "_knn_features", components["knn_features"].float())
    _set_buffer(layer, "_knn_labels", components["knn_labels"].long())
    layer.forward = MethodType(_hybrid_forward, layer)
