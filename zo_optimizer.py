"""
zo_optimizer.py - Conservative zero-order fine-tuning.

The initialized frozen-feature head is already strong, so the optimizer only
uses SPSA on the classifier bias. This keeps the submitted fine-tuning stage
inside the scalar-loss black-box rule while avoiding the overfitting observed
with larger class-wise and component-scale calibrations.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Bias-only SPSA optimizer using two scalar loss queries per step."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-6,
        eps: float = 1e-3,
        perturbation_mode: str = "rademacher",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.perturbation_mode = perturbation_mode
        self.layer_names: list[str] = ["fc.bias"]
        self.max_grad_scale = 5.0

    def _active_params(self) -> dict[str, nn.Parameter]:
        named_params = dict(self.model.named_parameters())
        missing = [name for name in self.layer_names if name not in named_params]
        if missing:
            raise KeyError(f"Unknown trainable parameter(s): {missing}")
        return {name: named_params[name] for name in self.layer_names}

    def _direction(self, param: nn.Parameter) -> torch.Tensor:
        if self.perturbation_mode == "gaussian":
            direction = torch.randn_like(param)
            return direction / direction.norm().clamp_min(1e-12)
        if self.perturbation_mode != "rademacher":
            raise ValueError(f"Unknown perturbation mode: {self.perturbation_mode}")
        return torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Estimate a pseudo-gradient from scalar losses and update in-place."""
        params = self._active_params()
        directions = {name: self._direction(param) for name, param in params.items()}

        with torch.no_grad():
            for name, param in params.items():
                param.add_(directions[name], alpha=self.eps)
        loss_plus = float(loss_fn())

        with torch.no_grad():
            for name, param in params.items():
                param.add_(directions[name], alpha=-2.0 * self.eps)
        loss_minus = float(loss_fn())

        grad_scale = (loss_plus - loss_minus) / (2.0 * self.eps)
        grad_scale = max(-self.max_grad_scale, min(self.max_grad_scale, grad_scale))

        with torch.no_grad():
            for name, param in params.items():
                param.add_(directions[name], alpha=self.eps)
                param.add_(directions[name], alpha=-self.lr * grad_scale)

        return 0.5 * (loss_plus + loss_minus)
