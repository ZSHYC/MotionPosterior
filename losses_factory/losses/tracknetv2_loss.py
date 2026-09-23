"""Numerically stable heatmap and trajectory losses used by TrackNet models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module
class TrackNetV2Loss(nn.Module):
    """Soft focal BCE with optional centre/visibility/uncertainty supervision.

    The model may pass either a probability tensor or the richer dictionary
    returned by ``TrackNetMotion(return_aux=True)``. Targets remain the
    project's 0..255 heatmaps, but their Gaussian values are preserved instead
    of being collapsed to a binary disk.
    """

    def __init__(self, reduction="mean", aux_weight=0.25, offset_weight=0.2,
                 visibility_weight=0.1, uncertainty_weight=0.05,
                 trajectory_weight=0.05):
        super().__init__()
        self.reduction = reduction
        self.aux_weight = aux_weight
        self.offset_weight = offset_weight
        self.visibility_weight = visibility_weight
        self.uncertainty_weight = uncertainty_weight
        self.trajectory_weight = trajectory_weight

    def _heatmap_loss(self, logits, targets):
        y = targets.float().div(255.0).clamp(0.0, 1.0)
        prob = torch.sigmoid(logits)
        focal_weight = y * (1.0 - prob).pow(2) + (1.0 - y) * prob.pow(2)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none") * focal_weight
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

    @staticmethod
    def _target_centres(targets):
        mass = targets.float().div(255.0).clamp_min(0.0)
        b, t, h, w = mass.shape
        ys = torch.linspace(0.0, 1.0, h, device=mass.device, dtype=mass.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=mass.device, dtype=mass.dtype)
        norm = mass.sum(dim=(-1, -2))
        visible = norm > 1e-5
        x = (mass * xs.view(1, 1, 1, w)).sum(dim=(-1, -2)) / norm.clamp_min(1e-5)
        y = (mass * ys.view(1, 1, h, 1)).sum(dim=(-1, -2)) / norm.clamp_min(1e-5)
        return torch.stack((x, y), dim=-1), visible

    def forward(self, prediction, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(prediction, dict):
            logits = prediction.get("heat_logits")
            if logits is None:
                logits = torch.logit(prediction["heatmap"].clamp(1e-5, 1 - 1e-5))
        else:
            probs = prediction.clamp(1e-5, 1 - 1e-5)
            logits = torch.logit(probs)

        heat_loss = self._heatmap_loss(logits, targets)
        if not isinstance(prediction, dict) or "offset" not in prediction:
            return heat_loss

        centres, visible = self._target_centres(targets)
        offsets = prediction["offset"]
        visibility_logits = prediction.get("visibility_logits")
        uncertainty = prediction.get("uncertainty")
        aux = logits.new_zeros(())
        if offsets is not None:
            offset_error = F.smooth_l1_loss(offsets, centres, reduction="none").mean(dim=-1)
            aux = aux + self.offset_weight * (offset_error * visible.float()).sum() / visible.float().sum().clamp_min(1.0)
        if visibility_logits is not None:
            aux = aux + self.visibility_weight * F.binary_cross_entropy_with_logits(
                visibility_logits, visible.float()
            )
        if uncertainty is not None and offsets is not None:
            squared_error = (offsets - centres).pow(2).sum(dim=-1).detach()
            heteroscedastic = torch.exp(-uncertainty).clamp_max(20.0) * squared_error + uncertainty
            aux = aux + self.uncertainty_weight * heteroscedastic.mean()
        if offsets is not None and offsets.shape[1] > 1:
            velocity = offsets[:, 1:] - offsets[:, :-1]
            target_velocity = centres[:, 1:] - centres[:, :-1]
            aux = aux + self.trajectory_weight * F.smooth_l1_loss(velocity, target_velocity)
            if offsets.shape[1] > 2:
                acceleration = velocity[:, 1:] - velocity[:, :-1]
                target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
                aux = aux + 0.5 * self.trajectory_weight * F.smooth_l1_loss(
                    acceleration, target_acceleration
                )
        return heat_loss + self.aux_weight * aux
