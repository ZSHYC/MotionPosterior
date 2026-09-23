"""Numerically stable heatmap and trajectory losses for TrackNet models.

The auxiliary heads use normalized ``(x, y)`` coordinates in ``[0, 1]``.
When a dataloader provides ``coords``/``visibility`` they are the authority;
the rendered heatmap is only the fallback target for old datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module
class TrackNetV2Loss(nn.Module):
    """Heatmap focal BCE plus visibility-gated geometric supervision."""

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

    @staticmethod
    def _as_logits(value):
        value = value.float()
        detached = value.detach()
        if torch.isfinite(detached).all() and detached.amin() >= 0 and detached.amax() <= 1:
            return torch.logit(value.clamp(1e-5, 1 - 1e-5))
        return value

    def _heatmap_loss(self, logits, targets):
        targets = targets.float()
        if targets.detach().amax() > 1.0:
            targets = targets.div(255.0)
        y = targets.clamp(0.0, 1.0)
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
        mass = targets.float()
        if mass.detach().amax() > 1.0:
            mass = mass.div(255.0)
        mass = mass.clamp_min(0.0)
        _, _, h, w = mass.shape
        ys = torch.linspace(0.0, 1.0, h, device=mass.device, dtype=mass.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=mass.device, dtype=mass.dtype)
        norm = mass.sum(dim=(-1, -2))
        visible = norm > 1e-5
        x = (mass * xs.view(1, 1, 1, w)).sum(dim=(-1, -2)) / norm.clamp_min(1e-5)
        y = (mass * ys.view(1, 1, h, 1)).sum(dim=(-1, -2)) / norm.clamp_min(1e-5)
        return torch.stack((x, y), dim=-1), visible

    @staticmethod
    def _stack_meta(value, batch, time, device, dtype):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            try:
                value = torch.stack([torch.as_tensor(v) for v in value], dim=0)
            except (TypeError, RuntimeError):
                return None
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim >= 2 and value.shape[0] == time and value.shape[1] == batch:
            value = value.transpose(0, 1)
        elif value.ndim >= 1 and value.shape[0] != batch and value.numel() == batch * time:
            value = value.reshape(batch, time, *value.shape[1:])
        return value

    @classmethod
    def _metadata_targets(cls, targets, coords, visibility):
        centres, heat_visible = cls._target_centres(targets)
        b, t = heat_visible.shape
        coords = cls._stack_meta(coords, b, t, targets.device, targets.dtype)
        visible = cls._stack_meta(visibility, b, t, targets.device, targets.dtype)
        if coords is not None and coords.shape[-1] >= 2:
            coords = coords[..., :2]
            h, w = targets.shape[-2:]
            scale = coords.new_tensor([max(w - 1, 1), max(h - 1, 1)])
            if coords.detach().abs().amax() > 1.5:
                coords = coords / scale
            finite = torch.isfinite(coords).all(dim=-1)
            centres = torch.nan_to_num(coords, nan=0.0).clamp(0.0, 1.0)
        else:
            finite = torch.ones_like(heat_visible, dtype=torch.bool)
        if visible is None:
            visible = heat_visible
        else:
            visible = visible > 0.5
        return centres, visible & finite

    @staticmethod
    def _masked_mean(value, mask):
        mask = mask.to(dtype=value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _delta_time(kwargs, b, t, device, dtype):
        dt = kwargs.get("dt")
        if dt is None and kwargs.get("fps") is not None:
            dt = 1.0 / torch.as_tensor(kwargs["fps"], device=device, dtype=dtype)
        if dt is None:
            return torch.ones((b, max(t - 1, 1)), device=device, dtype=dtype)
        dt = torch.as_tensor(dt, device=device, dtype=dtype)
        if dt.ndim == 0:
            return dt.expand(b, max(t - 1, 1)).clamp_min(1e-6)
        if dt.ndim == 1:
            if dt.numel() == b:
                return dt[:, None].expand(b, max(t - 1, 1)).clamp_min(1e-6)
            return dt[None, :].expand(b, -1).clamp_min(1e-6)
        return dt.reshape(b, -1)[:, :max(t - 1, 1)].clamp_min(1e-6)

    def forward(self, prediction, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(prediction, dict):
            logits = prediction.get("heat_logits")
            if logits is None:
                logits = self._as_logits(prediction["heatmap"])
        else:
            logits = self._as_logits(prediction)
        heat_loss = self._heatmap_loss(logits, targets)
        if not isinstance(prediction, dict) or "offset" not in prediction:
            return heat_loss
        offsets = prediction.get("offset")
        if offsets is None:
            return heat_loss
        centres, visible = self._metadata_targets(
            targets, kwargs.get("coords"), kwargs.get("visibility")
        )
        offsets = offsets.float()
        aux = logits.new_zeros(())
        if self.offset_weight:
            offset_error = F.smooth_l1_loss(offsets, centres, reduction="none").mean(dim=-1)
            aux = aux + self.offset_weight * self._masked_mean(offset_error, visible)
        visibility_logits = prediction.get("visibility_logits")
        if visibility_logits is not None and self.visibility_weight:
            aux = aux + self.visibility_weight * F.binary_cross_entropy_with_logits(
                visibility_logits.float(), visible.float()
            )
        log_variance = prediction.get("log_variance")
        if log_variance is not None and self.uncertainty_weight:
            log_variance = log_variance.float().clamp(-8.0, 4.0)
            squared_error = (offsets - centres).pow(2).sum(dim=-1)
            nll = 0.5 * (torch.exp(-log_variance) * squared_error + 2.0 * log_variance)
            aux = aux + self.uncertainty_weight * self._masked_mean(nll, visible)
        if self.trajectory_weight and offsets.shape[1] > 1:
            b, t = offsets.shape[:2]
            dt = self._delta_time(kwargs, b, t, offsets.device, offsets.dtype)
            velocity = (offsets[:, 1:] - offsets[:, :-1]) / dt[..., None]
            target_velocity = (centres[:, 1:] - centres[:, :-1]) / dt[..., None]
            valid_pairs = visible[:, 1:] & visible[:, :-1]
            velocity_error = F.smooth_l1_loss(velocity, target_velocity, reduction="none").mean(dim=-1)
            aux = aux + self.trajectory_weight * self._masked_mean(velocity_error, valid_pairs)
            if t > 2:
                acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt[:, 1:, None]
                target_acceleration = (target_velocity[:, 1:] - target_velocity[:, :-1]) / dt[:, 1:, None]
                valid_triplets = visible[:, 2:] & visible[:, 1:-1] & visible[:, :-2]
                acceleration_error = F.smooth_l1_loss(
                    acceleration, target_acceleration, reduction="none"
                ).mean(dim=-1)
                aux = aux + 0.5 * self.trajectory_weight * self._masked_mean(
                    acceleration_error, valid_triplets
                )
        return heat_loss + self.aux_weight * aux
