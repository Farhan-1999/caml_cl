"""Semantic segmentation metrics from one confusion matrix accumulated over a whole split."""
from __future__ import annotations

import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int, device: torch.device | str = "cpu"):
        self.n = num_classes
        self.ignore = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        valid = target != self.ignore
        idx = target[valid] * self.n + pred[valid]
        self.mat += torch.bincount(idx, minlength=self.n * self.n).view(self.n, self.n)

    def compute(self, rare_classes: list[int] | tuple = ()) -> dict:
        """IoU per class (None for classes absent from the ground truth of this split).

        miou        mean IoU over classes present in the ground truth
        miou_common the same without `rare_classes` (tiny classes whose IoU is mostly noise)
        """
        mat = self.mat.double().cpu()
        tp = mat.diag()
        gt = mat.sum(1)
        union = gt + mat.sum(0) - tp
        present = [c for c in range(self.n) if gt[c] > 0]
        iou = [float(tp[c] / union[c]) if c in present else None for c in range(self.n)]
        common = [c for c in present if c not in set(rare_classes)]
        mean = lambda cs: float(sum(iou[c] for c in cs) / len(cs)) if cs else float("nan")  # noqa: E731
        return {
            "miou": mean(present),
            "miou_common": mean(common),
            "pixel_accuracy": float(tp.sum() / max(1.0, float(gt.sum()))),
            "iou": iou,
            "present_classes": present,
            "pixels": int(gt.sum()),
        }
