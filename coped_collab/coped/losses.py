"""Losses: task cross-entropy and CAML's knowledge-distillation objective (App. A.5)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def task_loss(logits: torch.Tensor, label: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return F.cross_entropy(logits.float(), label, ignore_index=ignore_index)


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, label: torch.Tensor, alpha: float,
                      temperature: float, ignore_index: int) -> dict[str, torch.Tensor]:
    """L_S = (1 - alpha) * CE(student, label) + alpha * t^2 * L_KD, with
    L_KD = -sum_i softmax(z_T / t)_i * log softmax(z_S / t)_i, the soft cross-entropy of CAML App. A.5.

    Applied per pixel and averaged over every pixel: the teacher gives soft targets also where the
    label is unlabeled, while the hard term ignores those pixels.
    """
    t = temperature
    hard = task_loss(student, label, ignore_index)
    soft_targets = F.softmax(teacher.float() / t, dim=1)
    soft = -(soft_targets * F.log_softmax(student.float() / t, dim=1)).sum(dim=1).mean()
    total = (1 - alpha) * hard + alpha * t * t * soft
    return {"loss": total, "ce": hard.detach(), "kd": soft.detach()}
