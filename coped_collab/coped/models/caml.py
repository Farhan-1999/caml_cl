"""CAML for collaborative semantic segmentation (Liu et al., NeurIPS 2025, Sec. 4.2 and App. A.3).

Post-fusion CAML (the paper's main model):
    each robot encodes each modality locally            ResNet-18 -> 512 x 7 x 7 per (robot, modality)
    same-modality maps of all robots are concatenated   channel-wise, then a 1x1 conv back to 512
    the aggregated modality maps are fused              cross-attention between modalities
    an FCN decoder predicts the ego view                upsampled to the input size
Pre-fusion CAML (App. A.3.2): each robot first fuses its own modalities with the cross-attention,
then the fused per-robot maps are concatenated across robots and reduced by a 1x1 conv.

Choices the paper leaves open, made here:
  * one ResNet-18 per modality, shared by all robots. CAML reports 25.5M parameters for its 2-robot
    RGB+depth model, which matches two encoders (2 x 11.2M + fusion and decoder), not four;
  * ImageNet-pretrained encoders; 1-channel inputs arrive repeated to 3 channels;
  * for more than two modalities each modality attends to all the others and the outputs are
    averaged. With two modalities this is RGB<->depth cross-attention, with one it is skipped;
  * decoder = FCN head (3x3 conv, BN, ReLU, dropout, 1x1 conv) + bilinear upsampling.

Teacher and students are the same class with different `modalities`. A robot that is not
collaborating (`drop_robots`) contributes all-zero features, which is how the concatenation-based
model runs with fewer robots than it was trained with.

For deployment and timing the model splits into robot_forward (what each robot computes and
transmits) and server_forward (what the ego / edge server computes from the received messages).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def resnet18_trunk(pretrained: bool) -> nn.Sequential:
    """ResNet-18 without pooling and classifier: (B, 3, H, W) -> (B, 512, H/32, W/32)."""
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)


class RobotAggregation(nn.Module):
    """Concatenate the same feature map of every robot along channels, reduce back with a 1x1 conv."""

    def __init__(self, dim: int, num_robots: int):
        super().__init__()
        self.reduce = nn.Conv2d(dim * num_robots, dim, kernel_size=1)

    def forward(self, maps: list[torch.Tensor]) -> torch.Tensor:
        return self.reduce(torch.cat(maps, dim=1))


class CrossModalFusion(nn.Module):
    """Each modality's grid tokens attend to the tokens of the other modalities (shared attention,
    with learned position and modality embeddings on queries and keys); outputs are averaged."""

    def __init__(self, dim: int, heads: int, modalities: list[str], grid: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.modality = nn.ParameterDict({m: nn.Parameter(torch.zeros(1, 1, dim)) for m in modalities})
        nn.init.trunc_normal_(self.pos, std=0.02)
        for p in self.modality.values():
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, maps: dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(maps)
        if len(names) == 1:
            return maps[names[0]]
        b, c, h, w = maps[names[0]].shape
        tokens = {m: maps[m].flatten(2).transpose(1, 2) for m in names}           # (B, HW, C)
        keys = {m: tokens[m] + self.pos + self.modality[m] for m in names}
        fused = []
        for m in names:
            others = [n for n in names if n != m]
            out, _ = self.attn(keys[m], torch.cat([keys[n] for n in others], 1),
                               torch.cat([tokens[n] for n in others], 1), need_weights=False)
            fused.append(self.norm(tokens[m] + out))
        return torch.stack(fused).mean(0).transpose(1, 2).reshape(b, c, h, w)


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        inter = in_channels // 4
        super().__init__(
            nn.Conv2d(in_channels, inter, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(inter, num_classes, 1),
        )


class CAML(nn.Module):
    def __init__(self, robots: list[str], modalities: list[str], num_classes: int, fusion: str = "post",
                 pretrained: bool = True, dim: int = 512, heads: int = 8, grid: int = 7,
                 decoder_dropout: float = 0.1):
        super().__init__()
        if fusion not in ("post", "pre"):
            raise ValueError(f"fusion must be 'post' or 'pre', got {fusion!r}")
        self.robots, self.modalities, self.fusion = list(robots), list(modalities), fusion
        self.encoders = nn.ModuleDict({m: resnet18_trunk(pretrained) for m in self.modalities})
        self.modal_fusion = (CrossModalFusion(dim, heads, self.modalities, grid)
                             if len(self.modalities) > 1 else None)
        if fusion == "post":
            self.robot_agg = nn.ModuleDict({m: RobotAggregation(dim, len(self.robots)) for m in self.modalities})
        else:
            self.robot_agg = RobotAggregation(dim, len(self.robots))
        self.head = FCNHead(dim, num_classes, decoder_dropout)

    def _fuse_modalities(self, maps: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.modal_fusion(maps) if self.modal_fusion is not None else next(iter(maps.values()))

    def encode(self, inputs: dict) -> dict[tuple[str, str], torch.Tensor]:
        """Local per-robot features, keyed (robot, modality). Robots share each modality's encoder,
        so all robots go through it in one batch."""
        feats = {}
        for m in self.modalities:
            out = self.encoders[m](torch.cat([inputs[r][m] for r in self.robots], dim=0))
            for r, f in zip(self.robots, out.chunk(len(self.robots), dim=0)):
                feats[(r, m)] = f
        return feats

    def _to_messages(self, feats_by_modality: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not feats_by_modality:
            return {}
        if self.fusion == "post":
            return dict(feats_by_modality)
        return {"fused": self._fuse_modalities(feats_by_modality)}

    def robot_forward(self, robot_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """What one robot computes locally and transmits: one 512x7x7 map per modality (post-fusion)
        or a single fused map {"fused": ...} (pre-fusion). Only the model's modalities present in
        `robot_inputs` are used; a robot with none of them sends nothing."""
        return self._to_messages({m: self.encoders[m](robot_inputs[m]) for m in self.modalities if m in robot_inputs})

    def server_forward(self, messages: dict[str, dict[str, torch.Tensor]], size, drop_robots=()) -> torch.Tensor:
        """Fuse the messages of all robots (ego included) and decode the ego view at `size`.

        Missing inputs are zeros, the way CAML runs with fewer agents: a robot that is dropped or
        sends nothing contributes all-zero maps, and a modality only some robots send is zero for
        the others. A modality no robot sends is left out of the cross-modal fusion.
        """
        dropped = set(drop_robots)
        present = {r: messages[r] for r in self.robots if messages.get(r) and r not in dropped}
        if not present:
            raise ValueError("no robot contributes any feature")
        ref = next(iter(next(iter(present.values())).values()))

        def get(robot: str, key: str) -> torch.Tensor:
            msg = present.get(robot, {})
            return msg[key] if key in msg else torch.zeros_like(ref)

        if self.fusion == "post":
            sent = [m for m in self.modalities if any(m in msg for msg in present.values())]
            fused = self._fuse_modalities({m: self.robot_agg[m]([get(r, m) for r in self.robots]) for m in sent})
        else:
            fused = self.robot_agg([get(r, "fused") for r in self.robots])
        return F.interpolate(self.head(fused), size=size, mode="bilinear", align_corners=False)

    def forward_available(self, feats: dict, available: dict[str, list], size) -> torch.Tensor:
        """Prediction when each robot sends only `available[robot]` (a subset of the model's
        modalities), computed from encoder features of `encode` so many subsets share one encoding."""
        msgs = {r: self._to_messages({m: feats[(r, m)] for m in mods}) for r, mods in available.items() if mods}
        return self.server_forward(msgs, size)

    def forward(self, inputs: dict, drop_robots: tuple | list = ()) -> torch.Tensor:
        """Same result as robot_forward on every robot followed by server_forward, but the robots
        share one batched pass through each encoder."""
        size = inputs[self.robots[0]][self.modalities[0]].shape[-2:]
        feats = self.encode(inputs)
        messages = {r: self._to_messages({m: feats[(r, m)] for m in self.modalities}) for r in self.robots}
        return self.server_forward(messages, size, drop_robots)


def build_model(cfg: dict) -> CAML:
    d, m = cfg["data"], cfg["model"]
    return CAML(
        robots=m.get("robots") or [d["ego"], *d["collaborators"]],
        modalities=m["modalities"],
        num_classes=d["num_classes"],
        fusion=m.get("fusion", "post"),
        pretrained=m.get("pretrained", True),
        dim=m.get("feature_dim", 512),
        heads=m.get("attention_heads", 8),
        grid=d["image_size"] // 32,
        decoder_dropout=m.get("decoder_dropout", 0.1),
    )


def parameter_counts(model: CAML) -> dict:
    count = lambda mod: sum(p.numel() for p in mod.parameters()) if mod is not None else 0  # noqa: E731
    return {"total": count(model), "encoders": count(model.encoders), "robot_aggregation": count(model.robot_agg),
            "modal_fusion": count(model.modal_fusion), "decoder": count(model.head)}
