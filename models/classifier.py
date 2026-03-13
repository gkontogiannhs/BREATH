"""
ICBHI Respiratory Sound Classifier
Architecture:
  1. Stem convolution (channel mixing)
  2. N x Asymmetric Inception Blocks with residual connections
  3. Dual-Axis Attention (frequency then temporal)
  4. Attention-weighted temporal pooling (MIL aggregation)
  5. MLP classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AsymmetricInceptionBlock(nn.Module):
    """
    Inception block with asymmetric kernels designed to detect:
      - Horizontal structures (wheeze harmonics) via tall-narrow kernels
      - Vertical structures (crackle spikes) via wide-short kernels
      - Local context via 3x3
      - Extended temporal context via branch D

    Input:  (B, in_ch, F, T)
    Output: (B, out_ch, F, T)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Each branch produces out_ch // 4 channels; concat → out_ch
        branch_ch = out_ch // 4

        # Branch A: pointwise (1x1) — channel mixing, no spatial bias
        self.branch_a = ConvBNReLU(in_ch, branch_ch, kernel_size=1)

        # Branch B: 1x9 — temporal (crackle-oriented, vertical structures in T)
        self.branch_b = nn.Sequential(
            ConvBNReLU(in_ch, branch_ch, kernel_size=1),
            ConvBNReLU(branch_ch, branch_ch, kernel_size=(1, 9), padding=(0, 4)),
        )

        # Branch C: 9x1 — spectral (wheeze harmonic-oriented, horizontal structures in F)
        self.branch_c = nn.Sequential(
            ConvBNReLU(in_ch, branch_ch, kernel_size=1),
            ConvBNReLU(branch_ch, branch_ch, kernel_size=(9, 1), padding=(4, 0)),
        )

        # Branch D: local 3x3 → extended 1x9 (local then extended temporal)
        self.branch_d = nn.Sequential(
            ConvBNReLU(in_ch, branch_ch, kernel_size=1),
            ConvBNReLU(branch_ch, branch_ch, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNReLU(branch_ch, branch_ch, kernel_size=(1, 9), padding=(0, 4)),
        )

        # 1x1 projection to fuse branches and restore channel count cleanly
        self.fusion = ConvBNReLU(out_ch, out_ch, kernel_size=1)

        # Residual projection if in_ch != out_ch
        self.residual = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.residual_bn = nn.BatchNorm2d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        d = self.branch_d(x)

        out = torch.cat([a, b, c, d], dim=1)  # (B, out_ch, F, T)
        out = self.fusion(out)

        # Residual
        res = self.residual_bn(self.residual(x)) if isinstance(self.residual, nn.Conv2d) else self.residual(x)
        return F.relu(out + res, inplace=True)


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------

class FrequencyAttention(nn.Module):
    """
    1D self-attention applied across the frequency (F) dimension.
    For each time step independently, learns which frequency bands matter most.

    Input:  (B, C, F, T)
    Output: (B, C, F, T)  — frequency-reweighted
    """

    def __init__(self, channels: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        # Reshape: treat each (B*T) as a batch, attend over F
        x_t = x.permute(0, 3, 2, 1).contiguous()  # (B, T, F, C)
        x_t = x_t.view(B * T, F, C)               # (B*T, F, C)

        attn_out, _ = self.attn(x_t, x_t, x_t)    # (B*T, F, C)
        attn_out = self.norm(attn_out + x_t)        # residual + norm

        attn_out = attn_out.view(B, T, F, C)
        return attn_out.permute(0, 3, 2, 1).contiguous()  # (B, C, F, T)


class TemporalAttention(nn.Module):
    """
    1D self-attention applied across the time (T) dimension.
    Learns to attend to abnormal event frames and ignore normal frames.
    Returns attention weights for interpretability (event localization).

    Input:  (B, C, F, T)
    Output: (B, C, F, T), attn_weights (B, T)
    """

    def __init__(self, channels: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)

        # Learnable query for computing pooling weights (attention over T)
        self.pool_query = nn.Parameter(torch.randn(1, 1, channels))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads,
            dropout=0.0, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, F, T = x.shape

        # Average over frequency to get a time sequence: (B, T, C)
        x_time = x.mean(dim=2).permute(0, 2, 1)  # (B, T, C)

        # Self-attention over time
        attn_out, _ = self.attn(x_time, x_time, x_time)  # (B, T, C)
        attn_out = self.norm(attn_out + x_time)

        # Compute pooling weights: one query attends over T positions
        query = self.pool_query.expand(B, -1, -1)          # (B, 1, C)
        _, pool_weights = self.pool_attn(query, attn_out, attn_out)
        pool_weights = pool_weights.squeeze(1)              # (B, T)

        # Broadcast enhanced time features back into spatial map
        # attn_out: (B, T, C) → (B, C, 1, T) → broadcast to (B, C, F, T)
        attn_broadcast = attn_out.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, T)
        x_enhanced = x + attn_broadcast.expand_as(x)

        return x_enhanced, pool_weights  # (B, C, F, T), (B, T)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ICBHIClassifier(nn.Module):
    """
    Full ICBHI 4-class respiratory sound classifier.

    Input:  (B, 4, n_mels, T)  — 4-channel spectrogram
    Output: (B, 4)             — class logits
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_mels: int = 128,
        n_classes: int = 4,
        stem_channels: int = 32,
        inception_channels: list = None,  # channel counts per inception block
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.4,
        n_inception_blocks: int = 3,
    ):
        super().__init__()
        if inception_channels is None:
            inception_channels = [64, 128, 256]

        assert len(inception_channels) == n_inception_blocks, \
            "len(inception_channels) must equal n_inception_blocks"

        # --- Stem ---
        self.stem = ConvBNReLU(in_channels, stem_channels, kernel_size=3, padding=1)

        # --- Asymmetric Inception Blocks + Pooling ---
        self.inception_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        prev_ch = stem_channels
        for out_ch in inception_channels:
            self.inception_blocks.append(AsymmetricInceptionBlock(prev_ch, out_ch))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = out_ch

        self.final_channels = prev_ch  # channels after last inception block

        # --- Dual-Axis Attention ---
        self.freq_attn = FrequencyAttention(
            channels=self.final_channels,
            n_heads=attn_heads,
            dropout=attn_dropout,
        )
        self.temporal_attn = TemporalAttention(
            channels=self.final_channels,
            n_heads=attn_heads,
            dropout=attn_dropout,
        )

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.final_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 4, n_mels, T)

        Returns:
            logits: (B, n_classes)
            attn_weights: (B, T') — temporal attention weights for visualization
        """
        # Stem
        out = self.stem(x)  # (B, 32, n_mels, T)

        # Inception blocks with max pooling
        for block, pool in zip(self.inception_blocks, self.pool_layers):
            out = block(out)
            out = pool(out)  # halve spatial dims each time

        # Dual-axis attention
        out = self.freq_attn(out)                   # (B, C, F', T')
        out, attn_weights = self.temporal_attn(out) # (B, C, F', T'), (B, T')

        # MIL aggregation: attention-weighted mean pooling over time
        # attn_weights: (B, T') → softmax → (B, 1, 1, T')
        pool_w = F.softmax(attn_weights, dim=-1).unsqueeze(1).unsqueeze(2)
        # out: (B, C, F', T') → weighted sum over T
        pooled = (out * pool_w).sum(dim=-1)         # (B, C, F')
        pooled = pooled.mean(dim=-1)                # (B, C) — avg over F'

        logits = self.classifier(pooled)            # (B, n_classes)
        return logits, attn_weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns predicted class indices."""
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
