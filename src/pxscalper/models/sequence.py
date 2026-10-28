"""Deep sequence models for short-horizon FX prediction.

All three models share one contract so the inference layer, the RL environment
and the exporters are model-agnostic::

    y = model(x)         x: (B, C, L) float32   ->  logits (B, n_outputs)

``C`` = number of engineered features, ``L`` = lookback. Channels are *features*
(not multivariate prices), therefore every model runs over the **time** axis and
is fully causal -- future bars can never leak in, architecturally:

* **TCN** -- dilated *causal* convolutions with residual blocks (Bai, Kolmogorov
  & Koltun, 2018). Receptive field ``1 + 2*(k-1)*(2^n - 1)`` grows exponentially,
  so a 64-bar window is covered with 4 blocks and ~10k FLOPs/sample: the fastest
  of the three, which matters when the decision budget is milliseconds.
* **LSTM / GRU** -- the classic; a single layer with ``hidden=64`` already beats
  a TCN on long-memory regimes but costs a sequential loop (higher latency).
* **PatchTST-style Transformer** -- input split into patches (``patch_len``),
  patch embedding + learnable positional encoding, then a small pre-LN encoder.
  Patching cuts sequence length ~4x, which is the trick that makes transformers
  affordable for scalping; attention over 16 patches instead of 64 steps.

Everything is TorchScript-friendly: no data-dependent control flow, no Python
containers in ``forward``, fixed attention masks computed in ``__init__``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ModelConfig",
    "CausalConv1d",
    "TCNBlock",
    "TCN",
    "LSTMSeq",
    "PatchTSTLite",
    "build_sequence_model",
    "exportable_forecaster",
    "ForecasterHead",
]


@dataclass
class ModelConfig:
    n_features: int = 31
    lookback: int = 64
    n_outputs: int = 3            # 3-class direction, or 1 for regression
    task: str = "classification"  # classification | regression
    # TCN
    channels: tuple[int, ...] = (64, 64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.1
    # LSTM/GRU
    rnn_type: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 1
    # Transformer
    patch_len: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 2
    # shared
    head_hidden: int = 64
    use_layer_norm: bool = True

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["channels"] = list(self.channels)
        return d


# --------------------------------------------------------------------- building blocks
class CausalConv1d(nn.Module):
    """1-D convolution that never looks into the future.

    Left-padding of ``dilation * (kernel_size - 1)`` keeps the output length
    equal to the input length while guaranteeing ``y[t]`` depends only on
    ``x[<= t]``.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual block: two dilated causal convs, weight-normed, gated activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        # LayerNorm over the channel axis. Applied as (B, L, C) -> norm -> (B, C, L)
        # so TorchScript/ONNX see a single, static normalisation op.
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.down = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.drop(h)
        return F.gelu(h + self.down(x))


class ForecasterHead(nn.Module):
    """Pooling + MLP head. Flattened, so TorchScript export stays trivial."""

    def __init__(self, in_dim: int, hidden: int, n_outputs: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ----------------------------------------------------------------------------- TCN
class TCN(nn.Module):
    """Temporal Convolutional Network for price-direction prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        channels = tuple(config.channels)
        layers: list[nn.Module] = []
        in_ch = config.n_features
        for i, out_ch in enumerate(channels):
            layers.append(
                TCNBlock(in_ch, out_ch, config.kernel_size, dilation=2**i, dropout=config.dropout)
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(in_ch) if config.use_layer_norm else nn.Identity()
        self.head = ForecasterHead(in_ch, config.head_hidden, config.n_outputs, config.dropout)

    @property
    def receptive_field(self) -> int:
        k = self.config.kernel_size
        n = len(self.config.channels)
        return 1 + 2 * (k - 1) * (2**n - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L). The final dilated causal conv already has the widest
        # receptive field, so its last position summarises the whole window;
        # using it alone keeps the head small and the ONNX graph static.
        h = self.blocks(x)
        last = h[:, :, -1]
        return self.head(self.norm(last))


# --------------------------------------------------------------------- LSTM / GRU
class LSTMSeq(nn.Module):
    """LSTM or GRU encoder over the time axis with a pooled head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        rnn_cls = nn.LSTM if config.rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(config.hidden_size) if config.use_layer_norm else nn.Identity()
        self.head = ForecasterHead(
            config.hidden_size, config.head_hidden, config.n_outputs, config.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)                     # (B, L, C)
        out, _ = self.rnn(seq)                      # (B, L, H)
        last = out[:, -1, :]                        # summary at the newest bar
        return self.head(self.norm(last))


# ---------------------------------------------------------------- PatchTST-style
class TransformerBlock(nn.Module):
    """Pre-LN encoder block (Vaswani et al. 2017; Xiong et al. 2020 ordering)."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class PatchTSTLite(nn.Module):
    """A small PatchTST: patchify the lookback, embed, encode, pool, predict.

    Patching (Nie et al., 2023) reduces token count by ``patch_len`` and keeps
    local semantics; for ``lookback=64, patch_len=8`` we attend over 8 tokens,
    which is why this runs in microseconds on CPU via ONNX Runtime.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.lookback % config.patch_len != 0:
            raise ValueError("lookback must be divisible by patch_len")
        self.n_patches = config.lookback // config.patch_len
        patch_dim = config.patch_len * config.n_features
        self.embed = nn.Linear(patch_dim, config.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, config.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.ff_mult, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = ForecasterHead(config.d_model, config.head_hidden, config.n_outputs, config.dropout)
        # causal mask for patched attention: patch i may attend to patches <= i
        mask = torch.triu(torch.ones(self.n_patches, self.n_patches, dtype=torch.bool), diagonal=1)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, length = x.shape
        # (B, C, L) -> (B, n_patches, patch_len * C)
        patches = x.reshape(b, c, self.n_patches, self.config.patch_len)
        patches = patches.permute(0, 2, 1, 3).reshape(b, self.n_patches, -1)
        h = self.embed(patches) + self.pos
        for block in self.blocks:
            h = block(h, self.attn_mask)
        h = self.norm(h)
        return self.head(h[:, -1, :])   # last patch = most recent information


# ------------------------------------------------------------------------- factory
def build_sequence_model(arch: str, config: ModelConfig) -> nn.Module:
    arch = arch.lower()
    if arch in ("tcn", "temporal_conv"):
        return TCN(config)
    if arch in ("lstm", "gru"):
        config.rnn_type = arch
        return LSTMSeq(config)
    if arch in ("transformer", "patchtst", "patch_tst", "temporal_transformer"):
        return PatchTSTLite(config)
    raise ValueError(f"unknown arch {arch!r}; expected tcn | lstm | gru | transformer")


class _ExportWrapper(nn.Module):
    """Wraps a model so the exported graph returns probs + value in one shot.

    Exporting *one* graph with both outputs means the live loop performs a single
    ONNX Runtime call per tick instead of two, which halves the Python-side
    overhead that dominates at microsecond inference times.
    """

    def __init__(self, model: nn.Module, task: str) -> None:
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        if self.task == "classification":
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = logits
        # confidence = max prob (classification) or |prediction| (regression)
        if self.task == "classification":
            conf = torch.max(probs, dim=-1, keepdim=True).values
        else:
            conf = torch.abs(probs)
        return probs, conf


def exportable_forecaster(model: nn.Module, config: ModelConfig) -> nn.Module:
    model.eval()
    return _ExportWrapper(model, config.task).eval()
