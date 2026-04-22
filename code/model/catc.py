"""CATC model implementation for PKU-AIGI-500K.

This reconstructs the missing codec described in:
`PKU-AIGI-500K: A Neural Compression Benchmark and Model for
AI-Generated Images` (IEEE JETCAS 2024).

The public repository only shipped the training loop and dataset stub.
The implementation below follows the paper's published architecture:

- CAT blocks in the analysis / synthesis transforms
- CLIP text conditioning through cross-attention
- a ConvGRU-based channel-wise entropy model
- channel attention (CAtten) inside the entropy model

Portions of the state-dict buffer-loading helpers are adapted from the
MIT-licensed LIC_TCM reference implementation:
https://github.com/jmliu206/LIC_TCM
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.models import CompressionModel
from torch import Tensor

SCALES_MIN = 0.11
SCALES_MAX = 256.0
SCALES_LEVELS = 64


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def get_scale_table(
    minimum: float = SCALES_MIN,
    maximum: float = SCALES_MAX,
    levels: int = SCALES_LEVELS,
) -> Tensor:
    return torch.exp(torch.linspace(math.log(minimum), math.log(maximum), levels))


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


def _find_named_buffer(module: nn.Module, query: str) -> Optional[Tensor]:
    return next((buf for name, buf in module.named_buffers() if name == query), None)


def _update_registered_buffer(
    module: nn.Module,
    buffer_name: str,
    state_dict_key: str,
    state_dict: dict[str, Tensor],
    policy: str = "resize_if_empty",
    dtype: torch.dtype = torch.int32,
) -> None:
    new_size = state_dict[state_dict_key].size()
    registered = _find_named_buffer(module, buffer_name)

    if policy in {"resize_if_empty", "resize"}:
        if registered is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')
        if policy == "resize" or registered.numel() == 0:
            registered.resize_(new_size)
        return

    if policy == "register":
        if registered is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')
        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))
        return

    raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module: Optional[nn.Module],
    module_name: str,
    buffer_names: Iterable[str],
    state_dict: dict[str, Tensor],
    policy: str = "resize_if_empty",
    dtype: torch.dtype = torch.int32,
) -> None:
    if module is None:
        return

    valid_names = {name for name, _ in module.named_buffers()}
    for buffer_name in buffer_names:
        if buffer_name not in valid_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy=policy,
            dtype=dtype,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SpatialFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            conv1x1(channels, hidden),
            nn.GELU(),
            conv1x1(hidden, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MixTextFeatures(nn.Module):
    """Approximate the paper's Mix module using pooled image context."""

    def __init__(
        self,
        channels: int,
        text_dim: int,
        num_text_tokens: int,
        mix_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_text_tokens = num_text_tokens
        self.mix_hidden_dim = mix_hidden_dim or min(text_dim, max(32, min(channels, 128)))
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, self.mix_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mix_hidden_dim, self.mix_hidden_dim),
        )
        # A low-rank token generator keeps checkpoint size manageable while still
        # producing token-specific text features for cross-attention.
        self.token_embeddings = nn.Parameter(
            torch.empty(num_text_tokens, self.mix_hidden_dim)
        )
        self.token_proj = nn.Linear(self.mix_hidden_dim, channels)
        self.image_context = nn.Sequential(
            conv3x3(channels, channels),
            nn.GELU(),
            conv3x3(channels, channels),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels, self.mix_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mix_hidden_dim, self.mix_hidden_dim),
        )
        self.gate_embeddings = nn.Parameter(
            torch.empty(num_text_tokens, self.mix_hidden_dim)
        )
        self.gate_proj = nn.Linear(self.mix_hidden_dim, channels)
        self.token_norm = nn.LayerNorm(channels)
        nn.init.normal_(self.token_embeddings, std=0.02)
        nn.init.normal_(self.gate_embeddings, std=0.02)

    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        shared_text = self.text_mlp(text_features)
        local_text = self.token_proj(
            shared_text.unsqueeze(1) + self.token_embeddings.unsqueeze(0)
        )
        image_context = self.image_context(image_features).flatten(1)
        text_context = local_text.mean(dim=1)
        gate_context = self.gate_mlp(image_context * text_context)
        gate = self.gate_proj(
            gate_context.unsqueeze(1) + self.gate_embeddings.unsqueeze(0)
        )
        mixed = torch.sigmoid(gate) * local_text
        return self.token_norm(mixed)


class CrossAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        text_dim: int,
        num_heads: int,
        num_text_tokens: int,
        mix_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.query_proj = conv3x3(channels, channels)
        self.text_adapter = MixTextFeatures(
            channels,
            text_dim,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        batch_size, channels, height, width = image_features.shape
        queries = self.query_proj(image_features).flatten(2).transpose(1, 2)
        text_tokens = self.text_adapter(image_features, text_features)
        attended, _ = self.attn(queries, text_tokens, text_tokens, need_weights=False)
        attended = self.out_proj(attended)
        return attended.transpose(1, 2).reshape(batch_size, channels, height, width)


class CATBlock(nn.Module):
    """Cross-Attention Transformer block from Fig. 10."""

    def __init__(
        self,
        channels: int,
        text_dim: int,
        num_heads: int = 8,
        num_text_tokens: int = 32,
        mix_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pre = conv3x3(channels, channels)
        self.norm1 = LayerNorm2d(channels)
        self.cross_attention = CrossAttention(
            channels=channels,
            text_dim=text_dim,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.norm2 = LayerNorm2d(channels)
        self.mlp = SpatialFeedForward(channels)
        self.rb = ResidualBlock(channels, channels)

    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        x = self.pre(image_features)
        x = x + self.cross_attention(self.norm1(x), text_features)
        x = x + self.mlp(self.norm2(x))
        return self.rb(x)


class CAtten(nn.Module):
    """Channel attention module from Fig. 12."""

    def __init__(
        self,
        channels: int,
        text_dim: int,
        squeeze_channels: int = 192,
        num_heads: int = 8,
        num_text_tokens: int = 32,
        mix_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_proj = conv1x1(channels, squeeze_channels)
        self.local_branch = ResidualBlock(squeeze_channels, squeeze_channels)
        self.cross_branch = CATBlock(
            channels=squeeze_channels,
            text_dim=text_dim,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.post_rb = ResidualBlock(squeeze_channels, squeeze_channels)
        self.gate = nn.Sequential(conv1x1(squeeze_channels, squeeze_channels), nn.Sigmoid())
        self.out_proj = conv1x1(squeeze_channels, channels)

    def forward(self, x: Tensor, text_features: Tensor) -> Tensor:
        x = self.in_proj(x)
        local = self.local_branch(x)
        gate = self.cross_branch(x, text_features)
        gate = self.post_rb(gate)
        fused = x + local * self.gate(gate)
        return self.out_proj(fused)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 2,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        combined = torch.cat([x, hidden], dim=1)
        update_gate, reset_gate = self.gates(combined).chunk(2, dim=1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset_gate * hidden], dim=1))
        )
        return (1.0 - update_gate) * hidden + update_gate * candidate


class ParametersNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv(in_channels, 224, kernel_size=3, stride=1),
            nn.GELU(),
            conv(224, 128, kernel_size=3, stride=1),
            nn.GELU(),
            conv(128, out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LRPNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv(in_channels, 320, kernel_size=3, stride=1),
            nn.GELU(),
            conv(320, 192, kernel_size=3, stride=1),
            nn.GELU(),
            conv(192, out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AnalysisTransform(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        latent_channels: int,
        text_dim: int,
        num_heads: int,
        num_text_tokens: int,
        mix_hidden_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.down1 = ResidualBlockWithStride(3, hidden_channels, stride=2)
        self.cat1 = CATBlock(
            hidden_channels,
            text_dim,
            num_heads,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.down2 = ResidualBlockWithStride(hidden_channels, hidden_channels, stride=2)
        self.cat2 = CATBlock(
            hidden_channels,
            text_dim,
            num_heads,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.down3 = ResidualBlockWithStride(hidden_channels, hidden_channels, stride=2)
        self.cat3 = CATBlock(
            hidden_channels,
            text_dim,
            num_heads,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.out_conv = conv3x3(hidden_channels, latent_channels, stride=2)

    def forward(self, x: Tensor, text_features: Tensor) -> Tensor:
        x = self.cat1(self.down1(x), text_features)
        x = self.cat2(self.down2(x), text_features)
        x = self.cat3(self.down3(x), text_features)
        return self.out_conv(x)


class SynthesisTransform(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        latent_channels: int,
        text_dim: int,
        num_heads: int,
        num_text_tokens: int,
        mix_hidden_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.up1 = ResidualBlockUpsample(latent_channels, hidden_channels, 2)
        self.cat1 = CATBlock(
            hidden_channels,
            text_dim,
            num_heads,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.up2 = ResidualBlockUpsample(hidden_channels, hidden_channels, 2)
        self.cat2 = CATBlock(
            hidden_channels,
            text_dim,
            num_heads,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.up3 = ResidualBlockUpsample(hidden_channels, hidden_channels, 2)
        self.cat3 = CATBlock(
            hidden_channels,
            text_dim,
            num_heads,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.out_conv = subpel_conv3x3(hidden_channels, 3, 2)

    def forward(self, x: Tensor, text_features: Tensor) -> Tensor:
        x = self.cat1(self.up1(x), text_features)
        x = self.cat2(self.up2(x), text_features)
        x = self.cat3(self.up3(x), text_features)
        return self.out_conv(x)


class HyperAnalysis(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hyper_channels: int,
        text_dim: int,
        num_heads: int,
        num_text_tokens: int,
        mix_hidden_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.down = ResidualBlockWithStride(latent_channels, hyper_channels, stride=2)
        self.cat = CATBlock(
            hyper_channels,
            text_dim,
            num_heads,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.out_conv = conv3x3(hyper_channels, hyper_channels, stride=2)

    def forward(self, x: Tensor, text_features: Tensor) -> Tensor:
        x = self.down(x)
        x = self.cat(x, text_features)
        return self.out_conv(x)


class HyperSynthesis(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hyper_channels: int,
        text_dim: int,
        num_heads: int,
        num_text_tokens: int,
        mix_hidden_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.up = ResidualBlockUpsample(hyper_channels, hyper_channels, 2)
        self.cat = CATBlock(
            hyper_channels,
            text_dim,
            num_heads,
            num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.out_conv = subpel_conv3x3(hyper_channels, latent_channels * 2, 2)

    def forward(self, x: Tensor, text_features: Tensor) -> tuple[Tensor, Tensor]:
        x = self.up(x)
        x = self.cat(x, text_features)
        mean_scale = self.out_conv(x)
        return mean_scale.chunk(2, dim=1)


class CATC(CompressionModel):
    def __init__(
        self,
        hidden_channels: int = 128,
        latent_channels: int = 320,
        hyper_channels: int = 192,
        text_dim: int = 512,
        num_slices: int = 5,
        num_heads: int = 8,
        num_text_tokens: int = 32,
        squeeze_channels: int = 192,
        mix_hidden_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(entropy_bottleneck_channels=hyper_channels, **kwargs)
        if latent_channels % num_slices != 0:
            raise ValueError("latent_channels must be divisible by num_slices")

        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels
        self.text_dim = text_dim
        self.num_slices = num_slices
        self.slice_channels = latent_channels // num_slices
        self.mix_hidden_dim = mix_hidden_dim

        self.g_a = AnalysisTransform(
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            text_dim=text_dim,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.g_s = SynthesisTransform(
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            text_dim=text_dim,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.h_a = HyperAnalysis(
            latent_channels=latent_channels,
            hyper_channels=hyper_channels,
            text_dim=text_dim,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.h_s = HyperSynthesis(
            latent_channels=latent_channels,
            hyper_channels=hyper_channels,
            text_dim=text_dim,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )

        self.mean_gru = ConvGRUCell(self.slice_channels, latent_channels)
        self.scale_gru = ConvGRUCell(self.slice_channels, latent_channels)
        self.mean_catten = CAtten(
            latent_channels,
            text_dim,
            squeeze_channels=squeeze_channels,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.scale_catten = CAtten(
            latent_channels,
            text_dim,
            squeeze_channels=squeeze_channels,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            mix_hidden_dim=mix_hidden_dim,
        )
        self.mean_parameters = ParametersNet(latent_channels, self.slice_channels)
        self.scale_parameters = ParametersNet(latent_channels, self.slice_channels)
        self.lrp = LRPNet(latent_channels * 2 + self.slice_channels, self.slice_channels)

        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.gaussian_conditional = GaussianConditional(None)

    def _prepare_text(self, text_features: Tensor) -> Tensor:
        if text_features.dim() == 3:
            text_features = text_features.mean(dim=1)
        if text_features.dim() != 2:
            raise ValueError(
                "text_features must be shaped [B, D] or [B, T, D], "
                f"got {tuple(text_features.shape)}"
            )
        return text_features.float()

    def _crop_like(self, x: Tensor, reference: Tensor) -> Tensor:
        return x[:, :, : reference.size(2), : reference.size(3)]

    def _estimate_slice_parameters(
        self,
        mean_hidden: Tensor,
        scale_hidden: Tensor,
        text_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean_context = self.mean_catten(mean_hidden, text_features)
        scale_context = self.scale_catten(scale_hidden, text_features)
        mu = self.mean_parameters(mean_context)
        scale = F.softplus(self.scale_parameters(scale_context)) + 1e-6
        return mu, scale, mean_context, scale_context

    def _apply_lrp(
        self,
        mean_context: Tensor,
        scale_context: Tensor,
        y_hat_slice: Tensor,
    ) -> Tensor:
        lrp_input = torch.cat([mean_context, scale_context, y_hat_slice], dim=1)
        residual = 0.5 * torch.tanh(self.lrp(lrp_input))
        return y_hat_slice + residual

    def update(self, scale_table: Optional[Tensor] = None, force: bool = False) -> bool:
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x: Tensor, text_features: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        text_features = self._prepare_text(text_features)

        y = self.g_a(x, text_features)
        z = self.h_a(y, text_features)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        mean_hidden, scale_hidden = self.h_s(z_hat, text_features)
        mean_hidden = self._crop_like(mean_hidden, y)
        scale_hidden = self._crop_like(scale_hidden, y)

        y_hat_slices = []
        y_likelihoods = []
        prev_slice = None

        for y_slice in y.chunk(self.num_slices, dim=1):
            if prev_slice is not None:
                mean_hidden = self.mean_gru(prev_slice, mean_hidden)
                scale_hidden = self.scale_gru(prev_slice, scale_hidden)

            mu, scale, mean_context, scale_context = self._estimate_slice_parameters(
                mean_hidden,
                scale_hidden,
                text_features,
            )
            mu = self._crop_like(mu, y_slice)
            scale = self._crop_like(scale, y_slice)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_hat_slice = ste_round(y_slice - mu) + mu
            y_hat_slice = self._apply_lrp(mean_context, scale_context, y_hat_slice)

            y_hat_slices.append(y_hat_slice)
            y_likelihoods.append(y_slice_likelihood)
            prev_slice = y_hat_slice

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat, text_features).clamp_(0, 1)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": torch.cat(y_likelihoods, dim=1),
                "z": z_likelihoods,
            },
        }

    def compress(self, x: Tensor, text_features: Tensor) -> dict[str, list[bytes] | tuple[int, int]]:
        text_features = self._prepare_text(text_features)

        y = self.g_a(x, text_features)
        z = self.h_a(y, text_features)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        mean_hidden, scale_hidden = self.h_s(z_hat, text_features)
        mean_hidden = self._crop_like(mean_hidden, y)
        scale_hidden = self._crop_like(scale_hidden, y)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list: list[int] = []
        indexes_list: list[int] = []
        prev_slice = None

        for y_slice in y.chunk(self.num_slices, dim=1):
            if prev_slice is not None:
                mean_hidden = self.mean_gru(prev_slice, mean_hidden)
                scale_hidden = self.scale_gru(prev_slice, scale_hidden)

            mu, scale, mean_context, scale_context = self._estimate_slice_parameters(
                mean_hidden,
                scale_hidden,
                text_features,
            )
            mu = self._crop_like(mu, y_slice)
            scale = self._crop_like(scale, y_slice)

            indexes = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = self.gaussian_conditional.dequantize(y_q_slice, mu)
            y_hat_slice = self._apply_lrp(mean_context, scale_context, y_hat_slice)

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(indexes.reshape(-1).tolist())
            prev_slice = y_hat_slice

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()

        return {
            "strings": [[y_string], z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(
        self,
        strings: list[list[bytes] | list[Tensor]],
        text_features: Tensor,
        shape: tuple[int, int],
    ) -> dict[str, Tensor]:
        text_features = self._prepare_text(text_features)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        mean_hidden, scale_hidden = self.h_s(z_hat, text_features)

        y_height, y_width = mean_hidden.size(-2), mean_hidden.size(-1)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])

        y_hat_slices = []
        prev_slice = None
        device = z_hat.device

        for _ in range(self.num_slices):
            if prev_slice is not None:
                mean_hidden = self.mean_gru(prev_slice, mean_hidden)
                scale_hidden = self.scale_gru(prev_slice, scale_hidden)

            mu, scale, mean_context, scale_context = self._estimate_slice_parameters(
                mean_hidden,
                scale_hidden,
                text_features,
            )
            indexes = self.gaussian_conditional.build_indexes(scale)
            rv = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.tensor(rv, device=device).reshape(1, self.slice_channels, y_height, y_width)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)
            y_hat_slice = self._apply_lrp(mean_context, scale_context, y_hat_slice)

            y_hat_slices.append(y_hat_slice)
            prev_slice = y_hat_slice

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat, text_features).clamp_(0, 1)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True):  # type: ignore[override]
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)


__all__ = ["CATC", "get_scale_table"]
