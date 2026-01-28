# -------------------------------------------------------------------------
# Adapted from
# https://github.com/apple/ml-tarflow/blob/main/transformer_flow.py
# Licensed under Apple License
# -------------------------------------------------------------------------
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Modifications Copyright (c) 2025 transferable-samplers contributors
# Licensed under the MIT License (see LICENSE in the repository root).
# -------------------------------------------------------------------------

"""Shared model components: Transformer blocks, TarFlow, and utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_bert_mask(
    x_onehot: torch.Tensor, mask_ratio: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT-style masking by zeroing out random positions."""
    B, T, V = x_onehot.shape
    mask = torch.rand(B, T, device=x_onehot.device) < mask_ratio
    x_masked = x_onehot.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


class MLMHead(nn.Module):
    """Simple MLM prediction head: LayerNorm -> Linear."""
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x))


# =============================================================================
# BERT-style Transformer Components (for TextEncoder and Generator)
# =============================================================================

class Attention(nn.Module):
    """Multi-head self-attention with SDPA."""
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.proj(x.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, dim: int, n_heads: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class TextEncoder(nn.Module):
    """BERT-style bidirectional transformer encoder."""
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.blocks = nn.ModuleList([
            Block(hidden_dim, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.input_proj(x) + self.pos_embed
        hiddens = [] if return_intermediates else None
        for block in self.blocks:
            x = block(x)
            if return_intermediates:
                hiddens.append(x)
        out = self.ln_out(x)
        if return_intermediates:
            return out, hiddens
        return out


# =============================================================================
# TarFlow Components (for Normalizing Flow)
# =============================================================================

class Permutation(nn.Module):
    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError("Overload me")


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class FlowAttention(nn.Module):
    """Attention for TarFlow with KV caching support."""
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int, dropout: float = 0.0):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}
        self.v_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}

    def forward_spda(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
        temp: float = 1.0, which_cache: str = "cond",
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)
            v = torch.cat(self.v_cache[which_cache], dim=2)
        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.dropout(self.proj(x))
        return x

    def forward_base(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
        temp: float = 1.0, which_cache: str = "cond",
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)
        attn = torch.einsum("bmhd,bnhd->bmnh", q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum("bmnh,bnhd->bmhd", attn, v)
        x = x.reshape(B, T, C)
        x = self.dropout(self.proj(x))
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
        temp: float = 1.0, which_cache: str = "cond",
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class FlowMLP(nn.Module):
    def __init__(self, channels: int, expansion: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.main = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attention = FlowAttention(channels, head_channels, dropout=dropout)
        self.mlp = FlowMLP(channels, expansion, dropout=dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0, which_cache: str = "cond",
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


class MetaBlock(nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self, in_channels: int, channels: int, num_patches: int,
        permutation: Permutation, num_layers: int = 1, head_dim: int = 64,
        expansion: int = 4, nvp: bool = True, num_classes: int = 0, dropout: float = 0.0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, channels)
        self.pos_embed = nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        if num_classes:
            self.class_embed = nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2)
        else:
            self.class_embed = None
        self.attn_blocks = nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion, dropout=dropout) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer("attn_mask", torch.tril(torch.ones(num_patches, num_patches)))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None, return_affine: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        x = self.proj_in(x) + pos_embed
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        xa_clamped = xa.float().clamp(-10, 10)
        scale = (-xa_clamped).exp().type(xa.dtype)
        output = self.permutation((x_in - xb) * scale, inverse=True)
        logdet = -xa_clamped.mean(dim=[1, 2])
        if return_affine:
            return output, logdet, xa, xb
        return output, logdet

    def reverse_step(
        self, x: torch.Tensor, pos_embed: torch.Tensor, i: int,
        y: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = "cond",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]
        x = self.proj_in(x_in) + pos_embed[i : i + 1]
        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)
        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)
        x = self.proj_out(x)
        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, FlowAttention):
                m.sample = flag
                m.k_cache = {"cond": [], "uncond": []}
                m.v_cache = {"cond": [], "uncond": []}

    def reverse(
        self, x: torch.Tensor, y: torch.Tensor | None = None, guidance: float = 0,
        guide_what: str = "ab", attn_temp: float = 1.0, annealed_guidance: bool = False,
        return_logdets: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1)
        C = x.size(2)
        xs = [x[:, i] for i in range(x.size(1))]
        logdet_accum = torch.zeros(x.size(0), device=x.device)
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache="cond")
            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(x, pos_embed, i, None, attn_temp=attn_temp, which_cache="uncond")
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                if "a" in guide_what:
                    za = za + g * (za - za_u)
                if "b" in guide_what:
                    zb = zb + g * (zb - zb_u)
            za_clamped = za[:, 0].float().clamp(-10, 10)
            scale = za_clamped.exp().type(za.dtype)
            logdet_accum = logdet_accum + za_clamped.sum(dim=-1)
            xs[i + 1] = xs[i + 1] * scale + zb[:, 0]
            x = torch.stack(xs, dim=1)
        self.set_sample_mode(False)
        x_out = self.permutation(x, inverse=True)
        if return_logdets:
            logdet = logdet_accum / (T * C)
            return x_out, logdet
        return x_out


class TarFlow(nn.Module):
    def __init__(
        self, in_channels: int, img_size: int, patch_size: int, channels: int,
        num_blocks: int, layers_per_block: int, nvp: bool = True,
        num_classes: int = 0, dropout: float = 0.0, *args, **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size // patch_size // in_channels
        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels * patch_size, channels, self.num_patches,
                    permutations[i % 2], layers_per_block, nvp=nvp,
                    num_classes=num_classes, dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.in_channels = in_channels
        if self.in_channels != 1:
            assert not self.img_size % self.in_channels

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None,
        return_intermediates: bool = False, return_affine: bool = False, *args, **kwargs,
    ):
        original_channels = x.shape[-1]
        if self.in_channels != original_channels:
            x = x.reshape(x.shape[0], -1, self.in_channels)
        logdets = torch.zeros((), device=x.device)
        intermediates = [] if return_intermediates else None
        xa_list = [] if return_affine else None
        xb_list = [] if return_affine else None
        for block in self.blocks:
            if return_affine:
                x, logdet, xa, xb = block(x, y, return_affine=True)
                xa_list.append(xa)
                xb_list.append(xb)
            else:
                x, logdet = block(x, y)
            logdets = logdets + logdet
            if return_intermediates:
                intermediates.append(x.clone())
        if self.in_channels != original_channels:
            x = x.reshape(x.shape[0], -1, original_channels)
        if return_affine:
            return x, logdets, intermediates, xa_list, xb_list
        if return_intermediates:
            return x, logdets, intermediates
        return x, logdets

    def reverse(
        self, x: torch.Tensor, y: torch.Tensor | None = None, guidance: float = 0,
        guide_what: str = "ab", attn_temp: float = 1.0, annealed_guidance: bool = False,
        return_sequence: bool = False, return_logdets: bool = False, *args, **kwargs,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        seq = [x.detach().clone()]
        original_channels = x.shape[-1]
        if self.in_channels != original_channels:
            x = x.reshape(x.shape[0], -1, self.in_channels)
        logdets = torch.zeros((), device=x.device)
        for block in reversed(self.blocks):
            if return_logdets:
                x, logdet = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance, return_logdets=True)
                logdets = logdets + logdet
            else:
                x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance)
            seq.append(x.detach().clone())
        if self.in_channels != original_channels:
            x = x.reshape(x.shape[0], -1, original_channels)
        if return_logdets:
            return x, logdets
        if not return_sequence:
            return x
        return seq
