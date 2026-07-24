from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TransformerConfig:
    """Everything that defines the architecture.

    The paper's base model is d_model=512, n_layers=6, n_heads=8, d_ff=2048.
    The defaults here are ~100x smaller so it trains on a laptop CPU in minutes.
    """

    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3  # applies to BOTH encoder and decoder stacks
    d_ff: int = 512  # paper uses 4 * d_model
    dropout: float = 0.1
    max_len: int = 128  # longest sequence the positional encoding supports
    tie_embeddings: bool = True  # share src emb / tgt emb / output projection
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_model % 2 != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be even for sinusoidal encoding"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class SinusoidalPositionalEncoding(nn.Module):
    """
    Add fixed sinusoids to the embeddings so the model can tell positions apart.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(
            1
        )  # (max_len, 1)

        inv_freq = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * inv_freq)
        pe[:, 1::2] = torch.cos(position * inv_freq)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, D)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.pe.size(1)}"
            )
        # Dropout on the sum of embeddings and positional encodings
        return self.dropout(x + self.pe[:, :seq_len, :])


class MultiHeadAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_head)) V, run in H parallel heads.

    encoder self-attn:  q = k = v = encoder hidden states
    decoder self-attn:  q = k = v = decoder hidden states, + causal mask
    cross-attn:         q = decoder states, k = v = encoder output
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # W_O mixes the concatenated heads back into the residual-stream basis.
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Set store_attention=True to keep the last attention map for inspection.
        self.store_attention: bool = False
        self.last_attention: Tensor | None = None  # (B, H, Tq, Tk)

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, D) -> (B, H, T, Dh)"""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, H, T, Dh) -> (B, T, D)"""
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """query: (B, Tq, D); key/value: (B, Tk, D); mask: broadcastable to (B, H, Tq, Tk).

        Returns (B, Tq, D).
        """
        q = self._split_heads(self.w_q(query))  # (B, H, Tq, Dh)
        k = self._split_heads(self.w_k(key))  # (B, H, Tk, Dh)
        v = self._split_heads(self.w_v(value))  # (B, H, Tk, Dh)

        # Raw compatibility scores
        # (B, H, Tq, Dh) @ (B, H, Dh, Tk) -> (B, H, Tq, Tk)
        # Entry [b, h, i, j] = how much position i wants to read from position j
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Scale by 1/sqrt(d_head)
        scores = scores * self.scale

        # Mask illegal positions, then softmax over keys
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        if self.store_attention:
            self.last_attention = weights.detach()

        # Weighted sum of values
        out = torch.matmul(self.attn_dropout(weights), v)  # (B, H, Tq, Dh)
        return self.w_o(self._merge_heads(out))


class FeedForward(nn.Module):
    """FFN(x) = W2 * ReLU(W1 * x + b1) + b2, applied to each position independently.

    d_ff is 4x d_model, so this holds ~2/3 of each layer's parameters
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_in = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """(B, T, D) -> (B, T, D)"""
        return self.linear_out(self.dropout(F.relu(self.linear_in(x))))


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention, then FFN, each in a pre-LN residual block"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm_ff = nn.LayerNorm(cfg.d_model)
        self.feed_forward = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """x: (B, Ts, D) -> (B, Ts, D)"""
        h = self.norm_attn(x)
        x = x + self.dropout(self.self_attn(h, h, h, src_mask))

        h = self.norm_ff(x)
        x = x + self.dropout(self.feed_forward(h))
        return x


class DecoderLayer(nn.Module):
    """One decoder layer: masked self-attention, cross-attention, FFN"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.norm_self_attn = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)

        self.norm_cross_attn = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)

        self.norm_ff = nn.LayerNorm(cfg.d_model)
        self.feed_forward = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """x: (B, Tt, D); memory: (B, Ts, D) -> (B, Tt, D)"""
        # Masked self-attention: position i may only look at positions <= i
        h = self.norm_self_attn(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))

        # Cross-attention: Q from the decoder, K and V from the encoder
        h = self.norm_cross_attn(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, memory_mask))

        # Per-position computation
        h = self.norm_ff(x)
        x = x + self.dropout(self.feed_forward(h))
        return x


class Encoder(nn.Module):
    """N identical encoder layers, plus the final norm that pre-LN requires"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """N identical decoder layers, plus the final norm that pre-LN requires"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.src_embed = nn.Embedding(
            cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id
        )
        if cfg.tie_embeddings:
            # Shared source/target vocabulary
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(
                cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id
            )

        self.pos_encoding = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_len, cfg.dropout
        )
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.output_proj.weight = self.tgt_embed.weight

        self.embed_scale = math.sqrt(cfg.d_model)
        self._init_parameters()

    def _init_parameters(self) -> None:
        embed_tensors = {id(self.src_embed.weight), id(self.tgt_embed.weight)}

        for param in self.parameters():
            if param.dim() > 1 and id(param) not in embed_tensors:
                nn.init.xavier_uniform_(param)

        for embedding in {
            id(self.src_embed): self.src_embed,
            id(self.tgt_embed): self.tgt_embed,
        }.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=self.cfg.d_model**-0.5)
            with torch.no_grad():
                embedding.weight[self.cfg.pad_id].fill_(0.0)

    # -- masks --------------------------------------------------------------

    def pad_mask(self, tokens: Tensor) -> Tensor:
        """(B, Tk) token ids -> (B, 1, 1, Tk) bool"""
        return (tokens != self.cfg.pad_id).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """(1, 1, T, T) lower-triangular bool: position i may attend to j <= i"""
        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()[
            None, None
        ]

    # -- forward ------------------------------------------------------------

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """src: (B, Ts) token ids -> memory (B, Ts, D)"""
        if src_mask is None:
            src_mask = self.pad_mask(src)
        x = self.pos_encoding(self.src_embed(src) * self.embed_scale)
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt_in: Tensor,
        memory: Tensor,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """tgt_in: (B, Tt) token ids -> logits (B, Tt, vocab_size)"""
        tgt_mask = self.causal_mask(tgt_in.size(1), tgt_in.device)
        x = self.pos_encoding(self.tgt_embed(tgt_in) * self.embed_scale)
        h = self.decoder(x, memory, tgt_mask, memory_mask)
        return self.output_proj(h)

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        """src: (B, Ts), tgt_in: (B, Tt) -> logits (B, Tt, vocab_size)"""
        src_mask = self.pad_mask(src)
        memory = self.encode(src, src_mask)
        return self.decode(tgt_in, memory, src_mask)

    # INFERENCE

    @torch.no_grad()
    def generate(
        self,
        src: Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Tensor:
        """Autoregressive decoding. src: (B, Ts) -> generated ids (B, <=max_new_tokens).

        temperature = 0.0 -> greedy (argmax). > 0.0 -> sample from the softmax.
        """
        self.eval()
        budget = self.cfg.max_len - 1
        max_new_tokens = (
            budget if max_new_tokens is None else min(max_new_tokens, budget)
        )
        batch = src.size(0)
        device = src.device

        src_mask = self.pad_mask(src)
        memory = self.encode(src, src_mask)  # encoder runs ONCE

        tokens = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decode(tokens, memory, src_mask)  # (B, t, V)
            next_logits = logits[:, -1, :]  # only the last position matters

            if temperature <= 0.0:
                next_token = next_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Once a row has emitted EOS, keep it padded forever.
            next_token = torch.where(
                finished, torch.full_like(next_token, self.cfg.pad_id), next_token
            )
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if bool(finished.all()):
                break

        return tokens[:, 1:]  # drop the BOS we seeded with

    # INTROSPECTION

    def set_store_attention(self, enabled: bool) -> None:
        """Toggle keeping of attention maps on every attention module."""
        for module in self.modules():
            if isinstance(module, MultiHeadAttention):
                module.store_attention = enabled
                if not enabled:
                    module.last_attention = None

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        # Tied weights appear more than once in .parameters(); dedupe by identity.
        seen: dict[int, int] = {}
        for p in params:
            seen[id(p)] = p.numel()
        return sum(seen.values())


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=16, d_model=64, n_heads=4, n_layers=2, d_ff=128)
    model = Transformer(cfg)
    print(f"parameters: {model.num_parameters():,}")

    src = torch.randint(1, 16, (4, 9))
    src[0, 7:] = cfg.pad_id  # give row 0 some padding
    tgt_in = torch.randint(1, 16, (4, 6))

    logits = model(src, tgt_in)
    print("logits:", tuple(logits.shape))
    assert logits.shape == (4, 6, 16)
    assert torch.isfinite(logits).all(), "NaN/inf in logits -- check the masks"

    out = model.generate(src, bos_id=1, eos_id=2, max_new_tokens=5)
    print("generated:", tuple(out.shape))

    # Causality check: changing a future target token must not change an earlier logit.
    model.eval()
    with torch.no_grad():
        a = model(src, tgt_in)
        perturbed = tgt_in.clone()
        perturbed[:, -1] = (perturbed[:, -1] + 1) % 16
        b = model(src, perturbed)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6), "causal mask is leaking!"
    print("causal mask OK")
