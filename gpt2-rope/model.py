import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn  # noqa: PLR0402
from torch import Tensor
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    rope_theta: float = 10000.0  # RoPE base frequency


# Replace Everywhere with
# layer_norm = nn.LayerNorm(
#     normalized_shape=ndim,
#     elementwise_affine=True,
#     bias=bias,
# )
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def precompute_rope_cache(head_dim: int, seq_len: int, theta: float):
    """cos/sin tables of shape (seq_len, head_dim // 2) for rotary embeddings."""
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"
    # theta_i = 1 / (base ** (2i / d)) for i in [0, d/2)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (T, hd/2)

    return freqs.cos(), freqs.sin()


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate (B, nh, T, hd) queries/keys by the position-dependent angles in cos/sin.

    Pairs dimension i with dimension i + hd/2 (the GPT-NeoX / LLaMA layout), so each
    2D slice is rotated by m * theta_i where m is the absolute position of the token.
    Attention scores then depend only on the relative offset between query and key.
    """
    x1, x2 = x.float().chunk(2, dim=-1)  # each (B, nh, T, hd/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    return rotated.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        Q, K, V = self.c_attn(x).split(self.n_embd, dim=2)
        K = K.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        V = V.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # positional information enters here, not at the embedding layer
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        y = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MultiLayerPerceptron(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        # x = self.c_fc(x)
        # x = self.gelu(x)
        # x = self.c_proj(x)
        # x = self.dropout(x)

        # return x
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(
            cfg.n_embd, eps=1e-5, elementwise_affine=True, bias=cfg.bias
        )
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(
            cfg.n_embd, eps=1e-5, elementwise_affine=True, bias=cfg.bias
        )
        self.mlp = MultiLayerPerceptron(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.block_size is not None

        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            {
                # no "wpe": positions are injected by RoPE inside each attention block
                "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
                "drop": nn.Dropout(cfg.dropout),
                "h": nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                "ln_f": nn.LayerNorm(
                    cfg.n_embd, eps=1e-5, elementwise_affine=True, bias=cfg.bias
                ),
            }
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

        cos, sin = precompute_rope_cache(
            cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta
        )
        # non-persistent: derived from the config, so it stays out of the checkpoint
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        # non_embedding kept for API compatibility; with RoPE there are no positional
        # embedding parameters to subtract, and wte is tied to lm_head
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, t = idx.size()
        assert t <= self.cfg.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.cfg.block_size}"
        )
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]

        x = self.transformer.drop(self.transformer.wte(idx))

        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.cfg.block_size
        self.cfg.block_size = block_size
        self.rope_cos = self.rope_cos[:block_size]
        self.rope_sin = self.rope_sin[:block_size]
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},  # 124M params
            "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # 350M params
            "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},  # 774M params
            "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        # this model uses RoPE, so GPT-2's learned absolute position table has no home
        print("dropping pretrained transformer.wpe.weight (model uses RoPE)")
        sd_keys_hf = [k for k in sd_keys_hf if k != "transformer.wpe.weight"]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.cfg.block_size
                else idx[:, -self.cfg.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
