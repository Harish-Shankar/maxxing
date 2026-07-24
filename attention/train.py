"""
Train the Transformer on a toy task.

    python train.py                              # 3-digit addition, the default
    python train.py --task reverse --steps 1500  # instant gratification
    python train.py --task add --n-digits 4 --steps 8000 --d-model 192

Everything is small enough to run on a CPU. Watch `exact` (whole-sequence
accuracy) rather than the loss: for addition it sits near zero for a while, then
climbs steeply once the model works out carrying.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tasks import Dataset, Split, build_dataset
from torch import Tensor
from transformer import Transformer, TransformerConfig

# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def lr_at_step(step: int, peak_lr: float, warmup: int) -> float:
    """Linear warmup, then inverse-square-root decay.

    Same *shape* as the paper's schedule

        lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

    but parameterised by the peak learning rate instead of d_model, which is far
    easier to tune. (The two agree when peak_lr = d_model^-0.5 * warmup^-0.5.)

    Why warm up at all: Adam's second-moment estimates are garbage for the first
    few hundred steps, and a full-size step on garbage statistics can wreck the
    model. Pre-LN tolerates a much shorter warmup than the paper's post-LN, which
    is why 200-400 steps is plenty here instead of 4000.
    """
    step = max(step, 1)
    return peak_lr * min(step / warmup, math.sqrt(warmup / step))


# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------


def compute_loss(
    model: Transformer,
    src: Tensor,
    tgt_in: Tensor,
    tgt_out: Tensor,
    label_smoothing: float,
) -> Tensor:
    """Cross-entropy over every output position at once, ignoring padding.

    One forward pass produces logits for all Tt positions, each correctly
    conditioned only on earlier positions thanks to the causal mask. This is the
    parallelism the architecture exists to provide.

    Label smoothing (0.1 in the paper) deliberately makes the model less
    confident: it *hurts* perplexity but improves sequence-level accuracy,
    because an overconfident model commits to bad prefixes during decoding.
    """
    logits = model(src, tgt_in)  # (B, Tt, V)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=model.cfg.pad_id,
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def evaluate(
    model: Transformer,
    data: Dataset,
    split: Split,
    device: torch.device,
    max_examples: int = 512,
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    """Teacher-forced loss/token accuracy, plus true autoregressive exact match."""
    model.eval()
    n = min(len(split), max_examples)
    src, tgt_in, tgt_out = (
        split.src[:n].to(device),
        split.tgt_in[:n].to(device),
        split.tgt_out[:n],
    )
    tgt_out = tgt_out.to(device)

    logits = model(src, tgt_in)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=model.cfg.pad_id,
        label_smoothing=label_smoothing,
    )

    real = tgt_out != model.cfg.pad_id
    token_acc = ((logits.argmax(-1) == tgt_out) & real).sum().item() / max(
        real.sum().item(), 1
    )

    # The honest metric: generate from scratch and compare strings.
    generated = model.generate(
        src,
        bos_id=data.vocab.bos_id,
        eos_id=data.vocab.eos_id,
        max_new_tokens=split.tgt_out.size(1) + 2,
    )
    correct = sum(
        data.vocab.decode(generated[i]) == split.pairs[i][1] for i in range(n)
    )

    model.train()
    return {"loss": loss.item(), "token_acc": token_acc, "exact": correct / n}


@torch.no_grad()
def show_samples(
    model: Transformer, data: Dataset, device: torch.device, n: int = 6
) -> None:
    """Print a few validation predictions so you can see what it gets wrong."""
    model.eval()
    src = data.val.src[:n].to(device)
    generated = model.generate(
        src,
        bos_id=data.vocab.bos_id,
        eos_id=data.vocab.eos_id,
        max_new_tokens=data.val.tgt_out.size(1) + 2,
    )
    for i in range(n):
        source, target = data.val.pairs[i]
        prediction = data.vocab.decode(generated[i])
        mark = "ok " if prediction == target else "BAD"
        print(f"    {mark} {source:>10} = {prediction:<8} (want {target})")
    model.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tiny Transformer on a toy task."
    )
    # data
    parser.add_argument(
        "--task", default="add", choices=["add", "reverse", "sort", "copy"]
    )
    parser.add_argument("--n-digits", type=int, default=3)
    parser.add_argument("--n-examples", type=int, default=60_000)
    # architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--peak-lr", type=float, default=2e-3)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # bookkeeping
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="checkpoint.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # -- data ---------------------------------------------------------------
    data = build_dataset(
        task_name=args.task,
        n_examples=args.n_examples,
        n_digits=args.n_digits,
        seed=args.seed + 1234,
    )
    train = data.train.to(device)
    print(f"task={data.task_name} n_digits={data.n_digits} vocab={len(data.vocab)}")
    print(f"train={len(data.train):,} val={len(data.val):,} device={device}")
    print("examples: " + ", ".join(f"{s}={t}" for s, t in data.train.pairs[:4]))

    # -- model --------------------------------------------------------------
    cfg = TransformerConfig(
        vocab_size=len(data.vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=max(data.train.src.size(1), data.train.tgt_in.size(1)) + 8,
        pad_id=data.vocab.pad_id,
    )
    model = Transformer(cfg).to(device)
    print(f"parameters: {model.num_parameters():,}\n")

    # betas/eps from the paper; beta2=0.98 is a little more conservative than the
    # usual 0.999 and helps early stability.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01,
    )

    n_train = train.src.size(0)
    best_exact = -1.0
    start = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        lr = lr_at_step(step, args.peak_lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        indices = torch.randint(0, n_train, (args.batch_size,), device=device)
        src, tgt_in, tgt_out = train.batch(indices)

        loss = compute_loss(model, src, tgt_in, tgt_out, args.label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, data, data.val, device)
            elapsed = time.time() - start
            print(
                f"step {step:>6}/{args.steps}  lr {lr:.2e}  "
                f"train {loss.item():.3f}  val {metrics['loss']:.3f}  "
                f"tok {metrics['token_acc']:.3f}  exact {metrics['exact']:.3f}  "
                f"[{elapsed:.0f}s]"
            )
            if metrics["exact"] >= best_exact:
                best_exact = metrics["exact"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "task": data.task_name,
                        "n_digits": data.n_digits,
                        # itos[3:] are the real characters; the first three are specials.
                        "vocab_characters": "".join(data.vocab.itos[3:]),
                        "step": step,
                        "val_exact": metrics["exact"],
                    },
                    args.out,
                )

    print(f"\nbest exact-match: {best_exact:.3f}   saved to {Path(args.out).resolve()}")
    print("sample predictions:")
    show_samples(model, data, device)
    print(f"\nnow run:  python play.py --checkpoint {args.out}")


if __name__ == "__main__":
    main()
