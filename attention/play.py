"""
Poke at a trained model.

    python play.py                          # interactive REPL
    python play.py --once 347+821           # single query and exit
    python play.py --test 500               # accuracy on 500 fresh random examples

Inside the REPL:

    347+821          predict (greedy)
    :t 0.8 347+821   sample at temperature 0.8 instead of greedy
    :a 347+821       cross-attention map: what each output digit looked at
    :a 1 2 347+821   cross-attention for a specific layer and head
    :e 347+821       encoder self-attention map
    :test 200        accuracy on 200 fresh random examples
    :info            model / checkpoint details
    :q               quit
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tasks import TASKS, CharVocab
from torch import Tensor
from transformer import MultiHeadAttention, Transformer, TransformerConfig

SHADES = " .:-=+*#%@"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class Model:
    """A trained checkpoint plus everything needed to talk to it."""

    def __init__(self, path: str | Path, device: torch.device) -> None:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.cfg = TransformerConfig(**checkpoint["config"])
        self.vocab = CharVocab(checkpoint["vocab_characters"])
        self.task_name: str = checkpoint["task"]
        self.n_digits: int = checkpoint["n_digits"]
        self.step: int = checkpoint.get("step", -1)
        self.val_exact: float = checkpoint.get("val_exact", float("nan"))
        self.device = device

        self.net = Transformer(self.cfg).to(device)
        self.net.load_state_dict(checkpoint["model"])
        self.net.eval()

    # -- prediction ---------------------------------------------------------

    def encode_source(self, text: str) -> Tensor:
        ids = self.vocab.encode(text) + [self.vocab.eos_id]
        return torch.tensor([ids], dtype=torch.long, device=self.device)  # (1, Ts)

    @torch.no_grad()
    def predict(
        self, text: str, temperature: float = 0.0, max_new_tokens: int | None = None
    ) -> str:
        src = self.encode_source(text)
        generated = self.net.generate(
            src,
            bos_id=self.vocab.bos_id,
            eos_id=self.vocab.eos_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return self.vocab.decode(generated[0])

    def expected(self, text: str) -> str | None:
        """Ground truth, computed directly, so we can mark answers right or wrong."""
        try:
            if self.task_name == "add":
                left, right = text.split("+")
                return str(int(left) + int(right))
            if self.task_name == "reverse":
                return text[::-1]
            if self.task_name == "sort":
                return "".join(sorted(text))
            if self.task_name == "copy":
                return text
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            return None
        return None

    # -- introspection ------------------------------------------------------

    @torch.no_grad()
    def attention(
        self, text: str
    ) -> tuple[list[str], list[str], list[Tensor], list[Tensor]]:
        """Run the model, then re-run it teacher-forced on its own output so that
        every row of the attention maps is available in one pass.

        Returns (src_tokens, out_tokens, cross_per_layer, encoder_self_per_layer),
        each attention tensor shaped (H, Tq, Tk).
        """
        src = self.encode_source(text)
        prediction = self.predict(text)

        # Decoder input = BOS + the tokens it actually produced.
        tgt_in = torch.tensor(
            [[self.vocab.bos_id] + self.vocab.encode(prediction)],
            dtype=torch.long,
            device=self.device,
        )

        self.net.set_store_attention(True)
        self.net(src, tgt_in)
        self.net.set_store_attention(False)

        def collect(modules: list[MultiHeadAttention]) -> list[Tensor]:
            maps = []
            for module in modules:
                if module.last_attention is None:
                    raise RuntimeError("attention was not captured")
                maps.append(module.last_attention[0])  # drop the batch dim
            return maps

        cross = collect([layer.cross_attn for layer in self.net.decoder.layers])
        enc_self = collect([layer.self_attn for layer in self.net.encoder.layers])

        # Row i of a decoder map is the step that *predicts* output token i, so
        # label rows by the token produced there. The final row predicts EOS.
        src_tokens = list(text) + ["¶"]  # ¶ = the EOS appended to the source
        out_tokens = list(prediction) + ["¶"]
        return src_tokens, out_tokens, cross, enc_self

    def num_layers(self) -> int:
        return self.cfg.n_layers

    def num_heads(self) -> int:
        return self.cfg.n_heads


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_heatmap(
    weights: Tensor,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
) -> str:
    """ASCII heatmap of a (Tq, Tk) attention matrix. Rows are normalised to 1."""
    rows, cols = weights.shape
    lines = [title, "        " + " ".join(f"{label:>2}" for label in col_labels[:cols])]
    for i in range(rows):
        row = weights[i]
        cells = []
        for j in range(cols):
            level = int(row[j].clamp(0, 1).item() * (len(SHADES) - 1) + 0.5)
            cells.append(SHADES[level] * 2)
        top = row.argmax().item()
        lines.append(
            f"  {row_labels[i]:>3} | " + " ".join(cells) + f"   -> {col_labels[top]}"
        )
    lines.append(f"        (shading: '{SHADES[1]}' low .. '{SHADES[-1]}' = 1.0)")
    return "\n".join(lines)


def show_cross_attention(
    model: Model, text: str, layer: int | None = None, head: int | None = None
) -> None:
    src_tokens, out_tokens, cross, _ = model.attention(text)
    index = model.num_layers() - 1 if layer is None else layer
    if not 0 <= index < len(cross):
        print(f"  layer must be in 0..{len(cross) - 1}")
        return

    attn = cross[index]  # (H, Tq, Ts)
    if head is None:
        matrix, label = attn.mean(dim=0), f"decoder layer {index}, mean over heads"
    else:
        if not 0 <= head < attn.size(0):
            print(f"  head must be in 0..{attn.size(0) - 1}")
            return
        matrix, label = attn[head], f"decoder layer {index}, head {head}"

    print()
    print(
        render_heatmap(
            matrix,
            out_tokens,
            src_tokens,
            f"cross-attention ({label})\n  rows = output position, cols = source position",
        )
    )


def show_encoder_attention(model: Model, text: str, layer: int | None = None) -> None:
    src_tokens, _, _, enc_self = model.attention(text)
    index = model.num_layers() - 1 if layer is None else layer
    if not 0 <= index < len(enc_self):
        print(f"  layer must be in 0..{len(enc_self) - 1}")
        return
    print()
    print(
        render_heatmap(
            enc_self[index].mean(dim=0),
            src_tokens,
            src_tokens,
            f"encoder self-attention (layer {index}, mean over heads)",
        )
    )


# ---------------------------------------------------------------------------
# Bulk test
# ---------------------------------------------------------------------------


def run_test(model: Model, n: int, seed: int = 7) -> None:
    """Fresh random examples, generated the same way training data was."""
    task = TASKS[model.task_name]
    rng = random.Random(seed)
    wrong: list[tuple[str, str, str]] = []
    correct = 0

    for _ in range(n):
        source, target = task.sample(rng, model.n_digits)
        prediction = model.predict(source)
        if prediction == target:
            correct += 1
        elif len(wrong) < 10:
            wrong.append((source, prediction, target))

    print(f"  exact match: {correct}/{n} = {correct / n:.1%}")
    if wrong:
        print("  some failures:")
        for source, prediction, target in wrong:
            print(f"    {source:>10} -> {prediction:<8} (want {target})")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def answer(model: Model, text: str, temperature: float = 0.0) -> None:
    prediction = model.predict(text, temperature=temperature)
    truth = model.expected(text)
    if truth is None:
        print(f"  {text} -> {prediction}")
    elif prediction == truth:
        print(f"  {text} -> {prediction}   correct")
    else:
        print(f"  {text} -> {prediction}   WRONG (want {truth})")


def print_info(model: Model) -> None:
    print(f"  task           {model.task_name} ({model.n_digits} digits)")
    print(f"  d_model        {model.cfg.d_model}")
    print(
        f"  layers         {model.cfg.n_layers} encoder + {model.cfg.n_layers} decoder"
    )
    print(f"  heads          {model.cfg.n_heads} x {model.cfg.d_head} dims")
    print(f"  d_ff           {model.cfg.d_ff}")
    print(f"  parameters     {model.net.num_parameters():,}")
    print(f"  vocab          {len(model.vocab)}  {model.vocab.itos}")
    print(f"  saved at step  {model.step}  (val exact {model.val_exact:.3f})")


def repl(model: Model) -> None:
    print(
        f"\nLoaded a '{model.task_name}' model. Type an input, or :help. Ctrl-D to quit.\n"
    )
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt) as e:
            print(f"Error: {e}")
            print()
            return
        if not line:
            continue

        if line in (":q", ":quit", ":exit"):
            return
        if line in (":help", ":h", ":?"):
            print(__doc__)
            continue
        if line == ":info":
            print_info(model)
            continue

        parts = line.split()

        if parts[0] == ":test":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 200
            run_test(model, n)
            continue

        if parts[0] == ":t":  # :t <temperature> <input>
            if len(parts) < 3:
                print("  usage: :t <temperature> <input>")
                continue
            try:
                answer(model, parts[2], temperature=float(parts[1]))
            except ValueError as exc:
                print(f"  {exc}")
            continue

        if parts[0] in (":a", ":e"):  # optional layer / head BEFORE the input
            if len(parts) < 2:
                print(f"  usage: {parts[0]} [layer] [head] <input>")
                continue
            # The input is always the last token; anything before it is indices.
            *prefix, text = parts[1:]
            if not all(p.isdigit() for p in prefix) or len(prefix) > 2:
                print(f"  usage: {parts[0]} [layer] [head] <input>")
                continue
            layer = int(prefix[0]) if prefix else None
            head = int(prefix[1]) if len(prefix) > 1 else None
            try:
                if parts[0] == ":a":
                    show_cross_attention(model, text, layer, head)
                else:
                    show_encoder_attention(model, text, layer)
            except ValueError as exc:
                print(f"  {exc}")
            continue

        if line.startswith(":"):
            print("  unknown command; try :help")
            continue

        try:
            answer(model, line)
        except ValueError as exc:
            print(f"  {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interact with a trained Transformer.")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--once", help="answer a single input and exit")
    parser.add_argument(
        "--test", type=int, help="run accuracy on N fresh examples and exit"
    )
    parser.add_argument("--attn", help="print attention maps for one input and exit")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path.resolve()} -- run train.py first")

    model = Model(path, torch.device(args.device))

    if args.once:
        answer(model, args.once)
        return
    if args.test:
        run_test(model, args.test)
        return
    if args.attn:
        show_cross_attention(model, args.attn)
        show_encoder_attention(model, args.attn)
        return
    repl(model)


if __name__ == "__main__":
    main()
