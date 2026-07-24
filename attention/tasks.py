"""
Tiny synthetic sequence-to-sequence tasks. No downloads, no preprocessing.

Everything is character-level, so the vocabulary is ~14 tokens and the whole
dataset fits in a couple of tensors.

    add      "347+821" -> "1168"     (needs real computation; the fun one)
    reverse  "3491"    -> "1943"     (converges almost immediately)
    sort     "3491"    -> "1349"
    copy     "3491"    -> "3491"     (sanity check: if this fails, the code is broken)

Start with `reverse` to confirm the plumbing works end to end, then move to `add`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class CharVocab:
    """Character-level vocabulary. PAD is id 0 so it doubles as the padding value."""

    def __init__(self, characters: str) -> None:
        self.itos: list[str] = [PAD, BOS, EOS] + sorted(set(characters))
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(
                f"character {exc.args[0]!r} is not in the vocabulary"
            ) from exc

    def decode(self, ids: list[int] | Tensor) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        specials = {self.pad_id, self.bos_id, self.eos_id}
        return "".join(self.itos[i] for i in ids if i not in specials)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    name: str
    characters: str
    sample: Callable[[random.Random, int], tuple[str, str]]


def _sample_add(rng: random.Random, n_digits: int) -> tuple[str, str]:
    limit = 10**n_digits - 1
    a, b = rng.randint(0, limit), rng.randint(0, limit)
    return f"{a}+{b}", str(a + b)


def _random_digits(rng: random.Random, n_digits: int) -> str:
    length = rng.randint(2, max(2, n_digits))
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


def _sample_reverse(rng: random.Random, n_digits: int) -> tuple[str, str]:
    digits = _random_digits(rng, n_digits)
    return digits, digits[::-1]


def _sample_sort(rng: random.Random, n_digits: int) -> tuple[str, str]:
    digits = _random_digits(rng, n_digits)
    return digits, "".join(sorted(digits))


def _sample_copy(rng: random.Random, n_digits: int) -> tuple[str, str]:
    digits = _random_digits(rng, n_digits)
    return digits, digits


TASKS: dict[str, Task] = {
    "add": Task("add", "0123456789+", _sample_add),
    "reverse": Task("reverse", "0123456789", _sample_reverse),
    "sort": Task("sort", "0123456789", _sample_sort),
    "copy": Task("copy", "0123456789", _sample_copy),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Split:
    """Pre-tokenised, pre-padded tensors for one split.

    src     (N, Ts)  source ids + EOS, padded
    tgt_in  (N, Tt)  BOS + target ids, padded      <- decoder input (shifted right)
    tgt_out (N, Tt)  target ids + EOS, padded      <- what the loss compares against
    """

    src: Tensor
    tgt_in: Tensor
    tgt_out: Tensor
    pairs: list[tuple[str, str]]

    def __len__(self) -> int:
        return self.src.size(0)

    def batch(self, indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.src[indices], self.tgt_in[indices], self.tgt_out[indices]

    def to(self, device: torch.device) -> Split:
        return Split(
            self.src.to(device),
            self.tgt_in.to(device),
            self.tgt_out.to(device),
            self.pairs,
        )


@dataclass
class Dataset:
    task_name: str
    n_digits: int
    vocab: CharVocab
    train: Split
    val: Split


def _encode_pairs(pairs: list[tuple[str, str]], vocab: CharVocab) -> Split:
    src_seqs = [vocab.encode(s) + [vocab.eos_id] for s, _ in pairs]
    in_seqs = [[vocab.bos_id] + vocab.encode(t) for _, t in pairs]
    out_seqs = [vocab.encode(t) + [vocab.eos_id] for _, t in pairs]

    def pad(seqs: list[list[int]]) -> Tensor:
        width = max(len(s) for s in seqs)
        tensor = torch.full((len(seqs), width), vocab.pad_id, dtype=torch.long)
        for row, seq in enumerate(seqs):
            tensor[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return tensor

    return Split(pad(src_seqs), pad(in_seqs), pad(out_seqs), pairs)


def build_dataset(
    task_name: str = "add",
    n_examples: int = 60_000,
    n_digits: int = 3,
    val_fraction: float = 0.1,
    seed: int = 1234,
) -> Dataset:
    """Generate the dataset and split it. Deduplicated, so val examples are unseen."""
    if task_name not in TASKS:
        raise ValueError(f"unknown task {task_name!r}; choose from {sorted(TASKS)}")
    task = TASKS[task_name]
    rng = random.Random(seed)

    # Sample unique examples. Give up after a generous number of attempts in case
    # the task's space is smaller than n_examples (e.g. 2-digit addition).
    seen: dict[str, str] = {}
    attempts = 0
    max_attempts = n_examples * 50
    while len(seen) < n_examples and attempts < max_attempts:
        source, target = task.sample(rng, n_digits)
        seen.setdefault(source, target)
        attempts += 1

    pairs = list(seen.items())
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction))
    vocab = CharVocab(task.characters)

    return Dataset(
        task_name=task_name,
        n_digits=n_digits,
        vocab=vocab,
        train=_encode_pairs(pairs[n_val:], vocab),
        val=_encode_pairs(pairs[:n_val], vocab),
    )


if __name__ == "__main__":
    data = build_dataset("add", n_examples=2000, n_digits=3)
    print(f"task={data.task_name}  vocab={len(data.vocab)}  {data.vocab.itos}")
    print(f"train={len(data.train)}  val={len(data.val)}")
    print(
        f"src shape {tuple(data.train.src.shape)}  tgt shape {tuple(data.train.tgt_in.shape)}"
    )
    for source, target in data.train.pairs[:5]:
        print(f"  {source:>9}  ->  {target}")
    # Round-trip check.
    row = data.train.src[0]
    print("decoded src[0]:", data.vocab.decode(row))
