"""Recognition with a few bits per key on the keyed benchmark: acc9 and pairwise accuracy of checksum memories.

A memory that keeps one checksum per key accepts a candidate whose checksum matches the queried key's. Edits change 1-2
tokens, so the sum mod 16 (4 bits) catches every single change; adding a position-weighted sum (8 bits) catches most double
changes. A sequence stored under another key collides with probability about 2^-bits. Keeping the 8 checksums without their
keys (a set) can't reject other-key negatives at all, since those sequences were stored too. Ties share the win.

  /opt/homebrew/Caskroom/miniconda/base/envs/ai/bin/python reports/checksum_keyed.py [--data data_keyed]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
from generate_keyed import LENGTHS, N_EDIT, subset_name  # noqa: E402
from test_keyed import load  # noqa: E402


def sums(x, words):
    """`words` 4-bit checksums of x [..., L]: the sum mod 16, the sum with odd position weights 2i+1 (the 8-bit checksum of
    the single-sequence benchmark), then sums with fixed random weights. Further words must not be linear combinations of
    the first two (4i+1 = 2(2i+1) - 1 would add nothing)."""
    pos = np.arange(x.shape[-1])
    w = [np.ones_like(pos), (2 * pos + 1) % 16] + [np.random.default_rng(j).integers(1, 16, len(pos)) for j in range(2, words)]
    return np.stack([(x.astype(np.int64) * wi).sum(-1) % 16 for wi in w[:words]], -1)


def scores(seqs, query, cand, words, keyed=True):
    n = len(query)
    true = seqs[np.arange(n), query]
    c9 = np.concatenate([true[:, None], cand], 1)                         # [n, 9, L]
    sig = sums(c9, words)
    if keyed:
        match = (sig == sig[:, :1]).all(-1)
    else:                                                                  # any stored sequence's checksum
        stored = sums(seqs, words)                                         # [n, K, words]
        match = (sig[:, :, None, :] == stored[:, None, :, :]).all(-1).any(-1)
    acc9 = (match[:, 0] / match.sum(1)).mean()
    win = np.where(match[:, 1:], 0.5, 1.0)                                # the true one always matches
    return acc9, win[:, :N_EDIT].mean(), win[:, N_EDIT:].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_keyed")
    p.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    a = p.parse_args()
    print("| L | per key, 4 bits (32 total) | per key, 8 bits (64) | per key, 12 bits (96) | per key, 16 bits (128) | 8 bits, keys not kept (set of 8) |")
    print("|---|---|---|---|---|---|")
    for length in a.lengths:
        x = load(Path(a.data) / subset_name(length), "test", length)
        cells = [scores(x["sequences"], x["query"], x["candidates"], w) for w in (1, 2, 3, 4)]
        cells.append(scores(x["sequences"], x["query"], x["candidates"], 2, keyed=False))
        print(f"| {length} | " + " | ".join(f"{c[0]:.3f} ({c[1]:.3f} / {c[2]:.3f})" for c in cells) + " |", flush=True)
    print("\nacc9 (pairwise on edits / on other keys)")


if __name__ == "__main__":
    main()
