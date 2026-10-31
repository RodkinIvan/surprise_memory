"""Validate the generated benchmark files.

Run: python benchmark/test_dataset.py --data data
Checks structure, invariants and the corruption statistics of every subset/split.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from generate import CHUNK, LENGTHS, NUM_CORRUPTED, SPLITS, VOCAB, chunk_rng, generate_chunk, subset_name

P_MIN = 1e-6  # many tests run; keep the false-failure rate negligible


def load_split(d, split):
    files = sorted(d.glob(f"{split}-*.parquet"))
    assert files, f"no files for {d.name}/{split}"
    nshards = int(files[0].stem.split("-of-")[1])
    assert len(files) == nshards, f"{d.name}/{split}: {len(files)} files, expected {nshards}"
    seqs, cors, ncs = [], [], []
    for f in files:
        t = pq.read_table(f)
        assert t.column_names == ["sequence", "corrupted", "num_changed"], t.column_names
        n = t.num_rows
        seqs.append(t["sequence"].combine_chunks().flatten().to_numpy().reshape(n, -1))
        cors.append(t["corrupted"].combine_chunks().flatten().flatten().to_numpy().reshape(n, NUM_CORRUPTED, -1))
        ncs.append(t["num_changed"].combine_chunks().flatten().to_numpy().reshape(n, NUM_CORRUPTED))
    return np.concatenate(seqs), np.concatenate(cors), np.concatenate(ncs)


def chisq(observed, expected, what):
    observed, expected = np.asarray(observed, float), np.asarray(expected, float)
    expected = expected * observed.sum() / expected.sum()
    p = stats.chisquare(observed, expected).pvalue
    assert p > P_MIN, f"{what}: chi-square p={p:.2e}"
    return p


def binned_binomial(counts, n, p):
    """Observed vs expected for Binomial(n, p), merging bins with expectation < 5."""
    total = counts.sum()
    pmf = stats.binom.pmf(np.arange(n + 1), n, p) * total
    obs, exp, o_acc, e_acc = [], [], 0.0, 0.0
    for k in range(n + 1):
        o_acc += counts[k] if k < len(counts) else 0
        e_acc += pmf[k]
        if e_acc >= 5:
            obs.append(o_acc); exp.append(e_acc); o_acc = e_acc = 0.0
    obs[-1] += o_acc + counts[n + 1:].sum(); exp[-1] += e_acc
    return obs, exp


def check_split(d, length, split):
    seq, cor, nc = load_split(d, split)
    n = SPLITS[split]
    k = NUM_CORRUPTED

    # Structure and value range.
    assert seq.shape == (n, length), seq.shape
    assert cor.shape == (n, k, length), cor.shape
    assert seq.dtype == np.uint8 and cor.dtype == np.uint8
    assert seq.max() < VOCAB and cor.max() < VOCAB

    # Invariants: every corrupted copy differs, and num_changed is the true Hamming distance.
    diff = cor != seq[:, None, :]
    hamming = diff.sum(-1)
    assert (hamming == nc).all(), "num_changed does not match the actual number of changes"
    assert (hamming >= 1).all(), "found a corrupted sequence identical to the primary"

    # Files match the generator (deterministic seeding): regenerate the first and last chunks.
    nchunks = -(-n // CHUNK)
    for i in {0, nchunks - 1}:
        lo, hi = i * CHUNK, min(n, (i + 1) * CHUNK)
        rs, rc, rn = generate_chunk(chunk_rng(length, split, i), hi - lo, length)
        assert (rs == seq[lo:hi]).all() and (rc == cor[lo:hi]).all() and (rn == nc[lo:hi]).all(), \
            f"chunk {i} differs from generator"

    # Splits must not share their random streams.
    other = [s for s in SPLITS if s != split][0]
    assert not (generate_chunk(chunk_rng(length, other, 0), 100, length)[0] == seq[:100]).all()

    # Primary tokens: uniform overall and at every position.
    chisq(np.bincount(seq.ravel(), minlength=VOCAB), np.ones(VOCAB), "token frequencies")
    for pos in range(length):
        chisq(np.bincount(seq[:, pos], minlength=VOCAB), np.ones(VOCAB), f"token frequencies at position {pos}")

    # Number of changed tokens: 1 + Binomial(N-1, 1/(N-1)), mean exactly 2.
    extra = np.bincount((hamming - 1).ravel(), minlength=length)
    obs, exp = binned_binomial(extra, length - 1, 1 / (length - 1))
    chisq(obs, exp, "number of changed tokens")
    mean = hamming.mean()
    se = np.sqrt((1 - 1 / (length - 1)) / hamming.size)  # sd of Binomial(N-1, 1/(N-1)) is sqrt(1 - 1/(N-1))
    assert abs(mean - 2) < 6 * se, f"mean number of changes {mean:.5f} too far from 2"

    # Changed positions: uniform across positions.
    chisq(diff.sum((0, 1)), np.ones(length), "changed positions")

    # Replacement tokens: uniform over the V-1 other tokens (shift 1..V-1 mod V).
    shift = ((cor - seq[:, None, :]) % VOCAB)[diff]  # uint8 wraps mod 256, a multiple of VOCAB
    shifts = np.bincount(shift, minlength=VOCAB)
    assert shifts[0] == 0
    chisq(shifts[1:], np.ones(VOCAB - 1), "replacement tokens")

    # Replacement token does not depend on the original token.
    orig = np.broadcast_to(seq[:, None, :], cor.shape)[diff]
    table = np.zeros((VOCAB, VOCAB), int)
    np.add.at(table, (orig, cor[diff]), 1)
    assert np.diag(table).sum() == 0
    off = table[~np.eye(VOCAB, dtype=bool)]
    chisq(off, np.ones_like(off), "original -> replacement table")

    # Duplicate corrupted copies within a sample (informational; expected for small N).
    keys = np.ascontiguousarray(cor).view(np.dtype((np.void, length))).reshape(n, k)
    keys.sort(axis=1)
    dup_rate = (keys[:, 1:] == keys[:, :-1]).any(1).mean()

    return seq, mean, dup_rate


def unique_rows(a):
    return np.unique(np.ascontiguousarray(a).view(np.dtype((np.void, a.shape[1]))).ravel())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    args = p.parse_args()

    for length in args.lengths:
        d = Path(args.data) / subset_name(length)
        primaries = {}
        for split in SPLITS:
            seq, mean, dup = check_split(d, length, split)
            primaries[split] = seq
            print(f"[ok] {d.name:8s} {split:10s} mean changes={mean:.4f}  "
                  f"samples with duplicate corrupted copies={dup:.4%}", flush=True)

        # Splits are drawn independently: overlap must match the i.i.d. expectation.
        train = unique_rows(primaries["train"])
        log_space = length * np.log(VOCAB)
        for split in ("validation", "test"):
            overlap = np.isin(unique_rows(primaries[split]), train).mean()
            expected = -np.expm1(-SPLITS["train"] * np.exp(-log_space))
            tol = 0.01 if length <= 4 else 1e-3
            assert abs(overlap - expected) < tol, f"{d.name} {split}: train overlap {overlap} vs {expected}"
            print(f"[ok] {d.name:8s} {split:10s} unique primaries also in train={overlap:.4%} "
                  f"(i.i.d. expectation {expected:.4%})", flush=True)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
