"""Keyed recognition memory: 8 (key, sequence) pairs, then a query "key: candidate".

Each sample stores NUM_KEYS sequences of length L under distinct keys. The query names one key; its candidates are the true
sequence and 8 negatives: N_EDIT small edits of it (one forced change plus each other token with probability 1/(L-1),
as in the single-sequence benchmark) and N_OTHER sequences stored under other keys (a binding test). Recall: produce the
sequence for the key. Recognition: rank the true candidate first.

  python benchmark/generate_keyed.py --out data_keyed [--lengths 1 2 4 ...]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VOCAB = 16
NUM_KEYS = 8
KEY_POOL = 32
N_EDIT, N_OTHER = 4, 4
LENGTHS = [2 ** i for i in range(11)]          # 1 .. 1024
SPLITS = {"train": 100_000, "validation": 5_000, "test": 10_000}
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
BASE_SEED = 20261108
CHUNK = 10_000


def subset_name(length):
    return f"L{length}_K{NUM_KEYS}_V{VOCAB}"


def chunk_rng(length, split, i, seed=BASE_SEED):
    return np.random.default_rng(np.random.SeedSequence([seed, length, NUM_KEYS, VOCAB, SPLIT_IDS[split], i]))


def generate_chunk(rng, n, length):
    """keys [n, K], sequences [n, K, L], query [n], candidates [n, 8, L] (edits first), num_changed [n, 8]."""
    keys = np.argsort(rng.random((n, KEY_POOL)), axis=1)[:, :NUM_KEYS].astype(np.uint8)
    seqs = rng.integers(0, VOCAB, size=(n, NUM_KEYS, length), dtype=np.uint8)
    while True:     # the stored sequences of a sample are distinct, so a sequence under another key is always a negative
        enc = np.zeros((n, NUM_KEYS), dtype=np.int64) if length <= 15 else None
        if enc is None:
            break   # 16^16 values: a repeat among 8 is too rare to matter (checked by the tests)
        for t in range(length):
            enc = enc * VOCAB + seqs[:, :, t]
        s = np.sort(enc, axis=1)
        dup = (s[:, 1:] == s[:, :-1]).any(1)
        if not dup.any():
            break
        seqs[dup] = rng.integers(0, VOCAB, size=(int(dup.sum()), NUM_KEYS, length), dtype=np.uint8)
    query = rng.integers(0, NUM_KEYS, size=n).astype(np.uint8)
    rows = np.arange(n)
    true = seqs[rows, query]                                                      # [n, L]

    p = 1.0 / (length - 1) if length > 1 else 0.0
    mask = rng.random((n, N_EDIT, length), dtype=np.float32) < p
    np.put_along_axis(mask, rng.integers(0, length, size=(n, N_EDIT))[..., None], True, axis=-1)
    shift = rng.integers(1, VOCAB, size=(n, N_EDIT, length), dtype=np.uint8)
    edits = ((true[:, None, :] + shift * mask) % VOCAB).astype(np.uint8)

    r = rng.random((n, NUM_KEYS))
    r[rows, query] = np.inf                                                       # never the queried key itself
    other_idx = np.argsort(r, axis=1)[:, :N_OTHER]
    others = seqs[rows[:, None], other_idx]                                       # [n, N_OTHER, L]

    cand = np.concatenate([edits, others], axis=1)
    changed = (cand != true[:, None, :]).sum(-1).astype(np.uint16)
    return keys, seqs, query, cand, changed


def to_table(keys, seqs, query, cand=None, changed=None):
    n, k, length = seqs.shape
    cols = {"keys": pa.FixedSizeListArray.from_arrays(pa.array(keys.reshape(-1)), k),
            "sequences": pa.FixedSizeListArray.from_arrays(
                pa.FixedSizeListArray.from_arrays(pa.array(seqs.reshape(-1)), length), k),
            "query": pa.array(query)}
    if cand is not None:
        m = cand.shape[1]
        cols["candidates"] = pa.FixedSizeListArray.from_arrays(pa.FixedSizeListArray.from_arrays(pa.array(cand.reshape(-1)), length), m)
        cols["num_changed"] = pa.FixedSizeListArray.from_arrays(pa.array(changed.reshape(-1)), m)
    return pa.table(cols)


def write_split(out_dir, length, split):
    """One parquet shard per chunk; the train split stores no candidates (training draws its own negatives)."""
    d = Path(out_dir) / subset_name(length)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{split}-*.parquet"):
        old.unlink()
    nchunks = math.ceil(SPLITS[split] / CHUNK)
    for i in range(nchunks):
        m = min(CHUNK, SPLITS[split] - i * CHUNK)
        keys, seqs, query, cand, changed = generate_chunk(chunk_rng(length, split, i), m, length)
        t = to_table(keys, seqs, query, *((None, None) if split == "train" else (cand, changed)))
        pq.write_table(t, d / f"{split}-{i:05d}-of-{nchunks:05d}.parquet", row_group_size=2_000, compression="zstd")
    return nchunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_keyed")
    p.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    a = p.parse_args()
    for length in a.lengths:
        for split in SPLITS:
            n = write_split(a.out, length, split)
            print(f"{subset_name(length)} {split}: {SPLITS[split]} samples, {n} shard(s)", flush=True)


if __name__ == "__main__":
    main()
