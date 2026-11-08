"""Keyed recognition memory: N (key, sequence) pairs of fixed length L = 8, then a query "key: candidate".

Each sample stores N sequences of length L under distinct keys, N = 1, 2, 4, ..., 1024. The query names one key; its
candidates are the true sequence and 8 negatives: N_EDIT small edits of it (one forced change plus each other token with
probability 1/(L-1), about 2 changes), then N_FAR distant ones: sequences stored under other keys (a binding test) while there
are enough of them, min(N_FAR, N - 1), and fresh random sequences that were never stored for the rest (only when N < 5).
Recall: produce the sequence for the key. Recognition: rank the true candidate first.

  python benchmark/generate_keyed.py --out data_keyed [--counts 1 2 4 ...]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VOCAB = 16
LENGTH = 8
KEY_POOL = 2048                                # one token per key; N keys are drawn from the pool without replacement
N_EDIT, N_FAR = 4, 4
COUNTS = [2 ** i for i in range(11)]           # N = 1 .. 1024
SPLITS = {"train": 100_000, "validation": 5_000, "test": 10_000}
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
BASE_SEED = 20261108
CHUNK = 10_000


def n_other(n):
    """How many of the N_FAR distant negatives are sequences stored under other keys; the rest are fresh."""
    return min(N_FAR, n - 1)


def subset_name(n):
    return f"N{n}_L{LENGTH}_V{VOCAB}"


def chunk_rng(n, split, i, seed=BASE_SEED):
    return np.random.default_rng(np.random.SeedSequence([seed, n, LENGTH, VOCAB, KEY_POOL, SPLIT_IDS[split], i]))


def encode(x):
    """A sequence of LENGTH 4-bit tokens as one int64 (exact for LENGTH <= 15)."""
    enc = np.zeros(x.shape[:-1], dtype=np.int64)
    for t in range(x.shape[-1]):
        enc = enc * VOCAB + x[..., t]
    return enc


def generate_chunk(rng, m, n):
    """m samples with n pairs: keys [m, n], sequences [m, n, L], query [m], candidates [m, 8, L] (edits, then other keys,
    then fresh), num_changed [m, 8]."""
    keys = np.argsort(rng.random((m, KEY_POOL)), axis=1)[:, :n].astype(np.uint16)
    seqs = rng.integers(0, VOCAB, size=(m, n, LENGTH), dtype=np.uint8)
    while True:     # the stored sequences of a sample are distinct, so a sequence under another key is always a negative
        s = np.sort(encode(seqs), axis=1)
        dup = (s[:, 1:] == s[:, :-1]).any(1)
        if not dup.any():
            break
        seqs[dup] = rng.integers(0, VOCAB, size=(int(dup.sum()), n, LENGTH), dtype=np.uint8)
    query = rng.integers(0, n, size=m).astype(np.uint16)
    rows = np.arange(m)
    true = seqs[rows, query]                                                      # [m, L]

    mask = rng.random((m, N_EDIT, LENGTH), dtype=np.float32) < 1.0 / (LENGTH - 1)
    np.put_along_axis(mask, rng.integers(0, LENGTH, size=(m, N_EDIT))[..., None], True, axis=-1)
    shift = rng.integers(1, VOCAB, size=(m, N_EDIT, LENGTH), dtype=np.uint8)
    edits = ((true[:, None, :] + shift * mask) % VOCAB).astype(np.uint8)

    k = n_other(n)
    r = rng.random((m, n))
    r[rows, query] = np.inf                                                       # never the queried key itself
    others = seqs[rows[:, None], np.argsort(r, axis=1)[:, :k]]                    # [m, k, L]

    fresh = rng.integers(0, VOCAB, size=(m, N_FAR - k, LENGTH), dtype=np.uint8)
    stored = encode(seqs)
    while True:     # fresh negatives were never stored in the sample
        hit = (encode(fresh)[:, :, None] == stored[:, None, :]).any((1, 2))
        if not hit.any():
            break
        fresh[hit] = rng.integers(0, VOCAB, size=(int(hit.sum()), N_FAR - k, LENGTH), dtype=np.uint8)

    cand = np.concatenate([edits, others, fresh], axis=1)
    changed = (cand != true[:, None, :]).sum(-1).astype(np.uint16)
    return keys, seqs, query, cand, changed


def to_table(keys, seqs, query, cand=None, changed=None):
    _, n, length = seqs.shape
    cols = {"keys": pa.FixedSizeListArray.from_arrays(pa.array(keys.reshape(-1)), n),
            "sequences": pa.FixedSizeListArray.from_arrays(
                pa.FixedSizeListArray.from_arrays(pa.array(seqs.reshape(-1)), length), n),
            "query": pa.array(query)}
    if cand is not None:
        c = cand.shape[1]
        cols["candidates"] = pa.FixedSizeListArray.from_arrays(pa.FixedSizeListArray.from_arrays(pa.array(cand.reshape(-1)), length), c)
        cols["num_changed"] = pa.FixedSizeListArray.from_arrays(pa.array(changed.reshape(-1)), c)
    return pa.table(cols)


def write_split(out_dir, n, split):
    """One parquet shard per chunk; the train split stores no candidates (training draws its own negatives)."""
    d = Path(out_dir) / subset_name(n)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{split}-*.parquet"):
        old.unlink()
    nchunks = math.ceil(SPLITS[split] / CHUNK)
    for i in range(nchunks):
        m = min(CHUNK, SPLITS[split] - i * CHUNK)
        keys, seqs, query, cand, changed = generate_chunk(chunk_rng(n, split, i), m, n)
        t = to_table(keys, seqs, query, *((None, None) if split == "train" else (cand, changed)))
        pq.write_table(t, d / f"{split}-{i:05d}-of-{nchunks:05d}.parquet", row_group_size=2_000, compression="zstd")
    return nchunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_keyed")
    p.add_argument("--counts", type=int, nargs="+", default=COUNTS, help="numbers of stored pairs N")
    a = p.parse_args()
    for n in a.counts:
        for split in SPLITS:
            c = write_split(a.out, n, split)
            print(f"{subset_name(n)} {split}: {SPLITS[split]} samples, {c} shard(s)", flush=True)


if __name__ == "__main__":
    main()
