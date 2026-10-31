"""Generate the recognition-memory benchmark.

Each sample holds a primary sequence of uniformly random tokens and K corrupted
copies of it. A corrupted copy always changes one uniformly chosen position and
additionally changes every other position independently with probability
1/(N-1), so the expected number of changed tokens is exactly 2. A changed token
is replaced by a uniformly chosen *different* token.

Output layout (Hugging Face parquet, one config per subset):
    <out>/N{N}_V{V}/{split}-{shard:05d}-of-{nshards:05d}.parquet
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VOCAB = 16
NUM_CORRUPTED = 8
LENGTHS = [4, 8, 16, 32, 64, 128, 256]
SPLITS = {"train": 900_000, "validation": 50_000, "test": 100_000}
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
BASE_SEED = 20261030
CHUNK = 50_000
MAX_SHARD_BYTES = 256 * 2**20


def subset_name(length, vocab=VOCAB):
    return f"N{length}_V{vocab}"


def generate_chunk(rng, n, length, vocab=VOCAB, k=NUM_CORRUPTED):
    """Return (sequence [n, L], corrupted [n, k, L], num_changed [n, k])."""
    seq = rng.integers(0, vocab, size=(n, length), dtype=np.uint8)

    mask = rng.random((n, k, length), dtype=np.float32) < 1.0 / (length - 1)
    forced = rng.integers(0, length, size=(n, k))
    np.put_along_axis(mask, forced[..., None], True, axis=-1)

    # Adding a non-zero shift mod V gives a uniform choice among the other V-1 tokens.
    shift = rng.integers(1, vocab, size=(n, k, length), dtype=np.uint8)
    corrupted = ((seq[:, None, :] + shift * mask) % vocab).astype(np.uint8)
    return seq, corrupted, mask.sum(-1).astype(np.uint16)


def chunk_rng(length, split, i, vocab=VOCAB, seed=BASE_SEED):
    """RNG for chunk i of a split; chunks are independent and individually reproducible."""
    return np.random.default_rng(np.random.SeedSequence([seed, length, vocab, SPLIT_IDS[split], i]))


def generate_split(length, split, n=None, vocab=VOCAB, k=NUM_CORRUPTED, seed=BASE_SEED):
    """Generate a full split deterministically from (seed, length, split)."""
    n = SPLITS[split] if n is None else n
    parts = []
    for i in range(math.ceil(n / CHUNK)):
        m = min(CHUNK, n - i * CHUNK)
        parts.append(generate_chunk(chunk_rng(length, split, i, vocab, seed), m, length, vocab, k))
    return tuple(np.concatenate(p) for p in zip(*parts))


def to_table(seq, corrupted, num_changed):
    n, k, length = corrupted.shape
    seq_arr = pa.FixedSizeListArray.from_arrays(pa.array(seq.reshape(-1)), length)
    inner = pa.FixedSizeListArray.from_arrays(pa.array(corrupted.reshape(-1)), length)
    cor_arr = pa.FixedSizeListArray.from_arrays(inner, k)
    nc_arr = pa.FixedSizeListArray.from_arrays(pa.array(num_changed.reshape(-1)), k)
    return pa.table({"sequence": seq_arr, "corrupted": cor_arr, "num_changed": nc_arr})


def write_split(out_dir, length, split, **kwargs):
    seq, corrupted, num_changed = generate_split(length, split, **kwargs)
    d = Path(out_dir) / subset_name(length)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{split}-*.parquet"):
        old.unlink()

    n = len(seq)
    bytes_per_row = corrupted[0].nbytes + seq[0].nbytes + num_changed[0].nbytes
    nshards = max(1, math.ceil(n * bytes_per_row / MAX_SHARD_BYTES))
    bounds = np.linspace(0, n, nshards + 1).astype(int)
    for s in range(nshards):
        lo, hi = bounds[s], bounds[s + 1]
        table = to_table(seq[lo:hi], corrupted[lo:hi], num_changed[lo:hi])
        pq.write_table(table, d / f"{split}-{s:05d}-of-{nshards:05d}.parquet",
                       row_group_size=10_000, compression="zstd")
    return nshards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    args = p.parse_args()
    for length in args.lengths:
        for split in SPLITS:
            nshards = write_split(args.out, length, split)
            print(f"{subset_name(length)} {split}: {SPLITS[split]} samples, {nshards} shard(s)", flush=True)


if __name__ == "__main__":
    main()
