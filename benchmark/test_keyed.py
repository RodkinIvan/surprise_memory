"""Validate the keyed benchmark files: structure, invariants, negatives and reproducibility.

  python benchmark/test_keyed.py --data data_keyed [--lengths 1 2 4 ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from generate_keyed import (CHUNK, KEY_POOL, LENGTHS, N_EDIT, N_OTHER, NUM_KEYS, SPLITS, VOCAB, chunk_rng, generate_chunk,
                            subset_name)


def load(d, split, length):
    files = sorted(d.glob(f"{split}-*.parquet"))
    assert files and len(files) == int(files[0].stem.split("-of-")[1]), f"{d.name}/{split}: shards missing"
    out = {}
    for f in files:
        t = pq.read_table(f)
        n = t.num_rows
        cols = {"keys": t["keys"].combine_chunks().flatten().to_numpy().reshape(n, NUM_KEYS),
                "sequences": t["sequences"].combine_chunks().flatten().flatten().to_numpy().reshape(n, NUM_KEYS, length),
                "query": t["query"].to_numpy()}
        if "candidates" in t.column_names:
            cols["candidates"] = t["candidates"].combine_chunks().flatten().flatten().to_numpy().reshape(n, N_EDIT + N_OTHER, length)
            cols["num_changed"] = t["num_changed"].combine_chunks().flatten().to_numpy().reshape(n, N_EDIT + N_OTHER)
        for k, v in cols.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def check(d, length, split):
    x = load(d, split, length)
    n = len(x["query"])
    assert n == SPLITS[split], (split, n)
    keys, seqs, query = x["keys"], x["sequences"], x["query"]
    assert keys.max() < KEY_POOL and seqs.max() < VOCAB and query.max() < NUM_KEYS
    assert (np.sort(keys, 1)[:, 1:] != np.sort(keys, 1)[:, :-1]).all(), "keys repeat within a sample"
    flat = seqs.reshape(n * NUM_KEYS, length)
    _, inv = np.unique(flat, axis=0, return_inverse=True)
    inv = inv.reshape(n, NUM_KEYS)
    assert (np.sort(inv, 1)[:, 1:] != np.sort(inv, 1)[:, :-1]).all(), "a sample stores the same sequence twice"
    # chunk 0 regenerates bit for bit
    k0, s0, q0, c0, n0 = generate_chunk(chunk_rng(length, split, 0), min(CHUNK, n), length)
    m = len(q0)
    assert (k0 == keys[:m]).all() and (s0 == seqs[:m]).all() and (q0 == query[:m]).all(), "chunk 0 does not regenerate"
    msg = ""
    if "candidates" in x:
        cand, changed = x["candidates"], x["num_changed"]
        assert (c0 == cand[:m]).all() and (n0 == changed[:m]).all()
        true = seqs[np.arange(n), query]
        assert (changed == (cand != true[:, None]).sum(-1)).all()
        edits, others = changed[:, :N_EDIT], cand[:, N_EDIT:]
        assert (edits >= 1).all(), "an edit negative equals the true sequence"
        expect = 1.0 if length == 1 else 2.0
        mean = edits.mean()
        assert abs(mean - expect) < 0.05, f"mean changes {mean} vs {expect}"
        # every other-key negative is a sequence stored under a different key
        same = (others[:, :, None, :] == seqs[:, None, :, :]).all(-1)                # [n, N_OTHER, K]
        assert (same.sum(-1) == 1).all(), "an other-key negative is not exactly one stored sequence"
        assert not same[np.arange(n), :, query].any(), "an other-key negative is the queried sequence"
        msg = f"mean changes (edits) {mean:.4f}, mean Hamming distance (other keys) {changed[:, N_EDIT:].mean() / length:.4f}·L"
    else:
        assert split == "train"
    print(f"[ok] {d.name:<14} {split:<10} {n:>7} samples  {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_keyed")
    p.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    a = p.parse_args()
    for length in a.lengths:
        for split in SPLITS:
            check(Path(a.data) / subset_name(length), length, split)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
