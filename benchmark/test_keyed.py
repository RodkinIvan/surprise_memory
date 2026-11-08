"""Validate the keyed benchmark files: structure, invariants, negatives and reproducibility.

  python benchmark/test_keyed.py --data data_keyed [--counts 1 2 4 ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from generate_keyed import (CHUNK, COUNTS, KEY_POOL, LENGTH, N_EDIT, N_FAR, SPLITS, VOCAB, chunk_rng, encode,  # noqa: E402
                            generate_chunk, n_other, subset_name)


def load(d, split, n):
    files = sorted(d.glob(f"{split}-*.parquet"))
    assert files and len(files) == int(files[0].stem.split("-of-")[1]), f"{d.name}/{split}: shards missing"
    out = {}
    for f in files:
        t = pq.read_table(f)
        m = t.num_rows
        cols = {"keys": t["keys"].combine_chunks().flatten().to_numpy().reshape(m, n),
                "sequences": t["sequences"].combine_chunks().flatten().flatten().to_numpy().reshape(m, n, LENGTH),
                "query": t["query"].to_numpy()}
        if "candidates" in t.column_names:
            cols["candidates"] = t["candidates"].combine_chunks().flatten().flatten().to_numpy().reshape(m, N_EDIT + N_FAR, LENGTH)
            cols["num_changed"] = t["num_changed"].combine_chunks().flatten().to_numpy().reshape(m, N_EDIT + N_FAR)
        for k, v in cols.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def check(d, n, split):
    x = load(d, split, n)
    m = len(x["query"])
    assert m == SPLITS[split], (split, m)
    keys, seqs, query = x["keys"], x["sequences"], x["query"]
    assert keys.max() < KEY_POOL and seqs.max() < VOCAB and query.max() < n
    sk = np.sort(keys, 1)
    assert (sk[:, 1:] != sk[:, :-1]).all(), "keys repeat within a sample"
    stored = encode(seqs)                                                          # [m, n]
    ss = np.sort(stored, 1)
    assert (ss[:, 1:] != ss[:, :-1]).all(), "a sample stores the same sequence twice"
    # chunk 0 regenerates bit for bit
    k0, s0, q0, c0, n0 = generate_chunk(chunk_rng(n, split, 0), min(CHUNK, m), n)
    c = len(q0)
    assert (k0 == keys[:c]).all() and (s0 == seqs[:c]).all() and (q0 == query[:c]).all(), "chunk 0 does not regenerate"
    msg = ""
    if "candidates" in x:
        cand, changed = x["candidates"], x["num_changed"]
        assert (c0 == cand[:c]).all() and (n0 == changed[:c]).all()
        true = seqs[np.arange(m), query]
        assert (changed == (cand != true[:, None]).sum(-1)).all()
        edits = changed[:, :N_EDIT]
        assert (edits >= 1).all(), "an edit negative equals the true sequence"
        mean = edits.mean()
        assert abs(mean - 2.0) < 0.05, f"mean changes {mean} vs 2 (one forced, (L-1) x 1/(L-1) expected)"
        k = n_other(n)
        enc = encode(cand[:, N_EDIT:])                                             # [m, N_FAR]
        same = enc[:, :, None] == stored[:, None, :]                               # [m, N_FAR, n]
        assert (same[:, :k].sum(-1) == 1).all(), "an other-key negative is not exactly one stored sequence"
        assert not same[np.arange(m), :k, query].any(), "an other-key negative is the queried sequence"
        assert not same[:, k:].any(), "a fresh negative was stored"
        msg = f"mean changes (edits) {mean:.4f}; far negatives: {k} other-key, {N_FAR - k} fresh"
    else:
        assert split == "train"
    print(f"[ok] {d.name:<14} {split:<10} {m:>7} samples  {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_keyed")
    p.add_argument("--counts", type=int, nargs="+", default=COUNTS)
    a = p.parse_args()
    for n in a.counts:
        for split in SPLITS:
            check(Path(a.data) / subset_name(n), n, split)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
