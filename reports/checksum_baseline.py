"""Recognition with a few bits: acc9 of checksum memories on the test splits.

A memory that keeps only a checksum of the sequence accepts a candidate whose checksum matches. Every corruption
changes at least one token by a nonzero shift, so the plain sum mod 16 (4 bits) catches every single change, and the
pair (sum, sum with odd position weights) mod 16 (8 bits) catches most double changes. Ties share the win, as in acc9.

  /opt/homebrew/Caskroom/miniconda/base/envs/ai/bin/python reports/checksum_baseline.py [--data data]
"""

import argparse
import glob

import numpy as np
import pyarrow.parquet as pq


def load(data, n):
    tabs = [pq.read_table(f, columns=["sequence", "corrupted"]) for f in sorted(glob.glob(f"{data}/N{n}_V16/test-*.parquet"))]
    seq = np.concatenate([t["sequence"].combine_chunks().flatten().to_numpy().reshape(t.num_rows, -1) for t in tabs])
    cor = np.concatenate([t["corrupted"].combine_chunks().flatten().flatten().to_numpy().reshape(t.num_rows, 8, -1) for t in tabs])
    return np.concatenate([seq[:, None], cor], 1).astype(np.int64)   # [B, 9, N], candidate 0 is the true sequence


def acc9(match):
    return float(np.mean(match[:, 0] / match.sum(1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    a = p.parse_args()
    print("| N | 4-bit checksum (sum mod 16) | 8-bit checksum (sum, weighted sum) mod 16 |\n|---|---|---|")
    for n in [4, 8, 16, 32, 64, 128, 256]:
        cand = load(a.data, n)
        s1 = cand.sum(-1) % 16
        s2 = (cand * ((2 * np.arange(n) + 1) % 16)).sum(-1) % 16
        m4 = s1 == s1[:, :1]
        m8 = m4 & (s2 == s2[:, :1])
        print(f"| {n} | {acc9(m4):.4f} | {acc9(m8):.4f} |", flush=True)


if __name__ == "__main__":
    main()
