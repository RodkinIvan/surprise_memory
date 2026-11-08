"""Write the dataset card and upload the keyed benchmark to the Hugging Face Hub.

Run: python benchmark/push_keyed.py --data data_keyed --repo irodkin/recognition-memory-keyed
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from generate_keyed import (BASE_SEED, COUNTS, KEY_POOL, LENGTH, N_EDIT, N_FAR, SPLITS, VOCAB,  # noqa: E402
                            subset_name)


def card():
    configs = "".join(
        f"- config_name: {subset_name(n)}\n  data_files:\n"
        + "".join(f"  - split: {s}\n    path: {subset_name(n)}/{s}-*.parquet\n" for s in SPLITS)
        for n in COUNTS)
    subsets = ", ".join(f"`{subset_name(n)}`" for n in COUNTS)
    return f"""---
configs:
{configs}license: mit
task_categories:
- other
pretty_name: Keyed Recognition Memory
size_categories:
- 1M<n<10M
---

# Keyed Recognition Memory

A benchmark for comparing a memory that *recalls* with one that *recognises*, as the number of stored items grows. Each
sample stores N sequences of length {LENGTH} under N keys, N = 1, 2, 4, …, 1,024. A query names one key and gives a candidate:
recall means producing that key's sequence; recognition means telling whether the candidate is it. It is the keyed version of
[irodkin/recognition-memory](https://huggingface.co/datasets/irodkin/recognition-memory), which stores one sequence.

## Subsets

{subsets}: N sequences of length {LENGTH} over {VOCAB} tokens, N = 1, 2, 4, …, 1,024.
Splits: train {SPLITS['train']:,} / validation {SPLITS['validation']:,} / test {SPLITS['test']:,} samples.

## Fields

| field | type | description |
|---|---|---|
| `keys` | `uint16[N]` | N distinct key ids, drawn without replacement from `0..{KEY_POOL - 1}`; their order is the storage order |
| `sequences` | `uint8[N][{LENGTH}]` | the sequence stored under each key; tokens i.i.d. uniform over `0..{VOCAB - 1}`, the N sequences of a sample distinct |
| `query` | `uint16` | index (`0..N-1`) of the queried pair: the query key is `keys[query]`, the true candidate `sequences[query]` |
| `candidates` | `uint8[{N_EDIT + N_FAR}][{LENGTH}]` | validation and test only: {N_EDIT} small edits of the true sequence, then {N_FAR} distant negatives (below) |
| `num_changed` | `uint16[{N_EDIT + N_FAR}]` | validation and test only: positions where each candidate differs from the true sequence |

The train split has no candidates: draw training negatives with the same process (below), fresh each time.

## Negatives

- **Small edits** ({N_EDIT} per query). One position chosen uniformly at random is always changed, and each of the other {LENGTH - 1}
  positions with probability 1/{LENGTH - 1}, so about 2 tokens change. A changed token gets a uniformly random *different* value.
- **Other keys** (min({N_FAR}, N−1) per query). The sequences stored under other keys of the same sample, chosen at random. They
  test whether the memory binds a sequence to its key; a memory that only knows which sequences it stored can't reject them.
- **Fresh** ({N_FAR} − min({N_FAR}, N−1) per query, so only for N < {N_FAR + 1}). Uniformly random sequences that the sample does not
  store. With N = 1 all {N_FAR} distant negatives are fresh, with N = 2 three are, with N = 4 one is.

## Evaluation protocol

1. Give the memory the N (key, sequence) pairs, then the query key.
2. Score the true sequence and each of its {N_EDIT + N_FAR} candidates.
3. Report on the test split:
   - **acc9**: the share of samples where the true sequence scores strictly highest among the 9. Ties for the top score count
     as 1/(number tied), so a constant score gives 1/9.
   - **pairwise**, separately for edits, other keys and fresh sequences: the share of (true, negative) pairs ranked correctly, ties counting 1/2.
   - For recall: the share of samples where greedy decoding reproduces the whole sequence.

## Reproduction

Generated with `generate_keyed.py` (base seed {BASE_SEED}) and checked with `test_keyed.py`; every chunk of 10k samples is
seeded from `(seed, N, length, vocabulary, key pool, split, chunk)` and can be regenerated on its own.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_keyed")
    p.add_argument("--repo", default="irodkin/recognition-memory-keyed")
    p.add_argument("--token", default=str(Path.home() / "hf_token"))
    p.add_argument("--public", action="store_true")
    a = p.parse_args()
    data = Path(a.data)
    (data / "README.md").write_text(card())
    for f in ("generate_keyed.py", "test_keyed.py"):
        (data / f).write_text((Path(__file__).parent / f).read_text())
    api = HfApi(token=Path(a.token).read_text().strip())
    api.create_repo(a.repo, repo_type="dataset", private=not a.public, exist_ok=True)
    # replaces the whole repo content (old subsets included) in one commit
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(data), allow_patterns=["N*/*.parquet", "README.md", "*.py"],
                      delete_patterns=["*"], commit_message="Keyed benchmark: N = 1..1024 sequences of length 8")
    print(f"https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
