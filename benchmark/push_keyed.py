"""Write the dataset card and upload the keyed benchmark to the Hugging Face Hub.

Run: python benchmark/push_keyed.py --data data_keyed --repo irodkin/recognition-memory-keyed
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from generate_keyed import (BASE_SEED, KEY_POOL, LENGTHS, N_EDIT, N_OTHER, NUM_KEYS, SPLITS, VOCAB,  # noqa: E402
                            subset_name)


def card():
    configs = "".join(
        f"- config_name: {subset_name(n)}\n  data_files:\n"
        + "".join(f"  - split: {s}\n    path: {subset_name(n)}/{s}-*.parquet\n" for s in SPLITS)
        for n in LENGTHS)
    subsets = ", ".join(f"`{subset_name(n)}`" for n in LENGTHS)
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

A benchmark for comparing a memory that *recalls* with one that *recognises*, when it holds several items. Each sample
stores {NUM_KEYS} sequences under {NUM_KEYS} keys. A query names one key and gives a candidate: recall means producing that key's
sequence; recognition means telling whether the candidate is it. It is the keyed version of
[irodkin/recognition-memory](https://huggingface.co/datasets/irodkin/recognition-memory), which stores one sequence.

## Subsets

{subsets}: {NUM_KEYS} sequences of length L = 1, 2, 4, …, 1,024 over {VOCAB} tokens.
Splits: train {SPLITS['train']:,} / validation {SPLITS['validation']:,} / test {SPLITS['test']:,} samples.

## Fields

| field | type | description |
|---|---|---|
| `keys` | `uint8[{NUM_KEYS}]` | {NUM_KEYS} distinct key ids, drawn without replacement from `0..{KEY_POOL - 1}`; their order is the storage order |
| `sequences` | `uint8[{NUM_KEYS}][L]` | the sequence stored under each key; tokens i.i.d. uniform over `0..{VOCAB - 1}`, the {NUM_KEYS} sequences of a sample distinct |
| `query` | `uint8` | index (`0..{NUM_KEYS - 1}`) of the queried pair: the query key is `keys[query]`, the true candidate `sequences[query]` |
| `candidates` | `uint8[{N_EDIT + N_OTHER}][L]` | validation and test only: {N_EDIT} small edits of the true sequence, then {N_OTHER} sequences stored under other keys |
| `num_changed` | `uint16[{N_EDIT + N_OTHER}]` | validation and test only: positions where each candidate differs from the true sequence |

The train split has no candidates: draw training negatives with the same process (below), fresh each time.

## Negatives

- **Small edits** ({N_EDIT} per query). One position chosen uniformly at random is always changed, and each of the other L−1
  positions with probability 1/(L−1), so about 2 tokens change (exactly 1 at L=1). A changed token gets a uniformly random
  *different* value.
- **Other keys** ({N_OTHER} per query). The sequences stored under {N_OTHER} other keys of the same sample, chosen at random. They test
  whether the memory binds a sequence to its key; a memory that only knows which sequences it stored can't reject them.

## Evaluation protocol

1. Give the memory the {NUM_KEYS} (key, sequence) pairs, then the query key.
2. Score the true sequence and each of its {N_EDIT + N_OTHER} candidates.
3. Report on the test split:
   - **acc9**: the share of samples where the true sequence scores strictly highest among the 9. Ties for the top score count
     as 1/(number tied), so a constant score gives 1/9.
   - **pairwise**, separately for edits and other keys: the share of (true, negative) pairs ranked correctly, ties counting 1/2.
   - For recall: the share of samples where greedy decoding reproduces the whole sequence.

## Reproduction

Generated with `generate_keyed.py` (base seed {BASE_SEED}) and checked with `test_keyed.py`; every chunk of 10k samples is
seeded from `(seed, L, keys, vocabulary, split, chunk)` and can be regenerated on its own.
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
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(data), allow_patterns=["*.parquet", "README.md", "*.py"])
    print(f"https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
