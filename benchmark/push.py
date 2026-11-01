"""Write the dataset card and upload the generated parquet files to the Hugging Face Hub.

Run: python benchmark/push.py --data data --repo irodkin/recognition-memory
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from generate import BASE_SEED, LENGTHS, NUM_CORRUPTED, SPLITS, VOCAB, subset_name


def card():
    configs = "\n".join(
        f"- config_name: {subset_name(n)}\n"
        f"  data_files:\n"
        + "".join(f"  - split: {s}\n    path: {subset_name(n)}/{s}-*.parquet\n" for s in SPLITS)
        for n in LENGTHS
    )
    subsets = ", ".join(f"`{subset_name(n)}`" for n in LENGTHS)
    return f"""---
configs:
{configs}license: mit
task_categories:
- other
pretty_name: Recognition Memory
size_categories:
- 1M<n<10M
---

# Recognition Memory

A benchmark for memories that *recognise* rather than *recall*: the model sees a sequence
and must decide whether it is consistent with what it has stored, for example by using low
surprise as the familiarity signal.

## Subsets

{subsets}: sequences of length N over a vocabulary of {VOCAB} tokens.
Splits: train {SPLITS['train']:,} / validation {SPLITS['validation']:,} / test {SPLITS['test']:,} samples.

## Fields

| field | type | description |
|---|---|---|
| `sequence` | `uint8[N]` | primary sequence, tokens i.i.d. uniform over `0..{VOCAB - 1}` |
| `corrupted` | `uint8[{NUM_CORRUPTED}][N]` | {NUM_CORRUPTED} corrupted copies of `sequence` |
| `num_changed` | `uint16[{NUM_CORRUPTED}]` | number of tokens changed in each corrupted copy |

## Corruption process

For each corrupted copy, one position chosen uniformly at random is always changed, and each
of the other N-1 positions is changed independently with probability 1/(N-1). The number of
changed tokens is therefore `1 + Binomial(N-1, 1/(N-1))`, with expectation exactly 2. A changed
token is replaced by a uniformly random *different* token, so every corrupted copy differs from
the primary sequence. The {NUM_CORRUPTED} copies are drawn independently, so two copies of the
same sample can coincide (in about 4.4% of samples for N=4, 2.7% for N=8, 1.5% for N=16, and
roughly halving with each doubling of N after that).

Primary sequences are drawn i.i.d. and are not deduplicated. For N=4 there are only
16^4 = 65,536 possible sequences, so nearly all of them appear in train and every
validation/test sequence also appears in train. From N=16 up, overlap between splits is
essentially zero.

## Evaluation protocol

1. Give the memory the primary `sequence` (for example, in context before a separator).
2. Score `sequence` and each of its {NUM_CORRUPTED} `corrupted` copies with the memory's familiarity signal (for example,
   minus the summed surprise of the candidate, or a learned score).
3. Report, over the test split:
   - **acc9**: the share of samples where `sequence` scores strictly highest among the 9 candidates. A tie for the top
     score counts as 1/(number tied), so a constant score gives 1/9.
   - **pairwise**: the share of (`sequence`, corrupted copy) pairs where `sequence` scores higher, with ties counting
     as 1/2. Chance is 0.5.

Reference points: a memory that keeps only the sum of the tokens mod 16 (4 bits) gets acc9 about 0.85 at every N, and
one that also keeps a position-weighted sum (8 bits) about 0.97, because the corruptions are small edits.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("irodkin/recognition-memory", "N64_V16")
```

## Reproduction

Generated with `generate.py` (base seed {BASE_SEED}); every chunk of 50k samples is seeded from
`(seed, N, V, split, chunk)` and can be regenerated on its own.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--repo", default="irodkin/recognition-memory")
    p.add_argument("--token", default=str(Path.home() / "hf_token"))
    p.add_argument("--public", action="store_true")
    args = p.parse_args()

    data = Path(args.data)
    (data / "README.md").write_text(card())
    for f in ("generate.py", "test_dataset.py"):
        (data / f).write_text((Path(__file__).parent / f).read_text())

    api = HfApi(token=Path(args.token).read_text().strip())
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(data),
                            allow_patterns=["*.parquet", "README.md", "*.py"])
    print(f"https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
