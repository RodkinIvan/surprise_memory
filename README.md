# Recall vs recognition memory in a small Gated DeltaNet-2

Can a memory tell whether an input is familiar long after it can no longer reproduce it? This repo trains a 2-layer Gated
DeltaNet-2 with a shrinking recurrent state, either to copy a sequence (recall) or to say whether a candidate is the
sequence it saw (recognition), and compares how long each keeps working.

- **Results:** [reports/recall_vs_recognition.md](reports/recall_vs_recognition.md), with every experiment, its design,
  results, caveats and the exact result files and code versions.
- **Benchmark:** [`irodkin/recognition-memory`](https://huggingface.co/datasets/irodkin/recognition-memory) on the
  Hugging Face Hub, lengths 4 to 1,024. Built by [benchmark/generate.py](benchmark/generate.py), checked by
  [benchmark/test_dataset.py](benchmark/test_dataset.py).
- **Poster:** [posters/](posters/).
- **Open issues:** [REVIEW.md](REVIEW.md), written by a read-only review pass.

## Layout

| Path | What it is |
|---|---|
| `experiments/train_gdn2.py` | The trainer: one run, one result JSON (arguments, learning curve, test metrics, host, GPU, code version) |
| `experiments/make_sweep.py` | Writes every `experiments/jobs_*.json`: the exact command of every run in the report |
| `experiments/{analyze_sweep,compare_fix,compare_exp5,analyze_diag,compare_hard,capacity_table,probe_ckpt}.py` | Turn result files into the report's tables and statistics |
| `reports/make_figures.py`, `reports/checksum_baseline.py` | The report's figures and checksum baselines |
| `results/<experiment>/` | Every result JSON, plus the summaries the scripts write |
| `tools/hostq.py` | The job queue used to run on the GPU hosts (see below) |

## Running

Requirements: a CUDA GPU, Python 3.11 (the version used) and the packages in [requirements.txt](requirements.txt).

One run, for example the recognition model with a 16×4 state at N=256:

```
python experiments/train_gdn2.py --subset N256_V16 --objective cls --head_dim 16 --expand_v 0.25 \
    --epochs 10 --seed 0 --curriculum 0.5 --out results/x.json
```

The data is downloaded from the Hub into `hf_data/` on first use. Each jobs file lists its runs' commands in the `cmd`
field, with `{project_dir_rel}` already filled in, so any scheduler can run them:

```
python3 -c "import json; [print(j['cmd']) for j in json.load(open('experiments/jobs_sweep.json'))['jobs']]"
```

The commands start with the Python of the hosts' environment; replace it with yours. Then rebuild the tables with the
scripts above (each says what it reads and writes in its docstring) and the figures with `python reports/make_figures.py`.

## The job queue

`tools/hostq.py` keeps a queue on each GPU host, so runs survive the submitting machine going away. It needs `route.py`
from a separate router tool, which is not public, so from a clone it stops with "route.py not found". The commands in the
jobs files don't depend on it. `tools/hostq.py` is the source; the router repository carries a copy.

## Notes

- Training is not reproducible run to run at small states: the same code and seed can learn or collapse (report,
  experiment 5). Compare failure rates over several runs, not single seeds.
- Result files name the hosts the runs used, and the jobs files the hosts' environment path.
