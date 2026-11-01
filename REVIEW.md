# Open review issues

Last updated 2026-11-01. Only unresolved issues are listed; fixed ones have been removed. Reviewed state: commit `a731a87` (HEAD) plus untracked `results/exp5/`. `experiments/train_gdn2.py` hashes to edfda8…, the code experiment 5 runs with (commit `391fda4`).

Reviewed line by line: `experiments/{train_gdn2,make_sweep,analyze_sweep,compare_fix}.py`, `experiments/archive/train_gdn2_56a994.py`, `experiments/jobs_exp5.json`, `tools/hostq.py` (its 7 unit tests pass), `reports/{make_figures,checksum_baseline}.py`, `posters/make_poster_figures.py`, `benchmark/push.py`, and the report and poster text.

This file is written by the read-only review agent, which makes no code changes. The coding agent fixes these issues.

**Experiment 5 is still running.** 34 of 98 runs had finished at the last check. This iteration can't be closed until it finishes and is analysed (issue 3).

## 1. The code version is recorded at the end of a run, not the start (high, provenance bug)

`experiments/train_gdn2.py:265`: `version = json.loads(Path("CODE_VERSION.json").read_text())` runs **after training**.

- **How it goes wrong.** Every `hostq submit` re-syncs `train_gdn2.py` and rewrites `CODE_VERSION.json` on the hosts. A run that is training during such a sync finishes and records the *new* version, though Python loaded the *old* code at start-up. Jobs still queued on that host start with the new code.
- **It has already happened, so far harmlessly.** Comparing each result's `synced_at` with the start time in `logs/hostq/<job>.log` finds 4 runs that started before the sync they record:
  - `k128v128_N256_lm_s0` (started 17:23, records the 18:01 sync). That sync had the same hash (56a994…), so the record is still right.
  - `smoke_k128v128_N256_cls`, `smoke_k16v4_N256_cls` and `smoke_k16v4_N256_lm` (started 16:06, record the 16:08:45 sync, 774cc7…). They ran whatever was synced about 16:06. If the code changed between 16:06 and 16:08, these three round-1 smoke runs are mislabelled. That code is lost, so the report should say their version is uncertain.
- **Live risk now.** Experiment 5 has 97 runs queued or running. Changing `train_gdn2.py` and running `hostq submit` for any jobs file before experiment 5 finishes would silently give queued runs different code, and mislabel running ones.
- **To do:**
  - Read `CODE_VERSION.json` at the top of `main()`, before training. Better still, also record `sha256` of `__file__`, so the result names the code that actually ran.
  - Do **not** sync this change, or any other change to `train_gdn2.py`, until every experiment 5 run has finished.
  - Make `hostq submit` refuse, or require `--force`, when a target host still has queued or running jobs of the same project and the sync would change the code hash.

## 2. The repro check in experiment 5 has no pass criterion (medium, analysis)

`make_sweep.py` adds `repro_k16v4_N32_cls_s0` and `repro_k16v8_N128_cls_s0` to confirm that the current code reproduces the sweep.

- **The code side is verified.** `archive/train_gdn2_56a994.py` hashes to exactly 56a994…. Diffing it against the current file shows that the defaults (`--cur_min 16`, `--cls_loss bce`, `--cur_weight equal`) give identical training. The BCE path and the `lens` and stage formulas are unchanged.
- **What's missing is how to judge the reruns.** fla's Triton backward kernels are not guaranteed bit-deterministic. Both chosen configs are baseline *failures* (test acc9 0.875 and 0.870), the most seed-sensitive cases. A tiny numerical difference can grow into a very different final acc9.
- **To do:** decide the criterion before the results arrive. For example: the step-1000 and step-2000 curve points (`train_loss`, `recog_acc9`) match the sweep run to about 1e-3, while the final acc9 may differ. Write it in the report.

## 3. Nothing analyses experiment 5 yet (medium, analysis)

- `compare_fix.py` only reads `results/fix`.
- **`analyze_sweep.py` would merge variants.** Its rows hold (k, v, N, objective, seed) but not `cls_loss`, `cur_min`, `cur_weight`, `lr` or `epochs`. So on `results/exp5`, the `e5_base_*` and `e5_C_*` runs, and the 20-epoch `L` runs, would be indistinguishable apart from `name`. Its docstring says "nothing collides", which is true only for `results/sweep`.
- **To do:**
  - Add those args to each row, or write `compare_exp5.py`. It should give baseline vs C on fresh seeds 3–5, separately for the 11 selected and the 3 unselected configs, plus the repro check from issue 2.
  - State the decision rule in advance, e.g. "C replaces the baseline if its mean acc9 is higher and it has fewer runs < 0.95 on the fresh seeds".

## 4. Smaller issues (low)

- **The 20-epoch arms confound two changes.** `e5_baseL`/`e5_CL` change epochs (10 → 20) and `--cur_weight length` together. An improvement can't be attributed to either one. Say so in the report, or add a 20-epoch `--cur_weight equal` arm.
- **Triton is still 3.2.0** (fla recommends 3.3.0, and experiment 1 hit `Triton Error [CUDA]: invalid argument`). Keeping it for this iteration is right, so every run shares one environment. Upgrade before the next iteration, and record the torch, fla and triton versions in each result JSON.
- **The dataset card isn't pushed.** The evaluation protocol is committed in `benchmark/push.py` (`b2fa293`), but the Hub still serves the old card. Run `push.py --public`. The license (`mit`) is still the agent's choice; the owner should confirm it.
- **Dead code in `posters/make_poster_figures.py`.** `fig_full_state` and `fig_small_state_curves` are never called; their PDFs were deleted. `fig_small_state_curves` also hard-codes equal-time stages from 16. Delete both.
- **`hostq` edge cases (no effect on results so far):**
  - `Worker.start` launches the job *before* writing `state=running`. A worker crash in between would leave the job `queued`, and it would start twice.
  - `handle_kills` logs "killed" and rewrites the record even when the job had already finished.
