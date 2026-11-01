# Open review issues

Last updated 2026-11-01. Only unresolved issues are listed; fixed ones have been removed. Reviewed state: commit `48e8121` plus the working tree. `experiments/train_gdn2.py` hashes to 360c77…, the code that produced `results/fix/`.

## 1. Experiment 4's comparison is biased in favour of the fixes (high, analysis)

Report section "Experiment 4", and `experiments/make_sweep.py` (`FIX_CONFIGS`).

- **Configs were selected on baseline failures.** `FIX_CONFIGS` contains exactly the configs where some baseline seed fell short. Every config therefore has at least one baseline run selected for being low, and the fix runs are fresh draws. Even a fix with no effect would look better through regression to the mean, especially in "runs < 0.95: 8 → 1".
- **"Same seed" is not a real pairing.** `--cur_min 4` changes the stage boundaries, and `--cls_loss pair` halves the batch while still drawing the unused `label`/`pick`. Either way the RNG streams diverge from step 1, so "paired" seeds share no noise.
- **The N ≤ 128 cut was chosen after seeing the results.** The headline "C fixes everything up to N=128" rests on it.
- **Overall, the fixes are no better than the baseline.** Despite the bias in their favour, all three fixes have a lower overall mean (A 0.943, B 0.914, C 0.940, against 0.959), and C is 11 better / 12 worse.
- **To do:**
  - Run fresh seeds (e.g. 3–5) for **both** the baseline and C, on the 11 configs and on a few configs that were not selected.
  - Until then, state "the ranking loss fixes recognition training up to N=128" as a hypothesis in the report and the poster.

## 2. The baseline code is not in version control (medium, reproducibility)

- **No git copy of the experiment 3 code.** All of `results/sweep/` (144 runs) and the round-2 smoke runs come from code hash 56a994…, and round 1 from 774cc7…. Neither version is in git. The only commit touching `train_gdn2.py` (`05d929d`) is already 360c77…, and `code_version.commit` is `null` in every result.
- **Experiment 4's comparisons can't be pinned to their flags.** They assume the baseline differs from A only by `--cur_min`. That can't be verified, and the sweep's `args` lack the `cur_min`/`cls_loss` keys, so the code changed in at least that way.
- **`make_sweep.py` overwrites the job files.** It rewrites `jobs_sweep.json` and `jobs_seeds.json` as well. Resubmitting them now runs 360c77…, not the code behind the published sweep.
- **To do:**
  - Recover 56a994… from a host if it still exists. Otherwise rerun 2–3 baseline cls configs with the current code at default flags (`--curriculum 0.5`, no `--cur_min`/`--cls_loss`) and confirm they match.
  - Commit before every sync, so `commit` is recorded.

## 3. "Hard-negative curriculum" is a misnomer (medium, terminology)

Report section "Experiment 2", and the poster.

- The curriculum feeds **heavily corrupted** negatives first, up to half the tokens changed. `train_gdn2.py:119` labels them "(easy)". Their share falls 1→0, so training moves from easy negatives to the benchmark's hard 1–2-change ones.
- In standard usage, "hard negatives" means negatives close to the positive, which is the opposite.
- **To do:** rename it, e.g. "easy-negative warm-up" or "negative-difficulty curriculum".

## 4. Smaller issues (low)

- **Triton version:** every run, including the fix runs, warns that Triton 3.2.0 is below fla's recommended 3.3.0 ("errors may occur"). Experiment 1 hit `Triton Error [CUDA]: invalid argument`. Upgrade the env on the hosts.
- **Unchecked `--cur_min`:** the help text says it must be a power of 2, but nothing checks this. A value like 12 would quietly make the first stage 16. Add a check.
- **Weak "every seed" claims:** with n=3 per cell, "wins in every seed" is weak evidence, even though the 16×4 margins (0.39–0.41 mean acc9) are large. Say "3 of 3 runs" and give the spread.
- **Uncommitted poster:** `posters/` has uncommitted edits, including deleted `full_state.pdf` and `small_state_curves.pdf`. Commit them, and apply issues 1 and 3 to the poster text.
- **Stale `summarize.py`:** `experiments/summarize.py` keys rows by `(N, objective)` only, so it would silently drop seeds or states if pointed at `results/sweep`. Use `analyze_sweep.py` instead, or delete `summarize.py`.
- **Dataset card** (`irodkin/recognition-memory`): the license (`mit`) was picked by the agent, not chosen by the owner. The card also describes no evaluation protocol (store `sequence`, score it against the 8 `corrupted` copies, report acc9/pairwise). Add one, now that the metrics are fixed.
