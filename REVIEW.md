# Review: recognition-memory benchmark

Reviewed 2026-10-30. Files: `benchmark/generate.py`, `benchmark/test_dataset.py`, `benchmark/push.py`, `data/`, and the Hub repo `irodkin/recognition-memory`.

**Verdict: done correctly. No blocking issues.** The dataset meets the spec, is public, and the Hub copy is byte-identical to the local data that passed the tests.

## Requirements check

| Requirement | Status | Evidence |
|---|---|---|
| Random tokens, vocab 16 | OK | `rng.integers(0, 16)`; values in 0..15; uniformity chi-square passes overall and at every position |
| Subset name `N${LENGTH}_V16` | OK | Configs `N4_V16 … N256_V16` (lengths 4, 8, 16, 32, 64, 128, 256; the lengths were not specified, so the agent picked these) |
| Splits 900k / 50k / 100k | OK | Confirmed by `load_dataset` for every split |
| 8 corrupted sequences per sample | OK | `corrupted` is `uint8[8][N]` |
| One random position always corrupted | OK | `np.put_along_axis(mask, forced, True)`; minimum Hamming distance is 1 everywhere |
| Other positions corrupted with p = 1/(N-1) | OK | Number of extra changes fits Binomial(N-1, 1/(N-1)) (chi-square); the mean number of changes is 2.000 ± 0.001 across all subsets |
| A corrupted token actually changes | OK | Nonzero shift mod 16, so the new token is uniform over the other 15 values; checked to be independent of the original token |
| Tests run | OK | I re-ran `python benchmark/test_dataset.py --data data`: **ALL CHECKS PASSED** (33 s) |
| Pushed to HF, public | OK | Anonymous `HfApi(token=False).dataset_info`: `private=False`, `gated=False`; all 36 data/card/script files match local sha256 |
| Loads with `datasets` | OK | Anonymous `load_dataset(..., "N4_V16")` and `"N8_V16"` work. Features: `List(uint8, N)`, `List(List(uint8, N), 8)`, `List(uint16, 8)`. Hamming == `num_changed` in all rows |

## Correctness notes on the generator

- The forced position's own Bernoulli draw is overwritten with True, so the other N-1 positions keep independent Bernoulli(1/(N-1)) draws. That gives E[changes] = 1 + 1 = 2 exactly, as the spec intends.
- The mask uses a `float32` uniform compared with `1/(N-1)`. The rounding bias is about 1e-8, which is negligible.
- `uint8` arithmetic `(seq + shift*mask) % 16` never overflows: the maximum value is 15 + 15.
- Seeding is deterministic per `(seed, N, V, split, chunk)`, and the splits use separate streams. The test regenerates the first and last chunks of every split and gets the same bytes.

## Minor observations (not bugs; worth knowing)

1. **N4_V16 is degenerate as a train-vs-test task.** There are only 16^4 = 65,536 possible sequences, so 100% of validation/test primaries also appear in train. Every corrupted copy is also some train primary. This is fine if the protocol is per-sample (store `sequence`, then tell it apart from its corruptions). It would be meaningless if "familiar" means "seen in the train split". The dataset card already says this.
2. **Corrupted copies within a sample can repeat.** This happens in about 4.4% of samples for N=4, 2.7% for N=8, 1.5% for N=16, and less after that. The spec does not forbid it, but the effective number of distinct negatives is sometimes below 8. The card documents this, and the figures it gives match what I measured.
3. **The test checks generator equality only on the first and last 50k chunks.** Middle chunks are covered by the statistical tests, not byte-for-byte. This is acceptable.
4. **License `mit` was chosen by the agent**, not by you. Change it in the card if you want something else.
5. **The card does not spell out an evaluation protocol.** For example, it doesn't say "write `sequence` to memory; score `sequence` vs each `corrupted` copy; report accuracy/AUROC". You might add one once the method is set.
6. `data/.cache/huggingface/` holds leftover `upload_large_folder` state. It was not uploaded (`allow_patterns` excludes it), and it is safe to delete locally.
7. Subsets with N ≤ 2 would break the scheme (p = 1/(N-1) ≥ 1, or division by zero). They are not included, so this does not matter now.

---

# Review: GDN-2 scripts and experiments

Reviewed 2026-10-31. Files: `experiments/train_gdn2.py`, `experiments/summarize.py`, `experiments/jobs_{gdn2,smoke}.json` and their state files, `results/gdn2/*.json`, `logs/router/*.log`, `logs_router_gdn2.txt`. I did not review `router/` or `tools/hostq.py`, except to read their logs.

**Verdict: the code is mostly correct and all 14 runs finished. The results table should not be read as "recognition fails at N ≥ 64".** That row comes from an optimisation failure, not a capacity limit. The setup as it stands cannot test the capacity hypothesis.

## Results (seed 0, h=128, 2 layers, 730k params, 10 epochs)

| N | lm: exact recall | lm: surprise acc9 | cls: acc9 | cls: balanced acc |
|---|---|---|---|---|
| 4–16 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32 | 1.000 | 1.000 | 0.999 | 0.999 |
| 64 | 0.9999 | 1.000 | **0.005** | **0.500** |
| 128 | 0.9993 | 1.000 | **0.001** | **0.500** |
| 256 | 0.988 | 1.000 | **0.003** | **0.500** |

## Findings, most important first

### 1. The cls runs at N ≥ 64 never trained: optimisation failure, not capacity (high)
- At N = 64, 128 and 256, cls `train_loss` is 0.693 (= ln 2) at the first eval (step 1000) and stays there to step 35156. `balanced_acc` stays at 0.500 throughout, and `grad_norm` decays to about 0.03. The model collapsed to a constant output.
- The same backbone, trained as lm, separates the true sequence from all 8 corruptions perfectly at every N (surprise acc9 = 1.000). So the architecture has enough capacity. The cls objective just never escapes the plateau.
- N32 cls shows the plateau directly: it sits at 0.693 until about step 2000, then escapes, with grad-norm spikes of 4–16 against a clip of 1.0. At N ≥ 64 the signal is too weak to escape: one bit per sample, from about 2 differing tokens among N.
- **The table must not be read as "recognition is harder than recall".** Possible fixes:
  - a listwise softmax over the 9 candidates, which matches acc9 directly and uses all 8 negatives;
  - a curriculum over N, or warm-starting from the N32 cls checkpoint or the lm checkpoint;
  - a lower LR or a longer warmup;
  - several seeds.

### 2. `recog_acc9` is below chance because ties count as failures (medium)
- When a model's output does not depend on the candidate, all 9 scores tie. The strict `>` in `train_gdn2.py:131` then gives acc9 ≈ 0, not the chance level of 1/9 ≈ 0.111.
- The cls score is also produced in bf16 under autocast, which makes ties more likely.
- `recog_pairwise` already counts a tie as half a win. For acc9, count a tie as 1/(number tied for the top score), or break ties at random. This matters as soon as a run sits near chance.

### 3. This setup cannot test "recognition needs less capacity" (high, design)
- Both objectives see the full sequence in context. The memory is the GDN-2 recurrent state, which is at least 128×128 per layer (≥ 16k numbers).
- At N = 256 there are only 256 × 4 = 1024 bits to store, so no run is capacity-limited. lm recall is near perfect everywhere (exact 0.988 at N=256).
- To test the hypothesis, the state has to become the bottleneck. Possible ways:
  - sweep `head_dim`/`hidden` downwards (e.g. 8–64);
  - push N into the thousands;
  - plot the recall and recognition curves against state size, not against N at a fixed size.

### 4. N8 results come from a different code version (low)
- `results/gdn2/*N8*.json` have `files_sha256 = 52e31b…` (synced 2026-10-30 21:09). All the other runs have `41772b…` (synced 2026-10-31 09:53). The router reused N8 ("2 of 14 jobs already finished").
- The likely difference is the `eval_bs` cap. The N4 logs show two earlier crashes with `Triton Error [CUDA]: invalid argument` in fla's causal conv at the first eval, which fits a grid larger than 65535.
- Without git history I can't confirm the change was eval-only. Rerun N8 so the whole table comes from one version, or keep a copy of each version that is synced.

### 5. Smaller issues
- **Router bookkeeping:** the N4 logs hold three runs appended together (two crashes, then a success), but the state file records `attempt: 1` and `problems: []`. The final results are fine, but the crash history is lost from the state file.
- **Triton warning:** every run warns that Triton 3.2.0 is below fla's recommended 3.3.0 ("errors may occur"). The crash above may be related. Upgrading the remote env would be worth it.
- **Silent "success":** the router counted the failed N ≥ 64 cls runs as successes. A cheap check would catch them: for example, flag runs whose final `balanced_acc` is under 0.55, or whose loss is still about ln 2.
- **`summarize.py` collisions:** it keys rows by `(N, objective)` only. Once there are several seeds or hidden sizes in one directory, it will silently show an arbitrary one, because glob order is unspecified. Group by the full args, or average over seeds.
- **Single seed:** there is only one seed (s0), and nothing records whether escaping the plateau depends on the seed.

## Code checks that passed
- **Input layout:** `[BOS] seq [SEP] cand`, and `h[:, n+1:2n+1]` predicts `cand[0..N-1]` with no offset error. The cls readout at the last token sees the whole input under causal masking.
- **Teacher-forced metrics:** `recall_exact` is teacher-forced, but for a deterministic copy target it equals greedy decoding.
- **LR schedule:** linear warmup over 500 steps, then cosine decay from 1.0× to 0.1×. Training uses bf16 autocast, gradient clipping at 1.0, and no weight decay on embeddings, norms or biases. All fine.
- **Eval data:** the first 10k validation samples during training and the full 100k test set at the end. The output JSON is written atomically and records the code hash, host and GPU.
