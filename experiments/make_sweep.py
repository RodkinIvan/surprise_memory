"""Write the jobs files of the state-size sweep: recall (lm) vs recognition (cls) memory.

  python3 experiments/make_sweep.py   ->  experiments/jobs_{sweep,sweep_smoke,seeds,fix,exp5,diag,diag_smoke,probe,hard}.json
Submit with:  python3 tools/hostq.py submit experiments/jobs_sweep.json
"""

import json
from pathlib import Path

PY = "/mnt/data/users/ivan.rodkin/lab/envs/memory_amplifier/bin/python"
HOSTS = ["mbz_nlp2", "mbz_nlp3", "mbz_nlp2.07"]
SEED_HOSTS = ["mbz_nlp", "mbz_nlp2", "mbz_nlp2.07"]   # mbz_nlp3 was full of other jobs on 2026-10-31 19:20
# (key dim, value dim) of the single GDN-2 head; state = k * v numbers per layer.
# fla's chunk kernel needs head_dim >= 16, so the smallest states shrink the value dim only.
STATES = [(16, 4), (16, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
LENGTHS = [4, 8, 16, 32, 64, 128, 256]
OBJECTIVES = {"lm": "", "cls": " --curriculum 0.5"}
# fixes for the recognition objective, tested where some seed of the sweep fell short
FIXES = {"A": " --cur_min 4", "B": " --cur_min 4 --lr 3e-4", "C": " --cur_min 4 --cls_loss pair"}
FIX_CONFIGS = [(16, 4, n) for n in (16, 32, 64, 128, 256)] + [(16, 8, n) for n in (32, 64, 128, 256)] + [(16, 16, n) for n in (32, 64)]
# experiment 5: fresh seeds for baseline and C, on FIX_CONFIGS and on configs where no baseline seed fell short
UNSELECTED = [(16, 8, 16), (16, 16, 128), (16, 16, 256)]
FRESH_SEEDS = (3, 4, 5)
LONG = " --cur_weight length"   # with 20 epochs: longer training, curriculum time proportional to prefix length


def job(name, n, k, v, obj, epochs, extra="", seed=0, hosts=HOSTS):
    cmd = (f"{PY} experiments/train_gdn2.py --subset N{n}_V16 --objective {obj} --head_dim {k} --expand_v {v / k}"
           f" --epochs {epochs} --seed {seed}{OBJECTIVES[obj]}{extra} --out results/{{project_dir_rel}}/{name}.json")
    return {"name": name, "cmd": cmd, "gpus": 1, "gb": 8, "retries": 1, "hosts": hosts}


def jobs_file(out_dir, jobs):
    for j in jobs:
        j["cmd"] = j["cmd"].replace("{project_dir_rel}", out_dir)
        j["expect"] = [f"results/{out_dir}/{j['name']}.json"]
    return {"project": "surprise_memory", "sync": ["experiments/train_gdn2.py"],
            "fetch": [f"results/{out_dir}/{{name}}.json"], "jobs": jobs}


def main():
    here = Path(__file__).parent
    sweep = [job(f"k{k}v{v}_N{n}_{obj}_s0", n, k, v, obj, 10)
             for k, v in STATES for n in LENGTHS for obj in OBJECTIVES]
    smoke = [job("smoke_k128v128_N64_cls", 64, 128, 128, "cls", 3, " --test_samples 10000"),
             job("smoke_k128v128_N256_cls", 256, 128, 128, "cls", 3, " --test_samples 10000"),
             job("smoke_k16v4_N256_cls", 256, 16, 4, "cls", 1, " --test_samples 10000"),
             job("smoke_k16v4_N256_lm", 256, 16, 4, "lm", 1, " --test_samples 10000"),
             # second round: classifier with the length curriculum
             job("smoke2_k128v128_N128_cls", 128, 128, 128, "cls", 3, " --test_samples 10000"),
             job("smoke2_k128v128_N256_cls", 256, 128, 128, "cls", 3, " --test_samples 10000"),
             job("smoke2_k16v4_N256_cls", 256, 16, 4, "cls", 3, " --test_samples 10000")]
    # more seeds where the sweep is capacity-limited and the recognition results vary (states up to 16x16, N >= 16)
    seeds = [job(f"k{k}v{v}_N{n}_{obj}_s{seed}", n, k, v, obj, 10, seed=seed, hosts=SEED_HOSTS)
             for seed in (1, 2) for k, v in STATES[:3] for n in LENGTHS[2:] for obj in OBJECTIVES]
    (here / "jobs_sweep.json").write_text(json.dumps(jobs_file("sweep", sweep), indent=1))
    (here / "jobs_seeds.json").write_text(json.dumps(jobs_file("sweep", seeds), indent=1))
    fix = [job(f"fix{f}_k{k}v{v}_N{n}_cls_s{seed}", n, k, v, "cls", 10, flags, seed=seed, hosts=SEED_HOSTS)
           for f, flags in FIXES.items() for seed in (0, 1, 2) for k, v, n in FIX_CONFIGS]
    (here / "jobs_fix.json").write_text(json.dumps(jobs_file("fix", fix), indent=1))
    variants = {"base": "", "C": FIXES["C"]}
    exp5 = [job(f"e5_{name}_k{k}v{v}_N{n}_cls_s{seed}", n, k, v, "cls", 10, flags, seed=seed, hosts=SEED_HOSTS)
            for name, flags in variants.items() for seed in FRESH_SEEDS for k, v, n in FIX_CONFIGS + UNSELECTED]
    exp5 += [job(f"e5_{name}L_k{k}v{v}_N256_cls_s{seed}", 256, k, v, "cls", 20, flags + LONG, seed=seed, hosts=SEED_HOSTS)
             for name, flags in variants.items() for seed in FRESH_SEEDS for k, v in [(16, 4), (16, 8)]]
    # the current code at default flags must reproduce the sweep (whose code is experiments/archive/train_gdn2_56a994.py)
    exp5 += [job(f"repro_k{k}v{v}_N{n}_cls_s0", n, k, v, "cls", 10, hosts=SEED_HOSTS) for k, v, n in [(16, 4, 32), (16, 8, 128)]]
    (here / "jobs_exp5.json").write_text(json.dumps(jobs_file("exp5", exp5), indent=1))
    # collapse diagnosis: repeated runs of the sweep's recognition training, with gate/gradient/logit logs and saved weights
    diag_flags = " --diag_every 100 --save ckpt/diag/{name}.pt"
    diag = [job(f"diag_k{k}v{v}_N{n}_cls_s{seed}", n, k, v, "cls", 10, diag_flags, seed=seed, hosts=SEED_HOSTS)
            for k, v, n, runs in [(16, 4, 16, 12), (16, 4, 32, 8), (16, 8, 128, 8)] for seed in range(100, 100 + runs)]
    diag_smoke = [job("diag_smoke", 16, 16, 4, "cls", 0.3, diag_flags + " --eval_every 200", hosts=SEED_HOSTS)]
    for name, js in (("diag", diag), ("diag_smoke", diag_smoke)):
        spec = jobs_file(name, js)
        spec["fetch"].append("ckpt/diag/{name}.pt")
        (here / f"jobs_{name}.json").write_text(json.dumps(spec, indent=1))
    # experiment 7: keep a share of benchmark-like (1-2 token) negatives from the first step of the easy-negative warm-up;
    # fresh seeds for every variant, the baseline (hard_min 0) included, on the two configs that failed most in experiment 6
    hard = [job(f"hard{h}_k{k}v{v}_N{n}_cls_s{seed}", n, k, v, "cls", 10,
                f" --hard_min {h / 100} --diag_every 100 --save ckpt/hard/{{name}}.pt", seed=seed, hosts=SEED_HOSTS)
            for h in (0, 25, 50) for k, v, n in [(16, 4, 16), (16, 4, 32)] for seed in range(200, 212)]
    spec = jobs_file("hard", hard)
    spec["fetch"].append("ckpt/hard/{name}.pt")
    (here / "jobs_hard.json").write_text(json.dumps(spec, indent=1))
    # probe the saved models with swap / shuffle / sum-preserving negatives, on one host that gets all of them
    probe = {"project": "surprise_memory", "sync": ["experiments/train_gdn2.py", "experiments/probe_ckpt.py", "results/diag", "ckpt/diag"],
             "fetch": ["results/diag/probe.json"],
             "jobs": [{"name": "probe_diag", "cmd": f"{PY} experiments/probe_ckpt.py --results results/diag --ckpts ckpt/diag --out results/diag/probe.json",
                       "gpus": 1, "gb": 8, "retries": 1, "hosts": ["mbz_nlp2"], "expect": ["results/diag/probe.json"]}]}
    (here / "jobs_probe.json").write_text(json.dumps(probe, indent=1))
    (here / "jobs_sweep_smoke.json").write_text(json.dumps(jobs_file("sweep_smoke", smoke), indent=1))
    print(f"{len(sweep)} sweep jobs, {len(smoke)} smoke jobs, {len(seeds)} seed jobs, {len(fix)} fix jobs, {len(exp5)} exp5 jobs, {len(diag)} diag jobs, {len(hard)} hard jobs")


if __name__ == "__main__":
    main()
