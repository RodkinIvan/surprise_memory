"""Write the jobs files of the state-size sweep: recall (lm) vs recognition (cls) memory.

  python3 experiments/make_sweep.py   ->  experiments/jobs_{sweep,sweep_smoke,seeds,fix}.json
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
    (here / "jobs_sweep_smoke.json").write_text(json.dumps(jobs_file("sweep_smoke", smoke), indent=1))
    print(f"{len(sweep)} sweep jobs, {len(smoke)} smoke jobs, {len(seeds)} seed jobs, {len(fix)} fix jobs")


if __name__ == "__main__":
    main()
