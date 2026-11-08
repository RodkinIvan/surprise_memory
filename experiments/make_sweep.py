"""Write the jobs files of the state-size sweep: recall (lm) vs recognition (cls) memory.

  python3 experiments/make_sweep.py   ->  experiments/jobs_{sweep,sweep_smoke,seeds,fix,exp5,diag,diag_smoke,probe,hard,long,long_smoke,curr,repro2,bridge,epochs,keyed,keyed_smoke}.json
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
    return {"project": "surprise_memory", "sync": ["experiments/train_gdn2.py", "benchmark/generate.py"],
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
    # longer sequences (generated on each host when these ran, on the Hub since) where states still recall or
    # recognise at N=256; fresh negatives after the curriculum, since the stored train copies would take 3.7-7.4 GB per job
    # mbz_nlp's GPUs were full of other users' jobs on 2026-11-01 22:50; N=1024 first, as those take longest
    long = [job(f"long_k{k}v{v}_N{n}_{obj}_s{seed}", n, k, v, obj, 10, " --train_neg fresh", seed=seed, hosts=["mbz_nlp2", "mbz_nlp2.07"])
            for n in (1024, 512) for obj, states in (("cls", STATES), ("lm", STATES[2:])) for k, v in states
            for seed in ((0, 1, 2) if k * v <= 256 else (0,))]
    for j in long:
        j["gb"] = 18 if "_N1024_" in j["name"] else 10   # smoke run: 14.3 GB peak at N=1024 (128x128)
    (here / "jobs_long.json").write_text(json.dumps(jobs_file("long", long), indent=1))
    long_smoke = [job(f"long_smoke_k128v128_N1024_{obj}", 1024, 128, 128, obj, 0.05, " --train_neg fresh --test_samples 2000 --eval_every 100",
                      hosts=["mbz_nlp2.07"]) for obj in OBJECTIVES]
    for j in long_smoke:
        j["gb"] = 24
    (here / "jobs_long_smoke.json").write_text(json.dumps(jobs_file("long_smoke", long_smoke), indent=1))
    # review follow-ups (REVIEW.md 2026-11-08); every file is synced from one commit, so all runs record the same code
    hosts4 = ["mbz_nlp", "mbz_nlp2", "mbz_nlp2.07", "mbz_nlp3"]
    def named(prefix, n, k, v, obj, epochs, extra, seed, hosts=hosts4):
        j = job(f"{prefix}_k{k}v{v}_N{n}_{obj}_s{seed}", n, k, v, obj, epochs, extra, seed=seed, hosts=hosts)
        j["gb"] = 18 if n >= 1024 else 8
        return j
    # issue 2: length curriculum and easy-negative warm-up, separately. OBJECTIVES adds --curriculum 0.5 to every cls job,
    # so "neither" removes it; warm-up only = one stage at full length; length only = no heavily corrupted negatives
    arms = {"both": "", "warm": " --cur_min {n}", "len": " --hard_min 1", "none": None}
    curr = []
    for arm, flags in arms.items():
        for n in (64, 256, 1024):
            for k, v in [(16, 16), (128, 128)]:
                for seed in (0, 1, 2):
                    extra = (flags or "").format(n=n) + (" --train_neg fresh" if n >= 1024 else "")
                    j = named(f"curr_{arm}", n, k, v, "cls", 10, extra, seed)
                    if flags is None:
                        j["cmd"] = j["cmd"].replace(" --curriculum 0.5", "")
                    curr.append(j)
    (here / "jobs_curr.json").write_text(json.dumps(jobs_file("curr", curr), indent=1))
    # issue 3: the same config and seed twice per host, on both GPU models (mbz_nlp2.07: RTX 5880; the others: RTX 6000)
    repro = [job(f"repro2_{h.replace('.', '_')}_{rep}", 128, 16, 8, "cls", 10, " --diag_every 100", seed=0, hosts=[h])
             for h in ("mbz_nlp", "mbz_nlp2", "mbz_nlp2.07") for rep in "ab"]
    (here / "jobs_repro2.json").write_text(json.dumps(jobs_file("repro2", repro), indent=1))
    # issue 4: fresh negatives at N=256, against the 6 stored-negative runs per cell (sweep seeds 0-2, experiment 5 seeds 3-5)
    bridge = [named("bridge_fresh", 256, k, v, "cls", 10, " --train_neg fresh", seed) for k, v in [(16, 4), (16, 8)] for seed in range(6)]
    (here / "jobs_bridge.json").write_text(json.dumps(jobs_file("bridge", bridge), indent=1))
    # issue 7: 20 epochs with equal stages, to separate longer training from --cur_weight length (experiment 5's L arms)
    epochs = [named(f"e20eq_{name}", 256, k, v, "cls", 20, flags, seed) for name, flags in (("base", ""), ("C", FIXES["C"]))
              for k, v in [(16, 4), (16, 8)] for seed in FRESH_SEEDS]
    (here / "jobs_epochs.json").write_text(json.dumps(jobs_file("epochs", epochs), indent=1))
    # experiment 9: keyed recall vs recognition (8 sequences of length L = 1..1024), d_model 32, 2 layers, square states 4..32,
    # staged length curriculum; recall stops after the first stage below 0.5 exact copy. The data is synced with the code.
    keyed_sync = ["experiments/train_gdn2.py", "experiments/train_keyed.py", "benchmark/generate.py", "benchmark/generate_keyed.py", "data_keyed"]
    def keyed_job(name, k, obj, seed, extra=""):
        return {"name": name, "gpus": 1, "gb": 8, "retries": 1, "hosts": hosts4,
                "cmd": f"{PY} experiments/train_keyed.py --objective {obj} --k {k} --v {k} --seed {seed}{extra} --out results/keyed/{name}.json",
                "expect": [f"results/keyed/{name}.json"]}
    keyed = [keyed_job(f"keyed_k{k}_{obj}_s{seed}", k, obj, seed) for k in (4, 8, 16, 32) for obj in ("lm", "cls") for seed in (0, 1, 2)]
    keyed_smoke = [keyed_job(f"keyed_smoke_k{k}_{obj}", k, obj, 0, " --lengths 1 4 1024 --stage_steps 60 --test_samples 300")
                   for k in (4, 32) for obj in ("lm", "cls")]
    for name, js in (("keyed", keyed), ("keyed_smoke", keyed_smoke)):
        spec = {"project": "surprise_memory", "sync": keyed_sync, "fetch": ["results/keyed/{name}.json"], "jobs": js}
        (here / f"jobs_{name}.json").write_text(json.dumps(spec, indent=1))
    # probe the saved models with swap / shuffle / sum-preserving negatives, on one host that gets all of them
    probe = {"project": "surprise_memory", "sync": ["experiments/train_gdn2.py", "benchmark/generate.py", "experiments/probe_ckpt.py", "results/diag", "ckpt/diag"],
             "fetch": ["results/diag/probe.json"],
             "jobs": [{"name": "probe_diag", "cmd": f"{PY} experiments/probe_ckpt.py --results results/diag --ckpts ckpt/diag --out results/diag/probe.json",
                       "gpus": 1, "gb": 8, "retries": 1, "hosts": ["mbz_nlp2"], "expect": ["results/diag/probe.json"]}]}
    (here / "jobs_probe.json").write_text(json.dumps(probe, indent=1))
    (here / "jobs_sweep_smoke.json").write_text(json.dumps(jobs_file("sweep_smoke", smoke), indent=1))
    print(f"{len(sweep)} sweep jobs, {len(smoke)} smoke jobs, {len(seeds)} seed jobs, {len(fix)} fix jobs, {len(exp5)} exp5 jobs, {len(diag)} diag jobs, {len(hard)} hard jobs, {len(long)} long jobs, {len(curr)} curriculum, {len(repro)} repro, {len(bridge)} bridge, {len(epochs)} epoch, {len(keyed)} keyed jobs")


if __name__ == "__main__":
    main()
