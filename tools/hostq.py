#!/usr/bin/env python3
"""hostq: a job queue that lives on the GPU hosts, so running and queued jobs do not depend on this machine.

`submit` copies the code to the hosts (with router/route.py: same hosts file, jobs file and conda envs as the router), puts every job that is not on a
host yet into one host's queue, and starts a worker there in a tmux session. The worker starts queued jobs when a GPU has room, records exit codes,
retries failed jobs and exits after it has been idle for a while. Jobs run in their own process sessions, so they survive the worker, tmux and ssh.
This machine can sleep or go away at any point; `fetch` copies the results back whenever it runs again.

Local (from the project folder):
    python3 tools/hostq.py submit jobs.json [--dry-run] [--resubmit job ...] [--resubmit-failed]
    python3 tools/hostq.py status jobs.json
    python3 tools/hostq.py fetch  jobs.json [--wait] [--all]
    python3 tools/hostq.py kill   jobs.json [job ...]          # default: all jobs of the file

On a host, inside <root>/<project>/ (standard library only; started by `submit`):
    python3 .hostq/hostq.py worker | add | list | kill

Host layout: .hostq/jobs/<job>.json (the job and its state: queued / running / done / failed / killed), .hostq/run/<job>.sh and .exit,
.hostq/worker.log, logs/hostq/<job>.log. Local state: <jobs file stem>.hostq.state.json (which host has which job, what was fetched).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

QDIR = Path(".hostq")
JOBS, RUN, LOGS = QDIR / "jobs", QDIR / "run", Path("logs/hostq")
FINISHED = ("done", "failed", "killed")


def now() -> str:
    return time.strftime("%F %T")


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


# ======================================================================================================== host side
def read_jobs() -> dict:
    out = {}
    for p in sorted(JOBS.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except (OSError, ValueError):
            pass
    return out


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_state() -> dict:
    """GPU index -> (free GB, total GB) from nvidia-smi; {} without GPUs."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    gpus = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            gpus[int(parts[0])] = (float(parts[1]) / 1024, float(parts[2]) / 1024)
    return gpus


def job_gpu_usage() -> dict | None:
    """(session id, GPU index) -> GB used by the processes of that session, from nvidia-smi; None if it can't tell.

    A job runs in its own session whose id is the pid hostq recorded, so its processes are found by their session id."""
    try:
        idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if apps.returncode != 0:
        return None
    uuid = {u.strip(): int(i) for i, u in (line.split(",", 1) for line in idx.splitlines() if "," in line) if i.strip().isdigit()}
    usage = {}
    for line in apps.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit() or parts[1] not in uuid:
            continue
        try:
            sid = os.getsid(int(parts[0]))
            used = float(parts[2]) / 1024
        except (OSError, ValueError):
            continue
        usage[(sid, uuid[parts[1]])] = usage.get((sid, uuid[parts[1]]), 0.0) + used
    return usage


class Worker:
    def __init__(self, a):
        self.a = a
        self.children: dict[str, subprocess.Popen] = {}

    def log(self, msg: str) -> None:
        print(f"[{now()}] {msg}", flush=True)

    def finish(self, key: str, j: dict, rc: int, why: str = "") -> None:
        missing = [e for e in j.get("expect", []) if not Path(e).exists()] if rc == 0 else []
        ok = rc == 0 and not missing
        why = why or (f"expected files missing: {missing}" if missing else f"exit {rc}")
        j.update(exit=rc, finished=now())
        if not ok and j.get("attempt", 1) <= j.get("retries", 0):
            j["state"], j["note"] = "queued", f"attempt {j['attempt']} failed ({why}); retrying"
        else:
            j["state"], j["note"] = ("done", "") if ok else ("failed", why)
        write_json(JOBS / f"{key}.json", j)
        self.log(f"{key}: {j['state']} ({why})")

    def handle_kills(self, jobs: dict) -> None:
        for key, j in jobs.items():
            marker = JOBS / f"{key}.kill"
            if marker.exists():
                if j["state"] == "queued":
                    j.update(state="killed", finished=now(), note="killed while queued")
                elif j["state"] == "running":
                    try:
                        os.killpg(j["pid"], signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                    j.update(state="killed", finished=now(), killed_ts=time.time(), note="killed while running")
                write_json(JOBS / f"{key}.json", j)
                marker.unlink()
                self.log(f"{key}: killed")
            elif j["state"] == "killed" and j.get("pid") and time.time() - j.get("killed_ts", 0) > 30 and alive(j["pid"]):
                try:
                    os.killpg(j["pid"], signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    def check_running(self, jobs: dict) -> None:
        for key, p in list(self.children.items()):
            if p.poll() is not None:        # reap our own children so that a finished one does not look alive
                del self.children[key]
        for key, j in jobs.items():
            if j["state"] != "running":
                continue
            exit_file = RUN / f"{key}.exit"
            if exit_file.exists():
                text = exit_file.read_text().strip()
                self.finish(key, j, int(text) if text.lstrip("-").isdigit() else 1)
            elif j.get("boot_id") != boot_id() or not alive(j["pid"]):
                if not exit_file.exists():  # it may have been written between the two checks
                    self.finish(key, j, -1, "the process vanished without an exit code (host restart or killed)")

    def start(self, key: str, j: dict, gpus: list) -> None:
        g = ",".join(map(str, gpus))
        env = f"export HOSTQ_JOB={key} HOSTQ_GPU={g}\n" + (f"export CUDA_VISIBLE_DEVICES={g}\n" if gpus else "")
        path = f"export PATH={j['prefix']}/bin:$PATH\n" if j.get("prefix") else ""
        script = RUN / f"{key}.sh"
        script.write_text(f"#!/bin/bash\ncd {shlex.quote(os.getcwd())}\n{path}{env}"
                          f"( {j['cmd'].replace('{gpu}', g)} ) >> {LOGS}/{key}.log 2>&1\n"
                          f"echo $? > {RUN}/{key}.exit.tmp && mv {RUN}/{key}.exit.tmp {RUN}/{key}.exit\n")
        (RUN / f"{key}.exit").unlink(missing_ok=True)
        attempt = j.get("attempt", 0) + 1
        with open(LOGS / f"{key}.log", "a") as f:
            f.write(f"===== hostq: attempt {attempt} on {socket.gethostname()} GPU {g or '-'} at {now()}\n")
        p = subprocess.Popen(["bash", str(script)], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children[key] = p
        j.update(state="running", pid=p.pid, assigned=gpus, started=now(), started_ts=time.time(), boot_id=boot_id(), attempt=attempt)
        write_json(JOBS / f"{key}.json", j)
        self.log(f"{key}: started attempt {attempt} on GPU {g or '-'} (pid {p.pid})")

    def place(self, jobs: dict) -> None:
        queued = sorted(((k, j) for k, j in jobs.items() if j["state"] == "queued"), key=lambda kj: kj[1].get("order", 0))
        if not queued:
            return
        running = [j for j in jobs.values() if j["state"] == "running"]
        gpus = gpu_state()
        allowed = [i for i in gpus if self.a.gpus is None or i in self.a.gpus]
        load = {i: 0 for i in allowed}
        reserved = {i: 0.0 for i in allowed}
        usage = job_gpu_usage()
        for j in running:
            for i in j.get("assigned", []):
                load[i] = load.get(i, 0) + 1
                if usage is not None:   # hold back the part of its gb it doesn't use yet: memory can peak late (e.g. at evaluation)
                    reserved[i] = reserved.get(i, 0.0) + max(0.0, j.get("gb", 0.0) - usage.get((j.get("pid"), i), 0.0))
                elif time.time() - j.get("started_ts", 0) < self.a.ramp:   # its memory may not show in nvidia-smi yet
                    reserved[i] = reserved.get(i, 0.0) + j.get("gb", 0.0)
        cpu_running = sum(1 for j in running if not j.get("assigned"))
        for key, j in queued:
            need = j.get("gpus", 1)
            if need == 0:
                if cpu_running < self.a.cpu_slots:
                    self.start(key, j, [])
                    cpu_running += 1
                continue
            ok = sorted((load[i], -(gpus[i][0] - reserved[i]), i) for i in allowed
                        if load[i] < self.a.per_gpu and gpus[i][0] - reserved[i] >= j.get("gb", 0.0) + self.a.headroom)
            if len(ok) >= need:
                pick = sorted(x[2] for x in ok[:need])
                self.start(key, j, pick)
                for i in pick:
                    load[i] += 1
                    reserved[i] += j.get("gb", 0.0)

    def run(self) -> int:
        for d in (JOBS, RUN, LOGS):
            d.mkdir(parents=True, exist_ok=True)
        lock = open(QDIR / "worker.lock", "a")
        for _ in range(20):     # `list` takes the lock for an instant to see whether a worker runs
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.5)
        else:
            self.log("another worker holds the lock; exiting")
            return 0
        self.log(f"worker started (pid {os.getpid()}, per-gpu {self.a.per_gpu}, gpus {self.a.gpus or 'all'}, headroom {self.a.headroom} GB)")
        idle_since = time.time()
        while True:
            self.handle_kills(read_jobs())
            self.check_running(read_jobs())
            self.place(read_jobs())
            if any(j["state"] in ("queued", "running") for j in read_jobs().values()):
                idle_since = time.time()
            elif time.time() - idle_since > self.a.idle_exit:
                self.log("idle; exiting")
                return 0
            time.sleep(self.a.poll)


def host_add(a) -> int:
    """Queue the jobs given as a JSON list on stdin; a job that is queued or running already is left alone."""
    for d in (JOBS, RUN, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    existing, result = read_jobs(), {}
    for i, j in enumerate(json.loads(sys.stdin.read())):
        key = j["key"]
        old = existing.get(key)
        if old and old["state"] in ("queued", "running"):
            result[key] = f"already {old['state']}"
            continue
        j.update(state="queued", submitted=now(), order=time.time_ns() + i, attempt=0)
        (RUN / f"{key}.exit").unlink(missing_ok=True)
        (JOBS / f"{key}.kill").unlink(missing_ok=True)
        write_json(JOBS / f"{key}.json", j)
        result[key] = "queued"
    print(json.dumps(result))
    return 0


def host_list(a) -> int:
    for d in (JOBS, RUN, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    with open(QDIR / "worker.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker = False
        except BlockingIOError:
            worker = True
    print(json.dumps({"worker": worker, "jobs": read_jobs()}))
    return 0


def host_kill(a) -> int:
    jobs = read_jobs()
    for key in a.keys:
        if key in jobs and jobs[key]["state"] in ("queued", "running"):
            (JOBS / f"{key}.kill").touch()
    return 0


# ======================================================================================================== local side
def load_route():
    import importlib.util
    here = Path(__file__).resolve().parent
    candidates = [Path(os.environ["HOSTQ_ROUTE"])] if "HOSTQ_ROUTE" in os.environ else [here / "route.py", Path("router/route.py")]
    path = next((p.resolve() for p in candidates if p.exists()), None)
    if path is None:
        sys.exit(f"error: route.py not found in {', '.join(map(str, candidates))} (keep it next to hostq.py, or set HOSTQ_ROUTE)")
    spec = importlib.util.spec_from_file_location("route", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["route"] = mod
    spec.loader.exec_module(mod)
    return mod


class Local:
    def __init__(self, a):
        self.a, self.R = a, load_route()
        self.hosts_list, self.hosts_file = self.R.read_hosts(a)
        self.hosts = {h.name: h for h in self.hosts_list}
        self.jobs_file = Path(a.jobs).resolve()
        self.spec = self.R.load_spec(self.jobs_file.read_text())
        self.project_dir = Path(a.project_dir or ".").resolve()
        self.state_path = self.jobs_file.with_name(self.jobs_file.stem + ".hostq.state.json")
        self.reload()
        self.key = {j.name: self.R.sanitize(j.name) for j in self.spec.jobs}
        self.dirs: dict[str, str] = {}

    def reload(self) -> None:
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"jobs": {}}

    def set_job(self, name: str, rec: dict | None = None, **fields) -> None:
        """Replace (rec) or update (fields) one job's entry. The file is re-read under a lock, so a running
        `fetch --wait` and a `submit` in another terminal don't undo each other's changes."""
        import fcntl
        with open(self.state_path.with_name(self.state_path.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.reload()
            if rec is not None:
                self.state["jobs"][name] = rec
            else:
                self.state["jobs"][name].update(fields)
            write_json(self.state_path, self.state)

    def conn(self, host: str):
        return self.R.Conn(self.hosts[host], self.hosts_file)

    def remote_dir(self, host: str) -> str:
        if host not in self.dirs:
            h = self.hosts[host]
            base = self.conn(host).must(f"mkdir -p {h.root} && cd {h.root} && pwd", "resolve the root folder", timeout=60).strip()
            self.dirs[host] = f"{base}/{self.spec.project}"
        return self.dirs[host]

    def host_cmd(self, host: str, args: str, stdin: str | None = None, timeout: float = 120) -> str:
        d = self.remote_dir(host)
        return self.conn(host).must(f"cd {d} && python3 .hostq/hostq.py {args}", f"hostq {args.split()[0]}", timeout=timeout, stdin=stdin)

    def ensure_worker(self, host: str) -> None:
        h, d = self.hosts[host], self.remote_dir(host)
        sess = f"hq_{self.R.sanitize(self.spec.project)}"
        opts = f"--per-gpu {h.per_gpu} --headroom {self.a.headroom}" + (f" --gpus {','.join(map(str, h.gpus))}" if h.gpus else "")
        run = f"python3 .hostq/hostq.py worker {opts} >> .hostq/worker.log 2>&1"
        self.conn(host).must(f"cd {d} && (tmux has-session -t {sess} 2>/dev/null || tmux new-session -d -s {sess} '{run}')", "start the worker")

    def listing(self, hosts, quiet: bool = False) -> dict:
        """host -> {"worker": bool, "jobs": {...}} or None when the host does not answer."""
        import concurrent.futures as cf

        def one(h):
            try:
                return json.loads(self.host_cmd(h, "list", timeout=60))
            except Exception as e:  # noqa: BLE001 - a host that does not answer is reported, not fatal
                if not quiet:
                    print(f"{h}: not available ({str(e)[-200:]})", file=sys.stderr)
                return None
        with cf.ThreadPoolExecutor(8) as ex:
            return dict(zip(hosts, ex.map(one, hosts)))

    def used_hosts(self) -> list:
        return sorted({s["host"] for s in self.state["jobs"].values() if s.get("host") in self.hosts})

    # ---------------------------------------------------------------- submit
    def submit(self) -> int:
        import concurrent.futures as cf
        a, R = self.a, self.R
        names = {j.name for j in self.spec.jobs}
        for n in a.resubmit:
            if n not in names:
                sys.exit(f"error: no job {n!r} in {self.jobs_file.name}")
        if a.resubmit_failed:
            for host, lst in self.listing(self.used_hosts()).items():
                for name, k in self.key.items():
                    if lst and self.state["jobs"].get(name, {}).get("host") == host and lst["jobs"].get(k, {}).get("state") in ("failed", "killed"):
                        a.resubmit.append(name)
        todo = [j for j in self.spec.jobs if j.name not in self.state["jobs"] or j.name in a.resubmit]
        if not todo:
            self.revive()
            print("every job of the file is on a host already (status / fetch, or --resubmit)")
            return 0

        def prep(h):
            try:
                ctx = R.prepare_host(h, self.hosts_file, self.spec, self.project_dir, a, log=print)
                ctx.conn.must(f"mkdir -p {ctx.project_dir}/.hostq && cat > {ctx.project_dir}/.hostq/hostq.py", "copy the worker",
                              stdin=Path(__file__).read_text())
                gpus = ctx.conn.sh("nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits", timeout=60).stdout
                total = {int(x.split(",")[0]): float(x.split(",")[1]) / 1024 for x in gpus.splitlines() if "," in x}
                total = {i: t for i, t in total.items() if h.gpus is None or i in h.gpus}
                load = json.loads(ctx.conn.must(f"cd {ctx.project_dir} && python3 .hostq/hostq.py list", "list", timeout=60))["jobs"]
                return ctx, total, sum(1 for j in load.values() if j["state"] in ("queued", "running"))
            except Exception as e:  # noqa: BLE001
                print(f"{h.name}: left out ({str(e)[-300:]})")
                return None
        needed = {h for j in todo for h in (j.hosts or self.hosts)} & set(self.hosts)
        targets = [h for h in self.hosts_list if h.name in needed]
        with cf.ThreadPoolExecutor(8) as ex:
            ready = {h.name: r for h, r in zip(targets, ex.map(prep, targets)) if r}
        for name, (ctx, _, _) in ready.items():
            self.dirs[name] = ctx.project_dir

        # Spread the jobs: the host with the least queued + running work per GPU, among those whose GPUs can hold the job at all.
        load = {n: float(q) for n, (_, _, q) in ready.items()}
        plan: dict[str, list] = {}
        for j in todo:
            cands = [n for n, (ctx, total, _) in ready.items() if (not j.hosts or n in j.hosts) and
                     sum(1 for t in total.values() if t >= j.gb + a.headroom) >= j.gpus]
            if not cands:
                print(f"{j.name}: no ready host can run it (GPUs / memory / hosts); not submitted")
                continue
            best = min(cands, key=lambda n: (load[n] / max(len(ready[n][1]), 1), n))
            load[best] += 1
            plan.setdefault(best, []).append(j)
        for host, js in plan.items():
            print(f"{host}: {', '.join(j.name for j in js)}")
        if a.dry_run or not plan:
            return 0 if plan or not todo else 1
        for host, js in plan.items():
            ctx = ready[host][0]
            recs = []
            for j in js:
                v = {"python": ctx.python, "name": j.name, "host": host, "project_dir": ctx.project_dir}   # {gpu} is filled by the worker
                recs.append({"key": self.key[j.name], "name": j.name, "cmd": R.fill(j.cmd, **v), "gpus": j.gpus, "gb": j.gb,
                             "retries": j.retries, "expect": [R.fill(e, **v) for e in j.expect], "prefix": ctx.prefix})
            res = json.loads(self.host_cmd(host, "add", stdin=json.dumps(recs)))
            for j in js:
                if res.get(self.key[j.name]) == "queued":
                    self.set_job(j.name, {"host": host, "submitted": now()})
                else:
                    print(f"{j.name}: {res.get(self.key[j.name])} on {host}; left as it is")
            self.ensure_worker(host)
        print(f"submitted; the hosts run them from here on. `python3 tools/hostq.py fetch {self.a.jobs}` copies the results back")
        return 0

    def revive(self) -> None:
        """Start the worker again on hosts that have queued or running jobs of this file but no worker (it crashed, or the host restarted)."""
        for host, lst in self.listing(self.used_hosts()).items():
            mine = {self.key[n] for n, s in self.state["jobs"].items() if s.get("host") == host and n in self.key}
            if lst and not lst["worker"] and any(j["state"] in ("queued", "running") for k, j in lst["jobs"].items() if k in mine):
                self.ensure_worker(host)
                print(f"{host}: worker was not running; started it again")

    # ---------------------------------------------------------------- status / fetch / kill
    def collect(self, quiet: bool = False) -> dict:
        """job name -> its record on its host (or a placeholder)."""
        lists = self.listing(self.used_hosts(), quiet)
        out = {}
        for j in self.spec.jobs:
            s = self.state["jobs"].get(j.name)
            if not s:
                out[j.name] = {"state": "not submitted"}
            elif lists.get(s["host"]) is None:
                out[j.name] = {"state": "host not answering", "host": s["host"]}
            else:
                out[j.name] = {**lists[s["host"]]["jobs"].get(self.key[j.name], {"state": "missing on host"}), "host": s["host"],
                               "worker": lists[s["host"]]["worker"]}
        return out

    def status(self) -> int:
        recs = self.collect()
        for name, r in recs.items():
            extra = ""
            if r["state"] == "running":
                extra = f"GPU {','.join(map(str, r.get('assigned', [])))} since {r.get('started')} attempt {r.get('attempt')}"
            elif r["state"] in FINISHED:
                extra = f"exit {r.get('exit', '-')} at {r.get('finished')}" + (f" ({r['note']})" if r.get("note") else "")
            elif r["state"] == "queued":
                extra = ("" if r.get("worker") else "NO WORKER RUNNING (submit again) ") + (r.get("note") or "")
            fetched = " fetched" if self.state["jobs"].get(name, {}).get("fetched") else ""
            print(f"{name:<40} {r['state']:<10} {r.get('host', ''):<12} {extra}{fetched}")
        counts = {}
        for r in recs.values():
            counts[r["state"]] = counts.get(r["state"], 0) + 1
        print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
        return 0

    def fetch(self) -> int:
        R = self.R
        down: set = set()
        while True:
            self.reload()   # a submit elsewhere may have moved a job since the last poll
            recs = self.collect(quiet=self.a.wait)
            # --wait outlives network outages (e.g. a VPN that drops): report each change of reachability once, keep polling
            now_down = {r["host"] for r in recs.values() if r["state"] == "host not answering"}
            if now_down - down:
                print(f"[{now()}] connection lost: {', '.join(sorted(now_down - down))} not answering; "
                      f"{'retrying every ' + str(int(self.a.poll)) + ' s' if self.a.wait else 'try again later'}", flush=True)
            if down - now_down:
                print(f"[{now()}] connection back: {', '.join(sorted(down - now_down))}", flush=True)
            down = now_down
            for j in self.spec.jobs:
                r, s = recs[j.name], self.state["jobs"].get(j.name)
                if r["state"] not in FINISHED or (s.get("fetched") and not self.a.all):
                    continue
                host, d = s["host"], self.remote_dir(s["host"])
                conn = self.conn(host)
                v = {"name": j.name, "host": host, "project_dir": d}
                problems = []
                for p in [*(R.fill(x, **v) for x in [*self.spec.fetch, *j.fetch]), f"logs/hostq/{self.key[j.name]}.log"]:
                    res = conn.rsync("-R " + R.quote_glob(conn.remote(d + "/./" + p)) + " " + shlex.quote(str(self.project_dir) + "/"))
                    if res.returncode not in (0, 23, 24):
                        problems.append(f"fetching {p}: {res.stderr.strip()[-200:]}")
                if r["state"] == "done":
                    problems += [f"missing here: {e}" for e in (R.fill(x, **v) for x in j.expect) if not (self.project_dir / e).exists()]
                if not problems:
                    self.set_job(j.name, fetched=now())
                print(f"[{now()}] {j.name}: {r['state']}" + (f"; {'; '.join(problems)}" if problems else ", fetched"), flush=True)
            open_jobs = [n for n, r in recs.items() if r["state"] in ("queued", "running", "host not answering")]
            if not self.a.wait or not open_jobs:
                if open_jobs:
                    print(f"{len(open_jobs)} job(s) not finished yet")
                elif self.a.wait:
                    print(f"[{now()}] all {len(recs)} jobs finished", flush=True)
                return 0
            time.sleep(self.a.poll)

    def kill(self) -> int:
        names = self.a.names or list(self.key)
        by_host: dict[str, list] = {}
        for n in names:
            s = self.state["jobs"].get(n)
            if s:
                by_host.setdefault(s["host"], []).append(self.key[n])
        for host, keys in by_host.items():
            self.host_cmd(host, "kill " + " ".join(keys))
            self.ensure_worker(host)
            print(f"{host}: kill requested for {len(keys)} job(s); the worker stops them within a poll")
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["submit", "status", "fetch", "kill", "worker", "add", "list"])
    ap.add_argument("jobs", nargs="?", help="the jobs file (local commands)")
    ap.add_argument("names", nargs="*", help="kill: jobs to stop (default: all of the file); host kill: job keys")
    ap.add_argument("--hosts-file", default=None)
    ap.add_argument("--hosts", nargs="*", default=[], help="use only these hosts of the hosts file")
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--headroom", type=float, default=4.0, help="GB kept free on a GPU beyond a job's own gb")
    ap.add_argument("--dry-run", action="store_true", help="submit: show which host would get which job")
    ap.add_argument("--resubmit", nargs="*", default=[], help="submit: queue these jobs again even if they were submitted before")
    ap.add_argument("--resubmit-failed", action="store_true", help="submit: queue the failed and killed jobs again")
    ap.add_argument("--wait", action="store_true", help="fetch: keep fetching until every submitted job is finished")
    ap.add_argument("--all", action="store_true", help="fetch: fetch finished jobs again")
    ap.add_argument("--poll", type=float, default=None, help="seconds between polls (worker: 15, fetch --wait: 120)")
    # worker options (set by submit from the hosts file)
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",") if x], default=None)
    ap.add_argument("--cpu-slots", type=int, default=4)
    ap.add_argument("--ramp", type=float, default=180.0, help="worker: seconds a started job's gb stay reserved before nvidia-smi is trusted for it")
    ap.add_argument("--idle-exit", type=float, default=900.0, help="worker: exit after this many seconds with nothing queued or running")
    a = ap.parse_args(argv)

    # host side: worker / add / list, and kill given job keys instead of a jobs file
    if a.command in ("worker", "add", "list") or (a.command == "kill" and not (a.jobs or "").endswith(".json")):
        os.chdir(Path(__file__).resolve().parent.parent)    # the project dir is the parent of .hostq/
        if a.command == "worker":
            a.poll = a.poll or 15.0
            return Worker(a).run()
        if a.command == "kill":
            a.keys = [k for k in [a.jobs, *a.names] if k]
            return host_kill(a)
        return {"add": host_add, "list": host_list}[a.command](a)

    if not a.jobs:
        ap.error("the jobs file is required")
    a.poll = a.poll or 120.0
    a.recreate_env, a.strict_env, a.env_timeout = False, False, 3 * 3600   # what route.prepare_host reads
    local = Local(a)
    try:
        return {"submit": local.submit, "status": local.status, "fetch": local.fetch, "kill": local.kill}[a.command]()
    except local.R.RouterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
