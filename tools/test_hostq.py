"""Host-side tests of hostq: a worker on this machine runs CPU jobs from a temporary project dir.

Run: python3 tools/test_hostq.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent / "hostq.py"


class HostSide(unittest.TestCase):
    def setUp(self):
        self.proj = Path(tempfile.mkdtemp())
        (self.proj / ".hostq").mkdir()
        shutil.copy(SRC, self.proj / ".hostq" / "hostq.py")
        self.workers = []
        self.out = open(self.proj / "worker.out", "a")

    def tearDown(self):
        for w in self.workers:
            if w.poll() is None:
                w.kill()
        self.out.close()
        shutil.rmtree(self.proj)

    def hq(self, *args, stdin=None):
        r = subprocess.run([sys.executable, str(self.proj / ".hostq" / "hostq.py"), *args], input=stdin, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def add(self, *jobs):
        recs = [{"key": k, "name": k, "cmd": cmd, "gpus": 0, "gb": 0, "retries": retries, "expect": expect, "prefix": ""}
                for k, cmd, retries, expect in jobs]
        return json.loads(self.hq("add", stdin=json.dumps(recs)))

    def worker(self, *extra):
        w = subprocess.Popen([sys.executable, str(self.proj / ".hostq" / "hostq.py"), "worker", "--poll", "0.3", "--idle-exit", "2", *extra],
                             stdout=self.out, stderr=subprocess.STDOUT)
        self.workers.append(w)
        return w

    def jobs(self):
        return json.loads(self.hq("list"))["jobs"]

    def wait_for(self, cond, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond(self.jobs()):
                return
            time.sleep(0.2)
        self.fail(f"timed out; jobs: {json.dumps(self.jobs(), indent=1)}\nworker: {(self.proj / 'worker.out').read_text()}")

    def test_success_retry_expect_and_idle_exit(self):
        self.add(("ok", "mkdir -p results && echo hi > results/ok.txt", 0, ["results/ok.txt"]),
                 ("flaky", "exit 3", 1, []),
                 ("noout", "true", 0, ["results/never.txt"]))
        w = self.worker("--cpu-slots", "4")
        self.wait_for(lambda j: all(x["state"] in ("done", "failed") for x in j.values()))
        j = self.jobs()
        self.assertEqual(j["ok"]["state"], "done")
        self.assertEqual(j["flaky"]["state"], "failed")
        self.assertEqual(j["flaky"]["attempt"], 2)       # one retry
        self.assertEqual(j["flaky"]["exit"], 3)
        self.assertEqual(j["noout"]["state"], "failed")
        self.assertIn("expected files missing", j["noout"]["note"])
        self.assertIn("hostq: attempt 2", (self.proj / "logs/hostq/flaky.log").read_text())
        self.assertEqual(w.wait(timeout=30), 0)        # exits when idle
        self.assertIn("idle; exiting", (self.proj / "worker.out").read_text())

    def test_queue_limit_duplicates_and_kill(self):
        self.add(("a", "sleep 30", 0, []), ("b", "sleep 30", 0, []), ("c", "echo c", 0, []))
        self.assertEqual(self.add(("a", "echo again", 0, []))["a"], "already queued")
        self.worker("--cpu-slots", "1")
        self.wait_for(lambda j: j["a"]["state"] == "running" and j["b"]["state"] == "queued")
        pid = self.jobs()["a"]["pid"]
        self.hq("kill", "a", "b")
        self.wait_for(lambda j: j["a"]["state"] == "killed" and j["b"]["state"] == "killed")
        self.wait_for(lambda j: j["c"]["state"] == "done")       # the freed slot goes to the next job
        time.sleep(0.5)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_jobs_survive_worker_restart(self):
        self.add(("long", "sleep 3; echo done > out.txt", 0, ["out.txt"]))
        w = self.worker()
        self.wait_for(lambda j: j["long"]["state"] == "running")
        pid = self.jobs()["long"]["pid"]
        w.send_signal(signal.SIGKILL)                    # the worker dies (as with a killed tmux session)
        w.wait()
        os.kill(pid, 0)                                   # the job keeps running
        self.worker()                                     # a new worker adopts it
        self.wait_for(lambda j: j["long"]["state"] == "done")
        self.assertEqual((self.proj / "out.txt").read_text().strip(), "done")
        self.assertEqual(self.jobs()["long"]["attempt"], 1)

    def test_single_worker(self):
        self.add(("a", "sleep 13", 0, []))    # outlasts the ~10 s a new worker waits for the lock
        self.worker()
        time.sleep(1)
        second = self.worker()
        self.assertEqual(second.wait(timeout=30), 0)
        self.assertIn("another worker holds the lock", (self.proj / "worker.out").read_text())
        self.wait_for(lambda j: j["a"]["state"] == "done")

    def test_resubmit_after_finish(self):
        self.add(("a", "exit 1", 0, []))
        self.worker()
        self.wait_for(lambda j: j["a"]["state"] == "failed")
        self.assertEqual(self.add(("a", "true", 0, []))["a"], "queued")
        self.wait_for(lambda j: j["a"]["state"] == "done")
        self.assertEqual(self.jobs()["a"]["attempt"], 1)


class LocalState(unittest.TestCase):
    def test_stale_process_does_not_undo_a_resubmit(self):
        sys.path.insert(0, str(SRC.parent))
        import hostq
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)

        def local():
            lo = hostq.Local.__new__(hostq.Local)
            lo.state_path = d / "jobs.hostq.state.json"
            lo.reload()
            return lo
        first = local()
        first.set_job("a", {"host": "h1"})
        first.set_job("b", {"host": "h1"})
        waiting = local()                       # e.g. a `fetch --wait` that loaded the state earlier
        local().set_job("a", {"host": "h2"})    # a resubmit moves job a to another host
        waiting.set_job("b", fetched="now")     # the stale process records a fetch of job b
        state = json.loads((d / "jobs.hostq.state.json").read_text())["jobs"]
        self.assertEqual(state["a"], {"host": "h2"})
        self.assertEqual(state["b"], {"host": "h1", "fetched": "now"})

    def test_fetch_wait_rides_out_an_outage(self):
        import contextlib
        import io
        import types
        sys.path.insert(0, str(SRC.parent))
        import hostq
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        lo = hostq.Local.__new__(hostq.Local)
        lo.state_path = d / "jobs.hostq.state.json"
        lo.reload()
        lo.set_job("a", {"host": "h1", "fetched": "earlier"})     # already fetched: no rsync needed when it is done
        lo.a = types.SimpleNamespace(wait=True, poll=0, all=False)
        lo.R = None
        lo.spec = types.SimpleNamespace(jobs=[types.SimpleNamespace(name="a")])
        polls = iter([{"state": "running"}, {"state": "host not answering"}, {"state": "host not answering"},
                      {"state": "running"}, {"state": "done"}])
        lo.collect = lambda quiet=False: {"a": {**next(polls), "host": "h1"}}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(lo.fetch(), 0)
        lines = out.getvalue().splitlines()
        self.assertEqual(sum("connection lost: h1" in x for x in lines), 1)      # reported once, not every poll
        self.assertEqual(sum("connection back: h1" in x for x in lines), 1)
        self.assertIn("all 1 jobs finished", lines[-1])

    def test_place_holds_back_unused_gb_of_running_jobs(self):
        import types
        sys.path.insert(0, str(SRC.parent))
        import hostq
        w = hostq.Worker(types.SimpleNamespace(gpus=None, per_gpu=4, headroom=4.0, ramp=180.0, cpu_slots=1))
        started = []
        w.start = lambda key, j, gpus: started.append((key, gpus))
        old = time.time() - 3600       # long past the ramp: before the fix, its gb was no longer reserved
        jobs = {"run1": {"state": "running", "assigned": [0], "gb": 18, "pid": 111, "started_ts": old},
                "run2": {"state": "running", "assigned": [0], "gb": 18, "pid": 222, "started_ts": old},
                "new": {"state": "queued", "gb": 18, "order": 1}}
        orig = hostq.gpu_state, hostq.job_gpu_usage
        self.addCleanup(lambda: (setattr(hostq, "gpu_state", orig[0]), setattr(hostq, "job_gpu_usage", orig[1])))
        hostq.gpu_state = lambda: {0: (47.0 - 9 - 9, 47.0)}      # both use 9 GB so far, on their way to 14-18
        hostq.job_gpu_usage = lambda: {(111, 0): 9.0, (222, 0): 9.0}
        w.place(jobs)
        self.assertEqual(started, [])   # 29 GB free, but 18 GB are still owed to the running jobs: 11 < 18 + 4
        hostq.job_gpu_usage = lambda: {(111, 0): 18.0, (222, 0): 18.0}
        hostq.gpu_state = lambda: {0: (47.0 - 36, 47.0)}
        w.place(jobs)
        self.assertEqual(started, [])   # they use all of their gb: 11 GB free < 22
        hostq.gpu_state = lambda: {0: (47.0, 47.0)}
        hostq.job_gpu_usage = lambda: None                        # nvidia-smi can't tell: fall back to the ramp rule
        w.place(jobs)
        self.assertEqual(started, [("new", [0])])


if __name__ == "__main__":
    unittest.main(verbosity=2)
