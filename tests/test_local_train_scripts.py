import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "launch_scripts"


class LocalTrainScriptTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.scripts = self.root / "launch_scripts"
        self.scripts.mkdir()
        for name in (
            "launch_train_local.sh", "local_train_scheduler.py", "run_method_e.sh",
        ):
            shutil.copyfile(SCRIPTS / name, self.scripts / name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.uuid_prefix = f"GPU-test-{self.root.name}"
        for index in range(2):
            lock = Path(tempfile.gettempdir()) / "abstention-local-gpu-locks" / f"{self.uuid_prefix}-{index}.lock"
            self.addCleanup(lock.unlink, missing_ok=True)
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "TEST_ROOT": str(self.root),
            "UUID_PREFIX": self.uuid_prefix,
        }
        self.stub("nvidia-smi", """
import fcntl, json, os, sys
from pathlib import Path
root = Path(os.environ["TEST_ROOT"])
prefix = os.environ["UUID_PREFIX"]
with (root / "queries").open("a+") as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    stream.seek(0)
    previous = stream.readlines()
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if "--query-gpu=index,uuid" in sys.argv:
    print(f"0, {prefix}-0\\n1, {prefix}-1")
elif "--query-compute-apps=pid" in sys.argv:
    if os.environ.get("TEST_BUSY") and len(previous) < 3:
        print("12345")
elif "--query-gpu=memory.used" in sys.argv:
    print("1024" if os.environ.get("TEST_BUSY") and len(previous) < 6 else "0")
else:
    raise SystemExit("Unexpected nvidia-smi call")
""")
        self.stub("python", """
import fcntl, json, os, sys, time
from pathlib import Path
root = Path(os.environ["TEST_ROOT"])
gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
def event(kind):
    with (root / "events").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({
            "kind": kind, "gpus": gpus, "args": sys.argv[1:],
            "ray": os.environ.get("RAY_ADDRESS"),
        }) + "\\n")
for gpu in gpus:
    (root / gpu).touch(exist_ok=False)
event("start")
time.sleep(0.6)
for gpu in gpus:
    (root / gpu).unlink()
event("end")
raise SystemExit(7 if os.environ.get("TEST_FAIL") and
    sys.argv[sys.argv.index("--data-name") + 1] == "math_o1" else 0)
""")
        # The launcher uses the real interpreter; only training's `python` is stubbed.
        launcher = self.scripts / "launch_train_local.sh"
        text = launcher.read_text().replace(
            'exec python "$SCRIPT_DIR/local_train_scheduler.py"',
            f'exec "{sys.executable}" "$SCRIPT_DIR/local_train_scheduler.py"',
        )
        text = text.replace("POLL_SECONDS=5", "POLL_SECONDS=0.01")
        text = text.replace("GPU_POOL=(0 1 2 3)", "GPU_POOL=(0 1)")
        launcher.write_text(text)

    def stub(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n{body}")
        path.chmod(0o755)

    def configure(self, models, data_names):
        launcher = self.scripts / "launch_train_local.sh"
        text = launcher.read_text().replace(
            "MODELS=(qwen2.5-1.5b qwen2.5-3b qwen3-4b)",
            f"MODELS=({' '.join(models)})",
        ).replace("DATA_NAMES=(math_o3)", f"DATA_NAMES=({' '.join(data_names)})")
        launcher.write_text(text)

    def launch(self, *flags, **env):
        return subprocess.run(
            ["bash", str(self.scripts / "launch_train_local.sh"), *flags],
            cwd=self.root, env={**self.env, **env},
            capture_output=True, text=True, timeout=30,
        )

    def events(self):
        return [
            json.loads(line)
            for line in (self.root / "events").read_text().splitlines()
        ]

    def test_batch_runs_concurrently_and_reserves_multiple_gpus(self):
        self.configure(["qwen2.5-1.5b", "qwen3-8b"], ["math_o1", "math_o3"])
        result = self.launch("--n-samples", "4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        active = set()
        max_active = 0
        active_jobs = 0
        max_active_jobs = 0
        starts = []
        for event in self.events():
            if event["kind"] == "start":
                self.assertTrue(active.isdisjoint(event["gpus"]))
                active.update(event["gpus"])
                active_jobs += 1
                max_active_jobs = max(max_active_jobs, active_jobs)
                starts.append(event)
                max_active = max(max_active, len(active))
            else:
                active.difference_update(event["gpus"])
                active_jobs -= 1
        self.assertEqual(len(starts), 4)
        self.assertEqual(max_active, 2)
        self.assertEqual(max_active_jobs, 2)
        self.assertEqual(len(active), 0)
        names = []
        for event in starts:
            args = event["args"]
            self.assertEqual(args[:3], ["-m", "pipeline", "train_rl"])
            self.assertEqual(args[args.index("--method") + 1], "method_e")
            model = args[args.index("--run-id") + 1]
            self.assertEqual(len(event["gpus"]), 2 if model == "qwen3-8b" else 1)
            self.assertEqual(event["ray"], "local")
            self.assertEqual(args[-3:], ["--interleave-rl", "--n-samples", "4"])
            names.append(args[args.index("--experiment-name") + 1])
        self.assertEqual(len(set(names)), 4)
        directories = list((self.root / "logs/local_train").glob("*/*"))
        self.assertEqual(len(directories), 4)
        self.assertTrue(all((path / "exit_code").read_text() == "0\n" for path in directories))

    def test_waits_for_external_processes_and_memory_to_clear(self):
        self.configure(["qwen3-8b"], ["math_o3"])
        result = self.launch(TEST_BUSY="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        queries = [
            json.loads(line)
            for line in (self.root / "queries").read_text().splitlines()
        ]
        process_checks = [q for q in queries if "--query-compute-apps=pid" in q]
        self.assertGreater(len(process_checks), 2)
        self.assertEqual(len(self.events()), 2)

    def test_gpu_locks_prevent_overlap_across_launchers(self):
        self.configure(["qwen3-8b"], ["math_o3"])
        processes = [
            subprocess.Popen(
                ["bash", str(self.scripts / "launch_train_local.sh")],
                cwd=self.root, env=self.env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            ) for _ in range(2)
        ]
        try:
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stdout + stderr)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=15)
        self.assertEqual(
            [event["kind"] for event in self.events()],
            ["start", "end", "start", "end"],
        )

    def test_failure_is_reported_without_skipping_other_jobs(self):
        self.configure(["qwen3-8b"], ["math_o1", "math_o3"])
        result = self.launch(TEST_FAIL="1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(len(self.events()), 4)
        codes = sorted(
            path.read_text().strip()
            for path in (self.root / "logs/local_train").glob("*/*/exit_code")
        )
        self.assertEqual(codes, ["0", "7"])

    def test_invalid_gpu_pool_fails_before_training(self):
        launcher = self.scripts / "launch_train_local.sh"
        launcher.write_text(launcher.read_text().replace("GPU_POOL=(0 1)", "GPU_POOL=(4)"))
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU indices not visible", result.stderr)
        self.assertFalse((self.root / "events").exists())


if __name__ == "__main__":
    unittest.main()
