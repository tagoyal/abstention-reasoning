"""GPU scheduling for launch_train_local.sh; training stays in run_method_e.sh."""

import argparse
import csv
import fcntl
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def query_gpu(*args):
    result = subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return [
        [value.strip() for value in row]
        for row in csv.reader(result.stdout.splitlines()) if row
    ]


def reserve_gpus(pool, count, lock_root, max_used_memory_mib):
    handles = []
    try:
        for uuid in pool:
            handle = (lock_root / f"{uuid}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            handles.append((uuid, handle))
            processes = query_gpu(f"--id={uuid}", "--query-compute-apps=pid")
            memory = query_gpu(f"--id={uuid}", "--query-gpu=memory.used")
            if processes or not memory or int(memory[0][0]) > max_used_memory_mib:
                handles.pop()[1].close()
                continue
            if len(handles) == count:
                return handles
    except BaseException:
        for _, handle in handles:
            handle.close()
        raise
    for _, handle in handles:
        handle.close()
    return []


def stop_runs(running):
    # Each runner has its own process group, including Python, Ray and vLLM.
    for run in running:
        try:
            os.killpg(run["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while running and time.monotonic() < deadline:
        if all(run["process"].poll() is not None for run in running):
            break
        time.sleep(0.1)
    for run in running:
        try:
            os.killpg(run["process"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        run["process"].wait()
        (run["directory"] / "exit_code").write_text(
            f'{run["process"].returncode}\n'
        )
        run["log"].close()
        for _, handle in run["reservation"]:
            handle.close()


def run_batch(args):
    if not args.job:
        raise ValueError("No experiment combinations configured")
    if args.poll_seconds <= 0 or args.max_used_memory_mib < 0:
        raise ValueError("Poll interval must be positive and memory threshold nonnegative")
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("GPU_POOL must contain unique GPU indices")
    inventory = dict(query_gpu("--query-gpu=index,uuid"))
    missing = set(args.gpus) - inventory.keys()
    if missing:
        raise ValueError(f"GPU indices not visible in this pod: {sorted(missing)}")
    pool = [inventory[index] for index in args.gpus]
    for job in args.job:
        if not 1 <= int(job[6]) <= len(pool):
            raise ValueError(f"{job[1]} requests {job[6]} GPUs; pool has {len(pool)}")
        for value in (job[0], job[1], job[3]):
            if not value or Path(value).name != value or value in (".", ".."):
                raise ValueError(f"Invalid task/model/data name: {value!r}")

    args.log_root.mkdir(parents=True, exist_ok=True)
    batch = Path(tempfile.mkdtemp(
        prefix=time.strftime("batch-%Y%m%d-%H%M%S-"), dir=args.log_root,
    ))
    # Shared across launcher instances in this pod; never delete lock files.
    lock_root = Path(tempfile.gettempdir()) / "abstention-local-gpu-locks"
    lock_root.mkdir(exist_ok=True)
    pending = list(enumerate(args.job, start=1))
    running = []
    failed = False
    print(f"Batch logs: {batch}", flush=True)
    try:
        while pending or running:
            for run in running[:]:
                code = run["process"].poll()
                if code is None:
                    continue
                (run["directory"] / "exit_code").write_text(f"{code}\n")
                run["log"].close()
                for _, handle in run["reservation"]:
                    handle.close()
                running.remove(run)
                failed |= code != 0
                print(f'Finished {run["directory"].name}: exit={code}', flush=True)

            for number, job in pending[:]:
                reservation = reserve_gpus(
                    pool, int(job[6]), lock_root, args.max_used_memory_mib,
                )
                if not reservation:
                    continue
                name = f"{batch.name}-{number}-{job[0]}-{job[3]}-{job[1]}"
                directory = batch / name
                log = None
                try:
                    directory.mkdir()
                    command = [
                        "bash", str(REPO_ROOT / "launch_scripts/run_method_e.sh"),
                        *job[:6], "--experiment-name", name, *args.training_args,
                    ]
                    environment = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(uuid for uuid, _ in reservation),
                        # Do not attach simultaneous runs to one Ray cluster.
                        "RAY_ADDRESS": "local",
                        "PYTHONUNBUFFERED": "1",
                    }
                    (directory / "command.sh").write_text(
                        f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']} "
                        f"RAY_ADDRESS=local {shlex.join(command)}\n"
                    )
                    log = (directory / "train.log").open("w")
                    process = subprocess.Popen(
                        command, cwd=REPO_ROOT, env=environment,
                        stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    running.append({
                        "process": process, "directory": directory,
                        "log": log, "reservation": reservation,
                    })
                except BaseException:
                    if log is not None:
                        log.close()
                    for _, handle in reservation:
                        handle.close()
                    raise
                pending.remove((number, job))
                print(
                    f"Started {name}: GPUs={environment['CUDA_VISIBLE_DEVICES']} "
                    f"pid={process.pid} log={directory / 'train.log'}",
                    flush=True,
                )
            if pending or running:
                time.sleep(args.poll_seconds)
    finally:
        stop_runs(running)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--job", nargs=7, action="append")
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--max-used-memory-mib", type=int, default=512)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.training_args[:1] == ["--"]:
        args.training_args = args.training_args[1:]
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        return run_batch(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Local training launcher: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
