#!/usr/bin/env python3
r"""
chain_method_a.py -- adaptive verifier/solver chaining for method_a.

`evaluate --method method_ac` (the fixed-hint solver) and
`evaluate --method method_a` (the verifier) used to be run independently at
every hint level for every problem, then stitched together after the fact by
`combine_verifier_eval`: that means the (expensive) solver runs once per
problem *per hint level*, even though only one of those levels is ever used.

This script instead runs the verifier live, round by round, so the solver
only ever runs once per rollout, at the hint level that rollout actually
stopped at:

    Round 0 (no hints): run the verifier on every (problem, sample) pair.
        Verifier outputs <answer>1</answer> -> that rollout is done; it goes
        into ready_for_solver at hint_level=0.
        Verifier outputs <answer>0</answer> -> add one more hint and carry
        the rollout into round 1.
    Round 1 (1 hint): run the verifier again, but only on the rollouts still
        carried over. Same accept/continue split as round 0.
    ...continues until a rollout is accepted (predicts 1), or the verifier
    rejects it even at its own last possible round -- every available hint
    shown, nothing left to escalate to (each problem can have a different
    number of available hints, so rollouts drop out of the loop at different
    rounds; the verifier IS still queried at that final round, it just can't
    lead to a further round if it says 0).

Once every rollout has a hint level (accepted, or forced despite rejection at
its final round because there's nowhere further to go), all of
them are rendered with the method_ac (solver) template and run through the
solver model exactly once, and checked against the ground truth with the
task's own `check_correctness`.

Each round hands vLLM the *whole* round's prompt list in one call and lets it
do its own continuous batching/scheduling (sync `Generator.generate`, or
`AsyncGenerator.generate_async` with `--async`) -- there's no manual
--batch-size chunking, since pre-slicing into fixed-size chunks and calling
generate() once per chunk only stops vLLM from co-scheduling requests across
chunks.

Progress is checkpointed to `<output>.checkpoint.json` after every verifier
round, so a crash (or Ctrl-C) doesn't have to redo already-decided rounds --
rerunning the same command picks up from the last completed round. The
checkpoint is removed once the run finishes and the real output is written.

method_a's and method_c's fixed-hint solver is always the shared "method_ac"
generator/template -- only the verifier differs between the two methods (see
chain_method_c.py for the method_c verifier, which checks a candidate
*solution* instead of asking "can you solve this given the hints so far?").

Usage:
    python chain_method_a.py \
        --task countdown \
        --eval-dataset artifacts/countdown/problems/eval.json \
        --verifier-model artifacts/countdown/models/method_a_predictors/qwen2.5-1.5b/model \
        --solver-model artifacts/countdown/models/method_ac_models/qwen2.5-1.5b/model \
        --num-samples 1 \
        --max-hints 5 \
        --async \
        --output artifacts/countdown/models/method_a_predictors/qwen2.5-1.5b/evals/eval__chained_1s.json
"""

import argparse
import gc
from datetime import datetime
from pathlib import Path

from pipeline.commands.inference import compute_hint_metrics, count_truncation, format_hint_metrics
from pipeline.core.generator import AsyncGenerator, Generator, GenerationConfig
from pipeline.core.io import load_json, save_json
from pipeline.core.method import Method
from pipeline.core.utils import extract_answer
from pipeline.tasks import get_task

# The fixed-hint solver template/model group is shared by method_a and
# method_c -- only the verifier differs. Hardcoded rather than a CLI flag so
# the two chain_method_*.py scripts can't accidentally point at mismatched
# solver conventions.
SOLVER_METHOD_NAME = "method_ac"
VERIFIER_METHOD_NAME = "method_a"


def get_hints_list(primitive: dict) -> list[str]:
    """Available hints for a primitive, countdown or competition_math.

    Mirrors pipeline/commands/data.py's extraction exactly: countdown carries
    a flat `hint_exprs` list (each entry already the full cumulative partial
    expression up to that point); competition_math carries a `prefix_hints`
    dict of up to 6 standalone steps (`hint_1`..`hint_6`).
    """
    hints_list = list(primitive.get("hint_exprs", []))
    if not hints_list:
        prefix_hints = primitive.get("prefix_hints", {})
        for i in range(1, 7):
            key = f"hint_{i}"
            if key in prefix_hints:
                hints_list.append(prefix_hints[key])
    return hints_list


def hint_sequence_for_level(task_name: str, hints_list: list[str], level: int) -> str:
    """Render the `{hint_sequence}` placeholder text for a given hint level.

    Matches pipeline/commands/data.py exactly. Level 0 has no partial
    solution. competition_math's hints are independent steps, so the partial
    solution is every step up to `level` concatenated; countdown's hint_exprs
    are each already the full cumulative expression, so only the single entry
    at `level` is shown.
    """
    if level == 0:
        return "No partial solution"
    if task_name == "competition_math":
        return "\n".join(hints_list[:level])
    return hints_list[level - 1]


class RoundGenerator:
    """One loaded model, generating across one or more rounds -- sync or async.

    vLLM's LLM.generate() (sync) and AsyncLLMEngine (async, via
    AsyncGenerator.generate_async) both already do their own continuous
    batching over whatever prompt list they're handed, so `.run()` always
    submits a full round in one call rather than chunking it into a
    --batch-size.

    In async mode, a single event loop is created lazily on the first `.run()`
    call and reused for every subsequent round on this instance, rather than
    calling `asyncio.run()` fresh each time. AsyncLLMEngine spawns its
    background output-handler task bound to whichever loop is running when the
    engine is first touched; calling `asyncio.run()` again for round 2 tears
    down that loop and starts a brand new one, orphaning the engine's
    background task -- the next `engine.generate()` call then dies with
    `EngineDeadError` ("EngineCore proc ... died unexpectedly"). Reusing one
    loop for the engine's whole lifetime (mirrored by closing it in
    `.close()`) avoids that.
    """

    def __init__(self, config: GenerationConfig, use_async: bool):
        self.use_async = use_async
        self._gen = AsyncGenerator(config) if use_async else Generator(config)
        self._loop = None

    def run(self, prompts: list[list[dict]]) -> list[list[dict]]:
        if not prompts:
            return []
        if self.use_async:
            if self._loop is None:
                import asyncio
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(self._gen.generate_async(prompts, num_samples=1))
        return self._gen.generate(prompts)

    def close(self) -> None:
        """Release the loaded model and free its GPU memory."""
        if self.use_async:
            if self._loop is not None:
                self._loop.run_until_complete(self._async_close())
                self._loop.close()
                self._loop = None
            else:
                self._gen.close()
        else:
            self._gen._model = None
        del self._gen
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _async_close(self) -> None:
        # AsyncGenerator.close() is sync, but engine.shutdown() underneath
        # tears down asyncio background tasks -- run it while our own loop
        # (the one those tasks are bound to) is still alive and running.
        self._gen.close()


class ChainItem:
    """One (source_index, trial) rollout as it moves through verifier rounds."""

    __slots__ = (
        "source_index", "trial", "primitive", "ground_truth", "hints_list",
        "local_max_hints", "resolved", "forced", "selected_level", "trace",
    )

    def __init__(self, source_index, trial, primitive, ground_truth, hints_list, local_max_hints):
        self.source_index = source_index
        self.trial = trial
        self.primitive = primitive
        self.ground_truth = ground_truth
        self.hints_list = hints_list
        self.local_max_hints = local_max_hints
        self.resolved = False
        # Set alongside `resolved` once a decision is made: True only when
        # every hint was used up without the verifier ever accepting.
        self.forced = False
        # Fallback if the verifier never accepts: use every hint this
        # problem has. Overwritten the moment a round predicts 1.
        self.selected_level = local_max_hints
        self.trace = []


def build_items(task, primitives: list[dict], num_samples: int, max_hints: int) -> list[ChainItem]:
    items = []
    for primitive in primitives:
        hints_list = get_hints_list(primitive)
        local_max_hints = min(max_hints, len(hints_list))
        ground_truth = task.get_ground_truth(primitive)
        for t in range(num_samples):
            items.append(ChainItem(
                source_index=primitive["index"],
                trial=t,
                primitive=primitive,
                ground_truth=ground_truth,
                hints_list=hints_list,
                local_max_hints=local_max_hints,
            ))
    return items


# =============================================================================
# Checkpointing -- round-by-round progress, so a crash mid-run doesn't have to
# redo already-decided rounds. Only the verifier chain is checkpointed: it's
# the part that runs many rounds; the solver runs once, as a single call.
# =============================================================================

def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.checkpoint.json")


def save_checkpoint(path: Path, items: list[ChainItem]) -> None:
    data = [
        {
            "source_index": it.source_index,
            "trial": it.trial,
            "resolved": it.resolved,
            "forced": it.forced,
            "selected_level": it.selected_level,
            "trace": it.trace,
        }
        for it in items
    ]
    save_json(path, data)


def restore_from_checkpoint(items: list[ChainItem], checkpoint_data: list[dict]) -> int:
    by_key = {(d["source_index"], d["trial"]): d for d in checkpoint_data}
    restored = 0
    for it in items:
        d = by_key.get((it.source_index, it.trial))
        if d is None:
            continue
        it.resolved = d["resolved"]
        it.forced = d.get("forced", False)
        it.selected_level = d["selected_level"]
        it.trace = d["trace"]
        restored += 1
    return restored


def run_verifier_rounds(
    task,
    task_name: str,
    verifier: RoundGenerator,
    verifier_template: str,
    items: list[ChainItem],
    checkpoint_path: Path,
) -> None:
    """Adaptively resolve every item's hint level via live verifier rounds.

    Round `level` only includes items that are still unresolved, have not yet
    gone past their own hint budget (local_max_hints >= level -- note the
    verifier IS queried at level == local_max_hints, the "every available
    hint shown" level, since there's still a real accept/reject decision to
    make there; it's simply the last round an item can ever be escalated
    past), and -- so a resumed run doesn't repeat rounds a previous run
    already decided -- whose trace shows this is exactly their next round
    (len(trace) == level).
    """
    max_level = max((it.local_max_hints for it in items), default=0)

    for level in range(max_level + 1):
        round_items = [
            it for it in items
            if not it.resolved and it.local_max_hints >= level and len(it.trace) == level
        ]
        if not round_items:
            continue

        prompts = []
        for it in round_items:
            hint_seq = hint_sequence_for_level(task_name, it.hints_list, level)
            rendered = verifier_template.replace("{hint_sequence}", hint_seq)
            prompts.append(task.format_prompt(it.primitive, rendered, include_assistant_prefix=False))

        print(f"[verifier] round {level}: {len(prompts)} prompts "
              f"({sum(1 for i in items if i.resolved)} resolved so far)")
        results = verifier.run(prompts)

        n_unparseable = 0
        for it, samples in zip(round_items, results):
            gen_text = samples[0]["text"]
            answer = extract_answer(gen_text)
            if answer not in ("0", "1"):
                # Malformed verifier output ("can't solve yet") -- treat like
                # a "0": add a hint and keep going rather than aborting a long
                # run over one bad sample.
                n_unparseable += 1
                prediction = 0
            else:
                prediction = int(answer)
            it.trace.append({"hint_level": level, "generation": gen_text, "prediction": prediction})
            if prediction == 1:
                it.resolved = True
                it.selected_level = level

        if n_unparseable:
            print(f"[verifier] round {level}: {n_unparseable}/{len(round_items)} "
                  f"outputs were not <answer>0/1</answer>; counted as 0")

        save_checkpoint(checkpoint_path, items)

    # Anything still unresolved was rejected by the verifier even at its own
    # last available hint level (level == local_max_hints, just queried above)
    # -- there's nowhere further to escalate to, so it's forced to
    # local_max_hints (already the __init__ default) despite the rejection.
    n_forced = 0
    for it in items:
        if not it.resolved:
            it.resolved = True
            it.forced = True
            n_forced += 1
    if n_forced:
        print(f"[verifier] {n_forced} rollouts were rejected at every hint level, "
              f"including their max, and are forced to their max hint level")
    save_checkpoint(checkpoint_path, items)


def run_solver(task, task_name: str, solver: RoundGenerator, solver_template: str, items: list[ChainItem]):
    prompts = []
    for it in items:
        hint_seq = hint_sequence_for_level(task_name, it.hints_list, it.selected_level)
        rendered = solver_template.replace("{hint_sequence}", hint_seq)
        prompts.append(task.format_prompt(it.primitive, rendered, include_assistant_prefix=True))

    print(f"[solver] {len(prompts)} prompts, one per rollout at its selected hint level")
    return solver.run(prompts)


def build_details(task, items: list[ChainItem], solver_results: list[list[dict]]) -> list[dict]:
    details = []
    for it, samples in zip(items, solver_results):
        gen = samples[0]
        primitive_for_check = {"index": it.source_index, **it.ground_truth}
        is_correct, meta = task.check_correctness(primitive_for_check, gen["text"])

        detail = {
            "index": it.source_index,
            "source_index": it.source_index,
            "trial_index": it.trial,
            "variant": it.ground_truth.get("variant", "unknown"),
            "level": it.ground_truth.get("level", "unknown"),
            "ground_truth": it.ground_truth,
            "num_hints": it.selected_level,
            "selected_hint_level": it.selected_level,
            "forced_at_max_hints": it.forced,
            "correct": is_correct,
            "generation": gen["text"],
            "predicted_answer": meta.get("predicted_answer"),
            "error": meta.get("error"),
            "finish_reason": gen.get("finish_reason", "unknown"),
            "token_count": gen.get("token_count", 0),
            "verifier_trace": it.trace,
        }
        details.append(detail)
    return details


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="Task name (countdown, competition_math)")
    p.add_argument("--eval-dataset", required=True, type=Path,
        help="Path to the raw eval partition (artifacts/{task}/problems/eval.json)")
    p.add_argument("--verifier-model", required=True, help="Verifier (method_a) model path")
    p.add_argument("--solver-model", required=True, help="Solver (method_ac) model path")
    p.add_argument("--output", required=True, type=Path, help="Output path for combined results")
    p.add_argument("--num-samples", type=int, default=1,
        help="Independent rollouts per problem (default: 1)")
    p.add_argument("--max-hints", type=int, default=5,
        help="Cap on hint rounds; a problem with fewer available hints stops sooner (default: 5)")
    p.add_argument("--async", dest="use_async", action="store_true",
        help="Use vLLM's AsyncLLMEngine instead of the sync engine (optimal throughput; "
             "required for --data-parallel-size > 1)")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--data-parallel-size", type=int, default=1,
        help="Independent model replicas, prompts sharded across them. Requires --async.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    p.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    p.add_argument("--verifier-max-new-tokens", type=int, default=16,
        help="Verifier output is just <answer>0</answer>/<answer>1</answer> (default: 16)")
    p.add_argument("--verifier-temperature", type=float, default=1.0)
    p.add_argument("--verifier-top-p", type=float, default=1.0)
    p.add_argument("--solver-max-new-tokens", type=int, default=2048)
    p.add_argument("--solver-temperature", type=float, default=1.0)
    p.add_argument("--solver-top-p", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true", help="Print sample prompts during generation")
    return p


def main():
    args = build_arg_parser().parse_args()

    task = get_task(args.task)
    verifier_method = Method.load(VERIFIER_METHOD_NAME, args.task)
    solver_method = Method.load(SOLVER_METHOD_NAME, args.task)
    verifier_template = verifier_method.load_template(args.task, "eval")
    solver_template = solver_method.load_template(args.task, "eval")

    primitives = load_json(args.eval_dataset)
    print(f"Loaded {len(primitives)} primitives from {args.eval_dataset}")

    items = build_items(task, primitives, args.num_samples, args.max_hints)
    print(f"{len(items)} total rollouts ({len(primitives)} problems x {args.num_samples} samples)")

    checkpoint_path = checkpoint_path_for(args.output)
    if checkpoint_path.exists():
        restored = restore_from_checkpoint(items, load_json(checkpoint_path))
        print(f"Resuming from checkpoint {checkpoint_path}: {restored} rollouts restored "
              f"({sum(1 for it in items if it.resolved)} already resolved)")

    verifier_config = GenerationConfig(
        model_name=args.verifier_model,
        max_new_tokens=args.verifier_max_new_tokens,
        temperature=args.verifier_temperature,
        top_p=args.verifier_top_p,
        num_samples=1,  # each rollout `t` is already one independent sample
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose,
        seed=args.seed,
    )
    verifier = RoundGenerator(verifier_config, args.use_async)
    run_verifier_rounds(task, args.task, verifier, verifier_template, items, checkpoint_path)
    verifier.close()

    solver_config = GenerationConfig(
        model_name=args.solver_model,
        max_new_tokens=args.solver_max_new_tokens,
        temperature=args.solver_temperature,
        top_p=args.solver_top_p,
        num_samples=1,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose,
        seed=args.seed,
    )
    solver = RoundGenerator(solver_config, args.use_async)
    solver_results = run_solver(task, args.task, solver, solver_template, items)
    solver.close()

    details = build_details(task, items, solver_results)

    metrics = task.compute_metrics(details)
    metrics["truncation"] = count_truncation(details)
    hint_metrics = compute_hint_metrics(details)

    results = {
        "verifier_model": args.verifier_model,
        "solver_model": args.solver_model,
        "eval_dataset": str(args.eval_dataset),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_samples": args.num_samples,
            "max_hints": args.max_hints,
            "verifier": {
                "max_new_tokens": args.verifier_max_new_tokens,
                "temperature": args.verifier_temperature,
                "top_p": args.verifier_top_p,
            },
            "solver": {
                "max_new_tokens": args.solver_max_new_tokens,
                "temperature": args.solver_temperature,
                "top_p": args.solver_top_p,
            },
        },
        "metrics": metrics,
        "details": details,
        "hint_metrics": hint_metrics,
    }

    print(task.format_metrics(metrics, "method_a_chain"))
    print(format_hint_metrics(hint_metrics, details))

    save_json(args.output, results)
    print(f"\nSaved chained evaluation to {args.output}")

    # The run finished cleanly -- the checkpoint has served its purpose.
    checkpoint_path.unlink(missing_ok=True)

    return args.output


if __name__ == "__main__":
    main()
