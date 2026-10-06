"""
CLI entrypoint for the pipeline.

Usage:
    python -m pipeline <command> [options]

Commands:
    list_tasks                  List available tasks
    list_methods                List available methods for a task
    create_primitives           Generate raw puzzle data
    create_prompts              Create prompts from primitives
    create_ood_prompts          Create eval prompts from an OOD math benchmark
    generate                    Run model on prompts to create dataset
    train_sft                   Train SFT model on generated dataset
    train_classifier            Train verifier/abstain classifier on generate_tree output
    train_rl                    Train RL model using verl (GRPO)
    convert_checkpoint          Convert FSDP/Megatron checkpoint to HuggingFace format
    evaluate                    Evaluate model and compute metrics

Examples:
    # List available tasks and methods
    python -m pipeline list_tasks
    python -m pipeline list_methods --task countdown

    # Full workflow with --method (auto-derived paths)
    python -m pipeline create_primitives --task countdown --num-puzzles 5000
    python -m pipeline create_prompts --task countdown --method baseline
    python -m pipeline generate --task countdown --method baseline --model Qwen/Qwen3-14B
    python -m pipeline train_sft --task countdown --method baseline --base-model Qwen/Qwen2.5-3B --run-id 3b
    python -m pipeline train_rl --task countdown --method baseline --run-id 3b
    python -m pipeline evaluate --task countdown --method baseline --model sft --run-id 3b
    python -m pipeline evaluate --task countdown --method baseline --model rl --run-id 3b

    # Artifacts are organized by stage, not by method: prompts, datasets and
    # models each share one directory per task, and a method's files are told
    # apart by its name. Data (prompts/datasets) and models live in separate
    # roots so that data/ can be committed to git while models/ is pushed to
    # a model hub. --data-name/--models-name default to the task name and can
    # be overridden independently (e.g. --task math --data-name math_o1).
    # data/math/problems/primitives.json
    # data/math/problems_with_format/{split}__{method}.json
    # data/math/sft_datasets/{split}__{method}.json
    # models/math/{method}_{sft,rl}/{run_id}/{model,evals,...}
"""

# IMPORTANT: Set VLLM's multiprocessing method before any imports.
# This prevents "Cannot re-initialize CUDA in forked subprocess" errors
# when using VLLM's AsyncLLMEngine which spawns worker processes.
# VLLM reads this env var rather than Python's multiprocessing.set_start_method().
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
from pathlib import Path

from pipeline.tasks import list_tasks
from pipeline.core.method import Method
from pipeline import commands


def cmd_list_tasks(args):
    """List available tasks."""
    tasks = list_tasks()
    print("Available tasks:")
    for task in tasks:
        print(f"  - {task}")


def cmd_list_methods(args):
    """List available methods for a task."""
    names = Method.list_methods(args.task)
    if not names:
        print(f"No methods found for task '{args.task}'")
        print(f"Create method configs in: pipeline/configs/methods/{args.task}/")
        return

    print(f"Available methods for '{args.task}':")
    for name in names:
        print(f"  - {name}")


def cmd_create_primitives(args):
    """Create primitives."""
    output_path = Path(args.output) if args.output else None
    # Task-specific options are forwarded only when explicitly given, so that
    # tasks which don't accept them are not handed an unexpected keyword.
    task_options = {}
    if args.tracer is not None:
        task_options["tracer"] = args.tracer
    commands.create_primitives(
        task_name=args.task,
        output_path=output_path,
        num_puzzles=args.num_puzzles,
        seed=args.seed,
        data_name=args.data_name,
        **task_options,
    )


def cmd_create_partitions(args):
    """Split primitives into per-split problem files."""
    commands.create_partitions(
        task_name=args.task,
        primitives_path=Path(args.primitives) if args.primitives else None,
        output_dir=Path(args.output) if args.output else None,
        seed=args.seed,
        data_name=args.data_name,
    )


def cmd_create_prompts(args):
    """Create prompts."""
    output_dir = Path(args.output) if args.output else None
    primitives_path = Path(args.primitives) if args.primitives else None

    commands.create_prompts(
        task_name=args.task,
        method_name=args.method,
        primitives_path=primitives_path,
        output_dir=output_dir,
        split_name=args.split,
        seed=args.seed,
        include_assistant_prefix=not args.no_assistant_prefix,
        num_hints=args.num_hints,
        force_json=args.json,
        data_name=args.data_name,
    )


def cmd_create_verification_data(args):
    """Create binary predictor data from multi-sample solver aggregates."""
    commands.create_verification_data(
        task_name=args.task,
        generations_path=Path(args.generations) if args.generations else None,
        output_path=Path(args.output) if args.output else None,
        method_name=args.method,
        num_samples=args.num_samples,
        threshold=args.threshold,
        sft_fraction=args.sft_fraction,  # only relevant for method_c
        seed=args.seed,  # only relevant for method_c
        run_id=args.run_id,  # only relevant for method_c
        split=args.split,  # only relevant for method_c
        data_name=args.data_name,
        models_name=args.models_name,
        run_solver=args.run_solver,
        solver_method_name=args.solver_method,
        solver_run_id=args.solver_run_id,
        solver_split=args.solver_split,
        solver_output_path=Path(args.solver_output) if args.solver_output else None,
        solver_batch_size=args.solver_batch_size,
        solver_max_new_tokens=args.solver_max_new_tokens,
        solver_temperature=args.solver_temperature,
        solver_top_p=args.solver_top_p,
        solver_tensor_parallel_size=args.solver_tensor_parallel_size,
        solver_data_parallel_size=args.solver_data_parallel_size,
        solver_gpu_memory_utilization=args.solver_gpu_memory_utilization,
        solver_use_async=args.solver_async,
        solver_seed=args.solver_seed,
    )


def cmd_create_ood_prompts(args):
    """Create OOD evaluation prompts."""
    output_path = Path(args.output) if args.output else None
    commands.create_ood_prompts(
        task_name=args.task,
        dataset_name=args.dataset,
        method_name=args.method,
        output_path=output_path,
        num_problems=args.num_problems,
        seed=args.seed,
        include_assistant_prefix=not args.no_assistant_prefix,
        data_name=args.data_name,
    )



def cmd_generate(args):
    """Generate dataset."""
    prompts_path = Path(args.prompts) if args.prompts else None
    output_path = Path(args.output) if args.output else None

    # Parse force-hints-distribution string into dict {num_hints: probability}
    force_hints_distribution = None
    dist_str = getattr(args, "force_hints_distribution", None)
    if dist_str:
        force_hints_distribution = {}
        for pair in dist_str.split(","):
            num_hints, prob = pair.strip().split(":")
            force_hints_distribution[int(num_hints.strip())] = float(prob.strip())

    # Parse force-hints-policy string into dict {level: rate}
    force_hints_policy = None
    policy_str = getattr(args, "force_hints_policy", None)
    if policy_str:
        if not force_hints_distribution:
            # `parser` is a local of main(); referencing it here raised NameError
            # instead of the intended usage error.
            raise SystemExit(
                "error: --force-hints-policy requires --force-hints-distribution"
            )
        force_hints_policy = {}
        for pair in policy_str.split(","):
            level, rate = pair.strip().split(":")
            force_hints_policy[level.strip()] = float(rate.strip())

    gen_kwargs = dict(
        task_name=args.task,
        model_name=args.model,
        method_name=args.method,
        run_id=getattr(args, "run_id", None),
        prompts_path=prompts_path,
        output_path=output_path,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose,
        retry_incorrect=args.retry_incorrect,
        retry_truncated=getattr(args, "retry_truncated", False),
        max_retries=getattr(args, "max_retries", 10),
        answer_budget=getattr(args, "answer_budget", 0),
        seed=getattr(args, "seed", 42),
        multi_turn=args.multi_turn,  # None = use method config
        use_async=getattr(args, "use_async", False),
        force_hints_distribution=force_hints_distribution,
        force_hints_policy=force_hints_policy,
        sample_strategy=getattr(args, "sample_strategy", None),
        data_parallel_size=args.data_parallel_size,
        no_hints=getattr(args, "no_hints", False),
        hint_schedule_from=(Path(args.hint_schedule)
                            if getattr(args, "hint_schedule", None) else None),
        hint_buckets=getattr(args, "hint_buckets", None),
        hint_target_fraction=getattr(args, "hint_target_fraction", None),
        max_hints=getattr(args, "max_hints", 4),
        drop_exhausted=getattr(args, "drop_exhausted", False),
        data_name=args.data_name,
        models_name=args.models_name,
    )

    # A target correct rate turns generation into the phase-3 oversampling loop:
    # resample the problems that came back wrong until enough are right.
    target = getattr(args, "target_correct_rate", None)
    if target is not None:
        return commands.generate_until_target(
            target_correct_rate=target,
            max_oversample=getattr(args, "max_oversample", 5),
            **gen_kwargs,
        )
    commands.generate(**gen_kwargs)


def cmd_generate_tree(args):
    """Generate a tree-structured dataset."""
    commands.generate_tree(
        task_name=args.task,
        model_name=args.model,
        method_name=args.method,
        run_id=args.run_id,
        split=args.split,
        output_path=Path(args.output) if args.output else None,
        data_name=args.data_name,
        models_name=args.models_name,
        num_midpoints=args.num_midpoints,
        num_samples=args.num_samples,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_async=args.use_async,
        seed=args.seed,
    )


def cmd_evaluate(args):
    """Evaluate model."""
    prompts_path = Path(args.prompts) if args.prompts else None
    output_path = Path(args.output) if args.output else None

    commands.evaluate(
        task_name=args.task,
        model_name=args.model,
        method_name=args.method,
        run_id=args.run_id,
        split=args.split,
        prompts_path=prompts_path,
        output_path=output_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose,
        multi_turn=args.multi_turn,  # None = use method config
        use_async=getattr(args, "use_async", False),
        seed=args.seed,
        no_hints=args.no_hints,
        data_name=args.data_name,
        models_name=args.models_name,
    )


def cmd_combine_verifier_eval(args):
    """Combine verifier decisions with fixed-hint solver evaluations."""
    commands.combine_verifier_eval(
        task_name=args.task,
        solver_results_path=Path(args.solver_results),
        verifier_results_path=Path(args.verifier_results),
        output_path=Path(args.output),
        max_hints=args.max_hints,
    )


def cmd_analyze(args):
    """Analyze dataset accuracy by variant."""
    commands.analyze(
        dataset_path=Path(args.dataset),
        task_name=getattr(args, "task", None),
    )


def cmd_train_sft(args):
    """Train SFT model."""
    dataset_path = Path(args.dataset) if args.dataset else None
    eval_dataset_path = Path(args.eval_dataset) if args.eval_dataset else None
    output_path = Path(args.output) if args.output else None

    commands.train_sft(
        task_name=args.task,
        base_model=args.base_model,
        method_name=args.method,
        run_id=getattr(args, "run_id", None),
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
        output_path=output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        bf16=not args.no_bf16,
        report_to=args.report_to,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        include_wrong_valid_format=args.include_wrong_valid_format,
        upsample_hint=args.upsample_hint,
        strip_think_tokens=getattr(args, "strip_think_tokens", False),
        max_correct=args.max_correct,
        completion_only_loss=args.completion_only_loss,
        data_name=args.data_name,
        models_name=args.models_name,
    )


def cmd_train_classifier(args):
    """Train verifier/abstain classifier on generate_tree output."""
    dataset_path = Path(args.dataset) if args.dataset else None
    eval_dataset_path = Path(args.eval_dataset) if args.eval_dataset else None
    output_path = Path(args.output) if args.output else None

    commands.train_classifier(
        task_name=args.task,
        base_model=args.base_model,
        method_name=args.method,
        run_id=getattr(args, "run_id", None),
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
        output_path=output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        bf16=not args.no_bf16,
        report_to=args.report_to,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        completion_only_loss=args.completion_only_loss,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        depth_eval_batch_size=args.depth_eval_batch_size,
        depth_eval_max_new_tokens=args.depth_eval_max_new_tokens,
        data_name=args.data_name,
        models_name=args.models_name,
    )


def cmd_train_rl(args):
    """Train RL model using verl."""
    train_prompts_path = Path(args.train_prompts) if args.train_prompts else None
    val_prompts_path = Path(args.val_prompts) if args.val_prompts else None
    sft_model_path = Path(args.sft_model) if args.sft_model else None
    output_path = Path(args.output) if args.output else None
    reward_function_path = Path(args.reward_function) if args.reward_function else None
    resume_path = Path(args.resume) if args.resume else None

    # Parse reward kwargs overrides (KEY=VALUE pairs)
    reward_kwargs_overrides = {}
    if args.reward_kwargs:
        for item in args.reward_kwargs:
            key, _, value = item.partition("=")
            # Try to parse as float, then int, then keep as string
            try:
                value = float(value)
                if value == int(value):
                    value = int(value)
            except ValueError:
                pass
            reward_kwargs_overrides[key] = value

    commands.train_rl(
        task_name=args.task,
        method_name=args.method,
        run_id=args.run_id,
        base_model=args.base_model,
        train_prompts_path=train_prompts_path,
        val_prompts_path=val_prompts_path,
        sft_model_path=sft_model_path,
        output_path=output_path,
        reward_function_path=reward_function_path,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate,
        total_steps=args.total_steps,
        kl_coef=args.kl_coef,
        n_samples=args.n_samples,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        n_gpus_per_node=args.n_gpus_per_node,
        gpu_memory_utilization=args.gpu_memory_utilization,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        wandb=not args.no_wandb,
        resume_path=resume_path,
        cleanup_checkpoints=not args.keep_checkpoints,
        keep_state=args.keep_state,
        reward_kwargs_overrides=reward_kwargs_overrides,
        extra_overrides=args.override,
        shuffle_seed=args.shuffle_seed,
        overwrite=args.overwrite,
        continue_run=args.continue_run,
        save_best=args.save_best,
        best_metric=args.best_metric,
        max_ckpt_to_keep=args.max_ckpt_to_keep,
        data_name=args.data_name,
        models_name=args.models_name,
    )


def cmd_convert_checkpoint(args):
    """Convert FSDP/Megatron checkpoint to HuggingFace format."""
    output_path = Path(args.output) if args.output else None
    commands.convert_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        output_path=output_path,
        backend=args.backend,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Data pipeline for abstention reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list_tasks
    p = subparsers.add_parser("list_tasks", help="List available tasks")
    p.set_defaults(func=cmd_list_tasks)

    # list_methods
    p = subparsers.add_parser("list_methods", help="List available methods for a task")
    p.add_argument("--task", required=True, help="Task name")
    p.set_defaults(func=cmd_list_methods)

    # create_primitives
    p = subparsers.add_parser("create_primitives", help="Generate raw puzzle data")
    p.add_argument("--task", required=True, help="Task name (e.g., countdown)")
    p.add_argument("--output", help="Output path (default: data/{data_name}/problems/primitives.json)")
    p.add_argument("--num-puzzles", type=int, default=None, help="Number of puzzles (omit to use all available)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--tracer", choices=["original", "uniform"], default=None,
                   help="code_output only. Tracer for hint generation: 'uniform' (execution-uniform, "
                        "used for the shipped primitives) or 'original' (top-level). Default: uniform.")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.set_defaults(func=cmd_create_primitives)

    # create_partitions
    p = subparsers.add_parser("create_partitions",
        help="Split primitives into per-split problem files under problems/")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--primitives", help="Path to primitives.json (default: data/{data_name}/problems/primitives.json)")
    p.add_argument("--output", help="Output directory (default: data/{data_name}/problems/)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split assignment")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.set_defaults(func=cmd_create_partitions)

    # create_prompts
    p = subparsers.add_parser("create_prompts", help="Create prompts from primitives")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--method", help="Method name for auto-derived paths and templates")
    p.add_argument("--primitives", help="Path to primitives.json (default: data/{data_name}/problems/primitives.json)")
    p.add_argument("--output", help="Output directory (default: data/{data_name}/problems_with_format/)")
    p.add_argument("--split", default="all",
                   help="Split name, or 'all' for every split this task defines. Most tasks: "
                   "sft_train, sft_val, rl_train, rl_val, eval; "
                   "code_output has no rl_val.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split assignment")
    p.add_argument("--no-assistant-prefix", action="store_true", help="Don't include assistant prefix")
    p.add_argument("--num-hints", type=int, default=None,
        help="Maximum hint level. method_ac samples one level for SFT/RL and "
             "creates every level for eval; other methods include the first N hints.")
    p.add_argument("--json", action="store_true", help="Force JSON output for all splits (instead of parquet for RL)")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.set_defaults(func=cmd_create_prompts)

    # create_verification_data
    p = subparsers.add_parser(
        "create_verification_data",
        help="Create binary predictor data from multi-sample solver aggregates",
    )
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--method", default="method_a",
                   help="Predictor method/template (default: method_a)")
    p.add_argument("--generations",
                   help="Path to the multi-sample solver aggregate JSON. Required "
                        "unless --run-solver is set, which generates it instead.")
    p.add_argument("--output",
                   help="Output path for the predictor SFT dataset. Required for "
                        "method_a; auto-derived under problems_with_format/ for method_c")
    p.add_argument("--num-samples", type=int, default=10,
                   help="Required samples per source prompt (default: 10)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Label 1 when pass_rate >= threshold (default: 0.5)")
    p.add_argument("--sft-fraction", type=float, default=0.1,
                   help="method_c only: fraction of records held out as SFT prompts, "
                        "rest go to RL (default: 0.1)")
    p.add_argument("--seed", type=int, default=42,
                   help="method_c only: seed for the SFT/RL split")
    p.add_argument("--run-id",
                   help="method_c only: run-id identifier used in output filenames and "
                        "stored on each record")
    p.add_argument("--split", default="train", choices=["train", "val"],
                   help="method_c only: 'train' writes sft_train/rl_train, 'val' writes "
                        "sft_val/rl_val (default: train)")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.add_argument("--models-name", help="Models directory name (default: data-name)")
    p.add_argument("--run-solver", action="store_true",
                   help="Produce the multi-sample aggregate here instead of reading it "
                        "from --generations: runs `evaluate --model rl --method "
                        "<solver-method> --num-samples <num-samples>` first (keeping "
                        "every raw sample and its correctness), then builds the "
                        "verifier data from its output.")
    p.add_argument("--solver-method", default="method_ac",
                   help="Solver method whose RL model produces the aggregate "
                        "(default: method_ac). Only used with --run-solver.")
    p.add_argument("--solver-run-id",
                   help="Run identifier for the solver's RL model. Only used with --run-solver.")
    p.add_argument("--solver-split", default="sft_val",
                   help="Which prompt split the solver is evaluated on (default: sft_val). "
                        "Only used with --run-solver.")
    p.add_argument("--solver-output",
                   help="Output path for the solver's evaluate() results JSON (default: "
                        "auto-derived under the rl model's evals/ directory, same "
                        "convention as `pipeline evaluate`). Only used with --run-solver.")
    p.add_argument("--solver-batch-size", type=int, default=16, help="Only used with --run-solver.")
    p.add_argument("--solver-max-new-tokens", type=int, default=2048, help="Only used with --run-solver.")
    p.add_argument("--solver-temperature", type=float, default=0.7, help="Only used with --run-solver.")
    p.add_argument("--solver-top-p", type=float, default=0.9, help="Only used with --run-solver.")
    p.add_argument("--solver-tensor-parallel-size", type=int, default=1, help="Only used with --run-solver.")
    p.add_argument("--solver-data-parallel-size", type=int, default=1, help="Only used with --run-solver.")
    p.add_argument("--solver-gpu-memory-utilization", type=float, default=0.9, help="Only used with --run-solver.")
    p.add_argument("--solver-async", action="store_true", help="Use async generation for the solver. Only used with --run-solver.")
    p.add_argument("--solver-seed", type=int, default=42, help="Only used with --run-solver.")
    p.set_defaults(func=cmd_create_verification_data)

    # create_ood_prompts
    ood_names = ", ".join(sorted(commands.OOD_DATASETS.keys()))
    p = subparsers.add_parser("create_ood_prompts", help="Create eval prompts from an OOD math benchmark")
    p.add_argument("--task", required=True, help="Task whose templates to use (e.g., math)")
    p.add_argument("--dataset", required=True, help=f"OOD dataset name ({ood_names})")
    p.add_argument("--method", help="Method name for template selection and output path")
    p.add_argument("--output", help="Output path (default: data/{data_name}/problems_with_format/eval__{method}_ood-{dataset}.json)")
    p.add_argument("--num-problems", type=int, default=None, help="Limit number of problems (omit for all)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    p.add_argument("--no-assistant-prefix", action="store_true", help="Don't include assistant prefix")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.set_defaults(func=cmd_create_ood_prompts)

    # generate
    p = subparsers.add_parser("generate", help="Run model on prompts to create dataset")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--model", required=True, help="Model name or path")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--run-id", help="Run identifier: names the run directory under models/{method}_{sft,rl}/. Required with --method; nothing is derived from the base checkpoint.")
    p.add_argument("--prompts", help="Path to prompts file (default: data/{data_name}/problems_with_format/{split}__{method}.json)")
    p.add_argument("--output", help="Output path (default: data/{data_name}/sft_datasets/{split}__{method}.json)")
    p.add_argument("--split", default="sft_train", help="Which split to generate from (default: sft_train)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p")
    p.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt (best selected by --sample-strategy)")
    p.add_argument("--sample-strategy", default=None,
        choices=["shortest_cot", "most_hints", "random_correct", "random"],
        help="Selection strategy when --num-samples > 1 (default: most_hints for "
             "multi-turn, shortest_cot otherwise). 'random_correct' picks uniformly "
             "among correct samples and DROPS problems with no correct sample; the "
             "others fall back to an incorrect one, and the argmax strategies skew "
             "the achieved distribution (most_hints inflates hint counts). 'random' "
             "picks uniformly among ALL samples regardless of correctness and never "
             "drops a problem -- use when the representative sample's own "
             "correctness is the signal you want preserved (e.g. method_c).")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--data-parallel-size", type=int, default=1,
        help="Independent model replicas; prompts are sharded across them. The cluster "
             "floor is 4 GPUs, so a model that fits on one GPU wants --data-parallel-size 4 "
             "(~4x throughput) rather than --tensor-parallel-size 4.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    p.add_argument("--verbose", action="store_true", help="Print sample prompts during generation")
    p.add_argument("--retry-incorrect", action="store_true", help="Re-run incorrect examples from existing output")
    p.add_argument("--retry-truncated", action="store_true", help="Re-run truncated examples (finish_reason=length) with different seeds until all complete")
    p.add_argument("--max-retries", type=int, default=10, help="Max retry iterations for --retry-truncated (default: 10)")
    p.add_argument("--answer-budget", type=int, default=0, help="Tokens held back to force an answer when a generation runs out of room mid-thought (0 = off, --async only)")
    p.add_argument("--seed", type=int, default=42, help="Starting seed for generation (default: 42)")
    p.add_argument("--multi-turn", action="store_true", default=None, help="Enable multi-turn generation (auto-detected from method config)")
    p.add_argument("--async", dest="use_async", action="store_true", help="Use async generation (optimal throughput, processes all prompts concurrently)")
    p.add_argument("--force-hints-distribution", type=str, default=None,
        help="Distribution of forced hint counts, e.g. '1:0.5,2:0.3,3:0.2'. "
             "Maps number of hints to probability. Probabilities must sum to 1.")
    p.add_argument("--force-hints-policy", type=str, default=None,
        help="Per-level force rate, e.g. '1:0.05,2:0.05,3:0.05,4:0.15,5:0.15'. "
             "Only this fraction of examples at each level will get forced hints. "
             "Requires --force-hints-distribution.")
    # --- Difficulty-scheduled hints (3 phases) ------------------------------
    # Phase 1: probe. Measure how often the model solves each problem unaided.
    #   generate ... --no-hints --num-samples 8 --output PROBE.json
    # Phase 2+3: schedule from that probe and oversample until enough are right.
    #   generate ... --hint-schedule PROBE.json --target-correct-rate 0.5
    # The probe is an ordinary dataset; its pass rates are read straight out of
    # it, so there is no separate profile file to write, name or keep in sync.
    p.add_argument("--no-hints", dest="no_hints", action="store_true",
        help="Ban the hint request during decoding, so the model has to answer "
             "unaided. Phase 1 of the difficulty schedule: measures unaided pass "
             "rate. Pair with --num-samples.")
    p.add_argument("--hint-schedule", type=str, default=None,
        help="Path to a no-hint probe dataset (a 'generate --no-hints "
             "--num-samples 8' output). Schedules per-problem hint counts from its "
             "pass rates, so harder problems get more hints. Overrides "
             "--force-hints-distribution/--force-hints-policy.")
    p.add_argument("--hint-target-fraction", type=float, default=None,
        help="Fraction of problems that should request a hint, e.g. 0.5. Assigns "
             "hints by difficulty RANK so the fraction is hit exactly whatever the "
             "probe's pass rates look like. Preferred over --hint-buckets, whose "
             "achieved rate swings with teacher strength.")
    p.add_argument("--max-hints", type=int, default=4,
        help="Most hints scheduled for any single problem (default 4).")
    p.add_argument("--hint-buckets", type=str, default=None,
        help="Pass-rate to hint-count mapping for --hint-schedule, as "
             "'max_pass_rate:num_hints,...'. Default: '0.0:4,0.25:3,0.5:2,0.75:1,1.0:0' "
             "(never solved unaided -> 4 hints; always solved -> none).")
    p.add_argument("--drop-exhausted", action="store_true",
        help="Discard generations where the model asked past the last available "
             "hint and was served 'No more hints available.'. That placeholder is "
             "real training text; keeping it teaches the warm start to emit it.")
    p.add_argument("--target-correct-rate", type=float, default=None,
        help="Phase 3: keep resampling incorrect problems until this fraction is "
             "answered correctly (e.g. 0.5 for a majority). Stops early if a round "
             "makes no progress.")
    p.add_argument("--max-oversample", type=int, default=5,
        help="Cap on --target-correct-rate resampling rounds (default 5).")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.add_argument("--models-name", help="Models directory name under models/ (default: data-name)")
    p.set_defaults(func=cmd_generate)

    # generate_tree
    p = subparsers.add_parser("generate_tree", help="Run model on prompts to build a tree of solver rollouts with pass-rate labels")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--model", required=True, help="Model name or path (or 'sft'/'rl' shortcut)")
    p.add_argument("--method", required=True, help="Method name for auto-derived paths")
    p.add_argument("--run-id", help="Run identifier: names the run directory under models/{method}_{sft,rl}/. Required when --model is 'sft' or 'rl'.")
    p.add_argument("--split", required=True,
                   help="Which split to generate from. Required -- there is no default.")
    p.add_argument("--output", help="Output path (default: data/{data_name}/sft_datasets/{split}__{method}__tree.json, "
                                     "or ..._tree__{run_id}.json when --run-id is set)")
    p.add_argument("--num-midpoints", type=int, default=0,
                   help="Number of branch points between root and leaf (default: 0, "
                        "i.e. just root + leaf). Only the root step is implemented so far.")
    p.add_argument("--num-samples", type=int, default=10,
                   help="Rollouts per prompt at the root step (default: 10)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Label 1 when pass_rate >= threshold (default: 0.5)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--data-parallel-size", type=int, default=1, help="Independent model replicas; prompts are sharded across them")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    p.add_argument("--async", dest="use_async", action="store_true", help="Use async generation")
    p.add_argument("--seed", type=int, default=42, help="Seed for generation (default: 42)")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.add_argument("--models-name", help="Models directory name under models/ (default: data-name)")
    p.set_defaults(func=cmd_generate_tree)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate model")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--model", required=True, help="Model name/path, or 'sft'/'rl' to use method's model")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--run-id", help="Run identifier: names the run directory under models/{method}_{sft,rl}/. Required with --method; nothing is derived from the base checkpoint.")
    p.add_argument("--split", default="eval", help="Prompts split to evaluate (default: eval)")
    p.add_argument("--prompts", help="Path to eval prompts (default: data/{data_name}/problems_with_format/{split}__{method}.json)")
    p.add_argument("--output", help="Output path (default: models/{models_name}/{method}_{sft,rl}/{run_id}/evals/{split}.json)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    p.add_argument("--top-p", type=float, default=1.0, help="Top-p")
    p.add_argument("--num-samples", type=int, default=1, help="Samples per problem (default: 1)")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--data-parallel-size", type=int, default=1,
        help="Independent model replicas; prompts are sharded across them. The cluster "
             "floor is 4 GPUs, so a model that fits on one GPU wants --data-parallel-size 4 "
             "(~4x throughput) rather than --tensor-parallel-size 4.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    p.add_argument("--verbose", action="store_true", help="Print sample prompts during generation")
    p.add_argument("--multi-turn", action="store_true", default=None, help="Enable multi-turn generation (auto-detected from method config)")
    p.add_argument("--async", dest="use_async", action="store_true", help="Use async generation (optimal throughput)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for generation (default: 42)")
    p.add_argument("--no-hints", action="store_true", help="Counterfactual eval: block hint requests during decoding so the model must answer alone")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.add_argument("--models-name", help="Models directory name under models/ (default: data-name)")
    p.set_defaults(func=cmd_evaluate)

    # combine_verifier_eval
    p = subparsers.add_parser(
        "combine_verifier_eval",
        help="Apply verifier decisions to fixed-hint solver evaluation results",
    )
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--solver-results", required=True,
                   help="Method AC evaluation over every fixed hint level")
    p.add_argument("--verifier-results", required=True,
                   help="Method A verifier evaluation over fixed hint levels")
    p.add_argument("--output", required=True,
                   help="Output path for the combined pipeline evaluation")
    p.add_argument("--max-hints", type=int, default=5,
                   help="Maximum hint level, selected unconditionally if needed")
    p.set_defaults(func=cmd_combine_verifier_eval)

    # analyze
    p = subparsers.add_parser("analyze", help="Analyze dataset accuracy by variant (prints grid)")
    p.add_argument("--dataset", required=True, help="Path to dataset or eval results")
    p.add_argument("--task", help="Task name (for task-specific metrics on raw datasets)")
    p.set_defaults(func=cmd_analyze)


    # train_sft
    p = subparsers.add_parser("train_sft", help="Train SFT model on generated dataset")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--base-model", required=True, help="Base model to fine-tune")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--run-id", help="Run identifier: names the run directory under models/{method}_{sft,rl}/. Required with --method; nothing is derived from the base checkpoint.")
    p.add_argument("--dataset", help="Path to generated dataset (default: auto-detect from method)")
    p.add_argument("--eval-dataset", help="Path to held-out dataset for validation loss "
                   "(default: data/{data_name}/sft_datasets/sft_val__{method}.json if it exists; "
                   "validation is skipped if not found)")
    p.add_argument("--output", help="Output path (default: models/{models_name}/{method}_sft/{run_id}/model)")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    p.add_argument("--max-length", type=int, default=4096, help="Maximum sequence length")
    p.add_argument("--no-bf16", action="store_true", help="Disable bfloat16 training")
    p.add_argument("--report-to", default="wandb", help="Reporting integration (wandb, none)")
    p.add_argument("--project-name", help="Wandb project name (default: {task}-sft)")
    p.add_argument("--experiment-name", help="Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})")
    p.add_argument("--include-wrong-valid-format", action="store_true", help="Include wrong answers with valid format (task-specific, e.g., valid UCI but wrong move for chess)")
    p.add_argument("--upsample-hint", type=int, default=1, help="Upsample hint-containing examples by this factor (e.g., 4 = 4x copies)")
    p.add_argument("--max-correct", type=int, default=None, help="Downsample correct examples to at most this many (random subset, seed=42)")
    p.add_argument("--strip-think-tokens", action="store_true",
        help="Remove <think>/</think> from the tokenizer's added-token vocabulary "
             "so they encode as ordinary text. Qwen3 ships them as added tokens "
             "(1 id each); Qwen2.5 has no such entries and uses 6 ids for the pair. "
             "They are NOT special tokens, so nothing is being dropped -- this is "
             "purely so both base models see identical tokenization of the same "
             "data and their SFT runs stay comparable. Use for Qwen3-* models.")
    p.add_argument("--completion-only-loss", action="store_true", help="Mask the prompt and the injected <response> hints out of the loss. Off by default: every RL parent was trained on the full sequence, and the masked ablation scored lower")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.add_argument("--models-name", help="Models directory name under models/ (default: data-name)")
    p.set_defaults(func=cmd_train_sft)

    # train_classifier
    p = subparsers.add_parser("train_classifier", help="Train verifier/abstain classifier on generate_tree output")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--base-model", required=True, help="Base model to fine-tune")
    p.add_argument("--method", help="Method name for auto-derived paths")
    p.add_argument("--run-id", help="Run identifier: names the run directory under models/{method}_classifier/. Required with --method; nothing is derived from the base checkpoint.")
    p.add_argument("--dataset", help="Path to generate_tree output (default: auto-detect from method)")
    p.add_argument("--eval-dataset", help="Path to held-out tree dataset for validation loss")
    p.add_argument("--output", help="Output path (default: models/{models_name}/{method}_classifier/{run_id}/model)")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    p.add_argument("--max-length", type=int, default=4096, help="Maximum sequence length")
    p.add_argument("--no-bf16", action="store_true", help="Disable bfloat16 training")
    p.add_argument("--report-to", default="wandb", help="Reporting integration (wandb, none)")
    p.add_argument("--project-name", help="Wandb project name (default: {task}-classifier)")
    p.add_argument("--experiment-name", help="Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})")
    p.add_argument("--completion-only-loss", dest="completion_only_loss", action="store_true", default=True,
                   help="Mask the prompt out of the loss, training only on the gold label token(s) (default: on)")
    p.add_argument("--no-completion-only-loss", dest="completion_only_loss", action="store_false",
                   help="Train on the full sequence (prompt + label) instead of masking the prompt out of the loss")
    p.add_argument("--use-lora", action="store_true", help="Train a LoRA adapter instead of full fine-tuning")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (only used with --use-lora)")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (only used with --use-lora)")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (only used with --use-lora)")
    p.add_argument("--lora-target-modules", nargs="+", default=None,
                   help="Module names to adapt (only used with --use-lora); default lets peft pick the "
                        "base model's standard attention/MLP projections")
    p.add_argument("--depth-eval-batch-size", type=int, default=16,
                   help="Batch size for the per-depth greedy-decode accuracy pass run at every evaluation")
    p.add_argument("--depth-eval-max-new-tokens", type=int, default=4,
                   help="Max new tokens to generate per held-out example when computing depth accuracy")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.add_argument("--models-name", help="Models directory name under models/ (default: data-name)")
    p.set_defaults(func=cmd_train_classifier)

    # train_rl
    p = subparsers.add_parser("train_rl", help="Train RL model using verl (GRPO)")
    p.add_argument("--task", required=True, help="Task name")
    p.add_argument("--method", help="Method name for auto-derived paths, template, and reward config")
    p.add_argument("--run-id", help="Run identifier: names the run directory under models/{method}_{sft,rl}/. Required with --method; nothing is derived from the base checkpoint.")
    p.add_argument("--base-model", help="Base model for cold-start RL (mutually exclusive with --sft-model)")
    p.add_argument("--train-prompts", help="Path to RL train prompts (default: data/{data_name}/problems_with_format/rl_train__{method}.parquet)")
    p.add_argument("--val-prompts", help="Path to RL validation prompts (default: data/{data_name}/problems_with_format/rl_val__{method}.parquet)")
    p.add_argument("--sft-model", help="Path to SFT model (mutually exclusive with --base-model)")
    p.add_argument("--output", help="Output path (default: models/{models_name}/{method}_rl/{run_id}/model)")
    p.add_argument("--reward-function", help="Path to reward function (default: task's reward function)")
    p.add_argument("--train-batch-size", type=int, default=64, help="Training batch size")
    p.add_argument("--val-batch-size", type=int, default=64, help="Validation batch size")
    p.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    p.add_argument("--total-steps", type=int, default=400, help="Total training steps")
    p.add_argument("--kl-coef", type=float, default=0.001, help="KL divergence coefficient")
    p.add_argument("--n-samples", type=int, default=8, help="Number of samples per prompt (group size)")
    p.add_argument("--save-freq", type=int, default=None, help="Checkpoint save frequency (default: 25, or off when --save-best is set)")
    p.add_argument("--test-freq", type=int, default=None, help="Validation/logging frequency (default: same as save-freq)")
    p.add_argument("--max-prompt-length", type=int, default=2048, help="Maximum prompt length in tokens")
    p.add_argument("--max-response-length", type=int, default=2048, help="Maximum response length in tokens")
    p.add_argument("--max-model-len", type=int, default=8192, help="Maximum model context length (default: 8192). Set higher for multi-turn.")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM rollout (GPUs per rollout replica)")
    p.add_argument("--n-gpus-per-node", type=int, default=None, help="Total GPUs per node used by the trainer (actor/ref FSDP + rollout). Default: auto-detect all visible GPUs via torch.cuda.device_count(). Decoupled from --tensor-parallel-size.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="GPU memory utilization")
    p.add_argument("--project-name", help="Wandb project name (default: {task}-rl)")
    p.add_argument("--experiment-name", help="Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--resume", help="Path to resume from existing run")
    p.add_argument("--overwrite", action="store_true", help="Discard an existing run at this run-id and train from scratch")
    p.add_argument("--continue-run", action="store_true", help="Resume the existing run at this run-id from its latest checkpoint")
    p.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True, help="Also emit best/, the checkpoint with the highest validation score (last/ is always written). Use --no-save-best to disable (default: enabled)")
    p.add_argument("--best-metric", default="auto", help="Validation metric selecting the best checkpoint (default: auto, verl's headline val-core scalar)")
    p.add_argument("--max-ckpt-to-keep", type=int, default=None, help="Checkpoints to retain during training (default: 1, or 2 with --save-best so best and last both survive)")
    p.add_argument("--keep-checkpoints", action="store_true", help="Keep checkpoints and rollouts after training (by default they are deleted)")
    p.add_argument("--keep-state", action="store_true", help="Keep the last optimizer state checkpoint after training")
    p.add_argument("--reward-kwargs", nargs="*", metavar="KEY=VALUE", help="Override reward kwargs (e.g., --reward-kwargs hint_penalty=0.05 hint_bonus=0.1)")
    p.add_argument("--override", action="extend", nargs="*", default=None, metavar="KEY=VALUE", help="Raw hydra overrides appended verbatim. Repeatable: multiple --override flags accumulate (e.g., --override a=1 --override b=2)")
    p.add_argument("--shuffle-seed", type=int, default=None, help="Seed for shuffling training data (default: 1, set to randomize order across runs)")
    p.add_argument("--data-name", help="Data directory name under data/ (default: task name)")
    p.add_argument("--models-name", help="Models directory name under models/ (default: data-name)")
    p.set_defaults(func=cmd_train_rl)

    # convert_checkpoint
    p = subparsers.add_parser("convert_checkpoint", help="Convert FSDP/Megatron checkpoint to HuggingFace format")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (e.g., .../global_step_100/actor)")
    p.add_argument("--output", help="Output path for HF model (default: {checkpoint}_hf)")
    p.add_argument("--backend", default="fsdp", choices=["fsdp", "megatron"], help="Checkpoint backend")
    p.set_defaults(func=cmd_convert_checkpoint)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
