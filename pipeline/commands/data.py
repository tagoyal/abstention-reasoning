"""
Data commands - primitives and prompts creation.
"""

import inspect
import random
from pathlib import Path

# method_ac / method_a: how many hint levels to sample per problem, and how
# strongly to bias that sample toward the lower (less-hinted) levels. Weight
# of level l is decay**l, so decay < 1 favors 0, 1, 2, ... over higher levels;
# decay=1 would be uniform.
HINT_LEVELS_PER_PROBLEM = 2
HINT_LEVEL_DECAY = 0.7

from pipeline.core.io import load_json, save_json, save_parquet
from pipeline.core.method import TASKS_ROOT, Method, get_primitives_path, partition_path, resolve_data_name, resolve_models_name
from pipeline.tasks import get_task


def create_primitives(
    task_name: str,
    output_path: Path | None = None,
    num_puzzles: int | None = None,
    seed: int = 42,
    data_name: str | None = None,
    **kwargs,
) -> Path:
    """
    Generate raw puzzle data.

    Args:
        task_name: Name of task (e.g., "countdown")
        output_path: Where to save primitives.json (default: data/{data_name}/problems/primitives.json)
        num_puzzles: Number of puzzles to generate (None = all available)
        seed: Random seed
        data_name: Data directory name (default: task_name)
        **kwargs: Additional task-specific options (e.g., tracer="uniform" for code_output)

    Returns:
        Path to created primitives.json
    """
    data_name = resolve_data_name(task_name, data_name)
    task = get_task(task_name)

    # Validate task-specific options against the task's actual signature. Tasks
    # differ in what they accept (only code_output takes `tracer`), so an option
    # the task can't use must fail loudly rather than raise a bare TypeError or
    # be silently dropped.
    if kwargs:
        sig = inspect.signature(task.create_primitives)
        takes_var_kw = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if not takes_var_kw:
            unsupported = [key for key in kwargs if key not in sig.parameters]
            if unsupported:
                raise ValueError(
                    f"Task '{task_name}' does not support "
                    f"{', '.join(repr(k) for k in sorted(unsupported))}. "
                    f"That option applies to other tasks only."
                )

    # Default output path
    if output_path is None:
        output_path = get_primitives_path(data_name)

    count_str = str(num_puzzles) if num_puzzles is not None else "all"
    print(f"Generating {count_str} primitives for task '{task_name}'...")

    primitives = task.create_primitives(num_puzzles, seed, **kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, primitives)
    print(f"Saved {len(primitives)} primitives to {output_path}")

    return output_path


def create_partitions(
    task_name: str,
    primitives_path: Path | None = None,
    output_dir: Path | None = None,
    seed: int = 42,
    data_name: str | None = None,
) -> dict[str, Path]:
    """Split primitives into per-split problem files under problems/.

    One partition serves every method: the split a problem belongs to is a
    property of the dataset, not of whatever method happens to be formatting it.
    Writing it down also pins it -- the boundaries in a task's SPLITS table can
    change afterwards without silently reinterpreting artifacts already built
    against the old ones.

    Returns:
        {split: path} for every split the task defines.
    """
    data_name = resolve_data_name(task_name, data_name)
    task = get_task(task_name)
    if primitives_path is None:
        primitives_path = get_primitives_path(data_name)
    primitives = load_json(primitives_path)

    written = {}
    extra_splits = set(getattr(task, "EXTRA_SPLITS", ()))
    for split in task.supported_splits():
        if split in extra_splits:
            # Not a derivable [start, end) ratio range (e.g. rl_gen_train /
            # rl_ver_train are a stratified resample of rl_train produced by
            # scripts/split_dataset.py) -- create_partitions can't recompute
            # these, so skip and leave whatever partition file already exists.
            path = (output_dir / f"{split}.json") if output_dir else partition_path(data_name, split)
            if path.exists():
                print(f"  {split:<10} (pre-built partition, skipping) -> {path}")
                written[split] = path
            else:
                print(f"  {split:<10} skipped: no SPLITS ratio and no partition file at {path}")
            continue
        indices = set(task.get_split_indices(len(primitives), split, seed, primitives))
        rows = [p for p in primitives if p["index"] in indices]
        path = (output_dir / f"{split}.json") if output_dir else partition_path(data_name, split)
        save_json(path, rows)
        print(f"  {split:<10} {len(rows):>6} problems -> {path}")
        written[split] = path
    return written


def _prompts_filename(split: str, method, force_json: bool) -> str:
    """Filename for one split's prompts.

    Prompt files for every method share a single prompts/ directory, so the
    method has to be in the name: `{split}__{method}`. Without a method the
    caller has supplied its own output directory and gets the bare split name.
    rl_* is parquet because verl reads parquet; --json overrides that for
    inspection.
    """
    ext = ".json" if force_json else (".parquet" if split.startswith("rl") else ".json")
    stem = method.artifact_stem(split) if method is not None else split
    return f"{stem}{ext}"


def create_prompts(
    task_name: str,
    method_name: str | None = None,
    primitives_path: Path | None = None,
    output_dir: Path | None = None,
    split_name: str = "all",
    seed: int = 42,
    include_assistant_prefix: bool = True,
    num_hints: int | None = None,
    force_json: bool = False,
    data_name: str | None = None,
) -> Path | dict[str, Path]:
    """
    Create prompts from primitives for a given split (or all splits).

    The task's get_split_indices() method determines which primitives
    belong to each split.

    Args:
        task_name: Name of task
        method_name: Method name for auto-derived paths and template selection
        primitives_path: Path to primitives.json (default: data/{data_name}/problems/primitives.json)
        output_dir: Directory to save prompts (default: data/{data_name}/problems_with_format/)
        split_name: Name of split (sft_train, sft_val, rl_train,
            rl_val, eval, or 'all')
        seed: Random seed for split assignment
        include_assistant_prefix: Whether to include assistant's opening
        num_hints: Number of hints to extract from prefix_hints (0-6). If None, no hint injection.
        data_name: Data directory name (default: task_name)

    Returns:
        Path to created prompts file, or dict of paths if split="all"
    """
    data_name = resolve_data_name(task_name, data_name)
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Default primitives path
    if primitives_path is None:
        primitives_path = get_primitives_path(data_name)

    # Default output directory
    if output_dir is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        output_dir = method.formatted_dir(data_name)

    # Get template variant from method or task default
    template_variant = None
    if method is not None:
        template_variant = method.template_variant
    else:
        template_variant = getattr(task, "default_template_variant", None)

    assistant_prefix = getattr(task, "assistant_prefix", None)

    # Handle "all" splits
    if split_name == "all":
        output_dir.mkdir(parents=True, exist_ok=True)
        # Ask the task which splits it defines rather than hardcoding a list:
        # code_output has no rl_val, and a fixed list made "all" die partway
        # through, leaving a half-written prompts dir behind.
        splits = task.supported_splits()
        results = {}
        for split in splits:
            output_path = output_dir / _prompts_filename(split, method, force_json)
            results[split] = _create_prompts_single(
                task=task,
                task_name=task_name,
                data_name=data_name,
                primitives_path=primitives_path,
                output_path=output_path,
                split_name=split,
                template_variant=template_variant,
                seed=seed,
                include_assistant_prefix=include_assistant_prefix,
                assistant_prefix=assistant_prefix,
                method=method,
                num_hints=num_hints,
            )
        return results

    # Single split
    output_path = output_dir / _prompts_filename(split_name, method, force_json)
    return _create_prompts_single(
        task=task,
        task_name=task_name,
        data_name=data_name,
        primitives_path=primitives_path,
        output_path=output_path,
        split_name=split_name,
        template_variant=template_variant,
        seed=seed,
        include_assistant_prefix=include_assistant_prefix,
        assistant_prefix=assistant_prefix,
        method=method,
        num_hints=num_hints,
    )


def _extract_hints_list(primitive: dict) -> list[str]:
    """Hint text per level, from whichever field the task populates:
    `hint_exprs` (list, e.g. countdown) or `prefix_hints` (dict `hint_1..hint_6`,
    e.g. math)."""
    hints_list = primitive.get("hint_exprs") or []
    if not hints_list:
        prefix_hints = primitive.get("prefix_hints") or {}
        for i in range(1, 7):
            key = f"hint_{i}"
            if key in prefix_hints:
                hints_list.append(prefix_hints[key])
    return hints_list


def _sample_hint_levels(
    rng: random.Random,
    max_hint_level: int,
    k: int = HINT_LEVELS_PER_PROBLEM,
    decay: float = HINT_LEVEL_DECAY,
) -> list[int]:
    """Sample up to `k` distinct levels from [0, max_hint_level], weighted so
    lower levels (less hinted) are more likely than higher ones. Weighted
    sampling without replacement: each draw removes its level from the pool
    so the same level can't be picked twice."""
    levels = list(range(max_hint_level + 1))
    if len(levels) <= k:
        return levels
    pool = list(zip(levels, (decay**level for level in levels)))
    chosen = []
    for _ in range(k):
        total = sum(weight for _, weight in pool)
        r = rng.uniform(0, total)
        upto = 0.0
        for i, (level, weight) in enumerate(pool):
            upto += weight
            if upto >= r:
                chosen.append(level)
                pool.pop(i)
                break
    return chosen


def _hint_sequence_for_level(hints_list: list[str], hint_level: int, task_name: str) -> str:
    if hint_level == 0:
        return "No partial solution"
    if task_name in ("math", "sql"):
        return "\n".join(hints_list[:hint_level])
    return hints_list[hint_level - 1]


def _create_prompts_single(
    task,
    task_name: str,
    primitives_path: Path,
    output_path: Path,
    split_name: str,
    template_variant: str | None,
    seed: int,
    include_assistant_prefix: bool,
    assistant_prefix: str | None = None,
    method: "Method | None" = None,
    num_hints: int | None = None,
    data_name: str | None = None,
) -> Path:
    """Apply a method's template to one materialized partition."""
    data_name = resolve_data_name(task_name, data_name)
    # Read the partition rather than recomputing it. create_partitions wrote it
    # down precisely so that every method formats the *same* problems, and so
    # that editing a SPLITS boundary later cannot silently repartition work
    # that has already been generated against the old one.
    partition = partition_path(data_name, split_name)
    if not partition.exists():
        raise FileNotFoundError(
            f"No {split_name} partition at {partition}. "
            f"Run 'python -m pipeline create_partitions --task {task_name}' first."
        )
    primitives = load_json(partition)

    # Determine format from output path
    fmt = "parquet" if str(output_path).endswith(".parquet") else "json"
    uses_fixed_hint_levels = (
        method is not None and method.name in {"method_ac", "method_a"}
    )
    if uses_fixed_hint_levels and num_hints is None:
        raise ValueError(f"{method.name} requires --num-hints")
    if uses_fixed_hint_levels and num_hints is not None and num_hints < 0:
        raise ValueError(f"num_hints must be non-negative, got {num_hints}")
    if uses_fixed_hint_levels and num_hints == 0:
        raise ValueError(f"{method.name} requires --num-hints to be greater than 0")

    is_parquet = fmt == "parquet"

    # Interaction class name for multi-turn methods, e.g. "hint" -> "countdown_hint".
    # Only relevant to the parquet (runtime-templated) path.
    interaction_name = None
    if is_parquet and method is not None and method.multi_turn:
        interaction_name = f"{task_name}_{method.name}"

    template = None
    template_path = None
    if not is_parquet:
        # Resolve template path. Splits share one template per family: every
        # sft_* split renders sft.txt and both rl_* splits render rl.txt, so
        # the family name is the split name up to its first underscore.
        template_split = split_name.split("_")[0]
        if template_variant:
            template_path = TASKS_ROOT / task_name / "templates" / template_variant / f"{template_split}.txt"
        else:
            # Legacy fallback (no variant subdirectory)
            template_path = TASKS_ROOT / task_name / "templates" / f"{template_split}.txt"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}. "
                f"Check that --method is correct."
            )

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

    if is_parquet:
        print(f"Creating {split_name} data for {len(primitives)} primitives (template applied at runtime)...")
        if interaction_name is not None:
            print(f"  Multi-turn enabled: interaction_name={interaction_name}")
    else:
        print(f"Creating {split_name} prompts for {len(primitives)} primitives (template: {template_path})...")

    records = []
    for primitive in primitives:
        hints_list = None
        hint_levels = [None]
        if num_hints is not None:
            hints_list = _extract_hints_list(primitive)
            if uses_fixed_hint_levels:
                max_hint_level = min(num_hints, len(hints_list))
                hint_levels = _sample_hint_levels(
                    random.Random(seed + primitive["index"]), max_hint_level
                )
            if not is_parquet:
                # json path renders `{hints}` (if the template uses it) from the raw list
                primitive = {**primitive, "hints": hints_list[:num_hints]}

        # Enrich primitive with derived fields if task supports it (parquet only:
        # verl's runtime template does simple substitution, so we pre-compute fields)
        if is_parquet:
            if hasattr(task, "enrich_primitive_for_rl"):
                enriched_primitive = task.enrich_primitive_for_rl(primitive)
            else:
                enriched_primitive = primitive

        ground_truth = task.get_ground_truth(primitive)

        for i, hint_level in enumerate(hint_levels):
            hint_sequence = (
                _hint_sequence_for_level(hints_list, hint_level, task_name)
                if hint_level is not None
                else None
            )
            # Every sampled hint level for a primitive needs a distinct record
            # index; non-fixed-hint methods emit exactly one record per primitive.
            record_index = (
                primitive["index"] * HINT_LEVELS_PER_PROBLEM + i
                if uses_fixed_hint_levels
                else primitive["index"]
            )

            if is_parquet:
                record_primitive = (
                    {**enriched_primitive, "hint_sequence": hint_sequence}
                    if uses_fixed_hint_levels
                    else enriched_primitive
                )

                # Build extra_info with interaction_kwargs for SGLang multi-turn
                extra_info = {
                    "index": record_index,
                }
                if interaction_name is not None:
                    extra_info["interaction_kwargs"] = {
                        "name": interaction_name,
                        "ground_truth": ground_truth,
                    }

                record = {
                    "index": record_index,
                    "primitive": record_primitive,  # Store enriched primitive for runtime template
                    "ground_truth": ground_truth,
                    "variant": primitive.get("variant", "unknown"),
                    "split": split_name,
                    "data_source": task_name,
                    "assistant_prefix": assistant_prefix,  # For verl runtime consistency
                    "extra_info": extra_info,  # For SGLang interaction system
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ground_truth,
                    },
                }
            else:
                prompt_template = (
                    template.replace("{hint_sequence}", hint_sequence)
                    if uses_fixed_hint_levels
                    else template
                )
                prompt = task.format_prompt(primitive, prompt_template, include_assistant_prefix)
                record = {
                    "index": record_index,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "variant": primitive.get("variant", "unknown"),
                    "split": split_name,
                }
                if uses_fixed_hint_levels:
                    record["hint_sequence"] = hint_sequence

            if uses_fixed_hint_levels:
                record["source_index"] = primitive["index"]
                record["hint_level"] = hint_level
            records.append(record)

    # Print reminder for verl config
    if is_parquet and assistant_prefix:
        print(f"  Note: Set verl config data.runtime_assistant_prefix=\"{assistant_prefix}\"")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        save_parquet(output_path, records)
    else:
        save_json(output_path, records)

    print(f"Saved {len(records)} records to {output_path}")
    return output_path


def _create_verification_data_method_c(
    task_name: str,
    generations_path: Path,
    sft_fraction: float = 0.1,
    seed: int = 42,
    run_id: str | None = None,
    split: str = "train",
    data_name: str | None = None,
) -> Path:
    """Build verifier prompts for method_c from single representative solves.

    Unlike method_a, there is no aggregation across samples: each training
    example is one representative solve attempt, and the ground truth is
    simply whether that attempt was correct (1) or not (0). The input can be
    either shape `generations_path` points at:
      - an `evaluate --num-samples N` results file (e.g. from
        `create_verification_data --run-solver`): one representative sample
        is drawn at random per problem (seeded per-index, so reruns are
        stable), preserving that sample's own correctness as the label --
        this is "random" selection, not the solver's pass-rate.
      - the older flat list from `generate --sample-strategy random_correct`:
        each record already is the single representative solve.
    A `sft_fraction` slice of the resulting records is written out as SFT
    prompts (to warm-start the verifier); the rest is written as RL prompts.
    The actual verifier judgment (the <think>/<answer> generation) is
    produced later by running `pipeline generate` against these prompts.

    `split` selects "train" (writes sft_train + rl_train, the default) or
    "val" (writes sft_val + rl_val, e.g. for an RL --val-prompts set).

    Output paths are derived automatically (same convention as `create_prompts`):
    both files land in `problems_with_format/`, named
    `sft_{split}__method_c__{run_id}.json` and `rl_{split}__method_c__{run_id}.parquet`
    (run-id suffix omitted if not given).
    """
    data_name = resolve_data_name(task_name, data_name)
    if not 0 < sft_fraction < 1:
        raise ValueError(f"sft_fraction must be in (0, 1), got {sft_fraction}")
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    method = Method.load("method_c", task_name)
    output_path_sft = method.formatted_path(data_name, f"sft_{split}", desc=run_id)
    output_path_rl = method.formatted_path(data_name, f"rl_{split}", desc=run_id)

    task = get_task(task_name)
    template = method.load_template(task_name, "sft")
    generations = load_json(generations_path)
    if isinstance(generations, dict) and "details" in generations:
        generations = generations["details"]
    assistant_prefix = getattr(task, "assistant_prefix", None)

    rng = random.Random(seed)
    order = list(range(len(generations)))
    rng.shuffle(order)
    n_sft = max(1, int(len(order) * sft_fraction))
    sft_positions = set(order[:n_sft])

    sft_records = []
    rl_records = []
    n_correct = 0

    for pos, source in enumerate(generations):
        index = source.get("index")
        ground_truth = source.get("ground_truth")
        raw_samples = source.get("samples")
        if raw_samples is not None:
            # evaluate()-style multi-sample aggregate: draw one representative
            # sample per problem instead of expecting it pre-selected. Seeded
            # per-index (not the shared `rng` above, which drives the SFT/RL
            # split) so the pick is stable across reruns regardless of
            # problem order.
            chosen = random.Random(seed + index).choice(raw_samples)
            generation_text = chosen.get("generation")
            is_correct = chosen.get("correct")
        else:
            generation_text = source.get("generation")
            is_correct = source.get("correct")
        if not isinstance(ground_truth, dict):
            raise ValueError(f"Index {index}: missing ground_truth")
        if generation_text is None:
            raise ValueError(f"Index {index}: missing generation")
        if not isinstance(is_correct, bool):
            raise ValueError(f"Index {index}: missing/invalid 'correct' flag")

        label = int(is_correct)
        n_correct += label
        record_ground_truth = {"correct": label}

        rendered_template = template.replace("{solution}", generation_text)
        prompt = task.format_prompt(ground_truth, rendered_template, include_assistant_prefix=True)

        if pos in sft_positions:
            sft_records.append({
                "index": index,
                "run_id": run_id,
                "prompt": prompt,
                "ground_truth": record_ground_truth,
                "variant": source.get("variant", "unknown"),
                "split": f"sft_{split}",
            })
        else:
            rl_records.append({
                "index": index,
                "run_id": run_id,
                "primitive": {
                    "problem": ground_truth.get("problem"),
                    "solution": generation_text,
                },
                "ground_truth": record_ground_truth,
                "variant": source.get("variant", "unknown"),
                "split": f"rl_{split}",
                "data_source": task_name,
                "assistant_prefix": assistant_prefix,
                "extra_info": {"index": index},
                "reward_model": {"style": "rule", "ground_truth": record_ground_truth},
            })

    output_path_sft.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path_sft, sft_records)
    print(f"Saved {len(sft_records)} SFT verifier records to {output_path_sft}")

    output_path_rl.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(output_path_rl, rl_records)
    print(f"Saved {len(rl_records)} RL verifier records to {output_path_rl}")
    print(f"Overall correct: {n_correct}/{len(generations)}")

    return output_path_sft


def _create_verification_data_method_a(
    task_name: str,
    generations_path: Path,
    output_path: Path | None,
    method_name: str,
    num_samples: int,
    threshold: float,
) -> Path:
    """Build predictor SFT data from a multi-sample solver aggregate.

    Each record's label is a pass-rate threshold: 1 if the solver answered
    correctly in at least `threshold` of its `num_samples` rollouts, else 0.
    """
    if output_path is None:
        raise ValueError("--output is required for method_name != 'method_c'")
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    task = get_task(task_name)
    method = Method.load(method_name, task_name)
    template = method.load_template(task_name, "sft")
    generations = load_json(generations_path)
    # `evaluate --num-samples N` (the --run-solver path) writes a results dict
    # with a "details" list, one entry per problem, each carrying every raw
    # sample and its own correctness under "samples"/"n_samples"/"n_correct".
    # A flat list is the older `generate`-produced aggregate (one best sample
    # per problem, no raw per-sample record) -- still accepted for anyone
    # pointing --generations at a file built that way.
    if isinstance(generations, dict) and "details" in generations:
        generations = generations["details"]

    records = []
    positives = 0
    for source in generations:
        index = source.get("index")
        actual_samples = source.get("num_samples", source.get("n_samples"))
        correct_samples = source.get("num_correct_samples", source.get("n_correct"))
        raw_samples = source.get("samples")  # per-sample generation + correctness, when available

        if actual_samples != num_samples:
            raise ValueError(
                f"Index {index}: expected num_samples={num_samples}, "
                f"got {actual_samples!r}"
            )
        if (
            not isinstance(correct_samples, int)
            or not 0 <= correct_samples <= actual_samples
        ):
            raise ValueError(
                f"Index {index}: invalid num_correct_samples={correct_samples!r}"
            )

        pass_rate = correct_samples / actual_samples
        recorded_pass_rate = source.get("pass_rate")
        if recorded_pass_rate is not None and (
            not isinstance(recorded_pass_rate, (int, float))
            or abs(float(recorded_pass_rate) - pass_rate) > 1e-12
        ):
            raise ValueError(
                f"Index {index}: pass_rate={recorded_pass_rate!r} does not match "
                f"{correct_samples}/{actual_samples}"
            )

        hint_sequence = source.get("hint_sequence")
        ground_truth = source.get("ground_truth")
        if hint_sequence is None:
            raise ValueError(f"Index {index}: missing hint_sequence")
        if not isinstance(ground_truth, dict):
            raise ValueError(f"Index {index}: missing ground_truth")

        rendered_template = template.replace("{hint_sequence}", hint_sequence)
        prompt = task.format_prompt(
            ground_truth,
            rendered_template,
            include_assistant_prefix=False,
        )
        label = int(pass_rate >= threshold)
        positives += label
        record = {
            "index": index,
            "source_index": source.get("source_index", index),
            "hint_level": source.get("hint_level"),
            "hint_sequence": hint_sequence,
            "prompt": prompt,
            "generation": f"<answer>{label}</answer>",
            "label": label,
            # This completion is the gold classification label, so both classes
            # are valid SFT examples and pass the existing task-level filter.
            "correct": True,
            "pass_rate": pass_rate,
            "num_correct_samples": correct_samples,
            "num_samples": actual_samples,
            "variant": source.get("variant", "unknown"),
            "split": source.get("split"),
        }
        if raw_samples is not None:
            # Keeps the label auditable against the rollouts it was computed
            # from: each entry is one of the solver's N samples with its own
            # generation text and correctness flag.
            record["solver_samples"] = raw_samples
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, records)
    print(
        f"Saved {len(records)} verification records to {output_path} "
        f"(label 1: {positives}, label 0: {len(records) - positives})"
    )
    return output_path


def create_verification_data(
    task_name: str,
    generations_path: Path | None = None,
    output_path: Path | None = None,
    method_name: str = "method_a",
    num_samples: int = 10,
    threshold: float = 0.5,
    sft_fraction: float = 0.1,
    seed: int = 42,
    run_id: str | None = None,
    split: str = "train",
    data_name: str | None = None,
    models_name: str | None = None,
    run_solver: bool = False,
    solver_method_name: str = "method_ac",
    solver_run_id: str | None = None,
    solver_split: str = "sft_val",
    solver_output_path: Path | None = None,
    solver_batch_size: int = 16,
    solver_max_new_tokens: int = 2048,
    solver_temperature: float = 0.7,
    solver_top_p: float = 0.9,
    solver_tensor_parallel_size: int = 1,
    solver_data_parallel_size: int = 1,
    solver_gpu_memory_utilization: float = 0.9,
    solver_use_async: bool = False,
    solver_seed: int = 42,
) -> Path:
    """Convert multi-sample solver aggregates into binary predictor SFT data.

    When `run_solver` is set, the multi-sample solver aggregate isn't read
    from disk -- it's produced first by evaluating `num_samples` rollouts per
    prompt with the solver's RL model (equivalent to `pipeline evaluate
    --model rl --method {solver_method_name} --run-id {solver_run_id}
    --split {solver_split} --num-samples {num_samples}`), and that output
    feeds straight into the label-building logic below. `evaluate` (unlike
    `generate`) keeps every one of the N raw samples and its individual
    correctness rather than collapsing to one representative sample, which is
    exactly the per-sample signal the predictor's pass-rate label is built
    from -- and those raw samples are carried into the output records here too,
    so the label stays auditable against the rollouts that produced it.
    """
    if run_solver:
        if generations_path is not None:
            raise ValueError(
                "--generations is not allowed together with --run-solver -- "
                "the solver's own output becomes the generations file."
            )
        from pipeline.commands.inference import evaluate as run_evaluate
        from pipeline.commands.inference import resolve_eval_output_path

        resolved_data_name = resolve_data_name(task_name, data_name)
        resolved_models_name = resolve_models_name(resolved_data_name, models_name)
        solver_method = Method.load(solver_method_name, task_name)
        if solver_output_path is None:
            solver_output_path = resolve_eval_output_path(
                solver_method, "rl", resolved_models_name, solver_run_id,
                solver_split, num_samples,
            )

        if solver_output_path.exists():
            # Same solver config (method/run-id/split/num-samples) as an
            # earlier call -- e.g. method_a built this already and method_c
            # is reusing it -- so there's nothing new to generate.
            print(f"Reusing existing solver eval at {solver_output_path}; "
                  f"skipping re-run.")
            generations_path = solver_output_path
        else:
            print(
                f"Running solver ({solver_method_name} rl model, run_id="
                f"{solver_run_id}) for {num_samples} samples/prompt on split "
                f"'{solver_split}'..."
            )
            generations_path = run_evaluate(
                task_name=task_name,
                model_name="rl",
                method_name=solver_method_name,
                run_id=solver_run_id,
                split=solver_split,
                output_path=solver_output_path,
                num_samples=num_samples,
                batch_size=solver_batch_size,
                max_new_tokens=solver_max_new_tokens,
                temperature=solver_temperature,
                top_p=solver_top_p,
                tensor_parallel_size=solver_tensor_parallel_size,
                data_parallel_size=solver_data_parallel_size,
                gpu_memory_utilization=solver_gpu_memory_utilization,
                use_async=solver_use_async,
                seed=solver_seed,
                data_name=data_name,
                models_name=models_name,
            )
            print(f"Solver eval written to {generations_path}; building verifier data...")
    elif generations_path is None:
        raise ValueError("--generations is required unless --run-solver is set")

    if method_name == "method_c":
        return _create_verification_data_method_c(
            task_name=task_name,
            generations_path=generations_path,
            sft_fraction=sft_fraction,
            seed=seed,
            run_id=run_id,
            split=split,
            data_name=data_name,
        )
    elif method_name == "method_a":
        return _create_verification_data_method_a(
            task_name=task_name,
            generations_path=generations_path,
            output_path=output_path,
            method_name=method_name,
            num_samples=num_samples,
            threshold=threshold,
        )
    else:
        raise ValueError(f"Unsupported method_name: {method_name!r}")


def generate_tree(
    task_name: str,
    model_name: str,
    method_name: str,
    split: str,
    run_id: str | None = None,
    output_path: Path | None = None,
    data_name: str | None = None,
    models_name: str | None = None,
    num_midpoints: int = 0,
    num_samples: int = 10,
    threshold: float = 0.5,
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    use_async: bool = False,
    seed: int | None = 42,
) -> Path:
    """Build a tree of solver rollouts, probing pass rate at `num_midpoints`
    points between the root (no partial solution) and the leaf (one fully
    representative rollout).

    Root step (i == 0): runs the solver (`evaluate`) on the prompts already
    created for this method/split -- no hints or other method-specific
    machinery here, just `num_samples` independent rollouts per existing
    prompt. Each prompt's label is a pass-rate threshold exactly like
    `create_verification_data`'s method_a (1 if pass_rate >= threshold, else
    0), but the recorded "generation" is the bare label ("0"/"1"), not
    method_a's `<answer>{label}</answer>` completion -- there is no predictor
    template involved here, just the raw rollouts and their aggregate label.
    One of the root's `num_samples` rollouts is also picked uniformly at
    random (not conditioned on correctness) as this datapoint's single
    "representative" generation -- fixed for the rest of this call, used by
    every later step below as the one trajectory being probed at different
    points.

    Midpoint steps (0 < i <= num_midpoints): distance_i = i / (num_midpoints +
    1). The representative generation is truncated to its first
    `distance_i` fraction of *words* (plain whitespace split, no tokenizer),
    and that partial text is baked into an augmented prompts file as each
    prompt's assistant-turn prefix (original prefix + partial words). That
    file is then run through `evaluate` exactly like the root step -- same
    batched generation, same correctness checking (against the real
    ground truth, not a binary label) -- which keeps this step as a thin
    wrapper around the root's machinery rather than a separate generation
    path. The record stores the partial text plus evaluate's samples (with
    the partial prefix re-attached to each sample's generation text, so the
    full reasoning chain stays readable) -- i.e. "original input + partial
    generation" is the new datapoint, same convention as the root.

    Leaf step (i == num_midpoints + 1, distance == 1): no new generation --
    this step reuses the representative generation chosen at the root
    verbatim and labels it from that rollout's own (already known)
    correctness, since distance 1 means the "partial" generation already is
    the whole thing.

    The full accumulated record list (every depth so far) is rewritten to
    `output_path` after every i, so a later step failing never loses the
    ones before it.

    Args:
        task_name: Name of task
        model_name: Model to use for generation (or "sft"/"rl" shortcut, resolved
            against the method's run, same as `generate`)
        method_name: Method name -- required, same as every other method here;
            works with any method
        split: Which split to generate from. Required -- there is no default,
            since picking one silently would make this easy to point at the
            wrong partition.
        run_id: Run identifier for model resolution (used when model_name is
            "sft" or "rl")
        output_path: Where to save the generated tree (default:
            data/{data_name}/sft_datasets/{split}__{method}__tree.json)
        data_name: Data directory name (default: task_name)
        models_name: Models directory name (default: data_name)
        num_midpoints: Number of probe points strictly between root and leaf
            (default 0 -- just root + leaf, leaf then being distance 1 i.e.
            the representative rollout's own label).
        num_samples: Rollouts per prompt at the root step and at each midpoint.
        threshold: Label 1 when pass_rate >= threshold (default 0.5).
        ... generation config, forwarded to `evaluate`/the generator ...

    Returns:
        Path to created output file
    """
    if not split:
        raise ValueError("--split is required for generate_tree (no default split)")
    if num_midpoints < 0:
        raise ValueError(f"num_midpoints must be >= 0, got {num_midpoints}")

    data_name = resolve_data_name(task_name, data_name)
    models_name = resolve_models_name(data_name, models_name)

    method = Method.load(method_name, task_name)

    # Prompts already created for this method/split -- no hints or other
    # method-specific handling here, just whatever `create_prompts` produced.
    prompts_path = method.formatted_path(data_name, split)

    if output_path is None:
        output_path = method.dataset_path(data_name, split, desc="tree")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    num_steps = num_midpoints + 1  # number of i>0 iterations (midpoints + leaf)

    # Per-index state carried from the root to every later step: the original
    # conversation (minus its trailing assistant-prefix message, kept
    # separately), and the one representative rollout picked at the root --
    # fixed for the rest of this call.
    rep_state: dict = {}
    all_records: list = []

    for i in range(num_steps + 1):
        is_root = (i == 0)
        distance = 0.0 if is_root else i / num_steps
        is_leaf = (not is_root) and distance == 1.0

        if is_leaf:
            # distance == 1: the "partial" generation already is the whole
            # representative rollout -- no new sampling, just its own label.
            for index, state in rep_state.items():
                label = int(state["rep_correct"])
                assistant_content = state["assistant_prefix"]
                rep_text = " ".join(state["rep_words"])
                if rep_text:
                    assistant_content = f"{assistant_content} {rep_text}"
                all_records.append({
                    **state["extra_meta"],
                    "index": index,
                    "depth": i,
                    "distance": 1.0,
                    "ground_truth": state["ground_truth"],
                    "variant": state["variant"],
                    "prompt": state["base_conversation"] + [{"role": "assistant", "content": assistant_content}],
                    "num_samples": 1,
                    "num_correct_samples": label,
                    "pass_rate": float(label),
                    "label": label,
                    "generation": str(label),
                })
            save_json(output_path, all_records)
            print(f"Saved leaf-level records (depth {i}, distance 1.0) to {output_path}")
            continue

        # Root (distance 0) and midpoints (0 < distance < 1) both just run
        # `evaluate` on a prompts file and threshold the pass rate -- the
        # only difference is which prompts file. Root reuses the prompts
        # already created for this method/split verbatim; a midpoint bakes
        # the representative rollout's first `distance` fraction of words
        # into each prompt's assistant-turn prefix and writes that out as a
        # scratch prompts file first.
        from pipeline.commands.inference import evaluate

        if is_root:
            prompts_data = load_json(prompts_path)
            prompts_by_index = {p["index"]: p for p in prompts_data}
            eval_prompts_path = prompts_path
        else:
            prompts_by_index = {}
            augmented_prompts = []
            for index, state in rep_state.items():
                cutoff = round(distance * len(state["rep_words"]))
                partial_text = " ".join(state["rep_words"][:cutoff])
                assistant_content = state["assistant_prefix"]
                if partial_text:
                    assistant_content = f"{assistant_content} {partial_text}"
                prompt = state["base_conversation"] + [{"role": "assistant", "content": assistant_content}]
                prompts_by_index[index] = {
                    "index": index,
                    "prompt": prompt,
                    "ground_truth": state["ground_truth"],
                    "variant": state["variant"],
                }
                augmented_prompts.append(prompts_by_index[index])

            eval_prompts_path = method.scratch_path(data_name, split, desc=f"tree_depth{i}_prompts")
            save_json(eval_prompts_path, augmented_prompts)

        depth_eval_path = method.scratch_path(
            data_name, split, desc="tree_root_eval" if is_root else f"tree_depth{i}_eval"
        )
        results_path = evaluate(
            task_name=task_name,
            model_name=model_name,
            method_name=method_name,
            run_id=run_id,
            split=split,
            prompts_path=eval_prompts_path,
            output_path=depth_eval_path,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            use_async=use_async,
            seed=seed,
            data_name=data_name,
            models_name=models_name,
        )

        results = load_json(results_path)
        details = results["details"] if isinstance(results, dict) and "details" in results else results

        positives = 0
        for detail in details:
            index = detail.get("index")
            actual_samples = detail.get("n_samples")
            correct_samples = detail.get("n_correct")
            samples = detail.get("samples") or []
            if actual_samples != num_samples:
                raise ValueError(
                    f"Index {index}: expected n_samples={num_samples}, "
                    f"got {actual_samples!r}"
                )

            pass_rate = correct_samples / actual_samples
            label = int(pass_rate >= threshold)
            positives += label

            # Carry forward every other field from the original prompts file
            # (hint_sequence, source_index, hint_level, split, etc.)
            if is_root:
                extra_meta = {
                    k: v for k, v in prompts_by_index[index].items()
                    if k not in ("index", "prompt", "ground_truth", "variant")
                }
            else:
                extra_meta = rep_state[index]["extra_meta"]

            all_records.append({
                **extra_meta,
                "index": index,
                "depth": i,
                "distance": distance,
                "ground_truth": detail.get("ground_truth"),
                "variant": detail.get("variant", "unknown"),
                "prompt": prompts_by_index[index]["prompt"],
                "samples": samples,
                "num_samples": actual_samples,
                "num_correct_samples": correct_samples,
                "pass_rate": pass_rate,
                "label": label,
                "generation": str(label),
            })

            if is_root:
                # Pick this datapoint's one representative rollout -- uniform
                # at random, not conditioned on correctness -- and remember
                # everything later steps need to keep probing it.
                rep_sample = rng.choice(samples)
                conversation = prompts_by_index[index]["prompt"]
                if conversation and conversation[-1]["role"] == "assistant":
                    base_conversation = conversation[:-1]
                    assistant_prefix = conversation[-1]["content"]
                else:
                    base_conversation = conversation
                    assistant_prefix = ""

                rep_state[index] = {
                    "base_conversation": base_conversation,
                    "assistant_prefix": assistant_prefix,
                    "rep_words": rep_sample["generation"].split(),
                    "rep_correct": bool(rep_sample["correct"]),
                    "ground_truth": detail.get("ground_truth"),
                    "variant": detail.get("variant", "unknown"),
                    "extra_meta": extra_meta,
                }

        save_json(output_path, all_records)
        print(
            f"Saved depth {i} (distance {distance:.3f}) records to {output_path} "
            f"(label 1: {positives}, label 0: {len(details) - positives})"
        )

    print(f"\nLabel distribution by depth ({output_path}):")
    for depth in range(num_steps + 1):
        depth_records = [r for r in all_records if r["depth"] == depth]
        n = len(depth_records)
        if n == 0:
            continue
        n_pos = sum(r["label"] for r in depth_records)
        distance = depth_records[0]["distance"]
        print(
            f"  depth {depth} (distance {distance:.3f}): n={n}  "
            f"label 0: {(n - n_pos) / n:.2%}  label 1: {n_pos / n:.2%}"
        )

    return output_path


# === OOD (Out-of-Distribution) evaluation datasets ===


OOD_DATASETS = {
    "minerva_math": "Minerva Math university-level STEM (272 problems)",
    "math500": "MATH-500 test subset (500 problems)",
    "olympiad_bench": "OlympiadBench text-only English math (674 problems)",
    "gsm8k": "GSM8K grade-school math (1,319 test problems)",
    "aime2024": "AIME 2024 I + II (30 problems)",
    "unanswerable_math": "Synthetic Unanswerable Math (500 answerable + 500 unanswerable)",
}


def _load_ood_minerva_math(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load Minerva Math dataset (math-ai/minervamath). 272 university-level STEM problems."""
    import random
    from datasets import load_dataset

    ds = load_dataset("math-ai/minervamath", split="test")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        primitives.append({
            "index": idx,
            "variant": "minerva_math",
            "level": "university",
            "problem": row["question"],
            "answer": row["answer"],
        })
    return primitives


def _load_ood_math500(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load MATH-500 dataset (di-zhang-fdu/MATH500)."""
    import random
    from datasets import load_dataset

    ds = load_dataset("di-zhang-fdu/MATH500", split="test")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        primitives.append({
            "index": idx,
            "variant": row["subject"],
            "level": f"Level {row['level']}",
            "problem": row["problem"],
            "answer": row["answer"],
        })
    return primitives


def _load_ood_olympiad_bench(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load OlympiadBench text-only English math (Hothan/OlympiadBench)."""
    import random
    from datasets import load_dataset

    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        # final_answer is a list; join with comma for multi-answer problems
        answers = row["final_answer"]
        answer = answers[0] if len(answers) == 1 else ", ".join(answers)

        primitives.append({
            "index": idx,
            "variant": row.get("subfield", "unknown"),
            "level": row.get("difficulty", "unknown"),
            "problem": row["question"],
            "answer": answer,
            "answer_type": row.get("answer_type", "unknown"),
            "is_multiple_answer": row.get("is_multiple_answer", False),
        })
    return primitives


def _load_ood_gsm8k(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load GSM8K test set (openai/gsm8k)."""
    import random
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        # Extract numeric answer after "####"
        answer = row["answer"].split("####")[-1].strip()
        primitives.append({
            "index": idx,
            "variant": "gsm8k",
            "level": "grade_school",
            "problem": row["question"],
            "answer": answer,
        })
    return primitives


def _load_ood_aime2024(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load AIME 2024 (HuggingFaceH4/aime_2024)."""
    import random
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_problems is not None:
        rows = rows[:num_problems]

    primitives = []
    for idx, row in enumerate(rows):
        primitives.append({
            "index": idx,
            "variant": "aime",
            "level": "competition",
            "problem": row["problem"],
            "answer": str(row["answer"]),
        })
    return primitives


def _load_ood_unanswerable_math(num_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load Synthetic Unanswerable Math (lime-nlp/Synthetic_Unanswerable_Math).

    Each row contains an answerable and unanswerable version. By default,
    samples 500 rows producing 500 answerable + 500 unanswerable primitives
    (1000 total). If num_problems is given, that many rows are sampled and
    both versions are created from each row.
    """
    import random
    from datasets import load_dataset

    ds = load_dataset("lime-nlp/Synthetic_Unanswerable_Math", data_files="synthetic_unanswerable_math.parquet", split="train")
    rows = list(ds)

    rng = random.Random(seed)
    rng.shuffle(rows)
    num_rows = (num_problems or 500)
    rows = rows[:num_rows]

    primitives = []
    for idx, row in enumerate(rows):
        primitives.append({
            "index": idx,
            "variant": "answerable",
            "level": "unknown",
            "problem": row["answerable_question"],
            "answer": row["ground_truth"],
        })
        primitives.append({
            "index": num_rows + idx,
            "variant": "unanswerable",
            "level": "unknown",
            "problem": row["unanswerable_question"],
            "answer": None,
        })
    return primitives


_OOD_LOADERS = {
    "minerva_math": _load_ood_minerva_math,
    "math500": _load_ood_math500,
    "olympiad_bench": _load_ood_olympiad_bench,
    "gsm8k": _load_ood_gsm8k,
    "aime2024": _load_ood_aime2024,
    "unanswerable_math": _load_ood_unanswerable_math,
}


def create_ood_prompts(
    task_name: str,
    dataset_name: str,
    method_name: str | None = None,
    output_path: Path | None = None,
    num_problems: int | None = None,
    seed: int = 42,
    include_assistant_prefix: bool = True,
    data_name: str | None = None,
) -> Path:
    """
    Create evaluation prompts from an out-of-distribution dataset.

    Loads an external math benchmark, normalizes it to the primitives format,
    then formats prompts using the task's eval template. The resulting file
    can be passed to `evaluate --prompts <path>`.

    Args:
        task_name: Task whose templates/check_correctness to use (e.g., "math")
        dataset_name: OOD dataset key (math500, olympiad_bench, gsm8k, aime2024)
        method_name: Method name for template selection and output path derivation
        output_path: Explicit output path (default: data/{data_name}/problems_with_format/eval__{method}_ood-{dataset}.json)
        num_problems: Limit number of problems (None = all)
        seed: Random seed for shuffling
        include_assistant_prefix: Whether to include assistant's opening in prompt
        data_name: Data directory name (default: task_name)

    Returns:
        Path to created prompts file
    """
    data_name = resolve_data_name(task_name, data_name)
    if dataset_name not in _OOD_LOADERS:
        available = ", ".join(sorted(_OOD_LOADERS.keys()))
        raise ValueError(f"Unknown OOD dataset: {dataset_name}. Available: {available}")

    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Default output path
    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        output_path = method.formatted_dir(data_name) / (
            f"{method.artifact_stem('eval', desc=f'ood-{dataset_name}')}.json"
        )

    # Load template (always use eval template)
    template_variant = method.template_variant if method else None
    if template_variant:
        template_path = TASKS_ROOT / task_name / "templates" / template_variant / "eval.txt"
    else:
        template_path = TASKS_ROOT / task_name / "templates" / "eval.txt"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Load OOD dataset
    print(f"Loading OOD dataset '{dataset_name}' ({OOD_DATASETS[dataset_name]})...")
    primitives = _OOD_LOADERS[dataset_name](num_problems=num_problems, seed=seed)
    print(f"Loaded {len(primitives)} problems")

    # Format prompts
    print(f"Formatting prompts (template: {template_path})...")
    records = []
    for primitive in primitives:
        prompt = task.format_prompt(primitive, template, include_assistant_prefix)
        ground_truth = {
            "answer": primitive["answer"],
            "variant": primitive.get("variant", "unknown"),
            "level": primitive.get("level", "unknown"),
            "problem": primitive["problem"],
        }
        records.append({
            "index": primitive["index"],
            "prompt": prompt,
            "ground_truth": ground_truth,
            "variant": primitive.get("variant", "unknown"),
            "split": f"ood_{dataset_name}",
        })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, records)
    print(f"Saved {len(records)} prompts to {output_path}")
    return output_path
