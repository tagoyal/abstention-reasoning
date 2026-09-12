"""
Data commands - primitives and prompts creation.
"""

import inspect
import random
from pathlib import Path

from pipeline.core.io import load_json, save_json, save_parquet
from pipeline.core.method import TASKS_ROOT, Method, get_primitives_path, partition_path
from pipeline.tasks import get_task


def create_primitives(
    task_name: str,
    output_path: Path | None = None,
    num_puzzles: int | None = None,
    seed: int = 42,
    **kwargs,
) -> Path:
    """
    Generate raw puzzle data.

    Args:
        task_name: Name of task (e.g., "countdown")
        output_path: Where to save primitives.json (default: artifacts/{task}/primitives.json)
        num_puzzles: Number of puzzles to generate (None = all available)
        seed: Random seed
        **kwargs: Additional task-specific options (e.g., tracer="uniform" for code_output)

    Returns:
        Path to created primitives.json
    """
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
        output_path = get_primitives_path(task_name)

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
    task = get_task(task_name)
    if primitives_path is None:
        primitives_path = get_primitives_path(task_name)
    primitives = load_json(primitives_path)

    written = {}
    for split in task.supported_splits():
        indices = set(task.get_split_indices(len(primitives), split, seed, primitives))
        rows = [p for p in primitives if p["index"] in indices]
        path = (output_dir / f"{split}.json") if output_dir else partition_path(task_name, split)
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
) -> Path | dict[str, Path]:
    """
    Create prompts from primitives for a given split (or all splits).

    The task's get_split_indices() method determines which primitives
    belong to each split.

    Args:
        task_name: Name of task
        method_name: Method name for auto-derived paths and template selection
        primitives_path: Path to primitives.json (default: artifacts/{task}/primitives.json)
        output_dir: Directory to save prompts (default: artifacts/{task}/problems_with_format/)
        split_name: Name of split (sft_whole, sft_train, sft_val, rl_train,
            rl_val, eval, or 'all')
        seed: Random seed for split assignment
        include_assistant_prefix: Whether to include assistant's opening
        num_hints: Number of hints to extract from prefix_hints (0-6). If None, no hint injection.

    Returns:
        Path to created prompts file, or dict of paths if split="all"
    """
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Default primitives path
    if primitives_path is None:
        primitives_path = get_primitives_path(task_name)

    # Default output directory
    if output_dir is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        output_dir = method.formatted_dir(task_name)

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
) -> Path:
    """Apply a method's template to one materialized partition."""
    # Read the partition rather than recomputing it. create_partitions wrote it
    # down precisely so that every method formats the *same* problems, and so
    # that editing a SPLITS boundary later cannot silently repartition work
    # that has already been generated against the old one.
    partition = partition_path(task_name, split_name)
    if not partition.exists():
        raise FileNotFoundError(
            f"No {split_name} partition at {partition}. "
            f"Run 'python -m pipeline create_partitions --task {task_name}' first."
        )
    primitives = load_json(partition)

    # Determine format from output path
    fmt = "parquet" if str(output_path).endswith(".parquet") else "json"
    is_method_ac = method is not None and method.name == "method_ac"
    if is_method_ac and num_hints is None:
        raise ValueError("method_ac requires --num-hints")
    if is_method_ac and num_hints is not None and num_hints < 0:
        raise ValueError(f"num_hints must be non-negative, got {num_hints}")
    if is_method_ac and num_hints == 0:
        raise ValueError("method_ac requires --num-hints to be greater than 0")

    # For RL splits (parquet), store primitives only - template applied at runtime
    # For other splits (json), apply template now
    if fmt == "parquet":
        print(f"Creating {split_name} data for {len(primitives)} primitives (template applied at runtime)...")

        # Interaction class name for multi-turn methods, e.g. "hint" -> "countdown_hint"
        interaction_name = None
        if method is not None and method.multi_turn:
            interaction_name = f"{task_name}_{method.name}"
            print(f"  Multi-turn enabled: interaction_name={interaction_name}")

        records = []
        for primitive in primitives:
            if is_method_ac and num_hints is not None:
                hints_list = primitive.get("hint_exprs", [])
                if not hints_list:
                    prefix_hints = primitive.get("prefix_hints", {})
                    for i in range(1, 7):
                        key = f"hint_{i}"
                        if key in prefix_hints:
                            hints_list.append(prefix_hints[key])
                max_hint_level = min(num_hints, len(hints_list))
                if split_name == "eval":
                    hint_levels = range(max_hint_level + 1)
                else:
                    hint_levels = [
                        random.Random(seed + primitive["index"]).randint(
                            0, max_hint_level
                        )
                    ]

            # Enrich primitive with derived fields if task supports it
            # (verl's runtime template does simple substitution, so we pre-compute fields)
            if hasattr(task, 'enrich_primitive_for_rl'):
                enriched_primitive = task.enrich_primitive_for_rl(primitive)
            else:
                enriched_primitive = primitive

            ground_truth = task.get_ground_truth(primitive)

            if not is_method_ac:
                hint_levels = [None]

            for hint_level in hint_levels:
                hint_sequence = (
                    "No partial solution"
                    if hint_level == 0
                    else "\n".join(hints_list[:hint_level])
                    if task_name == "competition_math"
                    else hints_list[hint_level - 1]
                    if hint_level is not None
                    else None
                )
                record_index = (
                    primitive["index"] * (num_hints + 1) + hint_level
                    if is_method_ac and split_name == "eval"
                    else primitive["index"]
                )
                record_primitive = (
                    {**enriched_primitive, "hint_sequence": hint_sequence}
                    if is_method_ac
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
                if is_method_ac:
                    record["source_index"] = primitive["index"]
                    record["hint_level"] = hint_level
                records.append(record)

        # Print reminder for verl config
        if assistant_prefix:
            print(f"  Note: Set verl config data.runtime_assistant_prefix=\"{assistant_prefix}\"")
    else:
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

        print(f"Creating {split_name} prompts for {len(primitives)} primitives (template: {template_path})...")
        records = []
        for primitive in primitives:
            # Inject hints if --num-hints is specified
            # Supports both hint_exprs (list, countdown) and prefix_hints (dict, competition_math)
            if num_hints is not None:
                hints_list = primitive.get("hint_exprs", [])
                if not hints_list:
                    prefix_hints = primitive.get("prefix_hints", {})
                    for i in range(1, 7):
                        key = f"hint_{i}"
                        if key in prefix_hints:
                            hints_list.append(prefix_hints[key])
                if is_method_ac:
                    max_hint_level = min(num_hints, len(hints_list))
                    if split_name == "eval":
                        hint_levels = range(max_hint_level + 1)
                    else:
                        hint_levels = [
                            random.Random(seed + primitive["index"]).randint(
                                0, max_hint_level
                            )
                        ]
                primitive = {**primitive, "hints": hints_list[:num_hints]}

            if is_method_ac:
                ground_truth = task.get_ground_truth(primitive)
                for hint_level in hint_levels:
                    hint_sequence = (
                        "No partial solution"
                        if hint_level == 0
                        else "\n".join(hints_list[:hint_level])
                        if task_name == "competition_math"
                        else hints_list[hint_level - 1]
                    )
                    prompt = task.format_prompt(
                        primitive,
                        template.replace("{hint_sequence}", hint_sequence),
                        include_assistant_prefix,
                    )
                    record = {
                        "index": (
                            primitive["index"] * (num_hints + 1) + hint_level
                            if split_name == "eval"
                            else primitive["index"]
                        ),
                        "source_index": primitive["index"],
                        "hint_level": hint_level,
                        "hint_sequence": hint_sequence,
                        "prompt": prompt,
                        "ground_truth": ground_truth,
                        "variant": primitive.get("variant", "unknown"),
                        "split": split_name,
                    }
                    records.append(record)
                continue

            prompt = task.format_prompt(primitive, template, include_assistant_prefix)
            ground_truth = task.get_ground_truth(primitive)
            record = {
                "index": primitive["index"],
                "prompt": prompt,
                "ground_truth": ground_truth,
                "variant": primitive.get("variant", "unknown"),
                "split": split_name,
            }
            records.append(record)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        save_parquet(output_path, records)
    else:
        save_json(output_path, records)

    print(f"Saved {len(records)} records to {output_path}")
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
) -> Path:
    """
    Create evaluation prompts from an out-of-distribution dataset.

    Loads an external math benchmark, normalizes it to the primitives format,
    then formats prompts using the task's eval template. The resulting file
    can be passed to `evaluate --prompts <path>`.

    Args:
        task_name: Task whose templates/check_correctness to use (e.g., "competition_math")
        dataset_name: OOD dataset key (math500, olympiad_bench, gsm8k, aime2024)
        method_name: Method name for template selection and output path derivation
        output_path: Explicit output path (default: artifacts/{task}/problems_with_format/eval__{method}_ood-{dataset}.json)
        num_problems: Limit number of problems (None = all)
        seed: Random seed for shuffling
        include_assistant_prefix: Whether to include assistant's opening in prompt

    Returns:
        Path to created prompts file
    """
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
        output_path = method.formatted_dir(task_name) / (
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
