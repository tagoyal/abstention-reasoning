"""
Inference commands - generation and evaluation.
"""

import random
import re
from datetime import datetime
from pathlib import Path

from pipeline.core.io import load_json, save_json
from pipeline.core.generator import Generator, GenerationConfig, AsyncGenerator
from pipeline.core.method import Method
from pipeline.core.utils import extract_answer, model_short_name
from pipeline.tasks import get_task


def extract_generation_hints(text: str) -> list[str]:
    """Extract '(using the <category> of <name>)' hints from generated text.

    Handles nested parentheses in concept names like (a+b)^2.
    """
    hints = []
    pattern_start = re.compile(r'\(using the (\w+) of ', re.IGNORECASE)

    for match in pattern_start.finditer(text):
        category = match.group(1)
        start_pos = match.end()

        # Find matching closing paren by counting depth
        depth = 1
        pos = start_pos
        while pos < len(text) and depth > 0:
            if text[pos] == '(':
                depth += 1
            elif text[pos] == ')':
                depth -= 1
            pos += 1

        if depth == 0:
            name = text[start_pos:pos - 1]
            clean_name = name.strip().replace('**', '').replace('*', '')
            hint = f'{category} of {clean_name}'
            hints.append(hint)

    return hints


def classify_truncation(record: dict) -> str | None:
    """Which stage the token budget cut off, or None if it did not.

    - "think": ran out before </think>. Benign -- RL rollouts (and generate
      with --answer-budget) close the block and force a short answer, so only
      the reasoning is cut. Without forcing the sample is still malformed.
    - "answer": the answer itself is cut. Ran out after </think>, or the
      forced answer also ran out and the closing tag was appended. Nothing
      downstream repairs this.
    """
    if record.get("answer_truncated"):
        return "answer"
    if record.get("forced_answer"):
        return "think"
    if record.get("finish_reason") != "length":
        return None
    return "answer" if "</think>" in record.get("generation", "") else "think"


def count_truncation(records: list[dict]) -> dict[str, int]:
    kinds = [classify_truncation(r) for r in records]
    return {"think": kinds.count("think"), "answer": kinds.count("answer")}


def compute_hint_metrics(details: list[dict]) -> dict:
    """
    Compute hint usage metrics for multi-turn evaluation.

    Returns a dict with:
    - counts_by_hints_and_variant: {num_hints: {variant: count}}
    - correct_by_hints_and_variant: {num_hints: {variant: correct_count}}
    - counts_by_hints_and_level: {num_hints: {level: count}}
    - correct_by_hints_and_level: {num_hints: {level: correct_count}}
    - total_hints: total hints used across all examples
    - avg_hints: average hints per example
    """
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    correct_counts = defaultdict(lambda: defaultdict(int))
    counts_by_level = defaultdict(lambda: defaultdict(int))
    correct_by_level = defaultdict(lambda: defaultdict(int))
    total_hints = 0
    max_hints = 0

    for record in details:
        variant = record.get("variant", "unknown")
        level = record.get("level", "unknown")
        num_hints = record.get("num_hints", 0)
        counts[num_hints][variant] += 1
        total_hints += num_hints
        max_hints = max(max_hints, num_hints)
        if level != "unknown":
            counts_by_level[num_hints][level] += 1
        if record.get("correct", False):
            correct_counts[num_hints][variant] += 1
            if level != "unknown":
                correct_by_level[num_hints][level] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    counts_dict = {h: dict(counts[h]) for h in range(max_hints + 1)}
    correct_dict = {h: dict(correct_counts[h]) for h in range(max_hints + 1)}
    counts_level_dict = {h: dict(counts_by_level[h]) for h in range(max_hints + 1)}
    correct_level_dict = {h: dict(correct_by_level[h]) for h in range(max_hints + 1)}

    return {
        "counts_by_hints_and_variant": counts_dict,
        "correct_by_hints_and_variant": correct_dict,
        "counts_by_hints_and_level": counts_level_dict,
        "correct_by_hints_and_level": correct_level_dict,
        "total_hints": total_hints,
        "avg_hints": total_hints / len(details) if details else 0,
        "max_hints": max_hints,
    }


def _format_hint_tables(
    row_labels: list[str],
    counts_map: dict,
    correct_map: dict,
    max_hints: int,
    row_header: str,
    section_title: str,
    actual_max_hints: int | None = None,
) -> list[str]:
    """Format counts and accuracy tables for a set of row labels.

    Args:
        row_labels: Sorted list of row label strings (variants or levels).
        counts_map: {num_hints: {label: count}} dict.
        correct_map: {num_hints: {label: correct_count}} dict.
        max_hints: Maximum hint count to display columns for. The last column
            becomes a "N+" bucket if actual_max_hints > max_hints.
        row_header: Column header for the row label (e.g. "Type", "Level").
        section_title: Title prefix (e.g. "TYPE", "DIFFICULTY").
        actual_max_hints: True maximum hint count in the data (for folding overflow).
    """
    if actual_max_hints is None:
        actual_max_hints = max_hints
    has_overflow = actual_max_hints > max_hints

    def get_val(mapping, h, label):
        return mapping.get(str(h), {}).get(label, mapping.get(h, {}).get(label, 0))

    def get_val_with_overflow(mapping, h, label):
        """Get value, folding overflow hints into the last displayed column."""
        if h == max_hints and has_overflow:
            total = 0
            for oh in range(max_hints, actual_max_hints + 1):
                total += get_val(mapping, oh, label)
            return total
        return get_val(mapping, h, label)

    # Compute label column width from longest label
    label_w = max(len(row_header), max((len(l) for l in row_labels), default=8)) + 2
    num_col_w = 10
    acc_col_w = 16

    lines = []

    # --- Counts table ---
    total_w = label_w + num_col_w * (max_hints + 3)
    lines.extend([
        "",
        f"COUNTS BY {section_title} AND HINTS",
        "-" * total_w,
    ])

    header = f"{row_header:<{label_w}}"
    for h in range(max_hints + 1):
        col_label = f"{h}+" if (h == max_hints and has_overflow) else str(h)
        header += f"{col_label:>{num_col_w}}"
    header += f"{'Total':>{num_col_w}}{'Hints':>{num_col_w}}"
    lines.append(header)
    lines.append("-" * total_w)

    grand_total = 0
    grand_hints = 0
    col_totals = [0] * (max_hints + 1)

    for label in row_labels:
        row = f"{label:<{label_w}}"
        row_total = 0
        row_hints = 0
        for h in range(max_hints + 1):
            c = get_val_with_overflow(counts_map, h, label)
            row_total += c
            col_totals[h] += c
            row += f"{c:>{num_col_w}}"
        # Compute true hint count from all data (not capped)
        for h in range(actual_max_hints + 1):
            row_hints += h * get_val(counts_map, h, label)
        grand_total += row_total
        grand_hints += row_hints
        row += f"{row_total:>{num_col_w}}{row_hints:>{num_col_w}}"
        lines.append(row)

    lines.append("-" * total_w)
    totals_row = f"{'Total':<{label_w}}"
    for h in range(max_hints + 1):
        totals_row += f"{col_totals[h]:>{num_col_w}}"
    totals_row += f"{grand_total:>{num_col_w}}{grand_hints:>{num_col_w}}"
    lines.append(totals_row)

    # --- Accuracy table ---
    acc_total_w = label_w + acc_col_w * (max_hints + 2)
    lines.extend([
        "",
        "",
        f"ACCURACY BY {section_title} AND HINTS",
        "-" * acc_total_w,
    ])

    header = f"{row_header:<{label_w}}"
    for h in range(max_hints + 1):
        col_label = f"{h}+" if (h == max_hints and has_overflow) else str(h)
        header += f"{col_label:>{acc_col_w}}"
    header += f"{'Total':>{acc_col_w}}"
    lines.append(header)
    lines.append("-" * acc_total_w)

    overall_correct = 0
    overall_total = 0
    col_correct_totals = [0] * (max_hints + 1)
    col_count_totals = [0] * (max_hints + 1)

    for label in row_labels:
        row = f"{label:<{label_w}}"
        row_correct = 0
        row_total = 0
        for h in range(max_hints + 1):
            c = get_val_with_overflow(counts_map, h, label)
            corr = get_val_with_overflow(correct_map, h, label)
            row_correct += corr
            row_total += c
            col_correct_totals[h] += corr
            col_count_totals[h] += c
            if c > 0:
                cell = f"{corr}/{c} ({100*corr/c:.0f}%)"
                row += f"{cell:>{acc_col_w}}"
            else:
                row += f"{'--':>{acc_col_w}}"
        overall_correct += row_correct
        overall_total += row_total
        if row_total > 0:
            cell = f"{row_correct}/{row_total} ({100*row_correct/row_total:.0f}%)"
            row += f"{cell:>{acc_col_w}}"
        else:
            row += f"{'--':>{acc_col_w}}"
        lines.append(row)

    lines.append("-" * acc_total_w)
    totals_row = f"{'Total':<{label_w}}"
    for h in range(max_hints + 1):
        c = col_count_totals[h]
        corr = col_correct_totals[h]
        if c > 0:
            cell = f"{corr}/{c} ({100*corr/c:.0f}%)"
            totals_row += f"{cell:>{acc_col_w}}"
        else:
            totals_row += f"{'--':>{acc_col_w}}"
    if overall_total > 0:
        cell = f"{overall_correct}/{overall_total} ({100*overall_correct/overall_total:.0f}%)"
        totals_row += f"{cell:>{acc_col_w}}"
    lines.append(totals_row)

    return lines


def format_hint_metrics(hint_metrics: dict, details: list[dict]) -> str:
    """Format hint metrics as tables for display.

    Prints tables grouped by variant (problem type) and, if level data is
    available, by difficulty level.
    """
    actual_max_hints = hint_metrics["max_hints"]
    max_hints = min(actual_max_hints, 5)

    _variant_order = {"short": 0, "medium": 1, "long": 2}
    variants = sorted(set(d.get("variant", "unknown") for d in details),
                      key=lambda v: _variant_order.get(v, 999))
    levels = sorted(set(d.get("level", "unknown") for d in details) - {"unknown"})

    # Aggregate totals across all variants for a single "All" row
    totals_counts: dict[int | str, dict[str, int]] = {}
    totals_correct: dict[int | str, dict[str, int]] = {}
    for h, variant_map in hint_metrics["counts_by_hints_and_variant"].items():
        totals_counts.setdefault(h, {})["All"] = sum(variant_map.values())
    for h, variant_map in hint_metrics["correct_by_hints_and_variant"].items():
        totals_correct.setdefault(h, {})["All"] = sum(variant_map.values())

    lines = [
        "",
        "=" * 90,
        "HINT USAGE ANALYSIS",
        "=" * 90,
    ]

    # Per-variant table (short/medium/long) if more than one variant
    if len(variants) > 1:
        lines.extend(_format_hint_tables(
            row_labels=variants,
            counts_map=hint_metrics["counts_by_hints_and_variant"],
            correct_map=hint_metrics["correct_by_hints_and_variant"],
            max_hints=max_hints,
            row_header="Variant",
            section_title="VARIANT",
            actual_max_hints=actual_max_hints,
        ))
    else:
        # Single variant — just show aggregated "All" row
        lines.extend(_format_hint_tables(
            row_labels=["All"],
            counts_map=totals_counts,
            correct_map=totals_correct,
            max_hints=max_hints,
            row_header="Type",
            section_title="HINTS",
            actual_max_hints=actual_max_hints,
        ))

    # Tables by difficulty level (if available)
    if levels:
        lines.append("")
        lines.extend(_format_hint_tables(
            row_labels=levels,
            counts_map=hint_metrics["counts_by_hints_and_level"],
            correct_map=hint_metrics["correct_by_hints_and_level"],
            max_hints=max_hints,
            row_header="Level",
            section_title="DIFFICULTY",
            actual_max_hints=actual_max_hints,
        ))

    return "\n".join(lines)


def extract_cot_length(generation: str) -> int:
    """
    Extract the length of chain-of-thought reasoning from generation.

    Looks for content inside <think>...</think> tags.
    Returns character count, or 0 if no CoT found.
    """
    match = re.search(r'<think>(.*?)</think>', generation, re.DOTALL)
    if match:
        return len(match.group(1))
    return 0


EXHAUSTED_MARKER = "No more hints available."


def select_best_sample(
    samples: list[dict],
    task,
    primitive: dict,
    strategy: str = "shortest_cot",
    exclude_exhausted: bool = False,
    rng=None,
) -> dict | None:
    """
    Select the best sample from multiple generations.

    Strategies:
    - "shortest_cot": Among correct samples, pick shortest CoT. Falls back to
      shortest CoT among incorrect if none correct.
    - "most_hints": Among correct samples, pick the one with most hint requests.
      Falls back to most hints among incorrect if none correct. Useful for
      generating SFT data that teaches models to use hints.
    - "random_correct": Pick uniformly at random among correct samples, and
      return None when none are correct so the caller can drop the problem.
      The other strategies all fall back to an incorrect sample, and the
      argmax ones bias the result: "most_hints" keeps the hint-richest correct
      sample, which pushes the achieved hint distribution above the scheduled
      one. Random selection leaves the schedule's shape intact.
    - "random": Pick uniformly at random among ALL samples, correct or not,
      and never drop the problem. Use this when the representative sample's
      own correctness is the signal you want to preserve (e.g. method_c's
      verifier ground truth), rather than biasing toward a solved rollout.

    Args:
        samples: List of generation results with 'text', 'finish_reason', 'token_count'
        task: Task instance for checking correctness
        primitive: Primitive data with ground truth
        strategy: Selection strategy ("shortest_cot", "most_hints", "random_correct", "random")

    Returns:
        Best sample dict with added 'correct' and 'metadata' fields
    """
    # Check correctness for all samples
    evaluated_samples = []
    for sample in samples:
        is_correct, meta = task.check_correctness(primitive, sample["text"])
        cot_length = extract_cot_length(sample["text"])
        num_hints = sample["text"].count("<request>")
        evaluated_samples.append({
            **sample,
            "correct": is_correct,
            "metadata": meta,
            "cot_length": cot_length,
            "_num_hints": num_hints,
        })

    # Drop samples that ran the hint pool dry before choosing. This matters most
    # for strategy="most_hints", which maximises request count and would
    # otherwise actively prefer an exhausted sample (one extra request, answered
    # with the placeholder) over a clean one -- and the record-level filter would
    # then discard the whole problem even though a usable sample existed.
    # If every sample is exhausted, keep them: the record filter drops it, which
    # is the right outcome.
    if exclude_exhausted:
        clean = [s for s in evaluated_samples if EXHAUSTED_MARKER not in s.get("text", "")]
        if clean:
            evaluated_samples = clean

    # Separate correct and incorrect
    correct_samples = [s for s in evaluated_samples if s["correct"]]
    incorrect_samples = [s for s in evaluated_samples if not s["correct"]]

    if strategy == "random_correct":
        if not correct_samples:
            return None  # caller drops the problem entirely
        if rng is None:
            import random as _random
            rng = _random.Random(0)
        best_sample = rng.choice(correct_samples)
    elif strategy == "random":
        if rng is None:
            import random as _random
            rng = _random.Random(0)
        best_sample = rng.choice(evaluated_samples)
    elif strategy == "most_hints":
        candidates = correct_samples if correct_samples else incorrect_samples
        best_sample = max(candidates, key=lambda s: s["_num_hints"])
    else:
        candidates = correct_samples if correct_samples else incorrect_samples
        best_sample = min(candidates, key=lambda s: s["cot_length"])

    # Remove internal fields
    del best_sample["cot_length"]
    del best_sample["_num_hints"]

    # Fraction of samples that were correct. The selection strategies above all
    # collapse to a single sample, which throws away how *reliably* the model
    # solves this problem -- exactly the signal the difficulty probe needs.
    best_sample["pass_rate"] = len(correct_samples) / len(evaluated_samples)
    best_sample["num_correct_samples"] = len(correct_samples)
    best_sample["num_samples"] = len(evaluated_samples)

    return best_sample




def drop_exhausted_records(records: list[dict]) -> tuple[list[dict], int]:
    """Remove generations that ran the hint pool dry.

    When the model requests a hint past the last one available, the generator
    serves the literal string "No more hints available." as the response. That
    is fine at RL time -- the reward function prices exhausted requests via
    exhausted_penalty -- but in SFT data it is training text, and keeping it
    teaches the warm start to emit and expect the placeholder. It also inflates
    `num_hints`, which counts requests served rather than real hints given.
    """
    keep = [r for r in records if EXHAUSTED_MARKER not in (r.get("generation") or "")]
    return keep, len(records) - len(keep)


def copy_source_fields(record: dict, prompt_data: dict) -> None:
    """Preserve fields needed to derive later datasets from this generation."""
    for key in (
        "source_index",
        "hint_level",
        "hint_sequence",
        "ground_truth",
        "split",
    ):
        if key in prompt_data:
            record[key] = prompt_data[key]


def generate(
    task_name: str,
    model_name: str,
    method_name: str | None = None,
    run_id: str | None = None,
    prompts_path: Path | None = None,
    output_path: Path | None = None,
    split: str = "sft",
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_samples: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    verbose: bool = False,
    retry_incorrect: bool = False,
    retry_truncated: bool = False,
    max_retries: int = 10,
    answer_budget: int = 0,
    seed: int | None = 42,
    multi_turn: bool | None = None,
    use_async: bool = False,
    force_hints_distribution: dict[int, float] | None = None,
    force_hints_policy: dict[str, float] | None = None,
    sample_strategy: str | None = None,
    data_parallel_size: int = 1,
    no_hints: bool = False,
    hint_schedule_from: Path | None = None,
    hint_buckets: str | None = None,
    hint_target_fraction: float | None = None,
    max_hints: int = 4,
    drop_exhausted: bool = False,
    target_correct_rate: float | None = None,
    max_oversample: int = 5,
) -> Path:
    """
    Generate model outputs on prompts.

    Supports resumption - if output file exists, skips already-completed indices.
    Saves after each batch for crash recovery.

    Args:
        task_name: Name of task
        model_name: Model to use for generation
        method_name: Method name for auto-derived paths
        run_id: Run identifier for model resolution (used when model_name="sft" or "rl")
        prompts_path: Path to prompts file (default: artifacts/{task}/problems_with_format/{split}__{method}.json)
        output_path: Where to save dataset (default: artifacts/{task}/sft_sft_datasets/{split}__{method}.json)
        split: Which split to generate from (default: sft)
        retry_incorrect: If True, re-run incorrect examples
        answer_budget: Tokens held back for a forced answer. A generation that runs
            out of room mid-thought gets </think> closed for it and this many tokens
            to answer in, matching what RL does at rollout time. 0 disables forcing.
            Async single-turn generation only.
        multi_turn: Enable multi-turn generation with hint injection. If None, uses method config.
        use_async: Use async generation for optimal throughput.
        force_hints_distribution: Distribution of forced hint counts. Maps number of hints
            to probability, e.g. {1: 0.5, 2: 0.3, 3: 0.2}. Probabilities must sum to 1.
            If None, no hints are forced.
        force_hints_policy: Per-level probability of forcing hints. Maps level number
            to rate, e.g. {"1": 0.05, "2": 0.05, "3": 0.05, "4": 0.15, "5": 0.15}.
            If None but force_hints_distribution is set, all examples get forced hints.
        data_parallel_size: Independent model replicas to shard prompts across.
        no_hints: Ban the hint request, so the model has to answer unaided.
            Also phase 1 of the difficulty schedule: measures unaided pass rate.
        hint_schedule_from: Path to a no-hint probe dataset (a `generate
            --no-hints --num-samples 8` output). Per-problem hint counts are
            derived from its pass rates instead of sampled from a global
            distribution.
        hint_buckets: "max_pass_rate:num_hints,..." mapping for the schedule.
        hint_target_fraction: Fraction of problems that should request a hint.
            Assigns hints by difficulty rank so the fraction is hit exactly,
            regardless of how the probe's pass rates are distributed. Takes
            precedence over hint_buckets.
        max_hints: Most hints any single problem is scheduled (rank mode).
        drop_exhausted: Discard records where the model asked past the last
            available hint and was served the "No more hints available."
            placeholder. That string is real training text, so leaving it in
            teaches the warm start to produce and expect it.
        target_correct_rate: Keep resampling incorrect problems until this fraction
            is answered correctly (phase 3). None disables the loop.
        max_oversample: Cap on phase-3 resampling rounds.
        ... generation config ...

    Returns:
        Path to created dataset file
    """
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Resolve multi_turn from method config (CLI overrides if explicitly set)
    if multi_turn is None:
        multi_turn = method.multi_turn if method else False
    if force_hints_distribution or hint_schedule_from is not None:
        multi_turn = True  # any hint injection requires multi_turn
    if no_hints:
        # With hints banned there is nothing to multi-turn about. This also
        # matters mechanically: the multi-turn path serves one
        # generation per prompt regardless of --num-samples, which would collapse
        # every pass rate to 0.0 or 1.0 and destroy the difficulty signal the
        # schedule is built from. Single-turn honours num_samples via n=.
        if multi_turn:
            print("--no-hints: forcing single-turn so --num-samples yields a "
                  "graded pass rate (method config requested multi-turn).")
        multi_turn = False

    # Resolve model shortcuts (sft, rl)
    actual_model_name = model_name
    if method is not None:
        if model_name == "sft":
            actual_model_name = str(method.sft_model_path(task_name, run_id))
        elif model_name == "rl":
            actual_model_name = str(method.rl_model_path(task_name, run_id))

    # Default prompts path
    if prompts_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --prompts must be specified. "
                "Use --method to auto-derive paths, or --prompts for explicit paths."
            )
        prompts_path = method.formatted_path(task_name, split)

    # Default output path
    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        output_path = method.dataset_path(task_name, split)

    # Load prompts
    prompts_data = load_json(prompts_path)
    print(f"Loaded {len(prompts_data)} prompts from {prompts_path}")

    # Initialize generator (once, reused across retry iterations)
    config = GenerationConfig(
        model_name=actual_model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_samples=num_samples,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        verbose=verbose,
        ban_hint_requests=no_hints,
        seed=seed,
        hint_transition=method.hint_transition if method else True,
        nested_request=method.nested_request if method else False,
        answer_budget=answer_budget,
    )
    generator = Generator(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_attempts = (max_retries + 1) if retry_truncated else 1
    for attempt in range(total_attempts):
        if attempt > 0:
            config.seed = seed + attempt * 1000
            print(f"\n=== Retry truncated attempt {attempt}/{max_retries} (seed={config.seed}) ===")

        # Check for existing output (for resumption)
        records_by_index = {}
        if output_path.exists():
            existing_records = load_json(output_path)
            records_by_index = {r["index"]: r for r in existing_records}

            if retry_incorrect:
                incorrect_indices = {r["index"] for r in existing_records if not r.get("correct", False)}
                done_count = len(existing_records) - len(incorrect_indices)
                print(f"Retry mode: {done_count} correct, retrying {len(incorrect_indices)} incorrect")
                completed_indices = {r["index"] for r in existing_records if r.get("correct", False)}
            elif retry_truncated:
                truncated_indices = {r["index"] for r in existing_records if r.get("finish_reason") == "length"}
                done_count = len(existing_records) - len(truncated_indices)
                print(f"Retry truncated: {done_count} done, retrying {len(truncated_indices)} truncated")
                completed_indices = {r["index"] for r in existing_records if r.get("finish_reason") != "length"}
            else:
                completed_indices = set(records_by_index.keys())
                print(f"Resuming: found {len(completed_indices)} completed examples")
        else:
            completed_indices = set()

        # Filter to remaining prompts
        remaining_prompts = [p for p in prompts_data if p["index"] not in completed_indices]
        if not remaining_prompts:
            print("All prompts already completed!")
            return output_path

        print(f"Generating {len(remaining_prompts)} remaining prompts...")

        total_batches = (len(remaining_prompts) + batch_size - 1) // batch_size

        if attempt == 0:
            print(f"Generating with {actual_model_name}...")
            if use_async:
                print("Async mode enabled (optimal throughput)")
            if multi_turn:
                print("Multi-turn mode enabled")

        # Compute per-prompt force_hints. A difficulty schedule, when supplied,
        # takes precedence: it assigns each problem a hint count from its own
        # measured pass rate rather than sampling from a global distribution, so
        # hard problems reliably get more hints instead of getting them by luck.
        if hint_schedule_from is not None:
            from pipeline.core.hint_schedule import (
                DEFAULT_BUCKETS,
                build_profile,
                parse_buckets,
                schedule_by_rank,
                schedule_from_profile,
                summarize_schedule,
            )
            profile = build_profile(load_json(hint_schedule_from))
            if profile.num_samples < 2:
                print(f"WARNING: probe {hint_schedule_from} has num_samples="
                      f"{profile.num_samples}; pass rates are 0.0 or 1.0 only, "
                      f"which buckets every problem into the extremes. Re-probe "
                      f"with --num-samples 4+ for a usable difficulty signal.")
            if hint_target_fraction is not None:
                # Rank-based: hits the requested hint fraction exactly, whatever
                # the probe's pass-rate distribution turned out to look like.
                schedule = schedule_by_rank(remaining_prompts, profile,
                                            hint_target_fraction, max_hints)
            else:
                buckets = parse_buckets(hint_buckets) if hint_buckets else DEFAULT_BUCKETS
                schedule = schedule_from_profile(remaining_prompts, profile, buckets)
            force_hints_list = [schedule[p["index"]] for p in remaining_prompts]
            print(f"Hint schedule from {hint_schedule_from} "
                  f"({len(profile.pass_rates)} problems, {profile.num_samples} samples)")
            print(summarize_schedule(schedule, profile))
        elif force_hints_distribution:
            # NB: no local `import random` here. A function-scope import makes
            # `random` local to the whole of generate(), so the selection code
            # further down hits UnboundLocalError whenever this branch is not
            # taken (it killed job 187745 33 minutes in). The module-level
            # import at the top of the file is what everything uses.
            rng = random.Random(42)

            hint_counts = sorted(force_hints_distribution.keys())
            hint_probs = [force_hints_distribution[k] for k in hint_counts]
            cum_probs = []
            cumsum = 0.0
            for p in hint_probs:
                cumsum += p
                cum_probs.append(cumsum)

            def sample_hint_count():
                r = rng.random()
                for count, cp in zip(hint_counts, cum_probs):
                    if r < cp:
                        return count
                return hint_counts[-1]

            dist_str = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(force_hints_distribution.items()))

            if force_hints_policy:
                force_hints_list = []
                level_counts = {}
                for p in remaining_prompts:
                    level_str = p.get("ground_truth", {}).get("level", "")
                    level_num = level_str.replace("Level ", "") if level_str else ""
                    rate = force_hints_policy.get(level_num, 0.0)
                    if rng.random() < rate:
                        n = sample_hint_count()
                        force_hints_list.append(n)
                        level_counts[level_str] = level_counts.get(level_str, 0) + 1
                    else:
                        force_hints_list.append(0)
                forced_total = sum(1 for f in force_hints_list if f > 0)
                print(f"Force hints policy: {forced_total}/{len(remaining_prompts)} examples will get forced hints")
                print(f"  Distribution: {dist_str}")
                for level in sorted(level_counts.keys()):
                    print(f"  {level}: {level_counts[level]} forced")
            else:
                force_hints_list = [sample_hint_count() for _ in remaining_prompts]
                print(f"Force hints enabled for all {len(remaining_prompts)} examples")
                print(f"  Distribution: {dist_str}")
        else:
            force_hints_list = [0] * len(remaining_prompts)

        # For async multi-turn, process all remaining prompts at once
        if multi_turn and use_async:
            import asyncio

            all_prompts = [p["prompt"] for p in remaining_prompts]
            all_ground_truths = [p["ground_truth"] for p in remaining_prompts]

            # generate_with_hints_async has no num_samples parameter: the
            # multi-turn loop carries per-prompt state across turns, so one call
            # is one rollout. Replicate the prompt list instead and regroup after
            # -- each replica gets its own request index, and the sampling seed is
            # base_seed + index, so the copies diverge rather than repeating.
            # Without this, --num-samples is silently ignored here, and SFT's
            # correct-only filter then thins the hard, hint-heavy buckets.
            rep_prompts = [p for p in all_prompts for _ in range(num_samples)]
            rep_gts = [g for g in all_ground_truths for _ in range(num_samples)]
            rep_force = [f for f in force_hints_list for _ in range(num_samples)]
            print(f"Running async generation on {len(all_prompts)} prompts"
                  + (f" x {num_samples} samples = {len(rep_prompts)} rollouts..."
                     if num_samples > 1 else "..."))
            async_generator = AsyncGenerator(config)
            try:
                rep_results = asyncio.run(
                    async_generator.generate_with_hints_async(
                        rep_prompts,
                        rep_gts,
                        force_hints=rep_force,
                    )
                )
            finally:
                # Free the KV cache before anything builds another engine in this
                # process (the oversampling loop does, once per round).
                async_generator.close()

            # Regroup the flat replica list back into num_samples per prompt.
            all_results = [
                [r for j in range(num_samples) for r in rep_results[i * num_samples + j]]
                for i in range(len(all_prompts))
            ]

            sample_strategy = sample_strategy or ("most_hints" if multi_turn else "shortest_cot")
            n_no_correct = 0  # dropped by strategy="random_correct"
            for prompt_data, gen_samples in zip(remaining_prompts, all_results):
                primitive = {
                    "index": prompt_data["index"],
                    **prompt_data["ground_truth"],
                    "variant": prompt_data.get("variant", "unknown"),
                }

                if num_samples > 1:
                    # Seed per problem index so a rerun picks the same sample.
                    _rng = random.Random((seed or 0) + prompt_data["index"])
                    best_sample = select_best_sample(gen_samples, task, primitive,
                                                     strategy=sample_strategy,
                                                     exclude_exhausted=drop_exhausted,
                                                     rng=_rng)
                    if best_sample is None:
                        # strategy="random_correct" and nothing was correct.
                        n_no_correct += 1
                        records_by_index.pop(prompt_data["index"], None)
                        continue
                else:
                    sample = gen_samples[0]
                    is_correct, meta = task.check_correctness(primitive, sample["text"])
                    best_sample = {
                        **sample,
                        "correct": is_correct,
                        "metadata": meta,
                    }

                record = {
                    "index": prompt_data["index"],
                    "variant": prompt_data.get("variant", "unknown"),
                    "prompt": prompt_data["prompt"],
                    "generation": best_sample["text"],
                    "correct": best_sample["correct"],
                    "finish_reason": best_sample["finish_reason"],
                    "token_count": best_sample["token_count"],
                    "metadata": best_sample["metadata"],
                    "extracted_hints": extract_generation_hints(best_sample["text"]),
                    **{k: best_sample[k] for k in
                       ("pass_rate", "num_correct_samples", "num_samples")
                       if k in best_sample},
                }

                if "num_hints" in best_sample:
                    record["num_hints"] = best_sample["num_hints"]
                if "turns" in best_sample:
                    record["turns"] = best_sample["turns"]
                if "total_token_count" in best_sample:
                    record["total_token_count"] = best_sample["total_token_count"]

                copy_source_fields(record, prompt_data)
                records_by_index[prompt_data["index"]] = record

            records = sorted(records_by_index.values(), key=lambda r: r["index"])
            if n_no_correct:
                print(f"Dropped {n_no_correct} problem(s) with no correct sample "
                      f"(--sample-strategy random_correct).")
            if drop_exhausted:
                records, n_dropped = drop_exhausted_records(records)
                if n_dropped:
                    print(f"Dropped {n_dropped} record(s) that exhausted the hint "
                          f"pool (--drop-exhausted).")
                    # Keep the in-memory index consistent, or a later resume would
                    # treat the dropped indices as still-completed.
                    kept = {r["index"] for r in records}
                    records_by_index = {i: r for i, r in records_by_index.items() if i in kept}
            if not records:
                print("No records remain after filtering.")
                save_json(output_path, records)
                return output_path
            save_json(output_path, records)

            correct = sum(1 for r in records if r["correct"])
            print(f"Async generation complete: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")
            print(f"Saved dataset to {output_path}")
            if not retry_truncated:
                return output_path
            truncated = sum(1 for r in records if r.get("finish_reason") == "length")
            if truncated == 0:
                print(f"All records complete after {attempt + 1} attempt(s).")
                return output_path
            print(f"{truncated} truncated records remaining.")
            continue

        # Async regular generation (non-multi-turn)
        if use_async and not multi_turn:
            import asyncio

            # Same defaulting as the sync and multi-turn branches. Without it
            # sample_strategy stays None here and overrides the function default.
            sample_strategy = sample_strategy or "shortest_cot"
            n_no_correct = 0  # dropped by strategy="random_correct"

            async def _async_generate_with_retries():
                async_generator = AsyncGenerator(config)
                try:
                    return await _async_generate_inner(async_generator)
                finally:
                    async_generator.close()

            async def _async_generate_inner(async_generator):
                nonlocal n_no_correct
                current_prompts = remaining_prompts

                for retry_attempt in range(max_retries + 1 if retry_truncated else 1):
                    if retry_attempt > 0:
                        config.seed = seed + retry_attempt * 1000
                        # Filter to truncated only
                        truncated_indices = {idx for idx, r in records_by_index.items() if r.get("finish_reason") == "length"}
                        current_prompts = [p for p in prompts_data if p["index"] in truncated_indices]
                        if not current_prompts:
                            break
                        print(f"\n=== Retry truncated attempt {retry_attempt}/{max_retries} (seed={config.seed}, {len(current_prompts)} prompts) ===")

                    all_prompts = [p["prompt"] for p in current_prompts]
                    print(f"Running async generation on {len(all_prompts)} prompts...")
                    all_results = await async_generator.generate_async(all_prompts, num_samples=num_samples)

                    for prompt_data, gen_samples in zip(current_prompts, all_results):
                        primitive = {
                            "index": prompt_data["index"],
                            **prompt_data["ground_truth"],
                            "variant": prompt_data.get("variant", "unknown"),
                        }

                        if num_samples > 1:
                            _rng = random.Random((seed or 0) + prompt_data["index"])
                            best_sample = select_best_sample(
                                gen_samples, task, primitive,
                                strategy=sample_strategy,
                                exclude_exhausted=drop_exhausted, rng=_rng)
                            if best_sample is None:
                                # random_correct: nothing correct -> drop the problem
                                n_no_correct += 1
                                records_by_index.pop(prompt_data["index"], None)
                                continue
                        else:
                            sample = gen_samples[0]
                            is_correct, meta = task.check_correctness(primitive, sample["text"])
                            best_sample = {
                                **sample,
                                "correct": is_correct,
                                "metadata": meta,
                            }

                        record = {
                            "index": prompt_data["index"],
                            "variant": prompt_data.get("variant", "unknown"),
                            "prompt": prompt_data["prompt"],
                            "generation": best_sample["text"],
                            "correct": best_sample["correct"],
                            "finish_reason": best_sample["finish_reason"],
                            "token_count": best_sample["token_count"],
                            "metadata": best_sample["metadata"],
                            "extracted_hints": extract_generation_hints(best_sample["text"]),
                            **{k: best_sample[k] for k in
                               ("pass_rate", "num_correct_samples", "num_samples")
                               if k in best_sample},
                            "forced_answer": best_sample.get("forced_answer", False),
                            "answer_truncated": best_sample.get("answer_truncated", False),
                        }

                        copy_source_fields(record, prompt_data)
                        records_by_index[prompt_data["index"]] = record

                    records = sorted(records_by_index.values(), key=lambda r: r["index"])
                    save_json(output_path, records)

                    correct = sum(1 for r in records if r["correct"])
                    truncated_count = sum(1 for r in records if r.get("finish_reason") == "length")
                    forced_count = sum(1 for r in records if r.get("forced_answer"))
                    forced_note = f", {forced_count} forced" if forced_count else ""
                    trunc = count_truncation(records)
                    print(f"Async generation complete: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%), "
                          f"{trunc['think']} think-truncated, {trunc['answer']} answer-truncated{forced_note}")
                    print(f"Saved dataset to {output_path}")

                    if truncated_count == 0 or not retry_truncated:
                        if truncated_count == 0 and retry_attempt > 0:
                            print(f"All records complete after {retry_attempt + 1} attempt(s).")
                        break

            asyncio.run(_async_generate_with_retries())
            if n_no_correct:
                print(f"Dropped {n_no_correct} problem(s) with no correct sample "
                      f"(--sample-strategy random_correct).")
            return output_path

        # Standard batch processing (sync)
        for batch_idx in range(0, len(remaining_prompts), batch_size):
            batch_prompts_data = remaining_prompts[batch_idx:batch_idx + batch_size]
            batch_prompts = [p["prompt"] for p in batch_prompts_data]

            # Generate batch
            if multi_turn:
                if num_samples > 1:
                    expanded_prompts = [p for p in batch_prompts for _ in range(num_samples)]
                    expanded_gts = [p["ground_truth"] for p in batch_prompts_data for _ in range(num_samples)]
                    expanded_force = [fh for fh in force_hints_list[batch_idx:batch_idx + batch_size] for _ in range(num_samples)]
                    expanded_results = generator.generate_with_hints(
                        expanded_prompts,
                        expanded_gts,
                        force_hints=expanded_force,
                    )
                    batch_results = [
                        [r[0] for r in expanded_results[i * num_samples:(i + 1) * num_samples]]
                        for i in range(len(batch_prompts_data))
                    ]
                else:
                    batch_ground_truths = [p["ground_truth"] for p in batch_prompts_data]
                    batch_force_hints = force_hints_list[batch_idx:batch_idx + batch_size]
                    batch_results = generator.generate_with_hints(
                        batch_prompts,
                        batch_ground_truths,
                        force_hints=batch_force_hints,
                    )
            else:
                batch_results = generator.generate(batch_prompts)

            # Process results
            sample_strategy = sample_strategy or ("most_hints" if multi_turn else "shortest_cot")
            n_no_correct = 0  # dropped by strategy="random_correct"
            batch_correct = 0
            for prompt_data, gen_samples in zip(batch_prompts_data, batch_results):
                primitive = {
                    "index": prompt_data["index"],
                    **prompt_data["ground_truth"],
                    "variant": prompt_data.get("variant", "unknown"),
                }

                if num_samples > 1:
                    # Seed per problem index so a rerun picks the same sample.
                    _rng = random.Random((seed or 0) + prompt_data["index"])
                    best_sample = select_best_sample(gen_samples, task, primitive,
                                                     strategy=sample_strategy,
                                                     exclude_exhausted=drop_exhausted,
                                                     rng=_rng)
                    if best_sample is None:
                        # strategy="random_correct" and nothing was correct.
                        n_no_correct += 1
                        records_by_index.pop(prompt_data["index"], None)
                        continue
                else:
                    sample = gen_samples[0]
                    is_correct, meta = task.check_correctness(primitive, sample["text"])
                    best_sample = {
                        **sample,
                        "correct": is_correct,
                        "metadata": meta,
                    }

                if best_sample["correct"]:
                    batch_correct += 1

                record = {
                    "index": prompt_data["index"],
                    "variant": prompt_data.get("variant", "unknown"),
                    "prompt": prompt_data["prompt"],
                    "generation": best_sample["text"],
                    "correct": best_sample["correct"],
                    "finish_reason": best_sample["finish_reason"],
                    "token_count": best_sample["token_count"],
                    "metadata": best_sample["metadata"],
                    "extracted_hints": extract_generation_hints(best_sample["text"]),
                    **{k: best_sample[k] for k in
                       ("pass_rate", "num_correct_samples", "num_samples")
                       if k in best_sample},
                }

                if "num_hints" in best_sample:
                    record["num_hints"] = best_sample["num_hints"]
                if "turns" in best_sample:
                    record["turns"] = best_sample["turns"]
                if "total_token_count" in best_sample:
                    record["total_token_count"] = best_sample["total_token_count"]

                copy_source_fields(record, prompt_data)
                records_by_index[prompt_data["index"]] = record

            records = sorted(records_by_index.values(), key=lambda r: r["index"])
            save_json(output_path, records)

            batch_num = batch_idx // batch_size + 1
            print(f"Batch {batch_num}/{total_batches}: {batch_correct}/{len(batch_results)} correct (saved)")

        # Final stats for this attempt
        records = sorted(records_by_index.values(), key=lambda r: r["index"])
        correct = sum(1 for r in records if r["correct"])
        print(f"Generated {len(records)} examples: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")
        save_json(output_path, records)
        print(f"Saved dataset to {output_path}")

        if not retry_truncated:
            break
        truncated = sum(1 for r in records if r.get("finish_reason") == "length")
        if truncated == 0:
            print(f"All records complete after {attempt + 1} attempt(s).")
            break
        print(f"{truncated} truncated records remaining.")

    return output_path


def generate_until_target(
    target_correct_rate: float,
    max_oversample: int = 5,
    **generate_kwargs,
):
    """Phase 3: resample incorrect problems until enough are answered correctly.

    Wraps `generate` rather than threading another mode through its retry loop.
    Each round re-runs with `retry_incorrect=True`, which keeps the correct
    records and regenerates only the rest, at a fresh seed. Rounds stop
    as soon as the correct fraction reaches the target, when a round makes no
    progress, or at `max_oversample`.

    Bailing out on a no-progress round matters: the remaining problems are the
    ones the teacher genuinely cannot solve at this hint budget, and resampling
    them again just burns GPU hours to arrive at the same answer.
    """
    output_path = generate_kwargs.get("output_path")
    seed = generate_kwargs.pop("seed", 42)
    # The caller passes its own --retry-incorrect; this loop owns that flag
    # (round 0 generates fresh, later rounds retry only the incorrect), so drop
    # theirs rather than colliding with it.
    generate_kwargs.pop("retry_incorrect", None)

    prev_correct = -1
    for round_idx in range(max_oversample + 1):
        if round_idx > 0:
            print(f"\n=== Oversample round {round_idx}/{max_oversample} ===")
        output_path = generate(
            **generate_kwargs,
            seed=(seed if seed is None else seed + round_idx * 1000),
            retry_incorrect=(round_idx > 0),
        )
        generate_kwargs["output_path"] = output_path

        records = load_json(output_path)
        if not records:
            print("No records produced; stopping.")
            return output_path
        correct = sum(1 for r in records if r.get("correct"))
        rate = correct / len(records)
        print(f"Round {round_idx}: {correct}/{len(records)} correct ({rate:.1%}), "
              f"target {target_correct_rate:.1%}")

        if rate >= target_correct_rate:
            print(f"Target reached after {round_idx + 1} round(s).")
            return output_path
        if correct == prev_correct:
            print(f"No improvement over the previous round ({correct} correct); "
                  f"remaining problems look unsolvable at this hint budget. Stopping.")
            return output_path
        prev_correct = correct

    print(f"Hit max_oversample={max_oversample} without reaching "
          f"{target_correct_rate:.1%}. Best: {prev_correct}/{len(records)}.")
    return output_path


def _compute_multisample_metrics(details: list[dict]) -> dict:
    """Compute metrics for multi-sample evaluation.

    Each detail has n_samples and n_correct.
    Returns avg@k accuracy and raw counts, overall and by variant/level.
    """
    from collections import defaultdict

    def _group_metrics(records):
        n_problems = len(records)
        total_correct = sum(r["n_correct"] for r in records)
        total_samples = sum(r["n_samples"] for r in records)
        avg_accuracy = (
            sum(r["n_correct"] / r["n_samples"] for r in records) / n_problems
            if n_problems > 0 else 0
        )
        return {
            "n_problems": n_problems,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "avg_accuracy": avg_accuracy,
        }

    by_variant = defaultdict(list)
    by_level = defaultdict(list)
    for d in details:
        by_variant[d.get("variant", "unknown")].append(d)
        level = d.get("level", "unknown")
        if level != "unknown":
            by_level[level].append(d)

    metrics = _group_metrics(details)
    metrics["by_variant"] = {v: _group_metrics(recs) for v, recs in sorted(by_variant.items())}
    if by_level:
        metrics["by_level"] = {l: _group_metrics(recs) for l, recs in sorted(by_level.items())}

    return metrics


def _format_multisample_metrics(metrics: dict, model_name: str | None, num_samples: int) -> str:
    """Format multi-sample metrics as a table."""
    lines = [
        "",
        f"=== Multi-Sample Evaluation ({num_samples} samples) ===",
    ]
    if model_name:
        lines.append(f"Model: {model_name}")

    def _row(label, m):
        avg = f"{m['avg_accuracy']:.1%}"
        raw = f"{m['total_correct']}/{m['total_samples']}"
        return f"  {label:<20s}  {m['n_problems']:>5d}  {avg:>8s}  {raw:>14s}"

    lines.append(f"  {'':20s}  {'N':>5s}  {'Avg@'+str(num_samples):>8s}  {'Correct/Total':>14s}")
    lines.append("  " + "-" * 51)

    for variant, vm in sorted(metrics.get("by_variant", {}).items()):
        lines.append(_row(variant, vm))

    lines.append("  " + "-" * 51)
    lines.append(_row("Total", metrics))

    if "by_level" in metrics:
        lines.append("")
        lines.append(f"  {'':20s}  {'N':>5s}  {'Avg@'+str(num_samples):>8s}  {'Correct/Total':>14s}")
        lines.append("  " + "-" * 51)
        for level, lm in sorted(metrics["by_level"].items()):
            lines.append(_row(level, lm))
        lines.append("  " + "-" * 51)
        lines.append(_row("Total", metrics))

    return "\n".join(lines)


def evaluate(
    task_name: str,
    model_name: str,
    method_name: str | None = None,
    run_id: str | None = None,
    split: str = "eval",
    prompts_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_samples: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    verbose: bool = False,
    multi_turn: bool | None = None,
    use_async: bool = False,
    seed: int | None = 42,
    no_hints: bool = False,
) -> Path:
    """
    Evaluate a model on prompts and compute metrics.

    Args:
        task_name: Name of task
        model_name: Model to evaluate (can be "sft" or "rl" to use method's model paths)
        method_name: Method name for auto-derived paths
        run_id: Run identifier for model resolution (used when model_name="sft" or "rl")
        prompts_path: Path to eval prompts (default: artifacts/{task}/problems_with_format/{split}__{method}.json)
        output_path: Where to save results (default: artifacts/{task}/models/{method}_{sft,models}/{run_id}/evals/{split}.json)
        multi_turn: Enable multi-turn generation with hint injection. If None, uses method config.
        use_async: Use async generation for optimal throughput.
        no_hints: Counterfactual eval -- ban the hint request so the model must
            answer on its own. Everything else about the eval is unchanged, so
            the drop against a normal run measures what the hints were worth.
        ... generation config ...

    Returns:
        Path to results file
    """
    task = get_task(task_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Resolve multi_turn from method config (CLI overrides if explicitly set)
    if multi_turn is None:
        multi_turn = method.multi_turn if method else False

    # Resolve model shortcuts (sft, rl)
    actual_model_name = model_name
    if method is not None:
        if model_name == "sft":
            actual_model_name = str(method.sft_model_path(task_name, run_id))
        elif model_name == "rl":
            actual_model_name = str(method.rl_model_path(task_name, run_id))

    # Default prompts path
    if prompts_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --prompts must be specified. "
                "Use --method to auto-derive paths, or --prompts for explicit paths."
            )
        prompts_path = method.formatted_path(task_name, split)

    # Default output path
    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        # Results live inside the run that produced them, so the run directory
        # already carries the model identity and the filename only records the
        # split plus any deviation from the default eval settings.
        samples_suffix = f"_{num_samples}s" if num_samples > 1 else ""
        hint_suffix = "_nohint" if no_hints else ""
        suffix = f"{samples_suffix}{hint_suffix}"
        if model_name in ("sft", "rl"):
            output_path = method.eval_path(task_name, model_name, run_id, split, suffix)
        else:
            # A checkpoint evaluated outside any run of ours -- a stock
            # HuggingFace model, or a path handed in directly. It has no run
            # directory, so it gets one under _stock keyed by the model name.
            slug = model_short_name(model_name)
            output_path = (method.models_dir(task_name) / "_stock" / slug
                           / "evals" / f"{split}{suffix}.json")

    # Load prompts
    prompts_data = load_json(prompts_path)
    print(f"Loaded {len(prompts_data)} prompts from {prompts_path}")

    # Initialize generator
    config = GenerationConfig(
        model_name=actual_model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_samples=num_samples,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        verbose=verbose,
        seed=seed,
        nested_request=method.nested_request if method else False,
        ban_hint_requests=no_hints,
    )
    generator = Generator(config)

    # Generate
    prompts = [p["prompt"] for p in prompts_data]

    print(f"Evaluating {actual_model_name}...")
    if num_samples > 1:
        print(f"Multi-sample mode: {num_samples} samples per problem")
    if use_async:
        print("Async mode enabled (optimal throughput)")
    if multi_turn:
        print("Multi-turn mode enabled")
    if no_hints:
        print("Counterfactual mode: hint requests are blocked during decoding")

    # Async multi-turn generation
    if multi_turn and use_async:
        import asyncio

        ground_truths = [p["ground_truth"] for p in prompts_data]

        # generate_with_hints_async produces 1 trajectory per prompt,
        # so duplicate prompts for num_samples > 1
        if num_samples > 1:
            expanded_prompts = [p for p in prompts for _ in range(num_samples)]
            expanded_gts = [gt for gt in ground_truths for _ in range(num_samples)]
            print(f"Running async multi-turn generation on {len(prompts)} prompts x {num_samples} samples...")
            async_generator = AsyncGenerator(config)
            expanded_results = asyncio.run(
                async_generator.generate_with_hints_async(
                    expanded_prompts,
                    expanded_gts,
                )
            )
            # Group back: every num_samples consecutive results belong to the same prompt
            generations = [
                [r[0] for r in expanded_results[i * num_samples:(i + 1) * num_samples]]
                for i in range(len(prompts))
            ]
        else:
            print(f"Running async multi-turn generation on {len(prompts)} prompts...")
            async_generator = AsyncGenerator(config)
            generations = asyncio.run(
                async_generator.generate_with_hints_async(
                    prompts,
                    ground_truths,
                )
            )

    # Async regular generation
    elif use_async and not multi_turn:
        import asyncio

        print(f"Running async generation on {len(prompts)} prompts...")
        async_generator = AsyncGenerator(config)
        generations = asyncio.run(
            async_generator.generate_async(prompts, num_samples=num_samples)
        )

    # Sync multi-turn generation
    elif multi_turn:
        ground_truths = [p["ground_truth"] for p in prompts_data]
        if num_samples > 1:
            expanded_prompts = [p for p in prompts for _ in range(num_samples)]
            expanded_gts = [gt for gt in ground_truths for _ in range(num_samples)]
            expanded_results = generator.generate_with_hints_batched(
                expanded_prompts,
                expanded_gts,
            )
            generations = [
                [r[0] for r in expanded_results[i * num_samples:(i + 1) * num_samples]]
                for i in range(len(prompts))
            ]
        else:
            generations = generator.generate_with_hints_batched(
                prompts,
                ground_truths,
            )

    # Sync regular generation
    else:
        def progress_callback(batch_idx, _results):
            total_batches = (len(prompts) + batch_size - 1) // batch_size
            print(f"Batch {batch_idx + 1}/{total_batches} complete")

        generations = generator.generate_batched(prompts, callback=progress_callback)

    # The ban has two layers and this reports on both. The sampler suppresses the
    # "request" token, which is best-effort -- bad_words bans a token id and BPE
    # can spell the string another way. The guarantee is upstream of that: with
    # ban_hint_requests set the generator swaps in an unmatchable request tag, so
    # no tag the model writes can make the environment serve a hint. Emitting
    # <request> is therefore a failed attempt, not a leak, and the rate is worth
    # logging next to the accuracy it produced rather than aborting over.
    if no_hints:
        import re as _re

        def _texts(items):
            """Every generated string in `generations`, whatever shape it came in.

            The shape depends on the mode: bare strings, a list of samples per
            prompt, or the {"text": ...} sample dicts the async paths return.
            Reading the wrong one is not a loud failure -- `"<tag>" in some_dict`
            tests keys and quietly answers False -- so this walks all three.
            """
            out = []
            for item in items:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        out.append(item["text"])
                elif isinstance(item, (list, tuple)):
                    out.extend(_texts(item))
            return out

        flat = _texts(generations)
        if not flat:
            raise RuntimeError(
                "--no-hints could not read any generation text, so the ban is unverified"
            )
        # Blocked from asking, some models write a request-shaped tag anyway and
        # then answer their own <response>. Nothing is served back, so the run
        # stays hint-free -- but the model is now reading text it invented, and
        # that rate belongs in the log beside the accuracy it produced.
        asked = sum(1 for g in flat if "</request>" in g)
        near_miss = sum(1 for g in flat if _re.search(r"<[^<>]*request[^<>]*>", g))
        fabricated = sum(1 for g in flat if "<response>" in g)
        print(f"Hint ban held: 0/{len(flat)} generations were served a hint"
              + (f"; {asked} reached </request> anyway" if asked else "")
              + (f"; {near_miss} wrote a request-shaped tag" if near_miss else "")
              + (f"; {fabricated} invented their own <response>" if fabricated else ""))

    # --- Multi-sample path (num_samples > 1) ---
    if num_samples > 1:
        details = []
        for prompt_data, gen_samples in zip(prompts_data, generations):
            primitive = {
                "index": prompt_data["index"],
                **prompt_data["ground_truth"],
                "variant": prompt_data.get("variant", "unknown"),
            }

            samples = []
            for s in gen_samples:
                is_correct, meta = task.check_correctness(primitive, s["text"])
                samples.append({
                    "generation": s["text"],
                    "correct": is_correct,
                    "predicted_answer": meta.get("predicted_answer") if isinstance(meta, dict) else None,
                    "finish_reason": s.get("finish_reason", "unknown"),
                    "token_count": s.get("token_count", 0),
                })

            n_correct = sum(1 for s in samples if s["correct"])

            details.append({
                "index": prompt_data["index"],
                "variant": prompt_data.get("variant", "unknown"),
                "level": prompt_data.get("ground_truth", {}).get("level", "unknown"),
                "ground_truth": prompt_data["ground_truth"],
                "n_samples": len(samples),
                "n_correct": n_correct,
                "samples": samples,
            })

        # Compute multi-sample metrics
        metrics = _compute_multisample_metrics(details)

        results = {
            "model": actual_model_name,
            "model_alias": model_name,
            "prompts": str(prompts_path),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "num_samples": num_samples,
                "multi_turn": multi_turn,
            },
            "metrics": metrics,
            "details": details,
        }

        print(_format_multisample_metrics(metrics, model_name, num_samples))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(output_path, results)
        print(f"\nSaved results to {output_path}")

        return output_path

    # --- Single-sample path (num_samples == 1) ---
    details = []
    for prompt_data, gen_samples in zip(prompts_data, generations):
        # gen_samples is a list of samples (even if num_samples=1)
        # For evaluation, use the first sample
        gen_result = gen_samples[0]

        primitive = {
            "index": prompt_data["index"],
            **prompt_data["ground_truth"],
            "variant": prompt_data.get("variant", "unknown"),
        }

        is_correct, meta = task.check_correctness(primitive, gen_result["text"])
        finish_reason = gen_result.get("finish_reason", "unknown")
        token_count = gen_result.get("token_count", 0)

        detail = {
            "index": prompt_data["index"],
            "variant": prompt_data.get("variant", "unknown"),
            "level": prompt_data.get("ground_truth", {}).get("level", "unknown"),
            "correct": is_correct,
            "finish_reason": finish_reason,
            "token_count": token_count,
            "forced_answer": gen_result.get("forced_answer", False),
            "answer_truncated": gen_result.get("answer_truncated", False),
            "generation": gen_result["text"],
            "predicted_answer": meta.get("predicted_answer"),
            "ground_truth": prompt_data["ground_truth"],
            "error": meta.get("error"),
            **{k: v for k, v in meta.items() if k not in ("predicted_answer", "error")},
        }
        copy_source_fields(detail, prompt_data)

        detail["truncation"] = classify_truncation(detail)

        # Add multi-turn specific fields
        if "num_hints" in gen_result:
            detail["num_hints"] = gen_result["num_hints"]
        if "turns" in gen_result:
            detail["turns"] = gen_result["turns"]
        if "total_token_count" in gen_result:
            detail["total_token_count"] = gen_result["total_token_count"]
        if "hint_selections" in gen_result:
            detail["hint_selections"] = gen_result["hint_selections"]

        details.append(detail)

    # Compute metrics using task-specific logic
    metrics = task.compute_metrics(details)
    metrics["truncation"] = count_truncation(details)

    # Compute hint metrics if multi-turn
    hint_metrics = None
    if multi_turn:
        hint_metrics = compute_hint_metrics(details)

    # Build results
    results = {
        "model": actual_model_name,
        "model_alias": model_name,  # Keep the alias (sft, rl) if used
        "prompts": str(prompts_path),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "num_samples": 1,
            "multi_turn": multi_turn,
        },
        "metrics": metrics,
        "details": details,
    }

    # Add hint metrics if available
    if hint_metrics:
        results["hint_metrics"] = hint_metrics

    # Print summary using task-specific formatting
    print(task.format_metrics(metrics, model_name))
    trunc = metrics["truncation"]
    print(f"Truncated: {trunc['think']} in think (benign, forced-answer rescuable), "
          f"{trunc['answer']} in answer (malignant) of {len(details)}")

    # Print hint analysis if multi-turn
    if hint_metrics:
        print(format_hint_metrics(hint_metrics, details))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, results)
    print(f"\nSaved results to {output_path}")

    return output_path


def _combine_verifier_eval_single(
    task_name: str,
    solver_results: dict,
    verifier_results: dict,
    solver_by_key: dict,
    verifier_by_key: dict,
    source_indices: list[int],
    max_hints: int,
    width: int,
) -> list[dict]:
    """Original single-sample logic: one decision trace per problem."""
    final_details = []
    for source_index in source_indices:
        expected_solver = {
            (source_index, hint_level) for hint_level in range(width)
        }
        missing_solver = sorted(expected_solver - solver_by_key.keys())
        if missing_solver:
            raise ValueError(
                f"Source {source_index}: missing solver hint levels "
                f"{[hint for _, hint in missing_solver]}"
            )

        trace = []
        selected_level = max_hints
        selected_verifier = None
        for hint_level in range(max_hints):
            key = (source_index, hint_level)
            verifier_detail = verifier_by_key.get(key)
            if verifier_detail is None:
                raise ValueError(
                    f"Source {source_index}: missing verifier result for "
                    f"hint level {hint_level}"
                )
            answer = extract_answer(verifier_detail.get("generation", ""))
            if answer not in {"0", "1"}:
                raise ValueError(
                    f"Source {source_index}, hint level {hint_level}: verifier "
                    f"must output <answer>0</answer> or <answer>1</answer>, "
                    f"got {verifier_detail.get('generation')!r}"
                )
            prediction = int(answer)
            trace.append({
                "hint_level": hint_level,
                "generation": verifier_detail["generation"],
                "prediction": prediction,
            })
            if prediction == 1:
                selected_level = hint_level
                selected_verifier = verifier_detail
                break

        selected_solver = solver_by_key[(source_index, selected_level)]
        detail = {
            **selected_solver,
            "index": source_index,
            "source_index": source_index,
            "selected_candidate_index": selected_solver["index"],
            "num_hints": selected_level,
            "selected_hint_level": selected_level,
            "verifier_generation": (
                selected_verifier["generation"] if selected_verifier else None
            ),
            "verifier_prediction": 1 if selected_verifier else None,
            "verifier_trace": trace,
            "forced_at_max_hints": selected_level == max_hints,
        }
        final_details.append(detail)
    return final_details


def _combine_verifier_eval_multisample(
    solver_by_key: dict,
    verifier_by_key: dict,
    source_indices: list[int],
    max_hints: int,
    width: int,
) -> list[dict]:
    """Multi-sample logic: run `num_samples` independent decision rollouts per
    problem, using the same "same rollout index across hint levels" pairing.

    Rollout t at hint level h draws the t-th verifier sample generated for
    that (problem, hint_level) candidate. If it predicts 1 ("stop, answer
    now"), the t-th solver sample at that hint level becomes the rollout's
    final answer. If it predicts 0, the rollout moves to hint_level + 1 and
    repeats with the t-th sample there. If no hint level says "stop" before
    max_hints, the t-th solver sample at max_hints is used (forced answer).

    This produces `num_samples` independent full decision traces per problem
    (instead of just one), so downstream metrics can report variance across
    repeated rollouts of the whole solver+verifier pipeline.
    """
    def sample_count(detail: dict) -> int:
        samples = detail.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(
                f"Index {detail.get('index')}: expected multi-sample "
                f"'samples' list, found none"
            )
        return len(samples)

    final_details = []
    for source_index in source_indices:
        expected_solver = {
            (source_index, hint_level) for hint_level in range(width)
        }
        missing_solver = sorted(expected_solver - solver_by_key.keys())
        if missing_solver:
            raise ValueError(
                f"Source {source_index}: missing solver hint levels "
                f"{[hint for _, hint in missing_solver]}"
            )

        # Every candidate at this source index must offer the same number of
        # samples, so rollout index t is well-defined across all hint levels.
        counts = set()
        for hint_level in range(width):
            counts.add(sample_count(solver_by_key[(source_index, hint_level)]))
        for hint_level in range(max_hints):
            counts.add(sample_count(verifier_by_key[(source_index, hint_level)]))
        if len(counts) != 1:
            raise ValueError(
                f"Source {source_index}: inconsistent sample counts across "
                f"candidates: {sorted(counts)}"
            )
        num_samples = counts.pop()

        for t in range(num_samples):
            trace = []
            selected_level = max_hints
            selected_verifier_sample = None
            for hint_level in range(max_hints):
                key = (source_index, hint_level)
                verifier_detail = verifier_by_key.get(key)
                if verifier_detail is None:
                    raise ValueError(
                        f"Source {source_index}: missing verifier result for "
                        f"hint level {hint_level}"
                    )
                v_sample = verifier_detail["samples"][t]
                answer = extract_answer(v_sample.get("generation", ""))
                if answer not in {"0", "1"}:
                    raise ValueError(
                        f"Source {source_index}, hint level {hint_level}, "
                        f"rollout {t}: verifier must output "
                        f"<answer>0</answer> or <answer>1</answer>, got "
                        f"{v_sample.get('generation')!r}"
                    )
                prediction = int(answer)
                trace.append({
                    "hint_level": hint_level,
                    "generation": v_sample["generation"],
                    "prediction": prediction,
                })
                if prediction == 1:
                    selected_level = hint_level
                    selected_verifier_sample = v_sample
                    break

            selected_solver_detail = solver_by_key[(source_index, selected_level)]
            selected_solver_sample = selected_solver_detail["samples"][t]
            detail = {
                **selected_solver_sample,
                "index": source_index,
                "source_index": source_index,
                "trial_index": t,
                "variant": selected_solver_detail.get("variant", "unknown"),
                "level": selected_solver_detail.get("level", "unknown"),
                "ground_truth": selected_solver_detail.get("ground_truth"),
                "selected_candidate_index": selected_solver_detail["index"],
                "num_hints": selected_level,
                "selected_hint_level": selected_level,
                "verifier_generation": (
                    selected_verifier_sample["generation"]
                    if selected_verifier_sample else None
                ),
                "verifier_prediction": 1 if selected_verifier_sample else None,
                "verifier_trace": trace,
                "forced_at_max_hints": selected_level == max_hints,
            }
            final_details.append(detail)
    return final_details


def combine_verifier_eval(
    task_name: str,
    solver_results_path: Path,
    verifier_results_path: Path,
    output_path: Path,
    max_hints: int = 5,
) -> Path:
    """Apply verifier decisions to fixed-hint solver results.

    Supports both single-sample eval results (one generation per candidate)
    and multi-sample eval results (`evaluate --num-samples N > 1`). In the
    multi-sample case, each of the N samples is treated as an independent
    rollout through the same stop/continue decision logic, producing N
    independent final decisions per problem -- see
    `_combine_verifier_eval_multisample` for details.
    """
    if max_hints < 1:
        raise ValueError(f"max_hints must be positive, got {max_hints}")

    solver_results = load_json(solver_results_path)
    verifier_results = load_json(verifier_results_path)
    solver_details = solver_results.get("details")
    verifier_details = verifier_results.get("details")
    if not isinstance(solver_details, list):
        raise ValueError(f"{solver_results_path} has no details list")
    if not isinstance(verifier_details, list):
        raise ValueError(f"{verifier_results_path} has no details list")

    width = max_hints + 1

    def candidate_key(detail: dict) -> tuple[int, int]:
        index = detail.get("index")
        if not isinstance(index, int):
            raise ValueError(f"Evaluation detail has invalid index: {index!r}")
        source_index = detail.get("source_index", index // width)
        hint_level = detail.get("hint_level", index % width)
        if not isinstance(source_index, int) or not isinstance(hint_level, int):
            raise ValueError(
                f"Index {index}: invalid source_index/hint_level "
                f"{source_index!r}/{hint_level!r}"
            )
        return source_index, hint_level

    def index_candidates(details: list[dict], label: str) -> dict[tuple[int, int], dict]:
        indexed = {}
        for detail in details:
            key = candidate_key(detail)
            if key in indexed:
                raise ValueError(f"Duplicate {label} result for {key}")
            indexed[key] = detail
        return indexed

    solver_by_key = index_candidates(solver_details, "solver")
    verifier_by_key = index_candidates(verifier_details, "verifier")
    source_indices = sorted({source_index for source_index, _ in solver_by_key})

    # Multi-sample eval results (`evaluate --num-samples N > 1`) store a
    # "samples" list per candidate instead of a single "generation" -- detect
    # that shape here and switch to the per-rollout combination logic.
    any_solver_detail = next(iter(solver_by_key.values()), None)
    any_verifier_detail = next(iter(verifier_by_key.values()), None)
    solver_multisample = bool(any_solver_detail) and "samples" in any_solver_detail
    verifier_multisample = bool(any_verifier_detail) and "samples" in any_verifier_detail
    if solver_multisample != verifier_multisample:
        raise ValueError(
            "solver and verifier results must both be single-sample or both "
            "multi-sample (one has a 'samples' list, the other doesn't)"
        )

    if solver_multisample:
        final_details = _combine_verifier_eval_multisample(
            solver_by_key, verifier_by_key, source_indices, max_hints, width,
        )
    else:
        final_details = _combine_verifier_eval_single(
            task_name, solver_results, verifier_results,
            solver_by_key, verifier_by_key, source_indices, max_hints, width,
        )

    task = get_task(task_name)
    metrics = task.compute_metrics(final_details)
    metrics["truncation"] = count_truncation(final_details)
    hint_metrics = compute_hint_metrics(final_details)
    results = {
        "model": solver_results.get("model"),
        "model_alias": "method_a_pipeline",
        "prompts": solver_results.get("prompts"),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_hints": max_hints,
            "multisample": solver_multisample,
            "solver": solver_results.get("config", {}),
            "verifier": verifier_results.get("config", {}),
            "solver_results": str(solver_results_path),
            "verifier_results": str(verifier_results_path),
        },
        "metrics": metrics,
        "details": final_details,
        "hint_metrics": hint_metrics,
    }

    print(task.format_metrics(metrics, "method_a_pipeline"))
    print(format_hint_metrics(hint_metrics, final_details))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, results)
    print(f"\nSaved combined evaluation to {output_path}")
    return output_path


def analyze(dataset_path: Path, task_name: str | None = None) -> dict:
    """
    Analyze a dataset or eval results and print metrics.

    Delegates display formatting to task.format_metrics() for task-specific output.

    Args:
        dataset_path: Path to dataset.json or results.json
        task_name: Optional task name (required for task-specific formatting)

    Returns:
        Dict with metrics
    """
    data = load_json(dataset_path)
    model_name = data.get("model") if isinstance(data, dict) else None

    # Get task for formatting (if provided)
    task = get_task(task_name) if task_name else None

    # If eval results with stored metrics, use those
    if isinstance(data, dict) and "metrics" in data:
        metrics = data["metrics"]
        if task:
            print(task.format_metrics(metrics, model_name))
        else:
            # Fallback: basic display without task
            print(_format_basic_metrics(metrics, model_name))
        return metrics

    # Otherwise compute from records
    records = data["details"] if isinstance(data, dict) and "details" in data else data

    if task:
        metrics = task.compute_metrics(records)
        print(task.format_metrics(metrics, model_name))
        return metrics

    # Fallback: basic counts without task
    from collections import defaultdict
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        variant = r.get("variant", "unknown")
        counts[variant]["total"] += 1
        if r.get("correct", False):
            counts[variant]["correct"] += 1

    metrics = {"counts_by_variant": dict(counts)}
    print(_format_basic_metrics(metrics, model_name))
    return metrics


def _format_basic_metrics(metrics: dict, model_name: str | None = None) -> str:
    """Basic metrics formatting when no task is specified."""
    lines = ["", "=== Metrics ==="]
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append(f"Total: {metrics.get('total', 0)}")
    lines.append(f"Correct: {metrics.get('correct', 0)}")
    lines.append(f"Accuracy: {metrics.get('accuracy', 0):.1%}")

    by_variant = metrics.get("counts_by_variant") or metrics.get("by_variant", {})
    if by_variant:
        lines.append("")
        lines.append("By variant:")
        for variant, c in sorted(by_variant.items(), key=lambda x: str(x[0])):
            total = c.get("total", 0)
            correct = c.get("correct", 0)
            acc = correct / total if total > 0 else 0
            lines.append(f"  {variant}: {correct}/{total} ({acc:.0%})")

    return "\n".join(lines)
