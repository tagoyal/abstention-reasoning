"""File I/O utilities."""

import json
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> Any:
    """Load JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Save JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_parquet(path: Path | str, records: list[dict], max_shard_bytes: int = 45 * 1024 * 1024) -> list[Path]:
    """Save list of dicts as parquet file(s), sharded to stay under GitHub's
    100MB hard limit (and 50MB "soft warning" threshold) on file size.

    When the full dataset would serialize to more than `max_shard_bytes`,
    writes multiple numbered shards instead of one big file: `<stem>.shard000
    <suffix>`, `<stem>.shard001<suffix>`, etc, next to the originally
    requested `path`, and removes `path` itself (plus any stale shards from a
    previous run with a different shard count) so there's no ambiguity about
    which file(s) are current. verl's RLDataset natively accepts a list of
    parquet files, so training code just needs to glob for these shards
    instead of assuming a single path.

    Returns the list of file(s) actually written (length 1 if unsharded).
    """
    from datasets import Dataset

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use HuggingFace datasets to preserve nested structures
    ds = Dataset.from_list(records)

    # Clear out any stale output from a previous run before (re)writing,
    # since the new run may produce a different number of shards (or none).
    path.unlink(missing_ok=True)
    for stale in path.parent.glob(f"{path.stem}.shard*{path.suffix}"):
        stale.unlink()

    # Estimate on-disk parquet size via a single-shot write, then split only
    # if that exceeds the threshold -- avoids guessing row count up front.
    ds.to_parquet(str(path))
    if path.stat().st_size <= max_shard_bytes or len(records) <= 1:
        return [path]

    total_bytes = path.stat().st_size
    num_shards = -(-total_bytes // max_shard_bytes)  # ceil division
    rows_per_shard = -(-len(records) // num_shards)
    path.unlink()

    written = []
    for shard_idx, start in enumerate(range(0, len(records), rows_per_shard)):
        shard_records = records[start:start + rows_per_shard]
        shard_path = path.parent / f"{path.stem}.shard{shard_idx:03d}{path.suffix}"
        Dataset.from_list(shard_records).to_parquet(str(shard_path))
        written.append(shard_path)
    return written


def load_parquet_shards(path: Path | str) -> list[Path]:
    """Resolve a parquet path written by `save_parquet` back to its file(s).

    If `path` exists as a single file, returns `[path]` unchanged. Otherwise
    looks for `<stem>.shard*<suffix>` next to it (see `save_parquet`) and
    returns them in order. Raises FileNotFoundError if neither is found.
    """
    path = Path(path)
    if path.exists():
        return [path]
    shards = sorted(path.parent.glob(f"{path.stem}.shard*{path.suffix}"))
    if not shards:
        raise FileNotFoundError(
            f"No parquet file or shards found for {path} "
            f"(looked for {path.stem}.shard*{path.suffix} in {path.parent})"
        )
    return shards


def load_parquet_records(path: Path | str) -> list[dict]:
    """Load all rows of a parquet prompts file (resolving shards via
    `load_parquet_shards`) back into a list of dicts, mirroring the records
    `save_parquet` was given -- nested fields (`primitive`, `reward_model`,
    `extra_info`) come back as plain nested dicts/lists, not datasets.Dataset
    feature objects."""
    from datasets import concatenate_datasets, load_dataset

    shards = load_parquet_shards(path)
    dataframes = [load_dataset("parquet", data_files=str(shard))["train"] for shard in shards]
    ds = dataframes[0] if len(dataframes) == 1 else concatenate_datasets(dataframes)
    return ds.to_list()


def is_parquet_prompts_path(path: Path | str) -> bool:
    """True if `path` is (or has shards for) a parquet prompts file written
    by `create_prompts`'s rl_* path, rather than a fully-rendered JSON one."""
    path = Path(path)
    if path.suffix == ".parquet":
        return True
    return (not path.exists()) and bool(list(path.parent.glob(f"{path.stem}.shard*.parquet")))


def _render_runtime_prompt(primitive: dict, template: str) -> str:
    """Substitute a primitive's fields into a `{key}`/`{{key}}`-templated
    string, mirroring verl's RLHFDataset._apply_runtime_template exactly
    (single-brace first, double-brace second) so eval-time rendering matches
    what the model was actually trained on."""
    content = template
    for key, value in primitive.items():
        content = content.replace(f"{{{key}}}", str(value))
        content = content.replace(f"{{{{{key}}}}}", str(value))
    return content


def load_prompts(
    path: Path | str,
    template: str | None = None,
    system_message: str | None = None,
) -> list[dict]:
    """Load a prompts file the way `evaluate`/`generate_tree` need it: a list
    of records each carrying a rendered "prompt" chat conversation.

    Transparently handles both formats `create_prompts` can write:
    - JSON (sft_*, eval, etc): records already have a fully-rendered "prompt"
      conversation -- returned as-is via `load_json`.
    - Parquet (rl_*): records carry a raw "primitive" dict and no rendered
      prompt -- verl's RLHFDataset renders it at train time by substituting
      the primitive's fields into the method's rl.txt template and wrapping
      the result in a [system?, user, assistant-prefix?] conversation (see
      `_apply_runtime_template` in verl/verl/utils/dataset/rl_dataset.py).
      This replays that exact substitution so eval/tree generation sees the
      same conversations the model was actually trained on. `template` (the
      method's rl.txt content, e.g. `method.load_template(task_name, "rl")`)
      is required in this case; `system_message` (e.g. `task.system_message`)
      is optional, matching verl's `runtime_system_message` default of None.
    """
    path = Path(path)
    if not is_parquet_prompts_path(path):
        return load_json(path)

    if template is None:
        raise ValueError(
            f"{path} is a parquet prompts file (primitives + runtime "
            "template, written for rl_* splits) -- pass `template` (e.g. "
            "method.load_template(task_name, 'rl')) to render it into prompt "
            "conversations."
        )

    rendered = []
    for record in load_parquet_records(path):
        primitive = record.get("primitive") or {}
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": _render_runtime_prompt(primitive, template)})
        assistant_prefix = record.get("assistant_prefix")
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})

        rendered.append({
            **{k: v for k, v in record.items() if k != "primitive"},
            "prompt": messages,
        })
    return rendered
