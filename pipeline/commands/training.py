"""
Training commands - SFT, RL, and checkpoint conversion.
"""

import os
import re
import shutil
from pathlib import Path

from pipeline.core.io import load_json, load_parquet_shards, save_json
from pipeline.core.method import Method, resolve_data_name, resolve_models_name
from pipeline.core.utils import model_short_name as _get_model_short_name
from pipeline.tasks import get_task


def _model_project_tag(model_path: str) -> str:
    """Extract model family for wandb project names, stripping experiment suffixes.

    Examples:
        "Qwen/Qwen3-4B"           -> "qwen3-4b"
        ".../sft/qwen3-4b-up4x/model" -> "qwen3-4b"
        ".../sft/qwen2.5-1.5b/model"  -> "qwen2.5-1.5b"
        ".../rl/default/model"         -> "default"
    """
    name = _get_model_short_name(model_path)
    # Truncate after model size indicator (e.g., "qwen3-4b-up4x" -> "qwen3-4b")
    match = re.match(r'(.*?\d+\.?\d*b)', name, re.IGNORECASE)
    return match.group(1) if match else name


def _finalize_training_run(
    trainer,
    tokenizer,
    output_path: Path,
    has_eval: bool,
) -> Path:
    """Save trainer output as best/last/model (with eval) or model/ alone
    (without). Shared by train_sft and train_classifier so both checkpoint
    layouts stay identical.
    """
    run_root = output_path.parent

    def _clear(path: Path) -> None:
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    if not has_eval:
        # No validation signal -- nothing to pick a "best" epoch by, so
        # just save the final weights as before.
        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        print(f"Saved model to {output_path}")
        return output_path

    # Every epoch's weights already live on disk under run_root/checkpoint-N
    # (the trainer's own rotation keeps at most save_total_limit of these,
    # with load_best_model_at_end exempting the best one -- so this is never
    # "all" epochs, just the 1-2 that matter). Move those directories straight
    # into best/last rather than re-saving or copying full weights a second
    # time, so disk usage never doubles.
    checkpoint_dirs = sorted(
        (d for d in run_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    best_ckpt_path = (
        Path(trainer.state.best_model_checkpoint).resolve()
        if trainer.state.best_model_checkpoint else None
    )
    last_ckpt_dir = checkpoint_dirs[-1] if checkpoint_dirs else None
    best_ckpt_dir = next(
        (d for d in checkpoint_dirs if best_ckpt_path is not None and d.resolve() == best_ckpt_path),
        None,
    )

    best_dest = run_root / "best"
    last_dest = run_root / "last"
    _clear(best_dest)
    _clear(last_dest)

    if best_ckpt_dir is not None:
        shutil.move(str(best_ckpt_dir), str(best_dest))
        tokenizer.save_pretrained(str(best_dest))  # cheap -- just ensures tokenizer files are present
        print(f"Saved best model (eval_loss) to {best_dest}")
    else:
        # Rotation pruned the recorded best checkpoint (or there's none to
        # find, e.g. a single epoch) -- load_best_model_at_end already
        # reloaded those weights into trainer.model, so fall back to saving
        # from memory.
        trainer.save_model(str(best_dest))
        tokenizer.save_pretrained(str(best_dest))
        print(f"Saved best model (eval_loss) to {best_dest} (re-saved; raw checkpoint unavailable)")

    if last_ckpt_dir is not None and last_ckpt_dir == best_ckpt_dir:
        # Same epoch -- symlink instead of duplicating the weights on disk.
        last_dest.symlink_to("best", target_is_directory=True)
        print(f"last/ is the same checkpoint as best/ ({last_ckpt_dir.name}); symlinked")
    elif last_ckpt_dir is not None:
        shutil.move(str(last_ckpt_dir), str(last_dest))
        tokenizer.save_pretrained(str(last_dest))
        print(f"Saved last model (final epoch) to {last_dest}")
    else:
        # Shouldn't happen with save_strategy="epoch", but fall back to the
        # best model rather than erroring.
        last_dest.symlink_to("best", target_is_directory=True)

    # Clean up any raw checkpoint dirs still on disk (there should be none
    # left beyond what was just moved into best/last, but be defensive).
    for d in checkpoint_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    # model/ -> best/, mirroring train_rl: the canonical path points at the
    # checkpoint the run should be judged on.
    _clear(output_path)
    output_path.symlink_to("best", target_is_directory=True)
    print(f"Saved model to {run_root} (best/, last/, model/ -> best/)")

    return output_path


def train_sft(
    task_name: str,
    base_model: str,
    method_name: str | None = None,
    run_id: str | None = None,
    dataset_path: Path | None = None,
    eval_dataset_path: Path | None = None,
    output_path: Path | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 4096,
    bf16: bool = True,
    report_to: str = "wandb",
    project_name: str | None = None,
    experiment_name: str | None = None,
    include_wrong_valid_format: bool = False,
    upsample_hint: int = 1,
    max_correct: int | None = None,
    completion_only_loss: bool = False,
    strip_think_tokens: bool = False,
    data_name: str | None = None,
    models_name: str | None = None,
) -> Path:
    """
    Train an SFT model on generated dataset.

    Filters to correct examples, formats as prompt/completion pairs, and
    trains using TRL's SFTTrainer.

    When a validation set is available (see eval_dataset_path), checkpoints
    every epoch and keeps two: best/ (lowest eval_loss) and last/ (final
    epoch), with model/ symlinked to best/ -- mirroring train_rl's
    best/last/model layout. Without a validation set there is no signal to
    pick a "best" epoch by, so only the final weights are saved, directly
    at model/ (no best/last split).

    Args:
        task_name: Name of task
        base_model: Base model to fine-tune
        method_name: Method name for auto-derived paths
        run_id: Run identifier for organizing outputs (default: "default")
        dataset_path: Path to generated dataset (default: data/{data_name}/sft_datasets/sft_train__{method}.json)
        eval_dataset_path: Path to held-out dataset for validation loss (default:
            data/{data_name}/sft_datasets/sft_val__{method}.json if it exists;
            validation is skipped if neither this nor a method-derived default is found)
        output_path: Where to save trained model (default: models/{models_name}/{method}_sft/{run_id}/model,
            a symlink to best/ when a validation set is used)
        data_name: Data directory name (default: task_name)
        models_name: Models directory name (default: data_name)
        epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        max_length: Maximum sequence length
        bf16: Use bfloat16 training
        report_to: Reporting integration ("none", "wandb", etc.)
        project_name: Wandb project name (default: {task}-sft)
        experiment_name: Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})
        include_wrong_valid_format: Include wrong answers with valid format (task-specific, default: False)

    Returns:
        Path to trained model
    """
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoTokenizer

    data_name = resolve_data_name(task_name, data_name)
    models_name = resolve_models_name(data_name, models_name)

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Default dataset path - look for the method's generated sft_train dataset
    if dataset_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --dataset must be specified. "
                "Use --method to auto-derive paths, or --dataset for explicit paths."
            )
        # Never pick sft_val -- it is the held-out validation split, so
        # training on it would be training on the validation set.
        candidate = method.dataset_path(data_name, "sft_train")
        if candidate.exists():
            dataset_path = candidate
        else:
            raise FileNotFoundError(
                f"No SFT dataset for method '{method.name}' in "
                f"{method.datasets_dir(data_name)} (looked for "
                f"{method.artifact_stem('sft_train')}.json). "
                f"Run 'python -m pipeline generate --task {task_name} --method {method_name}' first."
            )

    # Default eval (validation-loss) dataset path: sft_val, if a generated
    # dataset exists for it. Optional -- unlike the train dataset, its absence
    # is not an error; we just skip validation.
    if eval_dataset_path is None and method is not None:
        candidate = method.dataset_path(data_name, "sft_val")
        if candidate.exists():
            eval_dataset_path = candidate

    # Default output path
    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        method.ensure_sft_run_dir(models_name, run_id)
        output_path = method.sft_model_path(models_name, run_id)

    # Generate project name if not provided: {task}-sft-{model_short_name}
    if project_name is None:
        project_name = f"{task_name}-sft-{_model_project_tag(base_model)}"

    # Generate experiment name if not provided
    # Format: {method}-{run_id}
    # e.g., "simple-default"
    if experiment_name is None:
        method_str = method_name if method_name else "default"
        run_id_str = run_id if run_id else "default"
        experiment_name = f"{method_str}-{run_id_str}"

    run_id_display = run_id or "default"
    print(f"=== SFT Training Configuration ===")
    print(f"Task: {task_name}")
    print(f"Method: {method_name or 'default'}")
    print(f"Run ID: {run_id_display}")
    print(f"Base Model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print(f"Eval Dataset: {eval_dataset_path or '(none -- validation loss disabled)'}")
    print(f"Output: {output_path}")
    print(f"Project: {project_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Report to: {report_to}")
    print(f"==================================")

    # Load and filter dataset
    print(f"Loading dataset from {dataset_path}")
    data = load_json(dataset_path)

    # Get task for potential custom filtering
    task = get_task(task_name)

    # Use task-specific filter if available, otherwise keep only correct
    if hasattr(task, 'filter_for_sft'):
        filtered_examples = task.filter_for_sft(
            data,
            include_wrong_valid_format=include_wrong_valid_format,
            **({"nested_request": True}
               if method is not None and method.nested_request else {}),
        )
        # Count categories for logging
        num_correct = sum(1 for ex in filtered_examples if ex.get("correct", False))
        num_wrong_valid = len(filtered_examples) - num_correct
        print(f"Loaded {len(data)} examples, keeping {len(filtered_examples)} "
              f"({num_correct} correct, {num_wrong_valid} wrong-valid-format)")
    else:
        # No custom filter: include_wrong_valid_format is task-specific, so a
        # task without one has no way to judge a wrong answer's format.
        filtered_examples = [ex for ex in data if ex.get("correct", False)]
        print(f"Loaded {len(data)} examples, {len(filtered_examples)} correct ({100*len(filtered_examples)/len(data):.1f}%)")

    if not filtered_examples:
        raise ValueError("No valid examples found in dataset!")

    # Downsample correct examples if requested
    if max_correct is not None:
        import random
        correct = [ex for ex in filtered_examples if ex.get("correct", False)]
        non_correct = [ex for ex in filtered_examples if not ex.get("correct", False)]
        if len(correct) > max_correct:
            rng = random.Random(42)
            correct = rng.sample(correct, max_correct)
            filtered_examples = correct + non_correct
            rng.shuffle(filtered_examples)
            print(f"Downsampled correct to {max_correct} → {len(filtered_examples)} total ({max_correct} correct, {len(non_correct)} non-correct)")

    # Upsample hint-containing examples
    if upsample_hint > 1:
        hint_examples = [ex for ex in filtered_examples if "<request>" in ex.get("generation", "")]
        if hint_examples:
            extra_copies = hint_examples * (upsample_hint - 1)
            filtered_examples = filtered_examples + extra_copies
            print(f"Upsampled {len(hint_examples)} hint examples {upsample_hint}x → "
                  f"{len(hint_examples) * upsample_hint} copies (total: {len(filtered_examples)})")
        else:
            print(f"--upsample-hint={upsample_hint} specified but no hint examples found")

    # Load tokenizer
    print(f"Loading tokenizer for {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if strip_think_tokens:
        # Qwen3 ships <think>/</think> as added tokens, so each is ONE id
        # (151667/151668). Qwen2.5 has no such entries and encodes the same
        # strings as ordinary text (6 ids for the pair). They are not marked
        # `special`, so nothing is being silently dropped -- the problem is
        # purely representational: the two base models would see structurally
        # different inputs for identical data, and their SFT runs would not be
        # comparable. Removing the entries makes Qwen3 tokenize the reasoning
        # tags exactly the way Qwen2.5 does.
        #
        # The entries live in tokenizer.json AND tokenizer_config.json; dropping
        # them from only one lets the reload put them back.
        import json as _json
        import tempfile as _tempfile
        _d = _tempfile.mkdtemp(prefix="tok_nothink_")
        tokenizer.save_pretrained(_d)
        _removed = 0
        _tj = os.path.join(_d, "tokenizer.json")
        if os.path.exists(_tj):
            _j = _json.load(open(_tj))
            _before = len(_j.get("added_tokens", []))
            _j["added_tokens"] = [a for a in _j.get("added_tokens", [])
                                  if a.get("content") not in ("<think>", "</think>")]
            _removed = _before - len(_j["added_tokens"])
            _json.dump(_j, open(_tj, "w"))
        _cf = os.path.join(_d, "tokenizer_config.json")
        if os.path.exists(_cf):
            _c = _json.load(open(_cf))
            _c["added_tokens_decoder"] = {
                k: v for k, v in _c.get("added_tokens_decoder", {}).items()
                if v.get("content") not in ("<think>", "</think>")}
            _json.dump(_c, open(_cf, "w"))
        if _removed:
            tokenizer = AutoTokenizer.from_pretrained(_d)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"  strip-think-tokens: removed {_removed} added-token entr"
              f"{'y' if _removed == 1 else 'ies'}; "
              f"'<think>' now encodes to "
              f"{len(tokenizer.encode('<think>', add_special_tokens=False))} token(s)")

    # Check if we need response masking
    mask_response_tokens = method.mask_response_tokens if method else False

    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Format data for SFT
    def format_examples(examples: list[dict]) -> list[dict]:
        formatted = []
        if mask_response_tokens:
            # Pre-tokenize with response masking for clean boundaries
            from pipeline.core.utils import tokenize_with_response_mask

            for ex in examples:
                # Apply chat template to prompt messages (excluding assistant prefix)
                messages = ex["prompt"]
                if messages and messages[-1]["role"] == "assistant":
                    conversation = messages[:-1]
                    assistant_prefix = messages[-1]["content"]
                else:
                    conversation = messages
                    assistant_prefix = ""

                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Completion is assistant prefix + generation
                completion = assistant_prefix + ex["generation"]

                # Tokenize prompt (all masked, completion_mask=0)
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

                # Tokenize completion with response masking
                completion_tokens, response_mask = tokenize_with_response_mask(
                    completion, tokenizer
                )

                # Combine: prompt (mask=0) + completion (mask from response_mask)
                input_ids = prompt_tokens + completion_tokens
                completion_mask = [0] * len(prompt_tokens) + response_mask

                formatted.append({
                    "input_ids": input_ids,
                    "completion_mask": completion_mask,
                })
        else:
            # Standard prompt/completion format (SFTTrainer handles tokenization)
            for ex in examples:
                messages = ex["prompt"]
                if messages and messages[-1]["role"] == "assistant":
                    conversation = messages[:-1]
                    assistant_prefix = messages[-1]["content"]
                else:
                    conversation = messages
                    assistant_prefix = ""

                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                completion = assistant_prefix + ex["generation"]

                formatted.append({
                    "prompt": prompt,
                    "completion": completion,
                })
        return formatted

    print("Formatting for SFT...")
    if mask_response_tokens:
        print("  Using segmented tokenization for response masking")
        print("  Masking \\n<response>...</response>\\n spans")

    train_dataset = Dataset.from_list(format_examples(filtered_examples))
    print(f"Train: {len(train_dataset)}")

    # Load, filter (correctness only -- no downsample/upsample, this is a
    # fixed held-out set for measuring validation loss, not a training
    # signal to rebalance), and format the eval set, if one was found/given.
    eval_dataset = None
    if eval_dataset_path is not None:
        print(f"Loading eval dataset from {eval_dataset_path}")
        eval_data = load_json(eval_dataset_path)
        if hasattr(task, 'filter_for_sft'):
            filtered_eval_examples = task.filter_for_sft(
                eval_data,
                include_wrong_valid_format=include_wrong_valid_format,
                **({"nested_request": True}
                   if method is not None and method.nested_request else {}),
            )
        else:
            filtered_eval_examples = [ex for ex in eval_data if ex.get("correct", False)]
        if filtered_eval_examples:
            eval_dataset = Dataset.from_list(format_examples(filtered_eval_examples))
            print(f"Eval: {len(eval_dataset)}")
        else:
            print("  No valid examples found in eval dataset -- skipping validation.")

    # Data collator - standard collator handles completion_mask
    data_collator = None

    # run_root holds best/, last/, and the model/ symlink, mirroring
    # train_rl's checkpoint layout. Raw trainer checkpoints land directly in
    # run_root too (as checkpoint-N/) and are cleaned up once best/last are
    # extracted.
    run_root = output_path.parent
    has_eval = eval_dataset is not None

    # Training config. With a validation set, checkpoint every epoch and
    # keep the best (lowest eval_loss) one alongside the last -- otherwise
    # there is no signal to pick a "best" epoch by, so only the final
    # weights are saved (previous behavior).
    training_args = SFTConfig(
        output_dir=str(run_root),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        # TRL infers this from the dataset columns: the pre-tokenized
        # mask_response_tokens path emits input_ids/completion_mask rather than
        # prompt/completion, so it lands on False and the completion_mask is
        # never applied. Passed explicitly so the choice is visible. Full
        # sequence is the default because every RL parent was trained that
        # way, and the masked ablation (qwen3-4b-up4x-tok) scored lower.
        completion_only_loss=completion_only_loss,
        logging_steps=10,
        save_strategy="epoch" if has_eval else "no",
        eval_strategy="epoch" if has_eval else "no",
        save_total_limit=2 if has_eval else None,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        bf16=bf16,
        report_to=report_to,
        run_name=experiment_name,
    )

    # Initialize wandb if enabled
    if report_to == "wandb":
        import wandb
        wandb.init(project=project_name, name=experiment_name, reinit=True)

    # Train
    print(f"Starting training: {base_model} -> {output_path}")
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    return _finalize_training_run(trainer, tokenizer, output_path, has_eval)


def train_classifier(
    task_name: str,
    base_model: str,
    method_name: str | None = None,
    run_id: str | None = None,
    dataset_path: Path | None = None,
    eval_dataset_path: Path | None = None,
    output_path: Path | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 4096,
    bf16: bool = True,
    report_to: str = "wandb",
    project_name: str | None = None,
    experiment_name: str | None = None,
    completion_only_loss: bool = True,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    depth_eval_batch_size: int = 16,
    depth_eval_max_new_tokens: int = 4,
    balance_train: bool = False,
    data_name: str | None = None,
    models_name: str | None = None,
) -> Path:
    """
    Train a verifier/continue-vs-abstain classifier on `generate_tree` output.

    Unlike train_sft (which filters to generations with `correct: True`),
    `generate_tree` records carry no `correct` field -- each record's
    `generation` is itself the gold label ("0"/"1", from thresholding
    pass_rate), with `samples` kept only as the raw rollouts the label was
    computed from. So every record is trained on; there is no correctness
    filter here.

    Each record also carries a `depth` (0 = root, increasing towards the
    leaf) at which its partial rollout was probed. Pooled eval_loss can look
    fine while the classifier is blind at a specific depth (e.g. it nails
    the root's "will this solve at all" call but can't tell a near-finished
    rollout from a stalled one) -- so every evaluation also greedily decodes
    the held-out set and reports accuracy broken out by depth, not just loss.

    Args:
        task_name: Name of task
        base_model: Base model to fine-tune
        method_name: Method name for auto-derived paths
        run_id: Run identifier for organizing outputs (default: "default")
        dataset_path: Path to generate_tree output (default:
            data/{data_name}/sft_datasets/sft_train__{method}__tree.json)
        eval_dataset_path: Path to held-out tree dataset for validation loss
            and depth-accuracy reporting (default:
            data/{data_name}/sft_datasets/sft_val__{method}__tree.json if it
            exists; both are skipped if neither this nor a method-derived
            default is found)
        output_path: Where to save trained model (default:
            models/{models_name}/{method}_classifier/{run_id}/model)
        data_name: Data directory name (default: task_name)
        models_name: Models directory name (default: data_name)
        epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        max_length: Maximum sequence length
        bf16: Use bfloat16 training
        report_to: Reporting integration ("none", "wandb", etc.)
        project_name: Wandb project name (default: {task}-classifier)
        experiment_name: Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})
        completion_only_loss: Mask the prompt out of the loss, training only
            on the gold "0"/"1" label token(s). Default True (unlike
            train_sft's full-sequence default) -- the prompt here is the
            entire partial rollout being judged, not something the
            classifier should learn to reproduce.
        use_lora: Train a LoRA adapter instead of full fine-tuning.
        lora_r: LoRA rank (only used when use_lora=True).
        lora_alpha: LoRA alpha (only used when use_lora=True).
        lora_dropout: LoRA dropout (only used when use_lora=True).
        lora_target_modules: Module names to adapt (only used when
            use_lora=True); default None lets peft pick the base model's
            standard attention/MLP projections.
        depth_eval_batch_size: Batch size for the per-depth greedy-decode
            accuracy pass run at every evaluation.
        depth_eval_max_new_tokens: Max new tokens to generate per held-out
            example when computing depth accuracy (the label is a single
            digit, so this only needs to be a few tokens).
        balance_train: Upsample the minority label in the train set (with
            replacement, randomly) so 0/1 are equally represented. Only
            affects train -- eval/depth-accuracy stays on the true
            distribution, since that's what reports realistic accuracy.

    Returns:
        Path to trained model
    """
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoTokenizer, TrainerCallback

    data_name = resolve_data_name(task_name, data_name)
    models_name = resolve_models_name(data_name, models_name)

    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # generate_tree names its output `{split}__{method}__tree__{run_id}.json`
    # when it was given a run_id, and `{split}__{method}__tree.json` otherwise
    # (pipeline/commands/data.py's own `desc = f"tree__{run_id}" if run_id
    # else "tree"`). Mirror that here: a run_id-suffixed dataset takes
    # priority (it's the one generate_tree would have produced for this same
    # run_id), falling back to the bare "tree" dataset shared across runs.
    def _tree_descs() -> list[str]:
        return [f"tree__{run_id}", "tree"] if run_id else ["tree"]

    if dataset_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --dataset must be specified. "
                "Use --method to auto-derive paths, or --dataset for explicit paths."
            )
        descs = _tree_descs()
        candidates = [method.dataset_path(data_name, "sft_train", desc=d) for d in descs]
        dataset_path = next((c for c in candidates if c.exists()), None)
        if dataset_path is None:
            looked_for = ", ".join(method.artifact_stem("sft_train", desc=d) + ".json" for d in descs)
            raise FileNotFoundError(
                f"No tree dataset for method '{method.name}' in "
                f"{method.datasets_dir(data_name)} (looked for {looked_for}). "
                f"Run 'python -m pipeline generate_tree --task {task_name} --method {method_name}' first."
            )

    if eval_dataset_path is None and method is not None:
        candidate = next(
            (c for c in (method.dataset_path(data_name, "sft_val", desc=d) for d in _tree_descs()) if c.exists()),
            None,
        )
        if candidate is not None:
            eval_dataset_path = candidate

    if output_path is None:
        if method is None:
            raise ValueError(
                "Either --method or --output must be specified. "
                "Use --method to auto-derive paths, or --output for explicit paths."
            )
        method.ensure_classifier_run_dir(models_name, run_id)
        output_path = method.classifier_model_path(models_name, run_id)

    if project_name is None:
        project_name = f"{task_name}-classifier-{_model_project_tag(base_model)}"

    if experiment_name is None:
        method_str = method_name if method_name else "default"
        run_id_str = run_id if run_id else "default"
        experiment_name = f"{method_str}-{run_id_str}"

    run_id_display = run_id or "default"
    print(f"=== Classifier Training Configuration ===")
    print(f"Task: {task_name}")
    print(f"Method: {method_name or 'default'}")
    print(f"Run ID: {run_id_display}")
    print(f"Base Model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print(f"Eval Dataset: {eval_dataset_path or '(none -- validation loss and depth accuracy disabled)'}")
    print(f"Output: {output_path}")
    print(f"Project: {project_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Report to: {report_to}")
    print(f"LoRA: {use_lora}")
    print(f"==========================================")

    # Load tokenizer
    print(f"Loading tokenizer for {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def _gold_label(ex: dict) -> int:
        # `label` is the int the generator thresholded pass_rate against;
        # `generation` is just str(label). Prefer `label` but fall back to
        # parsing `generation` so this also accepts hand-built tree-format
        # data that only set one of the two.
        if "label" in ex:
            return int(ex["label"])
        return int(ex["generation"])

    def balance_examples(examples: list[dict]) -> list[dict]:
        """Upsample (with replacement) the minority label so both classes
        are equally represented -- train only, by design; see
        balance_train's docstring entry for why eval stays untouched."""
        import random
        from collections import defaultdict

        by_label: dict = defaultdict(list)
        for ex in examples:
            by_label[_gold_label(ex)].append(ex)
        if len(by_label) < 2:
            return examples  # nothing to balance against
        majority_count = max(len(v) for v in by_label.values())
        balanced = []
        for label, group in by_label.items():
            balanced.extend(group)
            deficit = majority_count - len(group)
            if deficit > 0:
                balanced.extend(random.choices(group, k=deficit))
        random.shuffle(balanced)
        counts = {label: len(v) for label, v in by_label.items()}
        print(f"  Balancing train: {counts} -> {majority_count} each ({len(balanced)} total)")
        return balanced

    def format_examples(examples: list[dict]) -> list[dict]:
        """prompt/completion pairs for SFTTrainer. "<answer>" is appended to
        the prompt as a cue token (so the model learns it signals "output
        the 0/1 label now"); the completion/loss target is still the bare
        "0"/"1" label, exactly as generate_tree wrote it."""
        formatted = []
        for ex in examples:
            messages = ex["prompt"]
            if messages and messages[-1]["role"] == "assistant":
                conversation = messages[:-1]
                assistant_prefix = messages[-1]["content"]
            else:
                conversation = messages
                assistant_prefix = ""

            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            # "<answer>" is a cue appended to the prompt (not the loss
            # target) so the model learns that this token means "now output
            # the 0/1 label" -- the completion itself is still bare "0"/"1",
            # exactly as generate_tree wrote it.
            prompt = prompt + assistant_prefix + "<answer>"
            completion = ex["generation"]

            formatted.append({"prompt": prompt, "completion": completion})
        return formatted

    def format_for_depth_eval(examples: list[dict]) -> list[dict]:
        """Prompt-only (no completion) + gold label + depth, for the greedy
        decode accuracy pass -- paired 1:1 with format_examples's prompts."""
        depth_examples = []
        for ex in examples:
            messages = ex["prompt"]
            if messages and messages[-1]["role"] == "assistant":
                conversation = messages[:-1]
                assistant_prefix = messages[-1]["content"]
            else:
                conversation = messages
                assistant_prefix = ""

            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            depth_examples.append({
                "index": ex.get("index"),
                "prompt": prompt + assistant_prefix + "<answer>",
                "label": _gold_label(ex),
                "depth": ex.get("depth", 0),
            })
        return depth_examples

    # Load train
    print(f"Loading dataset from {dataset_path}")
    train_examples = load_json(dataset_path)
    print(f"Loaded {len(train_examples)} examples (all used -- gold labels, no correctness filter)")
    if balance_train:
        train_examples = balance_examples(train_examples)
    train_dataset = Dataset.from_list(format_examples(train_examples))
    print(f"Train: {len(train_dataset)}")

    # Load val
    eval_dataset = None
    depth_eval_examples: list[dict] = []
    if eval_dataset_path is not None:
        print(f"Loading eval dataset from {eval_dataset_path}")
        eval_examples = load_json(eval_dataset_path)
        if eval_examples:
            eval_dataset = Dataset.from_list(format_examples(eval_examples))
            depth_eval_examples = format_for_depth_eval(eval_examples)
            print(f"Eval: {len(eval_dataset)}")
        else:
            print("  Eval dataset is empty -- skipping validation.")

    has_eval = eval_dataset is not None

    # LoRA adapter instead of full fine-tuning, if requested.
    peft_config = None
    if use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            task_type="CAUSAL_LM",
        )
        print(f"Using LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, "
              f"target_modules={lora_target_modules or '(peft default for base model)'}")

    run_root = output_path.parent

    training_args = SFTConfig(
        output_dir=str(run_root),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        completion_only_loss=completion_only_loss,
        logging_steps=10,
        save_strategy="epoch" if has_eval else "no",
        eval_strategy="epoch" if has_eval else "no",
        save_total_limit=2 if has_eval else None,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        bf16=bf16,
        report_to=report_to,
        run_name=experiment_name,
    )

    if report_to == "wandb":
        import wandb
        wandb.init(project=project_name, name=experiment_name, reinit=True)

    class _DepthAccuracyCallback(TrainerCallback):
        """Greedily decodes the held-out set at every evaluation and reports
        label accuracy broken out by tree depth, alongside the pooled
        eval_loss HF Trainer already logs. A verifier can look fine on
        average while being blind at one specific depth, so depth is never
        averaged away.

        Nested inside train_classifier (rather than module-level) so it can
        subclass TrainerCallback without a top-level `transformers` import --
        CallbackHandler.call_event calls every event name unconditionally via
        getattr, so a plain duck-typed object (defining only on_evaluate)
        raises AttributeError on the other events.
        """

        def __init__(
            self,
            eval_examples: list[dict],
            batch_size: int,
            max_new_tokens: int,
            run_root: Path,
        ):
            self.eval_examples = eval_examples  # [{"index", "prompt", "label", "depth"}, ...]
            self.batch_size = batch_size
            self.max_new_tokens = max_new_tokens
            # Aggregate per-depth accuracy, one row per evaluation.
            self.history_path = run_root / "depth_accuracy_history.json"
            # Every held-out example's own prediction, one file per
            # evaluation -- the aggregate accuracy above is computed from
            # these, but the raw generations are what you need to look at
            # misclassified examples later rather than just the headline
            # number.
            self.generations_dir = run_root / "val_generations"

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            import torch
            from collections import defaultdict

            if model is None or not self.eval_examples:
                return control

            prev_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            was_training = model.training
            model.eval()

            predictions: list[str] = []
            device = next(model.parameters()).device
            with torch.no_grad():
                for i in range(0, len(self.eval_examples), self.batch_size):
                    batch = self.eval_examples[i:i + self.batch_size]
                    encoded = tokenizer(
                        [ex["prompt"] for ex in batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    new_tokens = generated[:, encoded["input_ids"].shape[1]:]
                    predictions.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

            tokenizer.padding_side = prev_padding_side
            if was_training:
                model.train()

            correct_by_depth: dict = defaultdict(int)
            total_by_depth: dict = defaultdict(int)
            per_example_records = []
            for ex, text in zip(self.eval_examples, predictions):
                stripped = text.strip()
                predicted_label = int(stripped) if stripped in ("0", "1") else None
                is_correct = predicted_label == ex["label"]
                total_by_depth[ex["depth"]] += 1
                if is_correct:
                    correct_by_depth[ex["depth"]] += 1
                per_example_records.append({
                    "index": ex.get("index"),
                    "depth": ex["depth"],
                    "label": ex["label"],
                    "generation": text,
                    "predicted_label": predicted_label,
                    "correct": is_correct,
                })

            log_payload = {}
            overall_correct = overall_total = 0
            print(f"\n=== Depth accuracy (step {state.global_step}) ===")
            for depth in sorted(total_by_depth):
                correct, total = correct_by_depth[depth], total_by_depth[depth]
                accuracy = correct / total if total else 0.0
                print(f"  depth {depth}: {accuracy:.2%} ({correct}/{total})")
                log_payload[f"eval/depth_{depth}_accuracy"] = accuracy
                overall_correct += correct
                overall_total += total
            overall_accuracy = overall_correct / overall_total if overall_total else 0.0
            print(f"  overall: {overall_accuracy:.2%} ({overall_correct}/{overall_total})")
            log_payload["eval/depth_overall_accuracy"] = overall_accuracy

            if state.is_world_process_zero:
                # Persisted locally regardless of --report-to, so the run is
                # still analyzable without wandb (or if report_to=none).
                self.generations_dir.mkdir(parents=True, exist_ok=True)
                generations_path = self.generations_dir / f"step_{state.global_step}.json"
                save_json(generations_path, per_example_records)
                print(f"  Saved {len(per_example_records)} val generations to {generations_path}")

                history = load_json(self.history_path) if self.history_path.exists() else []
                history.append({
                    "step": state.global_step,
                    "epoch": state.epoch,
                    **log_payload,
                })
                save_json(self.history_path, history)

                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(log_payload, step=state.global_step)
                except ImportError:
                    pass

            return control

    callbacks = []
    if depth_eval_examples:
        callbacks.append(_DepthAccuracyCallback(
            eval_examples=depth_eval_examples,
            batch_size=depth_eval_batch_size,
            max_new_tokens=depth_eval_max_new_tokens,
            run_root=run_root,
        ))

    print(f"Starting training: {base_model} -> {output_path}")
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()

    return _finalize_training_run(trainer, tokenizer, output_path, has_eval)

def _wandb_run_id(run_dir: Path | None) -> str | None:
    """Return a stable wandb run id for this run directory, minting one if absent.

    verl calls wandb.init() once per process, so a run that gets preempted,
    resumed after an allocation expires, or relaunched by hand shows up in wandb
    as several disconnected runs. Pinning the id to the run directory makes every
    launch of the same run_id append to one curve. --overwrite clears the run
    directory and therefore the id, which is the intended behaviour: a discarded
    run should not keep writing to the old wandb run.
    """
    if run_dir is None:
        return None
    id_file = run_dir / "wandb_run_id"
    if id_file.exists():
        return id_file.read_text().strip() or None
    # wandb moved generate_id out of wandb.util (gone by 0.29; the lockfile pins
    # 0.24.1, where it still exists). Try the current location first, fall back
    # to the old one, and finally to a local id -- the value only has to be a
    # stable unique string, so an outage in either API must not kill training.
    run_id = None
    try:
        from wandb.sdk.lib.runid import generate_id as _gen
        run_id = _gen()
    except Exception:
        try:
            import wandb.util
            run_id = wandb.util.generate_id()
        except Exception:
            import secrets
            run_id = secrets.token_hex(4)
    id_file.write_text(run_id + "\n")
    return run_id


def _guard_existing_run(
    run_dir: Path | None,
    checkpoints_dir: Path | None,
    output_path: Path | None,
    rollouts_dir: Path | None,
    *,
    overwrite: bool,
    continue_run: bool,
) -> None:
    """Refuse to start a fresh run on top of an existing one.

    Two ways a rerun silently destroys work. A finished run has had its
    checkpoints cleaned up but still holds model/ and rollouts/, so verl finds
    nothing to resume, trains from scratch, and the final conversion overwrites
    the old model/. An interrupted run still has checkpoints, so resume_mode=auto
    picks it up mid-flight under whatever hyperparameters this invocation passed.
    Neither is ever what a fresh run wants, so both require saying so explicitly.
    """
    import re
    import shutil

    ckpts = []
    if checkpoints_dir is not None and checkpoints_dir.is_dir():
        ckpts = sorted(
            (d for d in checkpoints_dir.iterdir()
             if d.is_dir() and re.fullmatch(r"global_step_\d+", d.name)),
            key=lambda d: int(d.name.rsplit("_", 1)[1]),
        )

    model_dirs = [
        d for d in (
            output_path,
            (run_dir / "best") if run_dir is not None else None,
            (run_dir / "last") if run_dir is not None else None,
        )
        if d is not None
    ]

    found = []
    for path in model_dirs:
        if path.is_symlink() or (path.is_dir() and any(path.iterdir())):
            found.append(f"model        {path}")
    if ckpts:
        found.append(f"{len(ckpts)} checkpoint(s)  {checkpoints_dir} (latest {ckpts[-1].name})")
    if rollouts_dir is not None and rollouts_dir.is_dir() and any(rollouts_dir.iterdir()):
        found.append(f"rollouts     {rollouts_dir}")

    if continue_run:
        if not ckpts:
            raise FileNotFoundError(
                f"--continue-run passed but no checkpoints to resume in {checkpoints_dir}.\n"
                f"  A completed run has its checkpoints cleaned up; only --overwrite "
                f"or a new --run-id can proceed."
            )
        print(f"Continuing existing run from {ckpts[-1].name}")
        return

    if not found:
        return

    if overwrite:
        print(f"--overwrite: clearing existing run at {run_dir}")
        for path in [checkpoints_dir, rollouts_dir, *model_dirs]:
            if path is None:
                continue
            if path.is_symlink():
                path.unlink()
                print(f"  removed {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"  removed {path}")
        return

    listing = "\n".join(f"    {item}" for item in found)
    raise FileExistsError(
        f"run directory already holds results: {run_dir}\n{listing}\n"
        f"  Training here would overwrite them. Pick a new --run-id, or pass\n"
        f"  --overwrite to discard them, or --continue-run to resume in place."
    )


def train_rl(
    task_name: str,
    method_name: str | None = None,
    run_id: str | None = None,
    base_model: str | None = None,
    train_prompts_path: Path | None = None,
    val_prompts_path: Path | None = None,
    sft_model_path: Path | None = None,
    output_path: Path | None = None,
    reward_function_path: Path | None = None,
    train_batch_size: int = 64,
    val_batch_size: int = 64,
    learning_rate: float = 1e-6,
    total_steps: int = 400,
    kl_coef: float = 0.001,
    n_samples: int = 8,
    save_freq: int | None = None,
    test_freq: int | None = None,
    max_prompt_length: int = 2048,
    max_response_length: int = 2048,
    max_model_len: int = 8192,
    tensor_parallel_size: int = 1,
    n_gpus_per_node: int | None = None,
    gpu_memory_utilization: float = 0.5,
    project_name: str | None = None,
    experiment_name: str | None = None,
    wandb: bool = True,
    resume_path: Path | None = None,
    cleanup_checkpoints: bool = True,
    keep_state: bool = False,
    reward_kwargs_overrides: dict | None = None,
    extra_overrides: list[str] | None = None,
    shuffle_seed: int | None = None,
    overwrite: bool = False,
    continue_run: bool = False,
    save_best: bool = False,
    best_metric: str = "auto",
    max_ckpt_to_keep: int | None = None,
    data_name: str | None = None,
    models_name: str | None = None,
) -> Path:
    """
    Train RL model using verl (GRPO algorithm).

    Args:
        task_name: Name of task
        method_name: Method name for auto-derived paths and reward config
        run_id: Run identifier for organizing outputs (default: "default")
        base_model: Base model for cold-start RL (mutually exclusive with sft_model_path)
        train_prompts_path: Path to RL train prompts parquet file
        val_prompts_path: Path to RL validation prompts parquet file
        sft_model_path: Path to SFT model to start from (mutually exclusive with base_model)
        output_path: Where to save RL model
        reward_function_path: Path to reward function (default: task's reward function)
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        learning_rate: Learning rate
        total_steps: Total training steps
        kl_coef: KL divergence coefficient
        n_samples: Number of samples per prompt
        save_freq: Checkpoint save frequency
        test_freq: Validation/logging frequency (default: same as save_freq)
        max_prompt_length: Maximum prompt length in tokens
        max_response_length: Maximum response length in tokens
        max_model_len: Maximum model context length (default: prompt + response length).
            Set higher than prompt + response for multi-turn hint mode.
        tensor_parallel_size: Tensor parallel size for vLLM rollout (how many GPUs each
            rollout replica is split across)
        n_gpus_per_node: Total GPUs per node used by the trainer (actor/ref FSDP +
            rollout). Default: auto-detected via torch.cuda.device_count() so all
            visible GPUs are used. Decoupled from tensor_parallel_size -- set this
            higher than tensor_parallel_size to add data-parallel rollout/training
            capacity rather than splitting the model further.
        gpu_memory_utilization: GPU memory utilization for vLLM
        project_name: Wandb project name (default: {task}-rl)
        experiment_name: Custom experiment name (default: {method}-{run_id}-{YYYYMMDD})
        wandb: Enable wandb logging
        resume_path: Path to resume from existing run (overrides output_path)
        cleanup_checkpoints: Delete checkpoints after training (default: True)
        keep_state: Keep the last optimizer state checkpoint after training (default: False)
        data_name: Data directory name (default: task_name)
        models_name: Models directory name (default: data_name)

    Returns:
        Path to trained model
    """
    import subprocess
    import os
    import torch

    from pipeline.tasks import get_task

    # Decoupled from tensor_parallel_size (vLLM rollout TP): defaults to all
    # visible GPUs so the trainer isn't silently pinned to 1 GPU.
    if n_gpus_per_node is None:
        n_gpus_per_node = torch.cuda.device_count() or 1

    data_name = resolve_data_name(task_name, data_name)
    models_name = resolve_models_name(data_name, models_name)

    # Get repo root (assumes we're running from repo root)
    repo_root = Path.cwd()

    # Load method config if specified
    method = None
    if method_name is not None:
        method = Method.load(method_name, task_name)

    # Validate mutual exclusion
    if base_model is not None and sft_model_path is not None:
        raise ValueError("Cannot specify both --base-model and --sft-model")

    # Frequencies. With --save-best the periodic save is off by default: the only
    # checkpoints worth keeping are the ones validation picked, and an explicit
    # --save-freq still wins. test_freq must not inherit a disabled save_freq or
    # validation would switch itself off along with it.
    if save_freq is None:
        save_freq = -1 if save_best else 25
    if test_freq is None:
        test_freq = save_freq if save_freq > 0 else 25
    if save_best and test_freq <= 0:
        raise ValueError("--save-best needs validation enabled; --test-freq must be > 0")

    # verl prunes checkpoints by recency. Best-saves only ever happen at
    # increasing steps and the final save is later still, so the two most recent
    # survivors are exactly the best and the last -- but only if room for two.
    if max_ckpt_to_keep is None:
        max_ckpt_to_keep = 2 if save_best else 1
    elif save_best and max_ckpt_to_keep < 2:
        print("Note: --save-best needs room for the best and the final checkpoint; using --max-ckpt-to-keep 2")
        max_ckpt_to_keep = 2

    # Default paths from method. method_ac trains on the "gen" half of
    # rl_train/rl_val (scripts/split_dataset.py's stratified subdivision) so
    # the "ver" half stays unseen by the solver's RL run, for later use by
    # create_verification_data's --run-solver step.
    if method is not None:
        train_split = "rl_gen_train" if method.name == "method_ac" else "rl_train"
        val_split = "rl_gen_val" if method.name == "method_ac" else "rl_val"
        if train_prompts_path is None:
            train_prompts_path = method.formatted_path(data_name, train_split)
        if val_prompts_path is None:
            candidate = method.formatted_path(data_name, val_split)
            if candidate.exists():
                val_prompts_path = candidate

    # Determine the actor model (base_model or sft_model_path)
    if base_model is not None:
        actor_model = base_model
    elif sft_model_path is not None:
        actor_model = str(sft_model_path)
    elif method is not None and run_id:
        actor_model = str(method.sft_model_path(models_name, run_id))
    else:
        raise ValueError("Either --base-model or --sft-model is required (or use --method and --run-id)")

    # Derive run directory structure from method
    run_dir = None
    checkpoints_dir = None
    rollouts_dir = None
    if method is not None and output_path is None:
        method.ensure_rl_run_dir(models_name, run_id)
        run_dir = method.rl_run_dir(models_name, run_id)
        checkpoints_dir = method.rl_checkpoints_dir(models_name, run_id)
        rollouts_dir = method.rl_rollouts_dir(models_name, run_id)
        output_path = method.rl_model_path(models_name, run_id)
    elif output_path is not None:
        # Custom output path - derive subdirectories from it
        run_dir = output_path.parent if output_path.name == "model" else output_path
        checkpoints_dir = run_dir / "checkpoints"
        rollouts_dir = run_dir / "rollouts"
        output_path = run_dir / "model"

    if resume_path is None:
        _guard_existing_run(
            run_dir, checkpoints_dir, output_path, rollouts_dir,
            overwrite=overwrite, continue_run=continue_run,
        )

    # Validate required paths
    if train_prompts_path is None:
        raise ValueError("--train-prompts is required (or use --method)")
    # val_prompts_path is optional — training will skip validation if not provided
    if output_path is None or checkpoints_dir is None or rollouts_dir is None:
        raise ValueError("--output is required (or use --method)")

    # Convert all paths to absolute paths for Ray workers (they run in different working directories)
    train_prompts_path = Path(train_prompts_path).resolve()
    if val_prompts_path is not None:
        val_prompts_path = Path(val_prompts_path).resolve()
    checkpoints_dir = Path(checkpoints_dir).resolve()
    rollouts_dir = Path(rollouts_dir).resolve()
    output_path = Path(output_path).resolve()

    # A prompts file that does not exist used to be handed straight to verl,
    # which then failed much later with an opaque error. Only explicitly-passed
    # paths can be missing here: the auto-derived val path above is assigned
    # solely when it exists. A prompts path may also be sharded (see
    # save_parquet/load_parquet_shards in pipeline/core/io.py) -- i.e. the
    # single file doesn't exist but `<stem>.shard000.parquet` etc do -- so
    # check for either before reporting it as missing.
    def _prompts_exist(p: Path) -> bool:
        if p.exists():
            return True
        return bool(list(p.parent.glob(f"{p.stem}.shard*{p.suffix}")))

    for flag, prompts_path in (
        ("--train-prompts", train_prompts_path),
        ("--val-prompts", val_prompts_path),
    ):
        if prompts_path is None or _prompts_exist(prompts_path):
            continue
        split = prompts_path.stem
        supported = get_task(task_name).supported_splits()
        if split not in supported:
            detail = (
                f"task '{task_name}' does not define the '{split}' split "
                f"(it has: {', '.join(supported)})"
            )
        else:
            detail = (
                f"run: python -m pipeline create_prompts --task {task_name} "
                f"--method {method.name if method else '<method>'} --split {split}"
            )
        raise FileNotFoundError(f"{flag} not found: {prompts_path}\n  {detail}")
    if not actor_model.startswith("/") and "/" in actor_model:
        # Relative path (not a HuggingFace model ID like "Qwen/Qwen2.5-1.5B")
        actor_model = str(Path(actor_model).resolve())

    # Get reward function name and other config from method
    reward_function_name = "compute_score"
    reward_kwargs = {}
    template_content = None
    allow_hint = False
    interaction_name = None
    max_turns = 6
    max_hints = None
    if method is not None:
        reward_function_name = method.reward_function
        reward_kwargs = {**method.reward_kwargs}
        # reward_kwargs is the only channel into the reward function, so a
        # method flag the scorer needs has to travel on it. nested_request
        # selects the inline-request grammar; without it the scorer validates
        # against v1's and marks every rollout malformed.
        if method.nested_request:
            reward_kwargs["nested_request"] = True
        allow_hint = method.allow_hint
        interaction_name = f"{task_name}_{method.name}" if method.multi_turn else None
        max_turns = method.max_turns
        max_hints = method.max_hints
        template_content = method.load_template(task_name, "rl")

    # Apply CLI overrides to reward kwargs
    if reward_kwargs_overrides:
        reward_kwargs.update(reward_kwargs_overrides)

    # Handle resume path
    if resume_path is not None:
        # Resume uses the checkpoint directory structure
        checkpoints_dir = resume_path
        run_dir = resume_path.parent
        rollouts_dir = run_dir / "rollouts"
        output_path = run_dir / "model"
        if experiment_name is None:
            experiment_name = run_dir.name
        print(f"Resuming from: {resume_path}")

    # Default reward function path
    if reward_function_path is None:
        reward_function_path = repo_root / f"verl/recipe/{task_name}/reward_function.py"
        if not reward_function_path.exists():
            raise FileNotFoundError(
                f"Reward function not found at verl/recipe/{task_name}/. "
                f"Please provide --reward-function."
            )

    # Create output directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate project name if not provided: {task}-rl-{model_short_name}
    if project_name is None:
        project_name = f"{task_name}-rl-{_model_project_tag(actor_model)}"

    # Generate experiment name if not provided
    # Format: {method}-{run_id}
    if experiment_name is None:
        method_str = method_name if method_name else "default"
        run_id_str = run_id if run_id else "default"
        experiment_name = f"{method_str}-{run_id_str}"

    # Get task's system message and assistant prefix for runtime template application
    task = get_task(task_name)
    system_message = getattr(task, "system_message", None)
    assistant_prefix = getattr(task, "assistant_prefix", None)

    # Logger config
    logger_config = "['wandb','console']" if wandb else "['console']"

    print(f"=== RL Training Configuration ===")
    print(f"Task: {task_name}")
    print(f"Method: {method_name or 'default'}")
    print(f"Run ID: {run_id or 'default'}")
    print(f"Actor Model: {actor_model}")
    print(f"Train Prompts: {train_prompts_path}")
    print(f"Val Prompts: {val_prompts_path or 'None (no validation)'}")
    print(f"Reward Function: {reward_function_path}:{reward_function_name}")
    print(f"Reward Kwargs: {reward_kwargs}")
    print(f"Multi-Turn: {allow_hint}")
    if interaction_name:
        print(f"Interaction: {interaction_name}")
    print(f"Max Turns: {max_turns}")
    print(f"Run Directory: {run_dir}")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Rollouts: {rollouts_dir}")
    print(f"Output Model: {output_path}")
    print(f"Project: {project_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Batch Size: {train_batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Total Steps: {total_steps}")
    print(f"GPUs per Node: {n_gpus_per_node} (tensor_parallel_size={tensor_parallel_size})")
    print(f"Wandb: {wandb}")
    if template_content:
        print(f"Template: {method.template_variant}/rl.txt")
    print(f"=================================")

    # Resolve train/val prompts to their actual file(s) on disk -- a parquet
    # prompts path written by save_parquet() may be sharded into
    # `<stem>.shard000.parquet`, etc (see pipeline/core/io.py) rather than
    # existing as a single file, since GitHub rejects individual files over
    # 100MB. verl's RLDataset natively accepts a list of parquet paths, so we
    # pass a Hydra list override `[a,b,c]` whenever there's more than one.
    def _hydra_files_value(p: Path) -> str:
        shards = load_parquet_shards(p)
        if len(shards) == 1:
            return str(shards[0])
        return "[" + ",".join(str(s) for s in shards) + "]"

    train_files_value = _hydra_files_value(train_prompts_path)
    val_files_value = _hydra_files_value(val_prompts_path) if val_prompts_path else train_files_value

    # Build verl command
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"hydra.run.dir={checkpoints_dir}",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_files_value}",
        f"data.val_files={val_files_value}",
        f"data.train_batch_size={train_batch_size}",
        f"data.val_batch_size={val_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        f"custom_reward_function.path={reward_function_path}",
        f"custom_reward_function.name={reward_function_name}",
        f"actor_rollout_ref.model.path={actor_model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch_size}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.ppo_micro_batch_size=8",
        f"actor_rollout_ref.rollout.n={n_samples}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size=4",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size=4",
        f"algorithm.kl_ctrl.kl_coef={kl_coef}",
        f"trainer.logger={logger_config}",
        "trainer.default_hdfs_dir=null",
        f"trainer.default_local_dir={checkpoints_dir}",
        f"trainer.n_gpus_per_node={n_gpus_per_node}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.resume_mode={'auto' if (continue_run or resume_path is not None) else 'disable'}",
        f"trainer.max_actor_ckpt_to_keep={max_ckpt_to_keep}",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.total_training_steps={total_steps}",
        f"trainer.rollout_data_dir={rollouts_dir}",
    ]

    # Resume into this run's own wandb run rather than forking a new one. Read
    # by the vendored verl tracking.Tracking; harmless when wandb is off.
    wandb_run_id = _wandb_run_id(run_dir) if wandb else None
    if wandb_run_id:
        cmd.append(f"+trainer.wandb_run_id={wandb_run_id}")

    # Checkpoint selection (vendored verl extensions). save_last guarantees the
    # final step is on disk regardless of the periodic save cadence.
    cmd.append("+trainer.save_last=True")
    if save_best:
        cmd.append("+trainer.save_best=True")
        cmd.append(f"+trainer.best_ckpt_metric={best_metric}")

    # Shuffle seed for training data ordering
    if shuffle_seed is not None:
        cmd.append(f"+data.seed={shuffle_seed}")

    # Add runtime template config if method is specified
    if template_content is not None:
        # Pass the text verbatim. cmd is an argv list (no shell) and Hydra keeps
        # real newlines inside quoted values, so only double quotes need
        # escaping. Never encode newlines as "\\n": Hydra does not decode them,
        # and the model would see a literal backslash-n.
        # Use + prefix to add new config keys (they don't exist in base config)
        escaped_template = template_content.replace('"', '\\"')
        cmd.append(f'+data.runtime_template="{escaped_template}"')
        if system_message:
            escaped_system = system_message.replace('"', '\\"')
            cmd.append(f'+data.runtime_system_message="{escaped_system}"')
        if assistant_prefix:
            escaped_prefix = assistant_prefix.replace('"', '\\"')
            cmd.append(f'+data.runtime_assistant_prefix="{escaped_prefix}"')

    # vLLM rollout (verl's default backend and default sync mode; the SGLang
    # backend and the async rollout modes were never used by any recorded run).
    # Add allow_hint flag for multi-turn hint generation
    if allow_hint:
        cmd.append("allow_hint=True")
    # Add max_hints limit for rollout
    if max_hints is not None:
        cmd.append(f"+actor_rollout_ref.rollout.max_hints={max_hints}")
    # Add max_turns for loop bound (model gets extra turns after hints exhausted)
    cmd.append(f"+actor_rollout_ref.rollout.max_turns={max_turns}")
    # Inline <request></request>: the rollout keeps the think block open across
    # hint exchanges instead of closing and reopening it around each one.
    if method is not None and method.nested_request:
        cmd.append("+actor_rollout_ref.rollout.nested_request=True")

    # Add reward kwargs if specified in method config
    # Use + prefix to add new config keys
    if reward_kwargs:
        for key, value in reward_kwargs.items():
            cmd.append(f"+custom_reward_function.reward_kwargs.{key}={value}")

    # Raw hydra overrides go last so they win over everything derived above.
    if extra_overrides:
        cmd.extend(extra_overrides)

    # Set environment variables
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

    print(f"Running verl training...")
    print(f"Command: {' '.join(cmd[:5])}...")

    # Run training
    result = subprocess.run(cmd, env=env, cwd=str(repo_root))

    if result.returncode != 0:
        raise RuntimeError(f"RL training failed with return code {result.returncode}")

    print(f"RL training complete. Checkpoints saved to {checkpoints_dir}")

    # Convert checkpoints to HuggingFace format. last/ is whatever the run
    # ended on and is always written; best/ appears when --save-best recorded
    # a winning step.
    import re
    checkpoint_dirs = [
        d for d in checkpoints_dir.iterdir()
        if d.is_dir() and re.match(r"global_step_\d+", d.name)
    ]

    hf_model_path = output_path
    if checkpoint_dirs:
        def get_step(d):
            match = re.search(r"global_step_(\d+)", d.name)
            return int(match.group(1)) if match else 0

        import json
        import shutil

        best_step = None
        best_record = checkpoints_dir / "best_checkpoint.json"
        if best_record.exists():
            try:
                best_step = int(json.loads(best_record.read_text())["step"])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                print(f"Warning: could not read {best_record}: {exc}")

        last_ckpt = max(checkpoint_dirs, key=get_step)
        best_ckpt = None
        if best_step is not None:
            best_ckpt = next((d for d in checkpoint_dirs if get_step(d) == best_step), None)
            if best_ckpt is None:
                print(
                    f"Warning: best checkpoint global_step_{best_step} was recorded but is "
                    f"no longer on disk; writing last/ only"
                )
            else:
                print(f"Best checkpoint by validation: {best_ckpt.name}")

        run_root = output_path.parent
        converted: dict[str, Path] = {}

        def _clear(path):
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)

        def _convert(label, ckpt):
            actor_path = ckpt / "actor"
            if not actor_path.exists():
                print(f"Warning: actor directory not found in {ckpt}; skipping {label}/")
                return
            dest = run_root / label
            _clear(dest)
            print(f"\nConverting {label} checkpoint: {ckpt.name}")
            converted[label] = Path(convert_checkpoint(actor_path, output_path=dest))
            print(f"HuggingFace model saved to: {converted[label]}")

        if best_ckpt is not None:
            _convert("best", best_ckpt)

        if best_ckpt is not None and best_ckpt == last_ckpt and "best" in converted:
            # The final step also won; symlink instead of duplicating the
            # converted weights on disk.
            dest = run_root / "last"
            _clear(dest)
            dest.symlink_to("best", target_is_directory=True)
            converted["last"] = dest
            print(f"last/ is the same checkpoint as best/ ({last_ckpt.name}); symlinked")
        else:
            _convert("last", last_ckpt)

        # model/ is what method.rl_model_path resolves to, so point it at the
        # checkpoint the run should be judged on and leave the other beside it.
        primary = "best" if "best" in converted else ("last" if "last" in converted else None)
        if primary is not None:
            _clear(output_path)
            output_path.symlink_to(primary, target_is_directory=True)
            hf_model_path = output_path
            print(f"model/ -> {primary}/")

    # Cleanup checkpoints if requested (rollouts are preserved for analysis)
    if cleanup_checkpoints and not keep_state:
        import shutil
        print("Cleaning up checkpoints...")
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
            print(f"  Removed: {checkpoints_dir}")
        print("Cleanup complete.")
    elif cleanup_checkpoints and keep_state:
        # Keep only the last checkpoint (with optimizer state), remove the rest
        import shutil
        import re as _re
        if checkpoints_dir.exists():
            ckpt_dirs = [
                d for d in checkpoints_dir.iterdir()
                if d.is_dir() and _re.match(r"global_step_\d+", d.name)
            ]
            if len(ckpt_dirs) > 1:
                def _get_step(d):
                    m = _re.search(r"global_step_(\d+)", d.name)
                    return int(m.group(1)) if m else 0
                ckpt_dirs.sort(key=_get_step)
                for d in ckpt_dirs[:-1]:
                    shutil.rmtree(d)
                    print(f"  Removed old checkpoint: {d.name}")
            if ckpt_dirs:
                print(f"  Kept last checkpoint: {ckpt_dirs[-1].name}")

    return hf_model_path




def convert_checkpoint(
    checkpoint_path: Path,
    output_path: Path | None = None,
    backend: str = "fsdp",
) -> Path:
    """
    Convert FSDP/Megatron checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to checkpoint (e.g., .../global_step_100/actor)
        output_path: Where to save HF model (default: models/rl/<model>_step<N>)
        backend: Checkpoint backend ("fsdp" or "megatron")

    Returns:
        Path to converted HuggingFace model
    """
    import subprocess
    import re
    import json

    # Find the huggingface config directory
    hf_config_path = checkpoint_path / "huggingface"
    if not hf_config_path.exists():
        raise FileNotFoundError(
            f"HuggingFace config not found at {hf_config_path}. "
            f"Expected structure: {checkpoint_path}/huggingface/config.json"
        )

    # Default output path: models/rl/<model>_step<N>
    if output_path is None:
        # Extract step number from path (e.g., global_step_100)
        step_match = re.search(r"global_step_(\d+)", str(checkpoint_path))
        step_num = step_match.group(1) if step_match else "unknown"

        # Get model type from config
        config_file = hf_config_path / "config.json"
        with open(config_file) as f:
            config = json.load(f)
        model_type = config.get("model_type", "model")

        # Find models/rl directory by walking up from checkpoint
        # Structure: models/rl/<run_name>/global_step_X/actor
        rl_dir = checkpoint_path.parent.parent.parent
        if rl_dir.name != "rl":
            # Fallback: just use parent of checkpoint
            rl_dir = checkpoint_path.parent.parent

        output_path = rl_dir / f"{model_type}_rl_step{step_num}"

    # Get repo root for verl scripts
    repo_root = Path.cwd()
    merger_script = repo_root / "verl/scripts/legacy_model_merger.py"
    if not merger_script.exists():
        raise FileNotFoundError(f"Model merger script not found at {merger_script}")

    print(f"=== Converting Checkpoint ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Backend: {backend}")
    print(f"Output: {output_path}")
    print(f"=============================")

    # Build command
    cmd = [
        "python3", str(merger_script),
        "merge",
        "--backend", backend,
        "--local_dir", str(checkpoint_path),
        "--hf_model_path", str(hf_config_path),
        "--target_dir", str(output_path),
    ]

    # Run conversion
    result = subprocess.run(cmd, cwd=str(repo_root))

    if result.returncode != 0:
        raise RuntimeError(f"Checkpoint conversion failed with return code {result.returncode}")

    print(f"Conversion complete. HuggingFace model saved to {output_path}")
    return output_path
