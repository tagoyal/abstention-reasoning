"""Token-strided solver prefixes with individual continuation labels."""

import asyncio
import random
from pathlib import Path

from pipeline.core.generator import AsyncGenerator, GenerationConfig, Generator
from pipeline.core.io import is_parquet_prompts_path, load_prompts, save_json
from pipeline.core.method import Method, resolve_data_name, resolve_models_name
from pipeline.tasks import get_task


def generate_tree_multisample(
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
    data_name: str | None = None,
    models_name: str | None = None,
    *,
    stride: int,
    num_points: int,
) -> Path:
    """Sample one trajectory, then up to N token-strided prefixes per problem.

    Positions are sampled without replacement from 0, K, 2K, ... <= the
    trajectory's token length. All available positions are used when fewer
    than N exist; neither root nor leaf is forced. Token counts exclude the
    original assistant prefix. Trajectories and continuations each have a
    max_new_tokens budget. Generation options and model/prompt path resolution
    match evaluate: multi_turn defaults to the method config, data_name defaults
    to task_name, and models_name defaults to data_name.

    Each prefix gets M fresh continuations and M records with the same input
    and independent binary correctness targets, never a thresholded aggregate.
    Correctness is checked on the full assistant text, including the prefix.
    The output is compatible with train_classifier via an explicit --dataset.
    """
    if not split:
        raise ValueError("--split is required for generate_tree_multisample")
    for name, value in (
        ("stride", stride),
        ("num_points", num_points),
        ("num_samples", num_samples),
        ("batch_size", batch_size),
        ("max_new_tokens", max_new_tokens),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if data_parallel_size > 1 and not use_async:
        raise ValueError("data_parallel_size > 1 requires --async")

    from transformers import AutoTokenizer

    data_name = resolve_data_name(task_name, data_name)
    models_name = resolve_models_name(data_name, models_name)
    method = Method.load(method_name, task_name) if method_name is not None else None
    task = get_task(task_name)
    if multi_turn is None:
        multi_turn = method.multi_turn if method else False
    actual_model = model_name
    if method is not None and model_name == "sft":
        actual_model = str(method.sft_model_path(models_name, run_id))
    elif method is not None and model_name == "rl":
        actual_model = str(method.rl_model_path(models_name, run_id))

    if prompts_path is None:
        if method is None:
            raise ValueError("Either --method or --prompts must be specified")
        prompts_path = method.formatted_path(data_name, split)
    template = (
        method.load_template(task_name, "rl")
        if method is not None and is_parquet_prompts_path(prompts_path)
        else None
    )
    prompts_data = load_prompts(
        prompts_path, template=template, system_message=getattr(task, "system_message", None)
    )
    if output_path is None:
        if method is None:
            raise ValueError("Either --method or --output must be specified")
        desc = f"tree_multisample__{run_id}" if run_id else "tree_multisample"
        output_path = method.dataset_path(data_name, split, desc=desc)
    if not prompts_data:
        save_json(output_path, [])
        return output_path

    tokenizer = AutoTokenizer.from_pretrained(actual_model)
    config = GenerationConfig(
        model_name=actual_model,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_samples=1,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        verbose=verbose,
        seed=seed,
        nested_request=method.nested_request if method else False,
        ban_hint_requests=no_hints,
    )
    rng = random.Random(seed)
    records = []

    async def run():
        generator = AsyncGenerator(config) if use_async else Generator(config)
        requests_sampled = 0

        async def sample(prompts, ground_truths, count):
            nonlocal requests_sampled
            # Do not replay a trajectory's RNG stream when probing its root.
            config.seed = seed + requests_sampled if seed is not None else None
            requests_sampled += len(prompts) * count if multi_turn else len(prompts)
            if multi_turn:
                expanded_prompts = [prompt for prompt in prompts for _ in range(count)]
                expanded_truths = [truth for truth in ground_truths for _ in range(count)]
                if use_async:
                    expanded = await generator.generate_with_hints_async(
                        expanded_prompts, expanded_truths
                    )
                else:
                    expanded = generator.generate_with_hints_batched(
                        expanded_prompts, expanded_truths
                    )
                if len(expanded) != len(expanded_prompts) or any(len(samples) != 1 for samples in expanded):
                    raise ValueError(f"Expected one sample for each of {len(expanded_prompts)} expanded prompts")
                results = [
                    [samples[0] for samples in expanded[i * count:(i + 1) * count]]
                    for i in range(len(prompts))
                ]
            elif use_async:
                results = await generator.generate_async(prompts, num_samples=count)
            else:
                config.num_samples = count
                results = generator.generate_batched(prompts)
            if len(results) != len(prompts) or any(len(samples) != count for samples in results):
                raise ValueError(f"Expected {count} samples for each of {len(prompts)} prompts")
            return results

        try:
            trajectories = await sample(
                [p["prompt"] for p in prompts_data],
                [p["ground_truth"] for p in prompts_data],
                1,
            )
            states = []
            for prompt_data, trajectory in zip(prompts_data, trajectories):
                text = trajectory[0]["text"]
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                candidates = range(0, len(token_ids) + 1, stride)
                positions = sorted(rng.sample(candidates, min(num_points, len(candidates))))
                conversation = prompt_data["prompt"]
                has_prefix = bool(conversation and conversation[-1]["role"] == "assistant")
                states.append({
                    "data": prompt_data,
                    "conversation": conversation[:-1] if has_prefix else conversation,
                    "assistant_prefix": conversation[-1]["content"] if has_prefix else "",
                    "token_ids": token_ids,
                    "positions": positions,
                })

            for point_index in range(max(len(state["positions"]) for state in states)):
                active = [state for state in states if point_index < len(state["positions"])]
                prompts = []
                for state in active:
                    position = state["positions"][point_index]
                    partial = tokenizer.decode(
                        state["token_ids"][:position],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    prompts.append(state["conversation"] + [{
                        "role": "assistant", "content": state["assistant_prefix"] + partial,
                    }])

                continuations = await sample(
                    prompts, [state["data"]["ground_truth"] for state in active], num_samples
                )
                for state, prompt, samples in zip(active, prompts, continuations):
                    source = state["data"]
                    position = state["positions"][point_index]
                    primitive = {
                        **source["ground_truth"],
                        "index": source["index"],
                        "variant": source.get("variant", "unknown"),
                    }
                    for sample_index, completion in enumerate(samples):
                        full_text = prompt[-1]["content"] + completion["text"]
                        correct, meta = task.check_correctness(primitive, full_text)
                        label = int(correct)
                        records.append({
                            **{k: v for k, v in source.items()
                               if k not in ("prompt", "generation", "correct")},
                            "prompt": prompt,
                            "depth": position // stride,
                            "distance": position / len(state["token_ids"]) if state["token_ids"] else 0.0,
                            "prefix_tokens": position,
                            "trajectory_tokens": len(state["token_ids"]),
                            "point_index": point_index,
                            "sample_index": sample_index,
                            "num_samples": 1,
                            "num_correct_samples": label,
                            "pass_rate": float(label),
                            "label": label,
                            "generation": str(label),
                            "samples": [{
                                "generation": full_text,
                                "continuation": completion["text"],
                                "correct": bool(correct),
                                "predicted_answer": meta.get("predicted_answer") if isinstance(meta, dict) else None,
                                "finish_reason": completion.get("finish_reason", "unknown"),
                                "token_count": completion.get("token_count", 0),
                            }],
                        })
                save_json(output_path, records)
                print(f"Saved prefix round {point_index + 1}: {len(records)} records to {output_path}")
        finally:
            if use_async:
                generator.close()

    asyncio.run(run())
    return output_path
