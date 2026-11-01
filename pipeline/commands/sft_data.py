"""Prepare continuation-label datasets for ordinary SFT training."""

from pathlib import Path

from pipeline.core.io import load_json, save_json


def prepare_multisample_sft(
    dataset_path: Path,
    output_path: Path,
    base_model: str,
) -> Path:
    """Render chat contexts ending in <continue>, with binary label targets.

    Pass the logical JSON path to read all its shards. Use the training model's
    tokenizer so the saved contexts use its chat template.
    """
    from transformers import AutoTokenizer

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    if dataset_path.resolve() == output_path.resolve():
        raise ValueError("Output must differ from the source dataset")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    records = []
    for row_number, row in enumerate(load_json(dataset_path)):
        label = row.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Record {row_number}: expected an integer label of 0 or 1")
        messages = row["prompt"]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(f"Record {row_number}: expected a trailing assistant prefix")
        context = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
        context += messages[-1]["content"] + "<continue>"
        records.append({
            **{key: value for key, value in row.items()
               if key not in ("prompt", "samples", "assistant_prefix", "generation", "correct")},
            "prompt": context,
            "generation": str(label),
            # This is a valid classification target even when the solver failed.
            "correct": True,
        })
    save_json(output_path, records)
    print(f"Saved {len(records)} continuation-label SFT examples to {output_path}")
    return output_path
