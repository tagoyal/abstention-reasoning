"""Online trajectory-prefix supervision for method_d."""

import torch
import torch.nn.functional as F

from verl import DataProto


def is_sft_step(step: int, interleave_rl: bool) -> bool:
    return not interleave_rl or step % 2 == 1


def classification_tokens(tokenizer):
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in ("0", "1")]
    if any(len(ids) != 1 for ids in label_ids) or label_ids[0] == label_ids[1]:
        raise ValueError("method_d requires distinct single-token encodings for 0 and 1")
    complete_ids = tokenizer.encode("<complete>", add_special_tokens=False)
    if not complete_ids:
        raise ValueError("method_d requires a nonempty encoding for <complete>")
    return complete_ids, [ids[0] for ids in label_ids]


def build_prefix_batch(rollouts, correctness, tokenizer, stride=128, max_model_len=None):
    """Use exact rollout tokens, not decoded/re-tokenized prefixes.

    Terminal EOS is a generation control token, not part of the textual output.
    Every trajectory contributes its full output once, including when its length
    is an exact multiple of the stride.
    """
    if stride <= 0:
        raise ValueError("method_d stride must be positive")
    if len(correctness) != len(rollouts):
        raise ValueError("Each method_d trajectory needs one correctness label")
    if any(label not in (0, 1, False, True) for label in correctness):
        raise ValueError("method_d correctness labels must be binary")
    complete_ids, label_ids = classification_tokens(tokenizer)
    prompt_length = rollouts.batch["prompts"].shape[1]
    contexts, targets = [], []
    for i, label in enumerate(correctness):
        mask = rollouts.batch["attention_mask"][i]
        prompt = rollouts.batch["prompts"][i][mask[:prompt_length].bool()].tolist()
        response = rollouts.batch["responses"][i][mask[prompt_length:].bool()].tolist()
        if response and response[-1] == tokenizer.eos_token_id:
            response = response[:-1]
        for end in [*range(stride, len(response), stride), len(response)]:
            context = prompt + response[:end] + complete_ids
            if max_model_len is not None and len(context) + 1 > max_model_len:
                raise ValueError(
                    "method_d prefix plus <complete> and label exceeds max_model_len; "
                    "increase --max-model-len or reduce prompt/response lengths"
                )
            contexts.append(context)
            targets.append(label_ids[int(label)])
    if not contexts:
        raise ValueError("method_d received no trajectories")
    width = max(map(len, contexts))
    input_ids = torch.full((len(contexts), width), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, context in enumerate(contexts):
        input_ids[i, -len(context) :] = torch.tensor(context)
        attention_mask[i, -len(context) :] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "sft_targets": torch.tensor(targets),
            "sft_weights": torch.ones(len(contexts)),
        },
        meta_info={"sft_label_ids": label_ids},
    )


def classification_statistics(logits, targets, weights, label_ids):
    """Full-vocabulary CE/token accuracy and binary-normalized accuracy."""
    logits = logits.float()
    losses = F.cross_entropy(logits, targets, reduction="none")
    binary_probs = logits[:, label_ids].softmax(-1)
    binary_predictions = torch.as_tensor(label_ids, device=logits.device)[binary_probs.argmax(-1)]
    return {
        "loss": (losses * weights).sum(),
        "token_accuracy": ((logits.argmax(-1) == targets) * weights).sum(),
        "accuracy": ((binary_predictions == targets) * weights).sum(),
        "num_examples": weights.sum(),
    }
