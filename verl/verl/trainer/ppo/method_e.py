"""Single-pass trajectory classification at multiple causal positions."""

from types import MethodType
from pathlib import Path

import torch
import torch.nn.functional as F

from verl import DataProto


def classification_forward(self, *args, method_e=False, head_only=False, **kwargs):
    if not method_e:
        return self._method_e_lm_forward(*args, **kwargs)
    with torch.set_grad_enabled(torch.is_grad_enabled() and not head_only):
        hidden = self.base_model(*args, **kwargs, return_dict=True).last_hidden_state
    return self.classification_head(hidden)


def attach_classification_head(model, checkpoint_path=None):
    """Keep normal LM parameter names and forward behavior for RL/checkpoints."""
    model.classification_head = torch.nn.Linear(
        model.config.hidden_size, 2,
        device=model.get_input_embeddings().weight.device,
        dtype=model.get_input_embeddings().weight.dtype,
    )
    model.config.method_e = True
    object.__setattr__(model, "_method_e_lm_forward", model.forward)
    model.forward = MethodType(classification_forward, model)
    if checkpoint_path is not None:
        from safetensors import safe_open
        state = {}
        for path in sorted(Path(checkpoint_path).glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as source:
                for key in ("classification_head.weight", "classification_head.bias"):
                    if key in source.keys():
                        state[key.split(".", 1)[1]] = source.get_tensor(key)
        if state:
            model.classification_head.load_state_dict(state)


def build_trajectory_batch(rollouts, correctness, tokenizer, stride=128, max_model_len=None, report=False):
    if stride <= 0:
        raise ValueError("method_e stride must be positive")
    if len(correctness) != len(rollouts):
        raise ValueError("Each method_e trajectory needs one correctness label")
    if any(label not in (0, 1, False, True) for label in correctness):
        raise ValueError("method_e correctness labels must be binary")
    prompt_width = rollouts.batch["prompts"].shape[1]
    contexts, positions = [], []
    for i in range(len(rollouts)):
        mask = rollouts.batch["attention_mask"][i].bool()
        prompt = rollouts.batch["prompts"][i][mask[:prompt_width]].tolist()
        response = rollouts.batch["responses"][i][mask[prompt_width:]].tolist()
        if response and response[-1] == tokenizer.eos_token_id:
            response.pop()
        context = prompt + response
        if not context:
            raise ValueError("method_e requires a nonempty context")
        if max_model_len is not None and len(context) > max_model_len:
            raise ValueError("method_e trajectory exceeds max_model_len")
        contexts.append(context)
        ends = [*range(stride, len(response), stride), len(response)]
        positions.append([len(prompt) + end - 1 for end in ends])
    if not contexts:
        raise ValueError("method_e received no trajectories")
    width = max(map(len, contexts))
    ids = torch.full((len(contexts), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    weights = torch.zeros_like(ids, dtype=torch.float)
    targets = torch.zeros_like(ids)
    for i, (context, points) in enumerate(zip(contexts, positions)):
        offset = width - len(context)
        ids[i, offset:] = torch.tensor(context)
        mask[i, offset:] = 1
        for point in points:
            weights[i, offset + point] = 1
            targets[i, offset + point] = int(correctness[i])
    result = DataProto.from_dict(tensors={
        "input_ids": ids, "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
        "sft_targets": targets, "sft_weights": weights,
    })
    if report:
        from verl.trainer.ppo.sft_reporting import input_hint_levels, report_groups
        levels = input_hint_levels(rollouts)
        first = torch.zeros_like(ids, dtype=torch.bool)
        final = torch.zeros_like(first)
        for i, (context, points) in enumerate(zip(contexts, positions)):
            offset = width - len(context)
            first[i, offset + points[0]] = True
            final[i, offset + points[-1]] = True
        result.batch["sft_hint_level"] = torch.tensor(levels).unsqueeze(-1).expand_as(ids)
        result.batch["sft_position_first"] = first
        result.batch["sft_position_final"] = final
        result.meta_info["sft_report_groups"] = report_groups(levels)
    return result


def classification_statistics(logits, targets, weights):
    selected = weights > 0
    logits, targets, weights = logits[selected].float(), targets[selected], weights[selected]
    return {
        "loss": (F.cross_entropy(logits, targets, reduction="none") * weights).sum(),
        "accuracy": ((logits.argmax(-1) == targets) * weights).sum(),
        "num_examples": weights.sum(),
    }
