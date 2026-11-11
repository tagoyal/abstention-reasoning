"""Prefix-success prediction alongside an unchanged causal-LM trajectory."""

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

SUCCESS_HEAD_FILE = "prefix_success/head.safetensors"
SUCCESS_POSITION_BUCKETS = (
    ("overall", 0, None),
    ("tokens_1_100", 0, 100),
    ("tokens_101_200", 100, 200),
    ("tokens_201_400", 200, 400),
    ("tokens_401_800", 400, 800),
    ("tokens_801_plus", 800, None),
)


@torch.no_grad()
def success_reporting_statistics(logits, response_mask, interval):
    """Keep per-rollout sufficient statistics so distributed/microbatch means are exact."""
    valid = response_mask.bool()
    positions = valid.long().cumsum(-1)
    points = valid & (positions.remainder(interval) == 0)
    probabilities = logits.float().sigmoid()
    counts, sums, squared_sums = [], [], []
    for _, lower, upper in SUCCESS_POSITION_BUCKETS:
        mask = points & (positions > lower)
        if upper is not None:
            mask = mask & (positions <= upper)
        counts.append(mask.sum(-1))
        sums.append(probabilities.masked_fill(~mask, 0).sum(-1))
        squared_sums.append(probabilities.square().masked_fill(~mask, 0).sum(-1))
    return {
        "success_report_count": torch.stack(counts, dim=-1),
        "success_report_sum": torch.stack(sums, dim=-1),
        "success_report_squared_sum": torch.stack(squared_sums, dim=-1),
        # A finished response is accepted before checking whether to decode more.
        "success_report_min_probability": probabilities.masked_fill(
            ~(points & (positions < valid.sum(-1, keepdim=True))), float("inf")
        ).amin(-1),
    }


@torch.no_grad()
def success_reporting_metrics(statistics, labels, prefix):
    """Average positions within each rollout, then rollouts within each label group."""
    counts = statistics["success_report_count"]
    labels = labels.to(device=counts.device, dtype=torch.float32)
    probabilities = statistics["success_report_sum"] / counts.clamp_min(1)
    squared_probabilities = statistics["success_report_squared_sum"] / counts.clamp_min(1)
    brier = squared_probabilities - 2 * labels[:, None] * probabilities + labels[:, None]
    metrics = {}
    for i, (name, _, _) in enumerate(SUCCESS_POSITION_BUCKETS):
        key = f"{prefix}/{name}"
        contributing = counts[:, i] > 0
        if contributing.any():
            metrics[f"{key}/brier"] = brier[contributing, i].mean().item()
        for label, group in ((1, "correct"), (0, "incorrect")):
            mask = contributing & (labels == label)
            if mask.any():
                metrics[f"{key}/prob_{group}"] = probabilities[mask, i].mean().item()
    return metrics


def success_policy_metrics(records, samples_per_prompt=1, expected_levels=None):
    """Replay full validation continuations; pair the nth sample across hint levels.

    Records are (data_source, source_index, hint_level, correct, min_probability).
    The minimum over pre-completion prediction points is sufficient to determine
    whether the first threshold crossing occurs before the response finishes.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for data_source, source_index, level, correct, minimum in records:
        if isinstance(level, bool) or not isinstance(level, int) or level < 0:
            raise ValueError("Policy replay requires non-negative integer hint levels")
        if correct not in (0, 1) or math.isnan(minimum) or minimum < 0 or (minimum > 1 and minimum != math.inf):
            raise ValueError("Policy replay requires binary correctness and valid success probabilities")
        grouped[(data_source, source_index)][level].append((bool(correct), minimum))
    if not grouped:
        raise ValueError("Policy replay requires validation trajectories")
    if expected_levels is not None and set(grouped) != set(expected_levels):
        raise ValueError("Policy validation lost problems during prompt filtering; use complete hint chains")

    zero_correct, zero_minimum, chains = [], [], []
    for source, levels in grouped.items():
        expected = set(range(max(levels) + 1)) if expected_levels is None else expected_levels[source]
        if set(levels) != expected or expected != set(range(max(expected) + 1)):
            raise ValueError(f"Policy replay requires every hint level from 0 to max for {source}")
        if any(len(samples) != samples_per_prompt for samples in levels.values()):
            raise ValueError(f"Policy replay requires {samples_per_prompt} samples at every hint level for {source}")
        for sample in range(samples_per_prompt):
            chain = [levels[level][sample] for level in range(len(levels))]
            chains.append(chain)
            zero_correct.append(chain[0][0])
            # At max hints the controller must accept, even if confidence is low.
            zero_minimum.append(chain[0][1] if len(chain) > 1 else math.inf)

    # Use float32 successors so the deployed float32 comparison also rejects
    # the observed minimum; a float64 epsilon would round back to that minimum.
    minima = torch.tensor(zero_minimum, dtype=torch.float32)
    successors = torch.nextafter(minima[minima < 1], torch.tensor(math.inf)).tolist()
    thresholds = sorted({0.0, *successors})
    ordered = sorted(zip(zero_minimum, zero_correct, strict=True))
    removed = 0
    accepted_count = len(chains)
    accepted_correct = sum(zero_correct)
    best = None
    best_key = None
    for threshold in thresholds:
        while removed < len(ordered) and ordered[removed][0] < threshold:
            accepted_correct -= ordered[removed][1]
            accepted_count -= 1
            removed += 1
        if not accepted_count:
            continue
        accuracy = accepted_correct / len(chains)
        precision = accepted_correct / accepted_count
        # Prefer retaining more correct answers, then the lower threshold on ties.
        key = ((accuracy + precision) / 2, accuracy, -threshold)
        if best_key is None or key > best_key:
            best_key = key
            best = (threshold, accuracy, precision)

    threshold, accuracy, precision = best

    def replay(tau):
        successes, hints = 0, 0
        for chain in chains:
            for level, (is_correct, probability) in enumerate(chain):
                if level == len(chain) - 1 or probability >= tau:
                    successes += is_correct
                    hints += level
                    break
        return successes / len(chains), hints / len(chains)

    final_accuracy, mean_hints = replay(threshold)
    metrics = {
        "val/success_policy/accuracy_at_0": accuracy,
        "val/success_policy/precision_at_0": precision,
        "val/success_policy/final_accuracy": final_accuracy,
        "val/success_policy/mean_hints": mean_hints,
        "val/success_policy/threshold": threshold,
    }
    # A compact threshold sweep for one secondary accuracy-vs-hints chart.
    frontier_thresholds = sorted({i / 100 for i in range(101)} | {threshold})
    frontier = [(tau, *replay(tau)) for tau in frontier_thresholds]
    return metrics, frontier


@dataclass
class SuccessCausalLMOutput(CausalLMOutputWithPast):
    # FSDP reconstructs dataclass outputs during backward-hook registration,
    # so the auxiliary tensor must be a declared field, not a dynamic dict key.
    success_logits: torch.Tensor | None = None


def validate_success_config(weight, interval, update_backbone):
    if weight is not None and (not math.isfinite(weight) or weight < 0):
        raise ValueError("success_loss_weight must be finite and non-negative")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("success_interval must be a positive integer")
    if not isinstance(update_backbone, bool):
        raise ValueError("success_update_backbone must be a boolean")


def attach_success_head(model, update_backbone=True, model_path=None):
    """Expose ``output.success_logits`` without retaining every layer's states.

    The LM-head input is the final hidden state, after the decoder's final norm.
    Hooks preserve the original LM parameter names and its forward signature.
    Attach before FSDP wrapping so the head participates in sharding, optimizer
    updates, and normal training checkpoints.
    """
    if hasattr(model, "success_head"):
        raise ValueError("Model already has a success head")
    output_layer = model.get_output_embeddings()
    if not isinstance(output_layer, nn.Linear):
        raise ValueError("Prefix success currently requires a linear LM output head")
    head = nn.Linear(
        output_layer.in_features,
        1,
        bias=True,
        device=output_layer.weight.device,
        dtype=output_layer.weight.dtype,
    )
    nn.init.normal_(head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(head.bias)
    head_path = Path(model_path) / SUCCESS_HEAD_FILE if model_path is not None else None
    if head_path is not None and head_path.exists():
        from safetensors.torch import load_file

        head.load_state_dict(load_file(str(head_path)))
    elif getattr(model.config, "prefix_success", None):
        raise FileNotFoundError(f"Trained success head missing: {head_path}")
    model.success_head = head
    model.config.prefix_success = {"hidden_size": output_layer.in_features}

    def project_prefix(module, args):
        hidden = args[0]
        if not update_backbone:
            hidden = hidden.detach()
        model._prefix_success_logits = model.success_head(hidden).squeeze(-1)

    def add_output(module, args, output):
        logits = model.__dict__.pop("_prefix_success_logits", None)
        if logits is None:
            raise RuntimeError("LM forward bypassed the output head; disable fused linear kernels")
        if not hasattr(output, "logits"):
            raise ValueError("Prefix success requires return_dict=True")
        return SuccessCausalLMOutput(**output, success_logits=logits)

    output_layer.register_forward_pre_hook(project_prefix)
    model.register_forward_hook(add_output)
    return head


def save_success_head(state_dict, output_dir):
    """Move auxiliary weights into a sidecar, leaving a standard HF LM export."""
    weight_key, bias_key = "success_head.weight", "success_head.bias"
    if weight_key not in state_dict:
        return
    from safetensors.torch import save_file

    head_state = {
        "weight": state_dict.pop(weight_key).detach().cpu().contiguous(),
        "bias": state_dict.pop(bias_key).detach().cpu().contiguous(),
    }
    head_path = Path(output_dir) / SUCCESS_HEAD_FILE
    head_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(head_state, str(head_path))


def rollout_success_labels(reward_extra_info, batch_size, device=None):
    """Use scorer correctness, never format rewards, penalties, or advantages."""
    if "correct" not in reward_extra_info:
        raise ValueError("Prefix success requires the reward function to return 'correct' for every rollout")
    labels = torch.as_tensor(reward_extra_info["correct"], device=device, dtype=torch.float32)
    if labels.shape != (batch_size,) or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Reward 'correct' must contain one binary label per rollout")
    return labels


def prefix_success_loss(logits, response_mask, labels, interval):
    """BCE after each L generated tokens, averaged within each rollout first."""
    if logits.shape != response_mask.shape or labels.shape != logits.shape[:1]:
        raise ValueError("Success logits, response mask, and rollout labels have incompatible shapes")
    valid = response_mask.bool()
    token_numbers = valid.long().cumsum(dim=-1)
    checkpoints = valid & (token_numbers.remainder(interval) == 0)
    counts = checkpoints.sum(-1)
    targets = labels.float().unsqueeze(-1).expand_as(logits)
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    per_rollout = (losses * checkpoints).sum(-1) / counts.clamp_min(1)
    loss = per_rollout.mean()

    with torch.no_grad():
        total = checkpoints.sum().clamp_min(1)
        accuracy = (((logits >= 0) == targets.bool()) & checkpoints).sum() / total
        brier = (((logits.float().sigmoid() - targets) ** 2) * checkpoints).sum() / total
        metrics = {
            "actor/success_loss": loss.item(),
            "actor/success_accuracy": accuracy.item(),
            "actor/success_brier": brier.item(),
            "actor/success_positive_rate": labels.float().mean().item(),
        }
    return loss, metrics
