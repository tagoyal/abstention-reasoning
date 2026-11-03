"""Validation accuracy breakdowns shared by online classification methods."""

import torch


def input_hint_levels(rollouts):
    levels = rollouts.non_tensor_batch.get("hint_level")
    if levels is None:
        # Baseline inputs have no injected hints.
        return [0] * len(rollouts)
    return [int(level) if level is not None else -1 for level in levels]


def report_groups(levels):
    return [f"hint_{level}" if level >= 0 else "hint_unknown"
            for level in sorted(set(levels))] + ["position_first", "position_final"]


def accumulate_groups(totals, groups, batch, weights, correct, token_correct=None):
    for i, group in enumerate(groups):
        if group.startswith("hint_"):
            level = -1 if group == "hint_unknown" else int(group.removeprefix("hint_"))
            selected = batch["sft_hint_level"] == level
        else:
            selected = batch[f"sft_{group}"]
        selected_weights = weights * selected
        totals[i, 0] += selected_weights.sum()
        totals[i, 1] += (correct * selected_weights).sum()
        if token_correct is not None:
            totals[i, 2] += (token_correct * selected_weights).sum()


def finalize_groups(totals, groups, token_accuracy=False):
    torch.distributed.all_reduce(totals)
    metrics = {}
    for group, values in zip(groups, totals):
        count = values[0].item()
        if count:
            prefix = f"sft_groups/{group}"
            metrics[f"{prefix}/num_examples"] = count
            metrics[f"{prefix}/accuracy"] = (values[1] / values[0]).item()
            if token_accuracy:
                metrics[f"{prefix}/token_accuracy"] = (values[2] / values[0]).item()
    return metrics
