"""Reward wrapper for composed methods: one rollout answers K independent subproblems.

A composed response looks like a single-problem response whose terminal
`<answer>...</answer>` is replaced by K numbered tags:

    [reasoning, possibly with request/response exchanges]</think>
    <answer_1>...</answer_1> ... <answer_K>...</answer_K>

Rather than re-implementing each task's scorer, the wrapper rewrites the
response into K single-answer responses -- the shared prefix up to the final
</think> plus `<answer>{answer_i}</answer>` -- and scores each with the base
reward function the method would use uncomposed. Every structural check, hint
count and penalty of the base scorer therefore applies unchanged, so any
method (baseline, method_b, ...) gets a composed variant by setting
`composed: true` in its config. train_rl routes the reward here and passes the
base scorer as `base_reward_path` / `base_reward_name`.

Loaded by path (custom_reward_function.path), so it imports nothing from the
pipeline package.
"""

import importlib.util
import re
from pathlib import Path

_NUMBERED_TAG = re.compile(r"</?answer_(\d+)>")
_NUMBERED_ANSWER = re.compile(r"<answer_(\d+)>(.*?)</answer_\1>", re.DOTALL)
_BASE_CACHE: dict = {}


def _load_base(path: str, name: str):
    key = (str(Path(path).resolve()), name)
    if key not in _BASE_CACHE:
        spec = importlib.util.spec_from_file_location(f"composed_base_{len(_BASE_CACHE)}", key[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASE_CACHE[key] = getattr(module, name)
    return _BASE_CACHE[key]


def split_composed_response(solution_str: str, num_subparts: int) -> tuple[str, list[str]] | None:
    """Return (prefix, [answer_1..answer_K]) or None if the answer block is malformed.

    The answer block is everything after the last </think>. It must hold exactly
    the tags <answer_1>..<answer_K>, each once, in order, and only whitespace
    between them. Numbered tags anywhere before it are malformed too.
    """
    cut = solution_str.rfind("</think>")
    if cut < 0:
        return None
    cut += len("</think>")
    prefix, block = solution_str[:cut], solution_str[cut:]
    if _NUMBERED_TAG.search(prefix):
        return None

    answers, pos = [], 0
    for match in _NUMBERED_ANSWER.finditer(block):
        if block[pos:match.start()].strip():
            return None
        answers.append((int(match.group(1)), match.group(2).strip()))
        pos = match.end()
    if block[pos:].strip():
        return None
    if [i for i, _ in answers] != list(range(1, num_subparts + 1)):
        return None
    return prefix, [a for _, a in answers]


def _as_list(value):
    # Parquet round-trips lists as numpy arrays.
    return list(value) if not isinstance(value, list) else value


def compute_score(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    base_reward_path: str = None,
    base_reward_name: str = "compute_score",
    compose_aggregate: str = "min",
    **kwargs,
) -> dict:
    """Score a composed rollout by scoring each subpart with the base scorer.

    compose_aggregate: "min" (default) pays full score only when every subpart
    is correct, matching the all-K correctness used for SFT filtering and eval;
    "mean" gives partial credit per subpart.
    All remaining kwargs are forwarded to the base scorer.
    """
    if base_reward_path is None:
        raise ValueError("composed compute_score needs base_reward_path")
    base = _load_base(base_reward_path, base_reward_name)
    subproblems = _as_list(ground_truth["subproblems"])
    k = int(ground_truth.get("num_subparts", len(subproblems)))

    parts = split_composed_response(solution_str, k)
    if parts is None:
        # Let the base scorer build its own malformed result so the returned
        # keys match well-formed rollouts (verl requires a fixed key set).
        cut = solution_str.rfind("</think>")
        prefix = solution_str[:cut + len("</think>")] if cut >= 0 else solution_str
        result = _as_dict(base(data_source, prefix, dict(subproblems[0]), extra_info, **kwargs))
        result.update(score=0, correct=False, malformed=True,
                      n_subparts_correct=0, subpart_accuracy=0.0)
        if "score_wo_hint_penalty" in result:
            result["score_wo_hint_penalty"] = 0
        return result

    prefix, answers = parts
    results = [
        _as_dict(base(data_source, f"{prefix}\n\n<answer>{answer}</answer>", dict(sub_gt), extra_info, **kwargs))
        for answer, sub_gt in zip(answers, subproblems)
    ]
    return _aggregate(results, compose_aggregate)


def _as_dict(result) -> dict:
    return dict(result) if isinstance(result, dict) else {"score": float(result)}


def _aggregate(results: list[dict], how: str) -> dict:
    if how not in ("min", "mean"):
        raise ValueError(f"unknown compose_aggregate {how!r}; expected min or mean")
    reduce = min if how == "min" else (lambda xs: sum(xs) / len(xs))
    # Prefix-derived fields (hint counts etc.) are identical across subparts,
    # so the first result supplies everything not aggregated below.
    out = dict(results[0])
    for key in ("score", "score_wo_hint_penalty"):
        if all(key in r for r in results):
            out[key] = reduce([r[key] for r in results])
    n_correct = sum(bool(r.get("correct", False)) for r in results)
    out["correct"] = n_correct == len(results)
    out["malformed"] = any(r.get("malformed", False) for r in results)
    if "abstained" in out:
        out["abstained"] = any(r.get("abstained", False) for r in results)
    out["n_subparts_correct"] = n_correct
    out["subpart_accuracy"] = n_correct / len(results)
    return out
