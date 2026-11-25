"""Composed prompts: K independent problems from one partition in one datapoint.

A composed method (`composed: true` in its config) groups each partition into
disjoint K-problem groups and asks for every subpart in one response, with
per-subpart answers in `<answer_i> ... </answer_i>` tags. Grading delegates
each subpart to the base task's own checker, so composed scoring can never
drift from single-problem scoring.
"""

import importlib.util
import random
import re
from collections import defaultdict
from pathlib import Path

from pipeline.core.method import TASKS_ROOT

COMPOSED_TASKS = ("math", "countdown")

_ANSWER_I = re.compile(r"<answer_(\d+)>(.*?)</answer_\1>", re.DOTALL | re.IGNORECASE)


_RECIPE_SHARED = Path(__file__).resolve().parents[2] / "verl" / "recipe" / "shared"
_REQUEST_ANY = re.compile(r"<request>(.*?)</request>", re.DOTALL)
EXHAUSTED_MARKER = "No more hints available."


def _load_recipe(name: str):
    # Recipes are standalone files (custom_reward_function.path), not a package.
    spec = importlib.util.spec_from_file_location(f"composed_{name}", _RECIPE_SHARED / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reward = _load_recipe("composed_reward")
_grammar = _load_recipe("nested_grammar")


def composed_hint_malformed(generation: str, k: int) -> str | None:
    """Why a subpart-request generation is malformed, or None if it is clean.

    Strict by design, for SFT filtering: any request other than a bare subpart
    id 1..K, a subpart requested twice, an exhausted-pool response, a bad
    <answer_1>..<answer_K> block, or a violation of the inline-request grammar
    the RL reward enforces (checked on the same rewrite the reward scores).
    """
    opens = generation.count("<request>")
    ids = _REQUEST_ANY.findall(generation)
    if opens != len(ids) or generation.count("</request>") != len(ids):
        return "request_format"
    if any(not re.fullmatch(r"\d+", i) or not 1 <= int(i) <= k for i in ids):
        return "request_id"
    if len(set(ids)) != len(ids):
        return "repeated_request"
    if EXHAUSTED_MARKER in generation:
        return "exhausted"
    parts = _reward.split_composed_response(_reward.normalize_requests(generation, k), k)
    if parts is None:
        return "answer_block"
    prefix, _ = parts
    if _grammar.has_malformed_structure_nested(f"{prefix}\n\n<answer>x</answer>"):
        return "grammar"
    return None


def check_composed_task(task_name: str) -> None:
    if task_name not in COMPOSED_TASKS:
        raise ValueError(
            f"Composed prompts are only implemented for {list(COMPOSED_TASKS)}, not '{task_name}'."
        )


def group_disjoint(problems: list[dict], k: int, seed: int) -> list[list[dict]]:
    """Shuffle, then chunk into floor(N/K) groups; each problem is used at most once."""
    if k < 1:
        raise ValueError(f"compose_k must be >= 1, got {k}")
    order = list(problems)
    random.Random(seed).shuffle(order)
    n_groups = len(order) // k
    return [order[i * k:(i + 1) * k] for i in range(n_groups)]


def load_subproblem_template(task_name: str, template_variant: str) -> str:
    path = TASKS_ROOT / task_name / "templates" / template_variant / "subproblem.txt"
    return path.read_text(encoding="utf-8").strip()


def render_fields(template: str, fields: dict) -> str:
    """Same substitution as verl's runtime template (and io._render_runtime_prompt),
    so JSON and parquet prompts render identically."""
    content = template
    for key, value in fields.items():
        content = content.replace(f"{{{key}}}", str(value))
        content = content.replace(f"{{{{{key}}}}}", str(value))
    return content


def build_composed_fields(group: list[dict], subproblem_template: str) -> dict:
    """Template fields for one composed datapoint.

    `subproblems` is deliberately last: substitution runs in key order, so
    problem text (math LaTeX can contain `{...}`) is inserted after every other
    placeholder has been filled and is never itself rescanned.
    """
    k = len(group)
    blocks = [
        f"Subpart {i}:\n{render_fields(subproblem_template, p)}"
        for i, p in enumerate(group, start=1)
    ]
    answer_tags = "\n".join(f"<answer_{i}> ... </answer_{i}>" for i in range(1, k + 1))
    return {
        "num_subparts": k,
        "answer_tags": answer_tags,
        "subproblems": "\n\n".join(blocks),
    }


def extract_composed_answers(generation: str, k: int) -> list[str | None]:
    """Answer for each subpart 1..k (last occurrence wins), None if missing."""
    answers: list[str | None] = [None] * k
    for match in _ANSWER_I.finditer(generation):
        i = int(match.group(1))
        if 1 <= i <= k:
            answers[i - 1] = match.group(2).strip()
    return answers


def _num_subparts(record: dict) -> int:
    gt = record.get("ground_truth") or {}
    return int(gt.get("num_subparts") or len(gt.get("subproblems") or []))


class ComposedTask:
    """Wraps a base task so generate/evaluate grade composed datapoints.

    Everything not overridden here (system_message, assistant_prefix,
    filter_for_sft, ...) is delegated to the base task.
    """

    def __init__(self, base, subpart_requests: bool = False):
        self.base = base
        self.name = base.name
        self.subpart_requests = subpart_requests

    def __getattr__(self, attr):
        return getattr(self.base, attr)

    def extract_answer(self, generation: str):
        return None

    def check_correctness(self, primitive: dict, generation: str) -> tuple[bool, dict]:
        subproblems = primitive["subproblems"]
        k = len(subproblems)
        answers = extract_composed_answers(generation, k)
        subpart_correct, subpart_meta = [], []
        for answer, sub_gt in zip(answers, subproblems):
            if answer is None:
                ok, meta = False, {"predicted_answer": None, "error": "no_answer_tag"}
            else:
                # Reuse the base checker, which reads a single <answer> tag.
                ok, meta = self.base.check_correctness(sub_gt, f"<answer>{answer}</answer>")
            subpart_correct.append(bool(ok))
            subpart_meta.append(meta)
        n_correct = sum(subpart_correct)
        meta = {
            "predicted_answer": answers,
            "subpart_correct": subpart_correct,
            "n_subparts_correct": n_correct,
            "num_subparts": k,
            "subpart_metadata": subpart_meta,
        }
        if all(a is None for a in answers):
            meta["error"] = "no_answer_tag"
        if self.subpart_requests:
            meta["malformed_reason"] = composed_hint_malformed(generation, k)
        return n_correct == k, meta

    def filter_for_sft(self, examples: list[dict], include_wrong_valid_format: bool = False,
                       nested_request: bool = False, **kwargs) -> list[dict]:
        """Subpart-request methods keep only correct, strictly well-formed
        generations; other composed methods use the base task's filter."""
        if not self.subpart_requests:
            return self.base.filter_for_sft(examples, include_wrong_valid_format=include_wrong_valid_format,
                                            nested_request=nested_request, **kwargs)
        return [
            ex for ex in examples
            if ex.get("correct", False)
            and composed_hint_malformed(ex.get("generation", ""), _num_subparts(ex)) is None
        ]

    def compute_metrics(self, results: list[dict]) -> dict:
        """`accuracy` = all K subparts correct; also per-subpart accuracy, the
        mean fraction of subparts correct, and accuracy by subpart position."""
        total = len(results)
        correct = sum(1 for r in results if r.get("correct"))
        n_sub = sum(r.get("num_subparts", 0) for r in results)
        n_sub_correct = sum(r.get("n_subparts_correct", 0) for r in results)
        by_position = defaultdict(lambda: {"correct": 0, "total": 0})
        dist = defaultdict(int)
        for r in results:
            flags = r.get("subpart_correct") or []
            for i, ok in enumerate(flags, start=1):
                by_position[i]["total"] += 1
                by_position[i]["correct"] += int(ok)
            dist[r.get("n_subparts_correct", 0)] += 1
        return {
            "accuracy": correct / total if total else 0,
            "total": total,
            "correct": correct,
            "subpart_accuracy": n_sub_correct / n_sub if n_sub else 0,
            "subparts_total": n_sub,
            "subparts_correct": n_sub_correct,
            "accuracy_by_position": {
                i: d["correct"] / d["total"] for i, d in sorted(by_position.items())
            },
            "n_subparts_correct_distribution": dict(sorted(dist.items())),
            **self._hint_metrics(results),
        }

    @staticmethod
    def _hint_metrics(results: list[dict]) -> dict:
        if not any("num_hints" in r or "malformed_reason" in r for r in results):
            return {}
        total = len(results)
        hinted = [r for r in results if r.get("num_hints", 0) > 0]
        unhinted = [r for r in results if r.get("num_hints", 0) == 0]
        reasons = defaultdict(int)
        for r in results:
            if r.get("malformed_reason"):
                reasons[r["malformed_reason"]] += 1

        def acc(rs):
            return sum(1 for r in rs if r.get("correct")) / len(rs) if rs else 0

        return {
            "avg_hints": sum(r.get("num_hints", 0) for r in results) / total if total else 0,
            "hint_rate": len(hinted) / total if total else 0,
            "accuracy_with_hints": acc(hinted),
            "accuracy_without_hints": acc(unhinted),
            "malformed_rate": sum(reasons.values()) / total if total else 0,
            "malformed_reasons": dict(sorted(reasons.items())),
        }

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        lines = ["", "=== Composed Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append(
            f"All-subparts accuracy: {metrics['accuracy']:.2%} "
            f"({metrics['correct']}/{metrics['total']})"
        )
        lines.append(
            f"Subpart accuracy:      {metrics['subpart_accuracy']:.2%} "
            f"({metrics['subparts_correct']}/{metrics['subparts_total']})"
        )
        lines.append("By subpart position:")
        for i, acc in metrics["accuracy_by_position"].items():
            lines.append(f"  {i}: {acc:.2%}")
        lines.append("Subparts correct per datapoint:")
        for n, count in metrics["n_subparts_correct_distribution"].items():
            lines.append(f"  {n}: {count}")
        if "avg_hints" in metrics:
            lines.append(f"Avg hints: {metrics['avg_hints']:.2f}  "
                         f"hint rate: {metrics['hint_rate']:.2%}")
            lines.append(f"Accuracy with hints: {metrics['accuracy_with_hints']:.2%}  "
                         f"without: {metrics['accuracy_without_hints']:.2%}")
            lines.append(f"Malformed: {metrics['malformed_rate']:.2%} {metrics['malformed_reasons']}")
        return "\n".join(lines)
