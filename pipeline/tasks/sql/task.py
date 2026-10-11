"""Text-to-SQL task implementation.

Standardized Text-to-SQL reasoning task across Spider, BIRD, SParC, and CoSQL.
Database schema DDL and natural language questions are provided as input,
and target answers are executable SQL queries.

In prompts, models are instructed to place their final SQL query in \\boxed{...}
inside <answer>...</answer>.

Evaluation checks execution equivalence against the underlying SQLite databases
by delegating directly to verl/recipe/sql/reward_function.py.
"""

import importlib.util
from collections import defaultdict
from pathlib import Path

from pipeline.tasks.base import BaseTask

# Single source of truth: import the verifier, extractor, and grammar directly
# from the reward function so execution logic is never duplicated.
_REWARD_PATH = (
    Path(__file__).resolve().parents[3]
    / "verl"
    / "recipe"
    / "sql"
    / "reward_function.py"
)
_spec = importlib.util.spec_from_file_location("sql_reward", _REWARD_PATH)
_reward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reward)

_verify_sql = _reward._verify_sql
extract_answer = _reward.extract_answer
_has_malformed_structure_nested = _reward.has_malformed_structure_nested


class SqlTask(BaseTask):
    """
    Text-to-SQL reasoning task.

    Inputs contain DDL database schemas and natural language questions.
    Target answers are executable SQLite queries.
    Difficulty levels: easy, medium, hard, extra.
    """

    name = "sql"

    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )

    assistant_prefix = "<think>\nLet me work through this problem step by step."

    def __init__(self):
        super().__init__()

    def extract_answer(self, generation: str) -> str | None:
        """Extract SQL answer from model generation."""
        return extract_answer(generation)

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format Text-to-SQL problem into chat messages."""
        content = template.replace("{problem}", primitive["problem"])
        content = content.replace("{level}", str(primitive.get("level", "")))
        content = content.replace("{type}", str(primitive.get("variant", "")))
        content = content.replace("{db_id}", str(primitive.get("variant", "")))

        # Format hints if present and template uses {hints}
        if "{hints}" in content:
            hints = primitive.get("hints", [])
            if hints:
                hints_str = "\n".join(f"- {h}" for h in hints)
            else:
                hints_str = "(no hints available)"
            content = content.replace("{hints}", hints_str)

        # Format hints_block (empty when no hints, block of text when hints present)
        if "{hints_block}" in content:
            hints = primitive.get("hints", [])
            if hints:
                hints_str = "Here are some hints to help you:\n" + "\n".join(f"- {h}" for h in hints) + "\n\n"
            else:
                hints_str = ""
            content = content.replace("{hints_block}", hints_str)

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": content},
        ]

        if include_assistant_prefix:
            messages.append({
                "role": "assistant",
                "content": self.assistant_prefix,
            })

        return messages

    def check_correctness(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[bool, dict]:
        """
        Check if the generated SQL correctly solves the Text-to-SQL problem.

        Answers are extracted from <answer>...</answer> and/or \\boxed{...}.
        Evaluation executes both predicted and gold SQL against the SQLite database.
        """
        predicted = self.extract_answer(generation)

        if predicted is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
            }

        # Verifier tasks (method_c) where expected label is 0 or 1
        if "correct" in primitive and "answer" not in primitive:
            expected_label = primitive.get("correct")
            is_correct = predicted in ("0", "1") and int(predicted) == int(expected_label)
            return is_correct, {
                "predicted_label": predicted,
                "expected_label": expected_label,
            }

        correct_answer = primitive.get("answer", "")
        if correct_answer is None:
            return False, {
                "predicted_answer": predicted,
                "correct_answer": None,
                "error": "no_ground_truth",
            }

        db_id = (
            primitive.get("db_id")
            or primitive.get("variant")
            or primitive.get("metadata", {}).get("db_id")
        )
        is_correct, meta = _verify_sql(db_id, predicted, correct_answer)

        meta["predicted_answer"] = predicted
        meta["correct_answer"] = correct_answer
        return is_correct, meta

        meta["predicted_answer"] = predicted
        meta["correct_answer"] = correct_answer
        return is_correct, meta

    def get_ground_truth(self, primitive: dict) -> dict:
        """Extract ground truth for embedding in prompts and RL interactions."""
        db_id = (
            primitive.get("variant")
            or primitive.get("db_id")
            or primitive.get("metadata", {}).get("db_id")
        )
        gt = {
            "level": primitive["level"],
            "variant": primitive["variant"],
            "db_id": db_id,
            "problem": primitive["problem"],
            "answer": primitive["answer"],
        }

        if "prefix_hints" in primitive and primitive["prefix_hints"]:
            prefix_hints = primitive["prefix_hints"]
            hint_exprs = []
            for i in range(1, 10):
                key = f"hint_{i}"
                if key in prefix_hints:
                    hint_exprs.append(prefix_hints[key])
            gt["hint_exprs"] = hint_exprs
            gt["prefix_hints"] = prefix_hints

        return gt

    def filter_for_sft(
        self,
        examples: list[dict],
        include_wrong_valid_format: bool = False,
        nested_request: bool = False,
    ) -> list[dict]:
        """
        Filter examples for SFT training.

        When include_wrong_valid_format is True, includes incorrect examples
        that used hints and gave an answer.
        """
        def has_hints_and_answer(ex):
            gen = ex.get("generation", "")
            return "<request></request>" in gen and "<answer>" in gen

        filtered = []
        for ex in examples:
            if ex.get("correct", False):
                filtered.append(ex)
            elif include_wrong_valid_format and has_hints_and_answer(ex):
                filtered.append(ex)

        if nested_request and _has_malformed_structure_nested is not None:
            filtered = [ex for ex in filtered
                        if not _has_malformed_structure_nested(ex.get("generation", ""))]

        return filtered

    def _categorize_result(self, r: dict) -> str:
        """Categorize a result into: correct, incomplete, wrong."""
        if r.get("correct", False):
            return "correct"
        elif r.get("finish_reason") == "length" or r.get("error") == "no_answer_tag":
            return "incomplete"
        else:
            return "wrong"

    def compute_metrics(self, results: list[dict]) -> dict:
        """Compute Text-to-SQL metrics grouped by difficulty level and dataset source."""
        metrics = super().compute_metrics(results)

        dist_by_level = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "wrong": 0})
        dist_by_source = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "wrong": 0})

        for r in results:
            level = r.get("level", "unknown")
            meta = r.get("metadata")
            source = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
            category = self._categorize_result(r)

            dist_by_level[level]["count"] += 1
            dist_by_level[level][category] += 1

            dist_by_source[source]["count"] += 1
            dist_by_source[source][category] += 1

        total_dist = {"count": 0, "correct": 0, "incomplete": 0, "wrong": 0}
        for v_dist in dist_by_level.values():
            for k in total_dist:
                total_dist[k] += v_dist[k]

        metrics["distribution_by_level"] = dict(dist_by_level)
        metrics["distribution_by_source"] = dict(dist_by_source)
        metrics["distribution"] = total_dist
        return metrics

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """Format Text-to-SQL metrics as tables by level and dataset source."""
        lines = ["", "=== Text-to-SQL Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append(f"Overall Accuracy: {metrics.get('accuracy', 0):.2%} ({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
        lines.append("")

        # By difficulty level
        lines.append("By Difficulty Level:")
        lines.append(f"{'Level':<12} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Wrong':>8} {'Acc (%)':>9}")
        lines.append("-" * 64)

        dist_by_level = metrics.get("distribution_by_level", {})
        for level in sorted(dist_by_level.keys()):
            d = dist_by_level[level]
            acc = (d['correct'] / d['count'] * 100) if d['count'] > 0 else 0
            lines.append(
                f"{level:<12} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['wrong']:>8} {acc:>8.1f}%"
            )

        lines.append("")

        # By dataset source
        dist_by_source = metrics.get("distribution_by_source", {})
        if dist_by_source:
            lines.append("By Dataset Source:")
            lines.append(f"{'Source':<16} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Wrong':>8} {'Acc (%)':>9}")
            lines.append("-" * 68)
            for source in sorted(dist_by_source.keys()):
                d = dist_by_source[source]
                acc = (d['correct'] / d['count'] * 100) if d['count'] > 0 else 0
                lines.append(
                    f"{source:<16} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['wrong']:>8} {acc:>8.1f}%"
                )

        return "\n".join(lines)

