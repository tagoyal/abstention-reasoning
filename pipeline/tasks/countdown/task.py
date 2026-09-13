"""Countdown task implementation."""

import importlib.util
import re
from pathlib import Path

from pipeline.tasks.base import BaseTask
from pipeline.core.utils import safe_eval

# The nested tag grammar is defined once, in verl/recipe/shared, so the SFT
# filter below and the reward function verl scores rollouts with cannot drift
# apart. Loaded by path because recipes are standalone files, not a package.
_GRAMMAR_PATH = (Path(__file__).resolve().parents[3]
                 / "verl" / "recipe" / "shared" / "nested_grammar.py")
_spec = importlib.util.spec_from_file_location("nested_grammar", _GRAMMAR_PATH)
_grammar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grammar)
_has_malformed_structure_nested = _grammar.has_malformed_structure_nested


class CountdownTask(BaseTask):
    """
    Countdown numbers game task.

    Goal: Use given numbers with +, -, *, / to reach a target value.
    Each number can only be used once.
    """

    name = "countdown"
    default_template_variant = "simple"

    # Default system message
    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )

    def create_primitives(self, num_puzzles: int | None, seed: int = 42) -> list[dict]:
        """Generate countdown puzzles."""
        from .generator import generate_puzzles

        if num_puzzles is None:
            raise ValueError("countdown task requires --num-puzzles (it generates puzzles, not downloads)")

        return generate_puzzles(
            num_puzzles=num_puzzles,
            seed=seed,
            operand_distribution={4: 0.33, 5: 0.33, 6: 0.34},
        )

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format countdown puzzle into chat messages."""
        # Substitute variables in template
        content = template.replace("{{target}}", str(primitive["target"]))
        content = content.replace("{target}", str(primitive["target"]))
        content = content.replace("{{numbers}}", str(primitive["numbers"]))
        content = content.replace("{numbers}", str(primitive["numbers"]))

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
        Check if the generated expression correctly solves the puzzle.

        Validates:
        1. Count hint requests (<request> tags)
        2. Answer can be parsed from <answer> tags
        3. Expression only uses available numbers
        4. Each number used at most once
        5. Expression evaluates to target

        method_c's verifier ground truth carries a "correct" label (0/1)
        instead of "numbers"/"target" -- the predicted <answer>0/1</answer>
        tag is compared directly against that label rather than evaluated as
        an expression.
        """
        # Count hint requests
        num_hints = generation.count("<request>")

        answer = self.extract_answer(generation)

        if answer is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
                "num_hints": num_hints,
            }

        if "correct" in primitive and "numbers" not in primitive:
            answer = answer.strip()
            expected_label = primitive.get("correct")
            is_correct = answer in ("0", "1") and int(answer) == int(expected_label)
            return is_correct, {
                "predicted_label": answer,
                "expected_label": expected_label,
                "num_hints": num_hints,
            }

        is_correct, meta = self._check_expression(primitive, answer)
        meta["num_hints"] = num_hints
        return is_correct, meta

    def _check_expression(self, primitive: dict, answer: str) -> tuple[bool, dict]:
        """Validate and evaluate a countdown expression against the puzzle."""
        try:
            numbers_used = [int(n) for n in re.findall(r'\b\d+\b', answer)]
            available = primitive["numbers"].copy()

            for num in numbers_used:
                if num not in available:
                    return False, {
                        "predicted_answer": answer,
                        "error": "invalid_number",
                        "invalid_number": num,
                    }
                available.remove(num)

            if available:
                return False, {
                    "predicted_answer": answer,
                    "error": "unused_numbers",
                    "unused_numbers": available,
                }

            result = safe_eval(answer)
            is_correct = (result == primitive["target"])

            return is_correct, {
                "predicted_answer": answer,
                "result": result,
                "target": primitive["target"],
            }

        except Exception as e:
            return False, {
                "predicted_answer": answer,
                "error": str(e),
            }

    def filter_for_sft(
        self,
        examples: list[dict],
        include_wrong_valid_format: bool = False,
        nested_request: bool = False,
    ) -> list[dict]:
        """
        Filter examples for SFT training.

        When include_wrong_valid_format is True, includes incorrect examples
        that used hints and gave an answer — valuable for teaching the hint
        request/response protocol.

        nested_request additionally drops generations that are correct but do
        not match the inline-request grammar. check_correctness only parses the
        <answer> expression, so it says nothing about tag structure: in the 14B
        run it kept 156 of 1106 correct generations that closed </think> before
        requesting a hint. Countdown produces far more of these than math does
        (39% of asking generations against 2%) — its ask moment is a give-up
        point in a long search, where the habit of ending reasoning with
        </think> fires — so training on them would teach back the very pattern
        the nested format exists to remove.
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

        if nested_request:
            filtered = [ex for ex in filtered
                        if not _has_malformed_structure_nested(ex.get("generation", ""))]

        return filtered

    def _categorize_result(self, r: dict) -> str:
        """Categorize a single result into one of: correct, incomplete, wrong.

        Correctness decides, with a truncated or answer-less rollout counted as
        "incomplete" rather than wrong.
        """
        if r.get("correct", False):
            return "correct"
        elif r.get("finish_reason") == "length" or r.get("error") == "no_answer_tag":
            return "incomplete"
        else:
            return "wrong"

    def compute_metrics(self, results: list[dict]) -> dict:
        """Compute metrics including the outcome distribution by variant.

        """
        from collections import defaultdict

        metrics = super().compute_metrics(results)

        # Track distribution by variant
        dist_by_variant = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "wrong": 0})

        for r in results:
            variant = r.get("variant", "unknown")
            category = self._categorize_result(r)

            dist_by_variant[variant]["count"] += 1
            dist_by_variant[variant][category] += 1

        # Compute totals
        total_dist = {"count": 0, "correct": 0, "incomplete": 0, "wrong": 0}
        for v_dist in dist_by_variant.values():
            for k in total_dist:
                total_dist[k] += v_dist[k]

        metrics["distribution_by_variant"] = dict(dist_by_variant)
        metrics["distribution"] = total_dist

        return metrics

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """
        Format countdown metrics as a table with outcome distribution by operand count.
        """
        lines = ["", "=== Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append("")

        # Table header
        lines.append(f"{'Mode':<14} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Wrong':>8}")
        lines.append("-" * 53)

        # By variant rows
        dist_by_variant = metrics.get("distribution_by_variant", {})
        for variant in sorted(dist_by_variant.keys(), key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 0):
            d = dist_by_variant[variant]
            # Convert variant name like "4_operands" to "4 operands"
            label = variant.replace("_", " ")
            lines.append(
                f"{label:<14} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['wrong']:>8}"
            )

        # Total row
        lines.append("-" * 53)
        d = metrics.get("distribution", {})
        lines.append(
            f"{'Total':<14} {d.get('count', 0):>7} {d.get('correct', 0):>10} {d.get('incomplete', 0):>12} {d.get('wrong', 0):>8}"
        )

        # Summary line
        lines.append("")
        lines.append(f"Accuracy: {metrics.get('accuracy', 0):.2%}")

        return "\n".join(lines)
