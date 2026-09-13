"""Competition Math task implementation.

Uses the MATH dataset (competition_math) from HuggingFace with 12,500 competition
mathematics problems across 7 categories and 5 difficulty levels.

Dataset: https://huggingface.co/datasets/qwedsacf/competition_math

Hints come from each primitive's prefix_hints (6-hint progressive system).
"""

import importlib.util
import re
from pathlib import Path

from pipeline.tasks.base import BaseTask

# The nested tag grammar is defined once, in the reward function verl scores
# rollouts with. Loading it here keeps SFT filtering and RL scoring from
# drifting apart; it is loaded by path because the recipe is a standalone file,
# not a package.
_REWARD_PATH = (Path(__file__).resolve().parents[3]
                / "verl" / "recipe" / "competition_math" / "reward_function.py")
_spec = importlib.util.spec_from_file_location("competition_math_reward", _REWARD_PATH)
_reward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reward)
_has_malformed_structure_nested = _reward.has_malformed_structure_nested


class CompetitionMathTask(BaseTask):
    """
    Competition Math reasoning task.

    Problems span 7 categories: Algebra, Counting & Probability, Geometry,
    Intermediate Algebra, Number Theory, Prealgebra, Precalculus.

    Difficulty levels: Level 1 (easiest) to Level 5 (hardest).
    """

    name = "competition_math"

    system_message = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer."
    )

    assistant_prefix = "<think>\nLet me work through this problem step by step."

    # Only include harder problem types (exclude Algebra, Prealgebra, Precalculus)
    # Ordered, not a set: iteration order feeds the selection list *before* the
    # seeded shuffle, and set iteration order varies across processes under
    # hash randomization. A set here made --seed non-reproducible.
    ALLOWED_TYPES = (
        "Intermediate Algebra",
        "Geometry",
        "Number Theory",
        "Counting & Probability",
    )

    def create_primitives(self, num_puzzles: int | None, seed: int = 42) -> list[dict]:
        """
        Load competition math problems from HuggingFace, balanced across types.

        Args:
            num_puzzles: Number of problems to load (None = all matching filter)
            seed: Random seed for shuffling
        """
        import random
        from datasets import load_dataset

        rng = random.Random(seed)

        ds = load_dataset("qwedsacf/competition_math", split="train")

        # Group by type
        by_type: dict[str, list] = {t: [] for t in self.ALLOWED_TYPES}
        for row in ds:
            if row["type"] in self.ALLOWED_TYPES:
                by_type[row["type"]].append(row)

        # Shuffle each type
        for t in by_type:
            rng.shuffle(by_type[t])

        # Determine per-type limits for balanced sampling. Integer division
        # alone truncates: it yielded 0 primitives for any num_puzzles below the
        # type count, and under-delivered by up to len(ALLOWED_TYPES)-1
        # otherwise. Spread the remainder so --num-puzzles N really means N.
        n_types = len(self.ALLOWED_TYPES)
        if num_puzzles is not None:
            base, remainder = divmod(num_puzzles, n_types)
            per_type_limits = [
                base + (1 if i < remainder else 0) for i in range(n_types)
            ]
        else:
            per_type_limits = [None] * n_types

        # Sample from each type (balanced)
        selected = []
        for t, limit in zip(self.ALLOWED_TYPES, per_type_limits):
            pool = by_type[t]
            if limit is not None:
                pool = pool[:limit]
            selected.extend(pool)

        # Final shuffle
        rng.shuffle(selected)

        primitives = []
        for idx, row in enumerate(selected):
            # Extract answer from solution's \boxed{} format
            answer = self._extract_boxed_answer(row["solution"])

            primitives.append({
                "index": idx,
                "variant": row["type"],  # Problem category
                "level": row["level"],   # Difficulty level (Level 1-5)
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": answer,
            })

        return primitives

    def _extract_boxed_answer(self, solution: str) -> str | None:
        """Extract the answer from \\boxed{...} in the solution."""
        # Handle nested braces by finding matching closing brace
        match = re.search(r'\\boxed\{', solution)
        if not match:
            return None

        start = match.end()
        depth = 1
        pos = start

        while pos < len(solution) and depth > 0:
            if solution[pos] == '{':
                depth += 1
            elif solution[pos] == '}':
                depth -= 1
            pos += 1

        if depth == 0:
            return solution[start:pos - 1]
        return None

    def format_prompt(
        self,
        primitive: dict,
        template: str,
        include_assistant_prefix: bool = True,
    ) -> list[dict]:
        """Format competition math problem into chat messages."""
        content = template.replace("{problem}", primitive["problem"])
        content = content.replace("{level}", primitive["level"])
        content = content.replace("{type}", primitive["variant"])

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

    def _verify_answer(self, predicted: str, correct_answer: str) -> bool:
        """Verify predicted answer against correct answer using math-verify."""
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

        # Parse gold (LaTeX from dataset) and predicted (plain symbolic from model)
        try:
            gold_parsed = parse(
                f"${correct_answer}$",
                extraction_config=[LatexExtractionConfig()],
            )
        except Exception:
            gold_parsed = []

        try:
            pred_parsed = parse(
                f"${predicted}$",
                extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            )
        except Exception:
            pred_parsed = []

        # Try verification if both parsed successfully
        is_correct = False
        if gold_parsed and pred_parsed:
            try:
                is_correct = verify(gold_parsed, pred_parsed)
            except Exception:
                pass

        # Fallback: try both as plain expressions (no LaTeX wrapping)
        if not is_correct:
            try:
                gold_plain = parse(
                    correct_answer,
                    extraction_config=[ExprExtractionConfig()],
                )
                pred_plain = parse(
                    predicted,
                    extraction_config=[ExprExtractionConfig()],
                )
                if gold_plain and pred_plain:
                    is_correct = verify(gold_plain, pred_plain)
            except Exception:
                pass

        return is_correct

    def check_correctness(
        self,
        primitive: dict,
        generation: str,
    ) -> tuple[bool, dict]:
        """
        Check if the generated answer matches the correct answer.

        Answers are read from <answer>...</answer>.

        Uses math-verify for robust symbolic equivalence checking.
        Gold answers are parsed as LaTeX, predicted answers as plain expressions.

        method_c's verifier ground truth carries a "correct" label (0/1)
        instead of a math "answer" -- the predicted <answer>0/1</answer> tag
        is compared directly against that label rather than math-verified.
        """
        # Standard format
        predicted = self.extract_answer(generation)

        if predicted is None:
            return False, {
                "predicted_answer": None,
                "error": "no_answer_tag",
            }

        predicted = predicted.strip()

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

        is_correct = self._verify_answer(predicted, correct_answer)
        return is_correct, {
            "predicted_answer": predicted,
            "correct_answer": correct_answer,
        }

    def get_ground_truth(self, primitive: dict) -> dict:
        """Extract ground truth for embedding in prompts and RL interactions.

        Hints come from the primitive's prefix_hints (6-hint progressive system).
        """
        gt = {
            "level": primitive["level"],
            "variant": primitive["variant"],
            "problem": primitive["problem"],
            "answer": primitive["answer"],
        }

        if "prefix_hints" in primitive and primitive["prefix_hints"]:
            prefix_hints = primitive["prefix_hints"]
            hint_exprs = []
            for i in range(1, 7):  # hint_1 through hint_6
                key = f"hint_{i}"
                if key in prefix_hints:
                    hint_exprs.append(prefix_hints[key])
            gt["hint_exprs"] = hint_exprs
            gt["prefix_hints"] = prefix_hints  # Keep original for reference

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
        that used hints and gave an answer — valuable for teaching the hint
        request/response protocol.

        nested_request additionally drops generations that are correct but do
        not match the inline-request grammar. Correctness alone is too weak a
        filter there: in the full 14B run it let through 20 structurally
        invalid generations, 4 of which closed </think> before requesting.
        Training on those teaches back the very pattern the nested format
        exists to remove, and a handful of examples is enough for the model to
        learn that </think> is a legal place to stop and ask.
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
        """Categorize a result into: correct, incomplete, wrong.

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
        """
        Compute competition_math-specific metrics.

        Groups by both problem type and difficulty level.
        """
        from collections import defaultdict

        metrics = super().compute_metrics(results)

        # Track distribution by level
        dist_by_level = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "wrong": 0})
        # Track distribution by type
        dist_by_type = defaultdict(lambda: {"count": 0, "correct": 0, "incomplete": 0, "wrong": 0})

        for r in results:
            level = r.get("level", "unknown")
            ptype = r.get("variant", "unknown")
            category = self._categorize_result(r)

            dist_by_level[level]["count"] += 1
            dist_by_level[level][category] += 1

            dist_by_type[ptype]["count"] += 1
            dist_by_type[ptype][category] += 1

        # Compute totals
        total_dist = {"count": 0, "correct": 0, "incomplete": 0, "wrong": 0}
        for v_dist in dist_by_level.values():
            for k in total_dist:
                total_dist[k] += v_dist[k]

        metrics["distribution_by_level"] = dict(dist_by_level)
        metrics["distribution_by_type"] = dict(dist_by_type)
        metrics["distribution"] = total_dist

        return metrics

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """Format competition_math metrics as tables by level and type."""
        lines = ["", "=== Evaluation Results ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append("")

        # By difficulty level
        lines.append("By Difficulty Level:")
        lines.append(f"{'Level':<12} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Wrong':>8}")
        lines.append("-" * 53)

        dist_by_level = metrics.get("distribution_by_level", {})
        for level in sorted(dist_by_level.keys()):
            d = dist_by_level[level]
            lines.append(
                f"{level:<12} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['wrong']:>8}"
            )

        lines.append("")

        # By problem type
        lines.append("By Problem Type:")
        lines.append(f"{'Type':<24} {'Count':>7} {'Correct':>10} {'Incomplete':>12} {'Wrong':>8}")
        lines.append("-" * 65)

        dist_by_type = metrics.get("distribution_by_type", {})
        for ptype in sorted(dist_by_type.keys()):
            d = dist_by_type[ptype]
            lines.append(
                f"{ptype:<24} {d['count']:>7} {d['correct']:>10} {d['incomplete']:>12} {d['wrong']:>8}"
            )

        # Summary
        lines.append("")
        d = metrics.get("distribution", {})
        total_count = d.get("count", 0)
        lines.append(f"Total: {total_count} | Correct: {d.get('correct', 0)} | Accuracy: {metrics.get('accuracy', 0):.2%}")

        return "\n".join(lines)
