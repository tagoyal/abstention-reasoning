"""Base task class with shared functionality."""

from collections import defaultdict

from pipeline.core.utils import extract_answer


class BaseTask:
    """
    Base class for tasks.

    Subclasses must implement:
        - name: str
        - create_primitives(num_puzzles, seed) -> list[dict]
        - format_prompt(primitive, template) -> list[dict]
        - check_correctness(primitive, generation) -> tuple[bool, dict]
    """

    name: str = "base"

    # Default assistant prefix for prompts (override in subclasses for task-specific prefixes)
    assistant_prefix: str = "<think> Let me solve this step by step."

    # === Required methods (must override) ===

    def create_primitives(self, num_puzzles: int | None, seed: int) -> list[dict]:
        """
        Generate or load raw puzzle data.

        Args:
            num_puzzles: Number of puzzles (None = all available for imported datasets)
            seed: Random seed

        Returns:
            List of dicts, each with at least 'index' and 'variant' fields.
        """
        raise NotImplementedError

    def format_prompt(self, primitive: dict, template: str) -> list[dict]:
        """
        Format primitive into chat messages using template.

        Returns:
            List of message dicts: [{"role": "...", "content": "..."}, ...]
        """
        raise NotImplementedError

    def check_correctness(self, primitive: dict, generation: str) -> tuple[bool, dict]:
        """
        Check if generation correctly solves the puzzle.

        Returns:
            (is_correct, metadata_dict)
        """
        raise NotImplementedError

    # === Optional methods (can override) ===

    # Fractional [start, end) ranges over the shuffled primitive order.
    # Tasks override this to declare their own layout; a task need not define
    # every split (code_output has no rl_val), so callers that sweep "all"
    # splits must ask supported_splits() rather than assume the set.
    SPLITS: dict[str, tuple[float, float]] = {
        "sft_train": (0.0, 0.27),
        "sft_val": (0.27, 0.3),
        "rl_train": (0.3, 0.65),
        "rl_val": (0.65, 0.7),
        "eval": (0.7, 1.0),
    }

    # rl_train/rl_val further subdivided into a "gen" half (used to run
    # solver generation) and a "ver" half (used to train/evaluate the
    # verifier), via scripts/split_dataset.py's stratified sampling. That
    # split isn't a contiguous [start, end) ratio range over the shuffled
    # primitive order -- it's a distribution-preserving resample within
    # rl_train/rl_val -- so these have no SPLITS entry and get_split_indices()
    # cannot (re)derive them. They only exist as partition files already
    # written to disk; create_prompts reads those directly.
    EXTRA_SPLITS: tuple[str, ...] = (
        "rl_gen_train",
        "rl_gen_val",
        "rl_ver_train",
        "rl_ver_val",
    )

    @classmethod
    def supported_splits(cls) -> list[str]:
        """Split names this task defines, in creation order."""
        return list(cls.SPLITS) + list(cls.EXTRA_SPLITS)

    def get_split_indices(
        self,
        total: int,
        split: str,
        seed: int = 42,
        primitives: list[dict] | None = None,
    ) -> list[int]:
        """
        Return indices for a given split.

        Ranges come from the task's SPLITS table; override that (not this
        method) to change the layout. Base layout:
        - sft_train: 27% (indices 0-27%)
        - sft_val: 3% (indices 27-30%)
        - rl_train: 35% (indices 30-65%)
        - rl_val: 5% (indices 65-70%)
        - eval: 30% (indices 70-100%)

        Raises ValueError if the task does not define `split`.

        Args:
            total: Total number of primitives
            split: Split name; must be a key of this task's SPLITS
            seed: Random seed for shuffling
            primitives: Optional list of primitives (for stratified sampling)

        Returns:
            List of indices belonging to this split
        """
        import random

        rng = random.Random(seed)
        indices = list(range(total))
        rng.shuffle(indices)

        splits = self.SPLITS

        if split not in splits:
            raise ValueError(
                f"Task '{self.name}' does not define split '{split}'. "
                f"Available: {list(splits)}"
            )

        start_ratio, end_ratio = splits[split]
        start = int(total * start_ratio)
        end = int(total * end_ratio)

        return indices[start:end]

    def get_ground_truth(self, primitive: dict) -> dict:
        """
        Extract ground truth for embedding in prompts.
        Default: all fields except 'index'.
        """
        return {k: v for k, v in primitive.items() if k != "index"}

    def compute_metrics(self, results: list[dict]) -> dict:
        """
        Compute aggregate metrics from evaluation results.
        Default: accuracy overall and by variant.
        """
        total = len(results)
        correct = sum(1 for r in results if r.get("correct", False))

        # Group by variant
        by_variant = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            variant = r.get("variant", "unknown")
            by_variant[variant]["total"] += 1
            if r.get("correct", False):
                by_variant[variant]["correct"] += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "accuracy_by_variant": {
                v: d["correct"] / d["total"] if d["total"] > 0 else 0
                for v, d in by_variant.items()
            },
            "total": total,
            "correct": correct,
            "counts_by_variant": dict(by_variant),
        }

    def format_metrics(self, metrics: dict, model_name: str | None = None) -> str:
        """
        Format metrics for display. Override for task-specific formatting.

        Args:
            metrics: Dict from compute_metrics()
            model_name: Optional model name to include in output

        Returns:
            Formatted string for printing
        """
        lines = ["", "=== Metrics ==="]
        if model_name:
            lines.append(f"Model: {model_name}")
        lines.append(f"Total: {metrics.get('total', 0)}")
        lines.append(f"Correct: {metrics.get('correct', 0)}")
        lines.append(f"Accuracy: {metrics.get('accuracy', 0):.2%}")

        # Print by variant if available
        by_variant = metrics.get("counts_by_variant") or metrics.get("by_variant", {})
        if by_variant:
            lines.append("")
            lines.append("By variant:")
            for variant in sorted(by_variant.keys(), key=str):
                c = by_variant[variant]
                total = c.get("total", 0)
                correct = c.get("correct", 0)
                acc = correct / total if total > 0 else 0
                lines.append(f"  {variant}: {correct}/{total} ({acc:.0%})")

        return "\n".join(lines)

    # === Utility methods ===

    def extract_answer(self, generation: str) -> str | None:
        """Extract answer from generation. Override for custom parsing."""
        return extract_answer(generation)

