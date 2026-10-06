"""Method configuration - bundles template variant + reward function + artifact paths."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Repo root, derived from this file's location: <repo>/pipeline/core/method.py.
# Config, template and artifact lookups used to be relative to the process's
# working directory, so every command silently required being run from the repo
# root and failed with "method not found" (or created a stray empty artifacts/)
# anywhere else.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Data and models used to share one `artifacts/{task}/...` tree. They are now
# two independent roots so that `data/` (problems, formatted prompts, SFT
# datasets -- small, text, meant for git) can be committed, while `models/`
# (checkpoints -- large, binary, meant for a HuggingFace hub instead) stays
# out of git entirely. Each is keyed by its own name rather than always the
# task name: by default data_name/models_name both fall back to task_name, so
# nothing changes for a task with one dataset. But a task can have more than
# one dataset variant under trial (e.g. `data/math_o1/` instead of
# `data/math/`), in which case passing --data-name picks that data directory,
# and --models-name (or, left unset, the data name) picks where its models go
# -- so experimental data and its models stay paired without colliding with
# the task's default data/models.
DATA_ROOT = REPO_ROOT / "data"
MODELS_ROOT = REPO_ROOT / "models"

# Roots for repo-owned lookups (method configs and task templates)
CONFIGS_ROOT = REPO_ROOT / "pipeline" / "configs" / "methods"
TASKS_ROOT = REPO_ROOT / "pipeline" / "tasks"

# External storage for models. If model weights should live outside the repo
# (e.g. on shared storage, since models/ is gitignored and can get huge), set
# EXTERNAL_MODELS_ROOT and a run that doesn't exist yet is created there and
# symlinked into models/ instead of being written in place.
EXTERNAL_MODELS_ROOT = (
    Path(os.environ["EXTERNAL_MODELS_ROOT"])
    if "EXTERNAL_MODELS_ROOT" in os.environ
    else None
)


def resolve_data_name(task_name: str, data_name: str | None = None) -> str:
    """data_name falls back to task_name when not given explicitly.

    Also exports DATABASE_PATH=<repo>/data/<data_name>/databases into the
    environment (unless already set), so task-specific scorers that can't
    take data_name as a normal argument (e.g.
    verl/recipe/sql/reward_function.py, loaded standalone by both the
    pipeline and verl's RL trainer) pick up the right dataset variant's
    databases/ folder without every call site having to set it by hand.

    DATABASE_PATH is deliberately a separate env var from --data-name /
    pipeline's own DATA_ROOT path constant: the actual .sqlite files don't
    have to live under data/<data_name>/ at all (e.g. they may live on
    different storage, kept out of the git-tracked dataset folders). If the
    caller has already exported DATABASE_PATH, we leave it untouched so that
    override keeps working.
    """
    data_name = data_name or task_name
    os.environ.setdefault("DATABASE_PATH", str(DATA_ROOT / data_name / "databases"))
    return data_name


def resolve_models_name(data_name: str, models_name: str | None = None) -> str:
    """models_name falls back to data_name when not given explicitly -- so a
    custom --data-name (e.g. "math_o1") also repoints the models default
    (models/math_o1/...) unless --models-name overrides it separately."""
    return models_name or data_name


@dataclass
class Method:
    """
    Configuration for a pipeline method.

    A method defines a consistent configuration across the full pipeline:
    - template_variant: Which template variant to use (e.g., "simple", "hint")
    - reward_function: Name of reward function for RL training
    - reward_kwargs: Additional arguments for reward function
    - multi_turn: Enable multi-turn hint generation in RL (default: False)
    - mask_response_tokens: Mask <response>...</response> tokens during SFT (default: False)

    Methods also provide auto-derived artifact paths based on task and method name.
    """

    name: str
    template_variant: str
    reward_function: str = "compute_score"
    reward_kwargs: dict[str, Any] = field(default_factory=dict)
    multi_turn: bool = False  # Enable multi-turn hint generation
    max_turns: int = 6  # RL-only: verl's rollout turn cap. generate/evaluate
                        # are bounded by max_new_tokens, not by a turn count.
    mask_response_tokens: bool = False  # Mask <response>...</response> in SFT
    max_hints: int | None = None  # Maximum number of hints to give during RL rollout (None = unlimited)
    hint_transition: bool = True  # Splice a canned "I'm stuck..." phrase before each
                                  # forced hint request during SFT data generation. False
                                  # cuts the CoT silently instead, so nothing before
                                  # <request></request> telegraphs the request.
    nested_request: bool = False  # Keep <request></request> inside the <think> block
                                  # instead of after it, so </think> stays the single
                                  # irreversible commit point right before <answer>.

    # Backwards compatibility alias
    @property
    def allow_hint(self) -> bool:
        """Alias for multi_turn (backwards compatibility)."""
        return self.multi_turn

    @classmethod
    def load(cls, name_or_path: str, task_name: str) -> "Method":
        """
        Load a method config.

        Args:
            name_or_path: Either a method name (e.g., "baseline", "method_b")
                          or a path to a YAML config file
            task_name: Task name (used to find config in standard location)

        Returns:
            Method instance

        Lookup order:
            1. If name_or_path is a file path, load directly
            2. Otherwise, look in pipeline/configs/methods/{task}/{name}.yaml

        Either way the method's name is the config file's stem.
        """
        path = Path(name_or_path)

        # Check if it's a direct path
        if path.exists() and path.is_file():
            config_path = path
        else:
            # Look in standard location
            config_path = CONFIGS_ROOT / task_name / f"{name_or_path}.yaml"
            if not config_path.exists():
                available = cls.list_methods(task_name)
                raise FileNotFoundError(
                    f"Method '{name_or_path}' not found for task '{task_name}'. "
                    f"Available methods: {available}"
                )

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # The filename is the name, and the only source of it. It decides the
        # templates, the __suffix on every prompt and dataset file, and the
        # model group directory, so a `name:` key free to disagree with the
        # file it sits in would be a second source of truth for all three.
        return cls(
            name=config_path.stem,
            template_variant=data["template_variant"],
            reward_function=data.get("reward_function", "compute_score"),
            reward_kwargs=data.get("reward_kwargs", {}),
            multi_turn=data.get("multi_turn", False),
            max_turns=data.get("max_turns", 6),
            mask_response_tokens=data.get("mask_response_tokens", False),
            max_hints=data.get("max_hints"),
            hint_transition=data.get("hint_transition", True),
            nested_request=data.get("nested_request", False),
        )

    @staticmethod
    def list_methods(task_name: str) -> list[str]:
        """List available methods for a task."""
        methods_dir = CONFIGS_ROOT / task_name
        if not methods_dir.exists():
            return []
        return sorted(p.stem for p in methods_dir.glob("*.yaml"))

    def get_template_path(self, task_name: str, split: str) -> Path:
        """Get the template path for a given split."""
        return TASKS_ROOT / task_name / "templates" / self.template_variant / f"{split}.txt"

    def load_template(self, task_name: str, split: str) -> str:
        """Load the template content for a given split."""
        template_path = self.get_template_path(task_name, split)
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        with open(template_path) as f:
            return f.read()

    # =========================================================================
    # Artifact path utilities
    # =========================================================================
    def data_dir(self, data_name: str) -> Path:
        """Root of one data tree: problems, formatted prompts and SFT datasets.
        Method-independent: everything in it lives in shared directories and is
        told apart by name, not by sitting under a per-method subtree."""
        return DATA_ROOT / data_name

    # -- prompts and datasets -------------------------------------------------

    def artifact_stem(self, split: str, desc: str | None = None) -> str:
        """`{split}__{method}`, or `{split}__{method}__{desc}`.

        Every field is separated by a double underscore. Method names contain
        single underscores (`method_b`, `method_ac`), so a single-underscore
        desc separator would make `sft_train__method_b_generations` ambiguous --
        method `method_b` with desc `generations`, or a method actually named
        `method_b_generations`. Hyphens stay legal *inside* a field, which is
        what carries model slugs (`qwen3-4b-base`) and run descs
        (`extend-quad-a0.5`).
        """
        stem = f"{split}__{self.name}"
        return f"{stem}__{desc}" if desc else stem

    def formatted_dir(self, data_name: str) -> Path:
        """`problems_with_format/` -- a partition with the method's template
        applied, and nothing generated yet. Model-ready input, no rollouts."""
        return self.data_dir(data_name) / "problems_with_format"

    def formatted_path(self, data_name: str, split: str, desc: str | None = None) -> Path:
        ext = ".parquet" if split.startswith("rl") else ".json"
        return self.formatted_dir(data_name) / f"{self.artifact_stem(split, desc)}{ext}"

    def datasets_dir(self, data_name: str) -> Path:
        """`sft_datasets/` -- generations with correctness labels. SFT-stage only:
        the RL parquets carry no generations, so they stay in the formatted layer
        and never reach here."""
        return self.data_dir(data_name) / "sft_datasets"

    def dataset_path(self, data_name: str, split: str, desc: str | None = None) -> Path:
        return self.datasets_dir(data_name) / f"{self.artifact_stem(split, desc)}.json"

    def scratch_path(self, data_name: str, split: str, desc: str | None = None) -> Path:
        """Same naming as a dataset, but under .scratch/ -- for intermediates
        that are read back by a later step and never trained on."""
        return scratch_dir(data_name) / f"{self.artifact_stem(split, desc)}.json"

    # -- models ---------------------------------------------------------------

    def models_dir(self, models_name: str) -> Path:
        return MODELS_ROOT / models_name

    # Directory suffix per training stage. `_sft` holds the intermediate a
    # method was initialized from; `_rl` holds the finished model, post-RL;
    # `_classifier` holds a verifier/continue-vs-abstain classifier trained
    # on tree-labeled data (see generate_tree), independent of the sft/rl
    # generation lineage.
    GROUP_SUFFIX = {"sft": "sft", "rl": "rl", "classifier": "classifier"}

    def group_dir(self, models_name: str, stage: str) -> Path:
        """`models/{method}_{sft,rl}` -- e.g. models/baseline_sft, models/method_b_rl."""
        try:
            suffix = self.GROUP_SUFFIX[stage]
        except KeyError:
            raise ValueError(
                f"stage must be one of {sorted(self.GROUP_SUFFIX)}, got {stage!r}"
            ) from None
        return self.models_dir(models_name) / f"{self.name}_{suffix}"

    def run_dir(self, models_name: str, stage: str, run_id: str) -> Path:
        """One training run: `models/{models_name}/{method}_{sft,rl}/{run_id}`.

        run_id is the directory name verbatim and is always required. Nothing
        derives it from the base checkpoint: the convention (`1.5b`, `4b`,
        `4b-instruct`, `1.5b__extend-quad-a0.5`) is a naming decision, not a
        fact about the model, so a guessed name would only drift from it.
        """
        if not run_id:
            raise ValueError(
                f"--run-id is required: it names the directory under "
                f"{self.group_dir(models_name, stage).relative_to(MODELS_ROOT.parent)}."
            )
        return self.group_dir(models_name, stage) / run_id

    def _ensure_run_dir(self, run_dir: Path, models_name: str) -> Path:
        """Create a run directory, or resolve it to external storage.

        With EXTERNAL_MODELS_ROOT set, a run that does not exist yet is created
        there and symlinked in, so new runs land on shared storage even when
        older ones are local.
        """
        # A convenience alias (models/baseline_rl/4b -> 4b__stdnorm) names a
        # sibling run in the same directory, so a short name resolves without
        # renaming anything. Training through one would write into the run it
        # points at and destroy it. Such aliases are relative and have no "/";
        # the external-storage links created below are absolute and remain
        # valid resume targets.
        if run_dir.is_symlink():
            target = os.readlink(run_dir)
            if "/" not in target:
                raise FileExistsError(
                    f"'{run_dir.name}' is a convenience symlink to sibling run "
                    f"'{target}', not a run directory of its own. Training here "
                    f"would overwrite that run.\n"
                    f"  Use --run-id {target} to train or resume it, "
                    f"or pick a new run id."
                )
            return run_dir

        if run_dir.exists():
            return run_dir

        if EXTERNAL_MODELS_ROOT is not None:
            rel_to_models = run_dir.relative_to(self.models_dir(models_name))
            external_path = EXTERNAL_MODELS_ROOT / models_name / rel_to_models
            external_path.mkdir(parents=True, exist_ok=True)
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            run_dir.symlink_to(external_path)
            print(f"Created symlink: {run_dir} -> {external_path}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def sft_run_dir(self, models_name: str, run_id: str) -> Path:
        return self.run_dir(models_name, "sft", run_id)

    def rl_run_dir(self, models_name: str, run_id: str) -> Path:
        return self.run_dir(models_name, "rl", run_id)

    def classifier_run_dir(self, models_name: str, run_id: str) -> Path:
        return self.run_dir(models_name, "classifier", run_id)

    def ensure_sft_run_dir(self, models_name: str, run_id: str) -> Path:
        return self._ensure_run_dir(self.sft_run_dir(models_name, run_id), models_name)

    def ensure_rl_run_dir(self, models_name: str, run_id: str) -> Path:
        return self._ensure_run_dir(self.rl_run_dir(models_name, run_id), models_name)

    def ensure_classifier_run_dir(self, models_name: str, run_id: str) -> Path:
        return self._ensure_run_dir(self.classifier_run_dir(models_name, run_id), models_name)

    def sft_model_path(self, models_name: str, run_id: str) -> Path:
        return self.sft_run_dir(models_name, run_id) / "model"

    def rl_model_path(self, models_name: str, run_id: str) -> Path:
        return self.rl_run_dir(models_name, run_id) / "model"

    def classifier_model_path(self, models_name: str, run_id: str) -> Path:
        return self.classifier_run_dir(models_name, run_id) / "model"

    def rl_checkpoints_dir(self, models_name: str, run_id: str) -> Path:
        return self.rl_run_dir(models_name, run_id) / "checkpoints"

    def rl_rollouts_dir(self, models_name: str, run_id: str) -> Path:
        return self.rl_run_dir(models_name, run_id) / "rollouts"

    # -- evaluations ----------------------------------------------------------

    def evals_dir(self, models_name: str, stage: str, run_id: str) -> Path:
        """`models/{method}_{sft,rl}/{run_id}/evals` -- results live inside the
        run that produced them, not in a task-wide results/ pool."""
        return self.run_dir(models_name, stage, run_id) / "evals"

    def eval_path(self, models_name: str, stage: str, run_id: str,
                  split: str, suffix: str = "") -> Path:
        """`.../evals/{split}{suffix}.json`. The run directory already carries
        the model identity, so the filename only has to say which split and
        under what deviation from the default settings."""
        return self.evals_dir(models_name, stage, run_id) / f"{split}{suffix}.json"


def scratch_dir(data_name: str) -> Path:
    """`data/{data_name}/.scratch` -- intermediates that feed a later step but
    are not themselves training data or results.

    The no-hint difficulty probe is the case that forced this: it is a real
    generate output, so it looks like a dataset, but nothing is ever trained on
    it and it only exists to be read back by --hint-schedule. Dot-prefixed so it
    sorts and greps out of the way of the artifacts that matter.
    """
    return DATA_ROOT / data_name / ".scratch"


def problems_dir(data_name: str) -> Path:
    """`data/{data_name}/problems` -- raw problems, shared by every method."""
    return DATA_ROOT / data_name / "problems"


def partition_path(data_name: str, split: str) -> Path:
    """`problems/{split}.json` -- the problems belonging to one split.

    Materialized rather than recomputed. The partition used to exist only as
    (seed, SPLITS table), so editing a boundary silently repartitioned every
    artifact ever produced, with no record of what the old one was. Writing it
    down makes the split a fact about the data instead of a fact about the code
    that happens to be checked out.
    """
    return problems_dir(data_name) / f"{split}.json"


def get_primitives_path(data_name: str) -> Path:
    """Get the shared primitives path for a data directory."""
    return problems_dir(data_name) / "primitives.json"
