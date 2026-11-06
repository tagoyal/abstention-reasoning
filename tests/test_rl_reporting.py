import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from pipeline.commands.training import train_rl
from pipeline.core.method import Method


class RlReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.run_dir = self.root / "existing-run"
        self.checkpoints = self.run_dir / "checkpoints"
        self.prompts = self.root / "train.parquet"
        self.prompts.touch()

    def saved_run(self, name):
        (self.checkpoints / "global_step_25").mkdir(parents=True)
        config_path = self.checkpoints / ".hydra" / "config.yaml"
        config_path.parent.mkdir()
        config_path.write_text(yaml.safe_dump({"trainer": {"experiment_name": name}}))

    def launch(self, converter=None, method=None, **overrides):
        kwargs = {
            "task_name": "sql",
            "run_id": "qwen3-4b",
            "base_model": "Qwen/Qwen3-4B",
            "train_prompts_path": self.prompts,
            "output_path": self.run_dir / "model",
            "reward_function_path": self.root / "reward.py",
            "n_gpus_per_node": 1,
            "cleanup_checkpoints": False,
        }
        kwargs.update(overrides)
        with (
            patch("pipeline.commands.training.datetime") as clock,
            patch("pipeline.commands.training._wandb_run_id", return_value="stable-id"),
            patch(
                "pipeline.commands.training.Method.load",
                side_effect=lambda name, task: Method(name=name, template_variant="baseline")
                if name in ("baseline", "method_ac") else method,
            ),
            patch("pipeline.commands.training.Method.load_template", return_value=None),
            patch("pipeline.core.method.MODELS_ROOT", self.root / "models"),
            patch("pipeline.core.method.EXTERNAL_MODELS_ROOT", None),
            patch("pipeline.tasks.get_task", return_value=SimpleNamespace()),
            patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as run,
            patch("pipeline.commands.training.convert_checkpoint", converter or Mock()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            clock.now.return_value = datetime(2026, 10, 21)
            train_rl(**kwargs)
        return run.call_args.args[0]

    def test_new_run_has_launch_date_prefix_without_changing_paths(self):
        command = self.launch()
        self.assertIn("trainer.experiment_name=10_21-default-qwen3-4b", command)
        self.assertIn(f"trainer.default_local_dir={self.checkpoints}", command)
        self.assertIn("+trainer.wandb_run_id=stable-id", command)

    def test_method_d_launches_online_sft_with_baseline_reward_and_binary_validation(self):
        method = Method(name="method_d", template_variant="baseline")
        command = self.launch(method=method, method_name="method_d", val_prompts_path=self.prompts,
                              n_samples=3, save_best=True)
        self.assertIn("+trainer.method_d=True", command)
        self.assertIn("+trainer.interleave_rl=False", command)
        self.assertIn("actor_rollout_ref.actor.use_kl_loss=False", command)
        self.assertIn("+trainer.best_ckpt_metric=sft_val/accuracy", command)
        self.assertIn("actor_rollout_ref.rollout.val_kwargs.n=3", command)
        self.assertTrue(any(arg.endswith("reward_function.name=compute_score") for arg in command))

    def test_method_d_interleaves_rl_and_preserves_baseline_best_metric(self):
        command = self.launch(
            method=Method(name="method_d", template_variant="baseline"),
            method_name="method_d", val_prompts_path=self.prompts,
            interleave_rl=True, sft_stride=64, sft_micro_batch_size=2, save_best=True,
        )
        self.assertIn("+trainer.interleave_rl=True", command)
        self.assertIn("actor_rollout_ref.actor.use_kl_loss=True", command)
        self.assertIn("+trainer.best_ckpt_metric=auto", command)
        self.assertIn("+trainer.sft_stride=64", command)
        self.assertIn("+actor_rollout_ref.actor.sft_micro_batch_size=2", command)

    def test_method_d_overrides_compose_with_real_verl_hydra_config(self):
        from hydra import compose, initialize_config_dir

        command = self.launch(
            method=Method(name="method_d", template_variant="baseline"),
            method_name="method_d", val_prompts_path=self.prompts, save_best=True,
        )
        config_dir = Path(__file__).resolve().parents[1] / "verl" / "verl" / "trainer" / "config"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(config_name="ppo_trainer", overrides=command[3:])
        self.assertTrue(config.trainer.method_d)
        self.assertFalse(config.trainer.interleave_rl)
        self.assertFalse(config.actor_rollout_ref.actor.use_kl_loss)
        self.assertEqual(config.actor_rollout_ref.rollout.n, 8)
        self.assertEqual(config.actor_rollout_ref.rollout.val_kwargs.n, 8)
        self.assertEqual(config.trainer.best_ckpt_metric, "sft_val/accuracy")

    def test_method_e_head_options_compose_with_real_hydra_config(self):
        from hydra import compose, initialize_config_dir

        for head_only, interleave in ((False, False), (True, True)):
            with self.subTest(head_only=head_only):
                command = self.launch(
                    method=Method(name="method_e", template_variant="baseline"),
                    method_name="method_e", val_prompts_path=self.prompts,
                    sft_head_only=head_only, interleave_rl=interleave, save_best=True,
                )
                config_dir = Path(__file__).resolve().parents[1] / "verl" / "verl" / "trainer" / "config"
                with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                    config = compose(config_name="ppo_trainer", overrides=command[3:])
                self.assertTrue(config.trainer.method_e)
                self.assertTrue(config.actor_rollout_ref.actor.method_e)
                self.assertTrue(config.actor_rollout_ref.actor.fsdp_config.use_orig_params)
                self.assertEqual(config.actor_rollout_ref.actor.sft_head_only, head_only)
                self.assertEqual(config.trainer.interleave_rl, interleave)
                self.assertEqual(config.actor_rollout_ref.rollout.n, 8)

    def test_method_d_requires_baseline_checkpoint_and_rejects_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "baseline checkpoint"):
            self.launch(
                method=Method(name="method_d", template_variant="baseline"),
                method_name="method_d", base_model=None,
            )
        with self.assertRaisesRegex(ValueError, "requires"):
            self.launch(interleave_rl=True)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.launch(
                method=Method(name="method_d", template_variant="baseline"),
                method_name="method_d", sft_stride=0,
            )

    def test_method_d_uses_method_ac_prompt_paths_including_validation_shards(self):
        with patch("pipeline.core.method.DATA_ROOT", self.root / "data"):
            method_ac = Method(name="method_ac", template_variant="baseline")
            train_path = method_ac.formatted_path("sql", "rl_train")
            val_path = method_ac.formatted_path("sql", "rl_val")
            train_path.parent.mkdir(parents=True)
            train_path.touch()
            val_shard = val_path.with_name(f"{val_path.stem}.shard000.parquet")
            val_shard.touch()
            command = self.launch(
                method=Method(name="method_d", template_variant="baseline"),
                method_name="method_d", train_prompts_path=None,
            )
        self.assertIn(f"data.train_files={train_path}", command)
        self.assertIn(f"data.val_files={val_shard}", command)

    def test_methods_d_and_e_use_full_method_ac_splits_including_shards(self):
        for method_name, sharded in (
            ("method_d", False), ("method_d", True), ("method_e", False), ("method_e", True),
        ):
            with (
                self.subTest(method=method_name, sharded=sharded),
                patch("pipeline.core.method.DATA_ROOT", self.root / method_name / str(sharded)),
            ):
                method_ac = Method(name="method_ac", template_variant="baseline")
                train_path = method_ac.formatted_path("sql", "rl_train")
                val_path = method_ac.formatted_path("sql", "rl_val")
                train_path.parent.mkdir(parents=True)
                paths = [
                    path.with_name(f"{path.stem}.shard000.parquet") if sharded else path
                    for path in (train_path, val_path)
                ]
                for path in paths:
                    path.touch()
                command = self.launch(
                    method=Method(name=method_name, template_variant="baseline"),
                    method_name=method_name, train_prompts_path=None,
                )
                self.assertIn(f"data.train_files={paths[0]}", command)
                self.assertIn(f"data.val_files={paths[1]}", command)

    def test_methods_d_and_e_preserve_explicit_prompt_overrides(self):
        for method_name in ("method_d", "method_e"):
            with self.subTest(method=method_name):
                command = self.launch(
                    method=Method(name=method_name, template_variant="baseline"),
                    method_name=method_name, val_prompts_path=self.prompts,
                )
                self.assertIn(f"data.train_files={self.prompts}", command)
                self.assertIn(f"data.val_files={self.prompts}", command)

    def test_custom_name_is_unchanged(self):
        command = self.launch(experiment_name="my-experiment")
        self.assertIn("trainer.experiment_name=my-experiment", command)

    def test_automatic_folder_exactly_matches_dated_wandb_name(self):
        method = Method(name="method_b", template_variant="hint")
        command = self.launch(
            method=method,
            method_name="method_b",
            data_name="sql_2",
            output_path=None,
            base_model=None,
        )
        name = "10_21-method_b-qwen3-4b"
        group = self.root / "models" / "sql_2"
        self.assertIn(f"trainer.experiment_name={name}", command)
        self.assertIn(f"trainer.default_local_dir={group / 'method_b_rl' / name / 'checkpoints'}", command)
        self.assertIn(f"trainer.rollout_data_dir={group / 'method_b_rl' / name / 'rollouts'}", command)
        self.assertIn(f"actor_rollout_ref.model.path={group / 'method_b_sft' / 'qwen3-4b' / 'model'}", command)
        self.assertTrue((group / "method_b_rl" / name / "model").is_dir())

    def test_custom_automatic_folder_matches_custom_wandb_name(self):
        method = Method(name="method_b", template_variant="hint")
        command = self.launch(method=method, method_name="method_b", output_path=None, experiment_name="custom-run")
        expected = self.root / "models" / "sql" / "method_b_rl" / "custom-run" / "checkpoints"
        self.assertIn("trainer.experiment_name=custom-run", command)
        self.assertIn(f"trainer.default_local_dir={expected}", command)

    def test_continuing_dated_folder_keeps_saved_name_and_sft_input(self):
        name = "10_20-method_b-qwen3-4b"
        root = self.root / "models" / "sql" / "method_b_rl" / name
        checkpoints = root / "checkpoints"
        (checkpoints / "global_step_25").mkdir(parents=True)
        config = checkpoints / ".hydra" / "config.yaml"
        config.parent.mkdir()
        sft_model = str(self.root / "models" / "sql" / "method_b_sft" / "qwen3-4b" / "model")
        config.write_text(
            yaml.safe_dump(
                {
                    "trainer": {"experiment_name": name},
                    "actor_rollout_ref": {"model": {"path": sft_model}},
                }
            )
        )
        command = self.launch(
            method=Method(name="method_b", template_variant="hint"),
            method_name="method_b",
            run_id=name,
            base_model=None,
            output_path=None,
            continue_run=True,
        )
        self.assertIn(f"trainer.experiment_name={name}", command)
        self.assertIn(f"trainer.default_local_dir={checkpoints}", command)
        self.assertIn(f"actor_rollout_ref.model.path={sft_model}", command)
        self.assertEqual([p.name for p in root.parent.iterdir()], [name])

    def test_resume_path_does_not_create_a_new_automatic_folder(self):
        self.saved_run("10_19-method_b-qwen3-4b")
        command = self.launch(
            method=Method(name="method_b", template_variant="hint"),
            method_name="method_b",
            resume_path=self.checkpoints,
            output_path=None,
        )
        self.assertIn("trainer.experiment_name=10_19-method_b-qwen3-4b", command)
        self.assertIn(f"trainer.default_local_dir={self.checkpoints}", command)
        self.assertFalse((self.root / "models").exists())

    def test_automatic_folder_rejects_path_separators(self):
        with self.assertRaisesRegex(ValueError, "single folder names"):
            self.launch(
                method=Method(name="method_b", template_variant="hint"),
                method_name="method_b",
                output_path=None,
                experiment_name="../outside",
            )

    def test_continue_run_preserves_original_date(self):
        self.saved_run("10_20-method_b-qwen3-4b")
        command = self.launch(continue_run=True)
        self.assertIn("trainer.experiment_name=10_20-method_b-qwen3-4b", command)
        self.assertIn("trainer.resume_mode=auto", command)

    def test_resume_path_preserves_saved_name(self):
        self.saved_run("10_19-method_b-qwen3-4b")
        command = self.launch(resume_path=self.checkpoints)
        self.assertIn("trainer.experiment_name=10_19-method_b-qwen3-4b", command)

    def test_custom_resume_name_takes_precedence(self):
        self.saved_run("10_19-method_b-qwen3-4b")
        command = self.launch(resume_path=self.checkpoints, experiment_name="custom-resume")
        self.assertIn("trainer.experiment_name=custom-resume", command)

    def test_legacy_resume_without_saved_config_keeps_previous_naming(self):
        command = self.launch(resume_path=self.checkpoints)
        self.assertIn("trainer.experiment_name=existing-run", command)

    def test_invalid_saved_name_is_rejected(self):
        self.saved_run(None)
        with self.assertRaisesRegex(ValueError, "Invalid trainer.experiment_name"):
            self.launch(continue_run=True)

    def test_multi_metric_selection_disables_periodic_and_worker_recency_saves(self):
        command = self.launch(save_best=True, save_hint_bests=True)
        self.assertIn("+trainer.save_hint_bests=True", command)
        self.assertIn("trainer.save_freq=-1", command)
        self.assertIn("trainer.test_freq=25", command)
        self.assertIn("trainer.max_actor_ckpt_to_keep=null", command)
        self.assertIn("+trainer.save_last=True", command)

    def test_multi_metric_selection_requires_save_best(self):
        with self.assertRaisesRegex(ValueError, "requires --save-best"):
            self.launch(save_hint_bests=True, save_best=False)

    def test_exports_each_winner_and_deduplicates_matching_steps(self):
        self.saved_run("10_21-method_b-qwen3-4b")
        for step in (25, 50, 75, 100):
            (self.checkpoints / f"global_step_{step}" / "actor").mkdir(parents=True)
        records = {
            "best": {"step": 25},
            "best_hint_0": {"step": 50},
            "best_combined": {"step": 75},
        }
        (self.checkpoints / "best_checkpoint.json").write_text(json.dumps(records["best"]))
        (self.checkpoints / "best_checkpoints.json").write_text(json.dumps(records))

        def convert(actor, output_path):
            output_path.mkdir()
            (output_path / "weights").write_text(str(actor))
            return output_path

        converter = Mock(side_effect=convert)
        self.launch(converter=converter, continue_run=True, save_best=True, save_hint_bests=True)
        self.assertEqual(converter.call_count, 4)
        for label, step in (("best", 25), ("best_hint_0", 50), ("best_combined", 75), ("last", 100)):
            self.assertEqual(
                (self.run_dir / label / "weights").read_text(), str(self.checkpoints / f"global_step_{step}" / "actor")
            )
        self.assertEqual((self.run_dir / "model").readlink(), Path("best"))

        records["best_hint_0"]["step"] = 25
        records["best_combined"]["step"] = 100
        (self.checkpoints / "best_checkpoints.json").write_text(json.dumps(records))
        # The multi-metric manifest is authoritative even if interruption left
        # the legacy single-metric record behind.
        (self.checkpoints / "best_checkpoint.json").write_text(json.dumps({"step": 50}))
        converter.reset_mock()
        self.launch(converter=converter, continue_run=True, save_best=True, save_hint_bests=True)
        self.assertEqual(converter.call_count, 2)
        self.assertEqual((self.run_dir / "best_hint_0").readlink(), Path("best"))
        self.assertEqual((self.run_dir / "last").readlink(), Path("best_combined"))
        self.assertEqual(
            (self.run_dir / "best" / "weights").read_text(), str(self.checkpoints / "global_step_25" / "actor")
        )


if __name__ == "__main__":
    unittest.main()
