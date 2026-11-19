import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from pipeline.commands.data import create_verification_data
from pipeline.core.io import load_json, save_json
from pipeline.tasks import get_task

REPO = Path(__file__).resolve().parents[1]


class MethodAVerificationDataTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.generations = self.root / "solver.json"
        self.output = self.root / "sft.json"

    def source(self, labels, index=12, hint_level=1):
        return {
            "index": index,
            "source_index": 4,
            "hint_level": hint_level,
            "hint_sequence": "1 + 2" if hint_level else "No partial solution",
            "ground_truth": {"numbers": [1, 2, 3], "target": 6},
            "n_samples": len(labels),
            "n_correct": sum(labels),
            "samples": [
                {"generation": f"rollout {i}", "correct": correct}
                for i, correct in enumerate(labels)
            ],
            "variant": "3_operands",
            "split": "rl_ver_train",
        }

    def create(self, data, **kwargs):
        save_json(self.generations, data)
        create_verification_data(
            "countdown",
            generations_path=self.generations,
            output_path=self.output,
            **kwargs,
        )
        return load_json(self.output)

    def test_retains_all_rollouts_and_both_labels_across_hint_levels(self):
        labels = [True] * 7 + [False] * 3
        sources = [self.source(labels), self.source(labels[::-1], index=13, hint_level=0)]
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                records = self.create({"details": sources} if wrapped else sources)
                self.assertEqual(len(records), 20)
                self.assertEqual([r["index"] for r in records], list(range(20)))
                self.assertEqual([r["label"] for r in records], list(map(int, labels + labels[::-1])))
                for offset, source in ((0, sources[0]), (10, sources[1])):
                    rows = records[offset:offset + 10]
                    self.assertTrue(all(r["prompt"] == rows[0]["prompt"] for r in rows))
                    self.assertIn(source["hint_sequence"], rows[0]["prompt"][1]["content"])
                    for i, record in enumerate(rows):
                        self.assertEqual(record["generation"], f"<answer>{int(source['samples'][i]['correct'])}</answer>")
                        self.assertEqual(record["solver_sample"], source["samples"][i])
                        self.assertEqual(record["sample_index"], i)
                        self.assertEqual(record["prompt_index"], source["index"])
                        self.assertEqual(record["source_index"], 4)
                        self.assertEqual(record["hint_level"], source["hint_level"])
                        self.assertEqual(record["split"], "rl_ver_train")
                self.assertEqual(get_task("countdown").filter_for_sft(records), records)
                self.assertNotEqual(records[0]["prompt"], records[10]["prompt"])

    def test_sample_count_is_configurable_and_uniform_outcomes_are_retained(self):
        sources = [self.source([flag] * 3, index=i) for i, flag in enumerate((True, False))]
        records = self.create({"details": sources}, num_samples=3)
        self.assertEqual([r["label"] for r in records], [1, 1, 1, 0, 0, 0])

    def test_rejects_missing_incomplete_or_inconsistent_rollout_results(self):
        source = self.source([True] * 7 + [False] * 3)
        cases = []
        missing = copy.deepcopy(source)
        del missing["samples"]
        cases.append((missing, "individual samples"))
        incomplete = copy.deepcopy(source)
        incomplete["samples"].pop()
        cases.append((incomplete, "individual samples"))
        for flag in (None, 1, "false"):
            invalid = copy.deepcopy(source)
            invalid["samples"][0]["correct"] = flag
            cases.append((invalid, "correct.*flag"))
        mismatch = copy.deepcopy(source)
        mismatch["n_correct"] = 8
        cases.append((mismatch, "does not match individual"))
        wrong_count = copy.deepcopy(source)
        wrong_count["n_samples"] = 9
        cases.append((wrong_count, "expected num_samples=10"))
        for data, message in cases:
            with self.subTest(message=message, data=data):
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    self.create({"details": [data]})
                self.assertFalse(self.output.exists())

    def test_run_solver_reuses_evaluation_and_expands_individual_rollouts(self):
        source = self.source([True] * 7 + [False] * 3)
        save_json(self.generations, {"details": [source]})
        with patch("pipeline.commands.inference.evaluate") as evaluate:
            create_verification_data(
                "countdown",
                output_path=self.output,
                data_name="countdown",
                run_solver=True,
                solver_run_id="qwen2.5-3b",
                solver_split="rl_ver_train",
                solver_output_path=self.generations,
            )
        evaluate.assert_not_called()
        self.assertEqual([r["label"] for r in load_json(self.output)], [1] * 7 + [0] * 3)


class MethodALaunchTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.log = self.root / "commands.jsonl"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("python", "kubectl"):
            stub = bin_dir / name
            stub.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "name = Path(sys.argv[0]).name\n"
                "with open(os.environ['COMMAND_LOG'], 'a') as log:\n"
                "    log.write(json.dumps({'name': name, 'args': sys.argv[1:], "
                "'stdin': sys.stdin.read() if name == 'kubectl' else ''}) + '\\n')\n"
                "if os.environ.get('FAIL_COMMAND') in sys.argv[1:]:\n"
                "    sys.exit(7)\n"
            )
            stub.chmod(0o755)
        self.env = dict(os.environ)
        for key in ("AC_RUN_ID", "AC_RUN_PREFIX", "RUN_SCRIPT", "TASKS", "MODELS", "DATA_NAME", "FAIL_COMMAND"):
            self.env.pop(key, None)
        self.env.update(PATH=f"{bin_dir}:{os.environ['PATH']}", COMMAND_LOG=str(self.log))

    def make_solver(self, run_id="qwen2.5-3b"):
        model = self.root / "models/countdown/method_ac_rl" / run_id / "model"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")

    def run_runner(self, **env):
        return subprocess.run(
            ["bash", str(REPO / "launch_scripts/run_train_a.sh"),
             "countdown", "qwen2.5-3b", "Qwen/Qwen2.5-3B", "countdown", "true"],
            cwd=self.root, env={**self.env, **env}, capture_output=True, text=True, check=False,
        )

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_runner_generates_both_held_out_splits_then_trains_from_selected_ac(self):
        run_id = "11_10-method_ac-qwen2.5-3b"
        self.make_solver(run_id)
        result = self.run_runner(AC_RUN_ID=run_id)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [entry["args"][2:] for entry in self.commands()]
        self.assertEqual([cmd[0] for cmd in commands], [
            "create_prompts", "create_verification_data",
            "create_prompts", "create_verification_data", "train_sft",
        ])
        from pipeline.__main__ import main

        with patch("pipeline.__main__.commands.create_prompts") as prompts, \
             patch("pipeline.__main__.commands.create_verification_data") as data, \
             patch("pipeline.__main__.commands.train_sft") as train:
            for command in commands:
                with patch.object(sys, "argv", ["pipeline", *command]):
                    main()
        self.assertEqual([c.kwargs["split_name"] for c in prompts.call_args_list],
                         ["rl_ver_train", "rl_ver_val"])
        for split, call in zip(("train", "val"), data.call_args_list, strict=True):
            self.assertEqual(call.kwargs["solver_split"], f"rl_ver_{split}")
            self.assertEqual(call.kwargs["solver_run_id"], run_id)
            self.assertEqual(call.kwargs["num_samples"], 10)
            self.assertTrue(call.kwargs["run_solver"])
            self.assertEqual(call.kwargs["output_path"],
                             Path(f"data/countdown/sft_datasets/sft_{split}__method_a__{run_id}.json"))
        train.assert_called_once()
        self.assertEqual(train.call_args.kwargs["base_model"],
                         f"models/countdown/method_ac_rl/{run_id}/model")
        self.assertEqual(train.call_args.kwargs["run_id"], run_id)
        self.assertTrue(train.call_args.kwargs["completion_only_loss"])
        self.assertEqual(train.call_args.kwargs["dataset_path"], data.call_args_list[0].kwargs["output_path"])
        self.assertEqual(train.call_args.kwargs["eval_dataset_path"], data.call_args_list[1].kwargs["output_path"])

    def test_runner_accepts_sharded_prompts_without_regenerating_them(self):
        self.make_solver()
        prompts = self.root / "data/countdown/problems_with_format"
        prompts.mkdir(parents=True)
        for split in ("train", "val"):
            (prompts / f"rl_ver_{split}__method_ac.shard000.parquet").touch()
        result = self.run_runner()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([entry["args"][2] for entry in self.commands()],
                         ["create_verification_data", "create_verification_data", "train_sft"])

    def test_runner_fails_before_generation_when_ac_model_is_missing(self):
        result = self.run_runner()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Trained Method AC model not found", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_runner_does_not_train_after_generation_failure(self):
        self.make_solver()
        result = self.run_runner(FAIL_COMMAND="create_verification_data")
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("train_sft", [entry["args"][2] for entry in self.commands()])

    def test_existing_launcher_renders_template_without_new_parameters(self):
        result = subprocess.run(
            ["bash", "launch_scripts/launch_train_jobs.sh"], cwd=REPO,
            env=self.env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.commands())
        for entry in self.commands():
            self.assertEqual(entry["args"], ["create", "-f", "-"])
            self.assertNotIn("__", entry["stdin"])
            manifest = yaml.safe_load(entry["stdin"])
            self.assertEqual(manifest["metadata"]["namespace"], "bonete52")
            self.assertEqual(manifest["spec"]["queue"], "bonete52")

    def test_launcher_uses_manually_selected_runner(self):
        scripts = self.root / "launch_scripts"
        scripts.mkdir()
        launcher = (REPO / "launch_scripts/launch_train_jobs.sh").read_text()
        launcher = launcher.replace(
            'RUN_TRAIN_SCRIPT="run_train_ac.sh"',
            'RUN_TRAIN_SCRIPT="run_train_a.sh"',
        )
        (scripts / "launch_train_jobs.sh").write_text(launcher)
        template = (REPO / "launch_scripts/submit_train_template.yaml").read_text()
        (scripts / "submit_train_template.yaml").write_text(template)
        result = subprocess.run(
            ["bash", "launch_scripts/launch_train_jobs.sh"], cwd=self.root,
            env=self.env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.commands())
        for entry in self.commands():
            self.assertNotIn("__", entry["stdin"])
            manifest = yaml.safe_load(entry["stdin"])
            container = manifest["spec"]["tasks"][0]["template"]["spec"]["containers"][0]
            self.assertIn("bash launch_scripts/run_train_a.sh ", container["command"][2])


if __name__ == "__main__":
    unittest.main()
