import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.commands.data import create_prompts, stitch_composed_sft
from pipeline.core.composed import ComposedTask
from pipeline.core.io import load_json, load_parquet_records, load_prompts, save_json
from pipeline.core.method import Method
from pipeline.tasks import get_task

REPO = Path(__file__).resolve().parents[1]


def load_composed_reward():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "composed_reward", REPO / "verl/recipe/shared/composed_reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def countdown_problems(n):
    return [
        {"index": i, "numbers": [i, 1], "target": i + 1, "answer": f"{i} + 1", "variant": "2_operands"}
        for i in range(n)
    ]


class ComposedPromptsTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.data_root = Path(temp_dir.name)
        root_patch = patch("pipeline.core.method.DATA_ROOT", self.data_root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def make(self, split, n, k):
        partition = self.data_root / "countdown" / "problems" / f"{split}.json"
        save_json(partition, countdown_problems(n))
        before = partition.read_text()
        output = create_prompts("countdown", method_name="baseline_composed", split_name=split, compose_k=k)
        self.assertEqual(partition.read_text(), before)
        return output

    def test_groups_are_disjoint_and_named_by_k(self):
        output = self.make("eval", 10, 3)
        self.assertEqual(output.name, "eval__baseline_composed__k3.json")
        records = load_json(output)
        self.assertEqual(len(records), 3)
        used = [i for r in records for i in r["source_indices"]]
        self.assertEqual(len(used), len(set(used)))
        content = records[0]["prompt"][1]["content"]
        self.assertIn("Subpart 3:", content)
        self.assertIn("<answer_3> ... </answer_3>", content)

    def test_parquet_renders_like_json(self):
        output = self.make("rl_val", 4, 2)
        method = Method.load("baseline_composed", "countdown")
        rendered = load_prompts(output, template=method.load_template("countdown", "rl"))
        self.assertEqual(list(load_parquet_records(output)[0]["primitive"])[-1], "subproblems")
        self.assertNotIn("{subproblems}", rendered[0]["prompt"][0]["content"])
        self.assertIn("Subpart 2:", rendered[0]["prompt"][0]["content"])

    def test_ac_expands_fixed_groups_into_every_hint_subset_on_all_splits(self):
        for split in ("sft_train", "sft_val", "rl_train", "rl_val", "eval"):
            for k in (1, 2, 3):
                with self.subTest(split=split, k=k):
                    problems = countdown_problems(7)
                    for p in problems:
                        p["hint_exprs"] = ["partial", p["answer"]]
                    partition = self.data_root / "countdown" / "problems" / f"{split}.json"
                    save_json(partition, problems)
                    before = partition.read_text()
                    baseline_path = create_prompts(
                        "countdown", "baseline_composed", split_name=split, compose_k=k)
                    baseline = (load_parquet_records(baseline_path) if split.startswith("rl")
                                else load_json(baseline_path))
                    output = create_prompts(
                        "countdown", "method_ac_composed", split_name=split, compose_k=k)
                    records = load_parquet_records(output) if split.startswith("rl") else load_json(output)
                    self.assertEqual(partition.read_text(), before)
                    self.assertEqual(len(records), (7 // k) * 2**k)
                    self.assertEqual(len({r["index"] for r in records}), len(records))
                    expected_subsets = [
                        list(subset) for n in range(k + 1)
                        for subset in itertools.combinations(range(1, k + 1), n)
                    ]
                    for group_index, base in enumerate(baseline):
                        rows = [r for r in records if r["source_index"] == group_index]
                        self.assertEqual([r["hinted_subparts"] for r in rows], expected_subsets)
                        for row in rows:
                            self.assertEqual(row["source_indices"], base["source_indices"])
                            self.assertEqual(row["ground_truth"], base["ground_truth"])
                            self.assertEqual(row["hint_level"], len(row["hinted_subparts"]))
                            if not row["hinted_subparts"]:
                                self.assertEqual(row["hint_sequence"], "No subpart solutions provided")
                            for i in range(1, k + 1):
                                label = f"Solution to Subpart {i}:\n"
                                self.assertEqual(label in row["hint_sequence"], i in row["hinted_subparts"])
                                if i in row["hinted_subparts"]:
                                    self.assertIn(label + problems[row["source_indices"][i - 1]]["answer"],
                                                  row["hint_sequence"])
                            self.assertNotIn("partial", row["hint_sequence"].replace("No subpart", ""))
                            if split.startswith("rl"):
                                self.assertEqual(row["extra_info"]["index"], row["index"])
                                self.assertNotIn("interaction_kwargs", row["extra_info"])


    def test_ac_math_solutions_accumulate_and_parquet_matches_json(self):
        problems = [
            {"index": i, "problem": f"Find x in the set {{1, 2}} for problem {i}.",
             "answer": "2", "level": "Level 1", "variant": "Algebra",
             "prefix_hints": {"hint_1": f"First step {i}.", "hint_2": f"Final step {i}."}}
            for i in (23, 42)
        ]
        for split in ("sft_train", "rl_train"):
            save_json(self.data_root / "math" / "problems" / f"{split}.json", problems)
            output = create_prompts("math", "method_ac_composed", split_name=split, compose_k=2)
            method = Method.load("method_ac_composed", "math")
            method.set_compose_k(2)
            records = load_prompts(output, template=method.load_template("math", split.split("_")[0]),
                                   system_message=get_task("math").system_message)
            for row in records:
                user = next(msg["content"] for msg in row["prompt"] if msg["role"] == "user")
                self.assertIn("{1, 2}", user)
                self.assertNotIn("{hint_sequence}", user)
                for i in row["hinted_subparts"]:
                    source = row["source_indices"][i - 1]
                    self.assertIn(f"Solution to Subpart {i}:\nFirst step {source}.\nFinal step {source}.", user)
            if split == "sft_train":
                json_records = records
            else:
                self.assertEqual([r["prompt"] for r in records], [r["prompt"] for r in json_records])
                self.assertEqual(list(load_parquet_records(output)[0]["primitive"])[-1], "subproblems")


    def test_ac_math_stitch_matches_source_with_different_hints_and_indices(self):
        task = get_task("math")
        problems = [
            {"index": i, "problem": f"Compute {i} + 1.", "answer": str(i + 1),
             "level": "Level 1", "variant": "Algebra",
             "prefix_hints": {"hint_1": f"New solution for {i}."}}
            for i in (23, 42)
        ]
        save_json(self.data_root / "math" / "problems" / "sft_train.json", problems)
        prompts_path = create_prompts("math", "method_ac_composed", split_name="sft_train", compose_k=2)
        prompts = load_json(prompts_path)
        source = []
        for i, problem in enumerate(problems):
            old_problem = {**problem, "prefix_hints": {"hint_1": "Old hints.", "hint_2": "More hints."}}
            source.append({
                "index": i, "ground_truth": task.get_ground_truth(old_problem),
                "generation": f"Add one.</think><answer>{problem['answer']}</answer>",
                "correct": True,
            })
        source.insert(0, {
            **source[0], "index": 23,
            "ground_truth": {**source[0]["ground_truth"], "problem": "Different problem."},
            "generation": "Unrelated reasoning.</think><answer>24</answer>",
        })
        src = self.data_root / "baseline.json"
        save_json(src, source)
        output = stitch_composed_sft("math", "method_ac_composed", 2, "sft_train", src)
        records = load_json(output)
        self.assertEqual(len(records), 4)
        for record, prompt in zip(records, prompts):
            self.assertEqual(record["prompt"], prompt["prompt"])
            self.assertEqual(record["ground_truth"], prompt["ground_truth"])
            self.assertNotIn("Unrelated reasoning", record["generation"])
            self.assertTrue(record["correct"])


    def test_ac_stitch_consumes_exact_prompt_rows_and_skips_incomplete_groups(self):
        task = get_task("countdown")
        problems = countdown_problems(6)
        for p in problems:
            p["hint_exprs"] = [p["answer"]]
        save_json(self.data_root / "countdown" / "problems" / "sft_train.json", problems)
        prompts_path = create_prompts(
            "countdown", "method_ac_composed", split_name="sft_train", compose_k=2, seed=7)
        prompts = load_json(prompts_path)
        # Keep only three subsets of the first group, in a non-default order.
        prompts = [prompts[3], prompts[1], prompts[0], *prompts[4:]]
        save_json(prompts_path, prompts)
        complete = prompts[0]["source_indices"]
        # Other solved problems belong to different incomplete groups; never regroup them.
        solved = [*complete, prompts[3]["source_indices"][0], prompts[7]["source_indices"][0]]
        source = [
            {"index": i, "ground_truth": task.get_ground_truth(problems[i]),
             "generation": f"Unaided reasoning for {i}.</think><answer>{i} + 1</answer>",
             "correct": True, "token_count": 10}
            for i in solved
        ]
        source.append({**source[0], "generation": source[0]["generation"].replace("Unaided", "Other")})
        missing = next(i for i in range(6) if i not in solved)
        # A mislabeled correct record must not make an incomplete group eligible.
        source.append({
            "index": missing, "ground_truth": task.get_ground_truth(problems[missing]),
            "generation": "Wrong reasoning.</think><answer>0</answer>", "correct": True,
        })
        src = self.data_root / "baseline.json"
        save_json(src, source)
        before = prompts_path.read_text()
        output = stitch_composed_sft(
            "countdown", "method_ac_composed", 2, "sft_train", src, seed=99)
        records = load_json(output)
        self.assertEqual(prompts_path.read_text(), before)
        self.assertEqual(len(records), 3)
        for record, prompt in zip(records, prompts[:3], strict=True):
            for field in ("index", "prompt", "source_indices", "ground_truth",
                          "hint_sequence", "hint_level", "hinted_subparts"):
                self.assertEqual(record[field], prompt[field])
            self.assertTrue(record["correct"])
            self.assertEqual(record["metadata"]["subpart_correct"], [True, True])
            self.assertEqual(record["token_count"], 20)
            for i, index in enumerate(complete, 1):
                self.assertIn(f"Subpart {i}: Unaided reasoning for {index}.", record["generation"])
                self.assertIn(f"<answer_{i}> {index} + 1 </answer_{i}>", record["generation"])
        self.assertEqual(len({r["generation"] for r in records}), 1)
        self.assertEqual(len(ComposedTask(task).filter_for_sft(records)), 3)


    def test_ac_stitch_requires_existing_prompts_and_one_round(self):
        with self.assertRaisesRegex(ValueError, "requires --rounds 1"):
            stitch_composed_sft("countdown", "method_ac_composed", 2, "sft_train",
                                self.data_root / "baseline.json", rounds=2)
        with self.assertRaisesRegex(FileNotFoundError, "Run create_prompts"):
            stitch_composed_sft("countdown", "method_ac_composed", 2, "sft_train",
                                self.data_root / "baseline.json")


    def test_requires_k_and_rejects_unsupported_task(self):
        with self.assertRaises(ValueError):
            create_prompts("countdown", method_name="baseline_composed", split_name="eval")
        with self.assertRaises(FileNotFoundError):
            create_prompts("sql", method_name="baseline_composed", split_name="eval", compose_k=2)


class ComposedGradingTests(unittest.TestCase):
    def test_all_subparts_must_be_correct(self):
        task = get_task("countdown")
        subs = [task.get_ground_truth(p) for p in countdown_problems(2)]
        composed = ComposedTask(task)
        primitive = {"index": 0, "subproblems": subs}

        ok, meta = composed.check_correctness(primitive, "<answer_2> 1 + 1 </answer_2><answer_1> 0 + 1 </answer_1>")
        self.assertTrue(ok)
        ok, meta = composed.check_correctness(primitive, "<answer_1> 0 + 1 </answer_1><answer_2> 1 * 1 </answer_2>")
        self.assertFalse(ok)
        self.assertEqual(meta["subpart_correct"], [True, False])
        ok, meta = composed.check_correctness(primitive, "<answer> 0 + 1 </answer>")
        self.assertEqual(meta["error"], "no_answer_tag")

    def test_ac_training_routes_to_baseline_composed_reward_and_rl_partitions(self):
        from pipeline.commands.training import train_rl

        class LaunchCaptured(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("pipeline.core.method.DATA_ROOT", root):
                method = Method.load("method_ac_composed", "countdown")
                method.set_compose_k(2)
                for split in ("rl_train", "rl_val"):
                    path = method.formatted_path("countdown", split)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                with patch("subprocess.run", side_effect=LaunchCaptured) as launch, \
                     self.assertRaises(LaunchCaptured):
                    train_rl("countdown", method_name="method_ac_composed", compose_k=2,
                             run_id="test", base_model=str(root / "base"),
                             output_path=root / "output", wandb=False)
                command = launch.call_args.args[0]
                self.assertIn(f"custom_reward_function.path={REPO}/verl/recipe/shared/composed_reward.py", command)
                self.assertIn("custom_reward_function.name=compute_score", command)
                self.assertIn("+custom_reward_function.reward_kwargs.base_reward_name=compute_score", command)
                self.assertIn("+custom_reward_function.reward_kwargs.compose_aggregate=min", command)
                self.assertTrue(any("rl_train__method_ac_composed__k2.parquet" in arg for arg in command))
                self.assertTrue(any("rl_val__method_ac_composed__k2.parquet" in arg for arg in command))
                self.assertFalse(any("subpart_requests=" in arg for arg in command))
                self.assertFalse(any("hint_penalty=" in arg for arg in command))



class ComposedACRunnerTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.log = self.root / "commands.jsonl"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "python"
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['COMMAND_LOG'], 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        stub.chmod(0o755)
        self.env = dict(os.environ)
        for key in ("SFT_MODEL", "COMPOSE_K", "COMPOSE_AGGREGATE", "BASELINE_SFT_DIR",
                    "MAX_RESPONSE_LENGTH", "SFT_MAX_LENGTH"):
            self.env.pop(key, None)
        self.env.update(PATH=f"{bin_dir}:{os.environ['PATH']}", COMMAND_LOG=str(self.log))

    def commands(self):
        return [json.loads(line)[2:] for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def run_script(self, name, args, **env):
        return subprocess.run(
            ["bash", str(REPO / "launch_scripts" / name), *args], cwd=self.root,
            env={**self.env, **env}, capture_output=True, text=True, check=False,
        )

    def test_training_selects_latest_dated_single_problem_baseline(self):
        for group, run in (
            ("baseline_rl", "10_05-baseline-qwen3-4b"),
            ("baseline_rl", "11_17-baseline-qwen3-4b"),
            ("baseline_rl", "11_25-baseline-qwen2.5-3b"),
            ("baseline_rl", "qwen3-4b"),
            ("baseline_composed_rl", "11_25-baseline_composed-qwen3-4b-k2"),
        ):
            model = self.root / "models/countdown" / group / run / "model"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}")
        result = self.run_script("run_train_composer_ac.sh",
                                 ["countdown", "qwen3-4b", "unused", "countdown", "true", "2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertEqual([c[0] for c in commands], ["train_sft", "create_prompts", "create_prompts", "train_rl"])
        sft, *_, rl = commands
        self.assertEqual(sft[sft.index("--base-model") + 1],
                         "models/countdown/baseline_rl/11_17-baseline-qwen3-4b/model")
        self.assertIn("--strip-think-tokens", sft)
        self.assertIn("data/countdown/sft_datasets/sft_train__method_ac_composed__k2__stitched.json", sft)
        self.assertIn("compose_aggregate=min", rl)
        self.assertIn("models/countdown/method_ac_composed_sft/qwen3-4b-k2-stitched/model", rl)
        from pipeline.__main__ import main
        with patch("pipeline.__main__.commands.train_sft") as train_sft, \
             patch("pipeline.__main__.commands.create_prompts"), \
             patch("pipeline.__main__.commands.train_rl") as train_rl:
            for command in commands:
                with patch.object(sys, "argv", ["pipeline", *command]):
                    main()
        self.assertEqual(train_sft.call_args.kwargs["compose_k"], 2)
        self.assertEqual(train_rl.call_args.kwargs["reward_kwargs_overrides"], {"compose_aggregate": "min"})

    def test_training_fails_without_dated_baseline_and_can_skip_sft(self):
        args = ["math", "qwen2.5-3b", "unused", "math_o3"]
        result = self.run_script("run_train_composer_ac.sh", args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No dated baseline_rl checkpoint", result.stderr)
        self.assertEqual(self.commands(), [])
        result = self.run_script("run_train_composer_ac.sh", args, SFT_MODEL="existing/model")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([c[0] for c in self.commands()], ["create_prompts", "create_prompts", "train_rl"])
        self.assertIn("existing/model", self.commands()[-1])

    def test_data_runner_uses_existing_baseline_sources_and_fixed_stitching(self):
        sources = self.root / "data/countdown/sft_datasets"
        sources.mkdir(parents=True)
        (sources / "sft_train__baseline.json").write_text("[]")
        (sources / "sft_val__baseline.shard000.json").write_text("[]")
        result = self.run_script("run_sft_data_composer_ac.sh", ["countdown", "2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertEqual([c[0] for c in commands],
                         ["create_prompts", "stitch_composed_sft", "create_prompts", "stitch_composed_sft"])
        self.assertEqual([c[c.index("--source") + 1] for c in commands if c[0] == "stitch_composed_sft"], [
            "data/countdown/sft_datasets/sft_train__baseline.json",
            "data/countdown/sft_datasets/sft_val__baseline.json",
        ])
        from pipeline.__main__ import main
        with patch("pipeline.__main__.commands.create_prompts"), \
             patch("pipeline.__main__.commands.stitch_composed_sft") as stitch:
            for command in commands:
                with patch.object(sys, "argv", ["pipeline", *command]):
                    main()
        self.assertEqual(stitch.call_count, 2)
        self.assertTrue(all(c.kwargs["method_name"] == "method_ac_composed" and
                            c.kwargs["rounds"] == 1 for c in stitch.call_args_list))

    def test_runners_preserve_existing_sharded_prompt_datapoints(self):
        sources = self.root / "data/countdown/sft_datasets"
        sources.mkdir(parents=True)
        prompts = self.root / "data/countdown/problems_with_format"
        prompts.mkdir(parents=True)
        for split in ("sft_train", "sft_val"):
            (sources / f"{split}__baseline.json").write_text("[]")
            (prompts / f"{split}__method_ac_composed__k2.shard000.json").write_text("[]")
        for split in ("rl_train", "rl_val"):
            (prompts / f"{split}__method_ac_composed__k2.shard000.parquet").touch()
        result = self.run_script("run_sft_data_composer_ac.sh", ["countdown", "2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_script("run_train_composer_ac.sh",
                                 ["countdown", "qwen2.5-3b", "unused", "countdown"],
                                 SFT_MODEL="existing/model")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([c[0] for c in self.commands()],
                         ["stitch_composed_sft", "stitch_composed_sft", "train_rl"])



class ComposedRewardTests(unittest.TestCase):
    def setUp(self):
        self.reward = load_composed_reward()
        self.gt = {"num_subparts": 2, "subproblems": [
            {"numbers": [3, 1], "target": 4}, {"numbers": [5, 2], "target": 3}]}
        self.kwargs = {"base_reward_path": str(REPO / "verl/recipe/countdown/reward_function.py")}

    def score(self, answers, **kwargs):
        sol = "reasoning</think>\n\n" + answers
        return self.reward.compute_score("countdown", sol, self.gt, {}, **self.kwargs, **kwargs)

    def test_scores_each_subpart_with_base_scorer(self):
        both = self.score("<answer_1>3 + 1</answer_1>\n<answer_2>5 - 2</answer_2>")
        self.assertEqual((both["score"], both["correct"], both["n_subparts_correct"]), (1.0, True, 2))
        one = self.score("<answer_1>3 + 1</answer_1>\n<answer_2>5 + 2</answer_2>")
        self.assertEqual((one["score"], one["correct"], one["n_subparts_correct"]), (0.1, False, 1))
        mean = self.score("<answer_1>3 + 1</answer_1>\n<answer_2>5 + 2</answer_2>", compose_aggregate="mean")
        self.assertAlmostEqual(mean["score"], 0.55)
        self.assertEqual(sorted(both), sorted(one))

    def test_rejects_malformed_answer_blocks(self):
        for answers in ("<answer_2>5 - 2</answer_2><answer_1>3 + 1</answer_1>",
                        "<answer_1>3 + 1</answer_1>",
                        "<answer_1>3 + 1</answer_1> junk <answer_2>5 - 2</answer_2>",
                        "<answer>3 + 1</answer>"):
            result = self.score(answers)
            self.assertEqual((result["score"], result["malformed"]), (0, True), answers)


if __name__ == "__main__":
    unittest.main()
