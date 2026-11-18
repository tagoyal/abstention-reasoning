import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.commands.data import create_prompts
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
