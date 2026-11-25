# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Turn rules and driver equivalence for the multi-turn hint rollout.

No GPU and no vLLM: the engine is a script of canned generations and the
tokenizer is character-level, so a token is a character and truncating at
max_tokens is exactly truncating the string. That correspondence is what lets
the tests exercise budget behaviour honestly.

The central claim under test is that run_continuous is a scheduling change and
nothing more -- given the same generations it must produce byte-identical tapes,
masks and flags to run_rounds.
"""

import random

import pytest

from verl.workers.rollout.hint_loop import (
    Generation,
    HintLoop,
    HintLoopConfig,
    LoopEngine,
)


class CharTokenizer:
    """One token per character, so token counts and string lengths agree."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


TOKENIZER = CharTokenizer()


def prompt_for(index):
    """Prompts carry their rollout index so the scripted engine can look up
    which generation comes next without being told."""
    return [ord(c) for c in f"P{index}|"]


def index_of(prompt):
    text = "".join(chr(t) for t in prompt)
    return int(text[1:text.index("|")])


class ScriptedEngine(LoopEngine):
    """Replays a fixed script of generations, one entry per rollout per turn.

    ``step_size`` and ``shuffle`` control how the continuous driver sees
    completions: the point of the equivalence test is that neither the number
    of completions per step nor their order changes the result.
    """

    def __init__(self, script, answers=None, step_size=3, shuffle=False, seed=0):
        self.script = script
        self.answers = answers or {}
        self.turns = {}
        self.queue = []
        self.step_size = step_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.max_concurrent = 0
        self.batch_sizes = []

    def _generate(self, prompt, max_tokens, mode):
        index = index_of(prompt)
        if mode == "answer":
            text = self.answers.get(index, "42</answer>")
        else:
            turn = self.turns.get(index, 0)
            self.turns[index] = turn + 1
            texts = self.script.get(index, [])
            text = texts[turn] if turn < len(texts) else ""
        text = text[:max_tokens]
        return Generation(text=text, token_ids=[ord(c) for c in text])

    # -- round driver ------------------------------------------------------
    def generate_batch(self, prompts, max_tokens, mode="turn"):
        if mode == "turn":
            self.batch_sizes.append(len(prompts))
        return [self._generate(p, m, mode) for p, m in zip(prompts, max_tokens)]

    # -- continuous driver -------------------------------------------------
    def submit(self, request_id, prompt, max_tokens):
        self.queue.append((request_id, list(prompt), max_tokens))
        self.max_concurrent = max(self.max_concurrent, len(self.queue))

    def step(self):
        if not self.queue:
            return []
        self.batch_sizes.append(len(self.queue))
        if self.shuffle:
            self.rng.shuffle(self.queue)
        taking, self.queue = self.queue[: self.step_size], self.queue[self.step_size :]
        return [(rid, self._generate(p, m, "turn")) for rid, p, m in taking]

    def busy(self):
        return bool(self.queue)


def make_loop(engine, **overrides):
    config = HintLoopConfig(
        max_hints=overrides.pop("max_hints", 3),
        max_turns=overrides.pop("max_turns", 4),
        max_model_len=overrides.pop("max_model_len", 4096),
        max_response_len=overrides.pop("max_response_len", 1024),
        answer_budget=overrides.pop("answer_budget", 0),
        nested_request=overrides.pop("nested_request", False),
        subpart_requests=overrides.pop("subpart_requests", 0),
    )
    assert not overrides, f"unknown overrides: {overrides}"
    return HintLoop(engine=engine, tokenizer=TOKENIZER, config=config)


def run(driver, script, answers=None, engine_kwargs=None, **config):
    engine = ScriptedEngine(script, answers=answers, **(engine_kwargs or {}))
    loop = make_loop(engine, **config)
    states = loop.make_states(
        [prompt_for(i) for i in range(len(script))],
        [[f"hint{i}-{k}" for k in range(4)] for i in range(len(script))],
    )
    getattr(loop, driver)(states)
    loop.force_answers(states)
    return states, loop, engine


def snapshot(states):
    """Everything downstream of the loop actually reads."""
    return [
        {
            "partials": st.partials,
            "masks": st.masks,
            "num_hints": st.num_hints,
            "last_given": st.last_given,
            "response_len": st.response_len,
            "truncated": st.truncated,
            "forced": st.forced,
            "think_truncated": st.think_truncated,
            "answer_truncated": st.answer_truncated,
            "malformed": st.malformed,
            "completed": st.completed,
            "turns": st.turns,
        }
        for st in states
    ]


# ---------------------------------------------------------------- turn rules


def test_answer_completes_in_one_turn():
    states, _, _ = run("run_rounds", {0: ["thinking...</think><answer>7</answer>"]})
    st = states[0]
    assert st.completed and st.num_hints == 0
    assert st.truncated == 0.0 and not st.malformed
    assert st.masks == [[1] * len(st.partials[1])]


def test_valid_request_injects_hint_and_continues():
    script = {0: ["stuck <request></request>", "<answer>7</answer>"]}
    states, _, _ = run("run_rounds", script)
    st = states[0]
    assert st.num_hints == 1 and st.last_given == 0
    # prompt, generation, injected hint, generation
    assert len(st.partials) == 4
    assert TOKENIZER.decode(st.partials[2]) == "\n<response>hint0-0</response>\n<think>\n"
    # The injected turn is not trained on.
    assert set(st.masks[1]) == {0}


def test_injected_hint_is_exempt_from_the_response_budget():
    script = {0: ["<request></request>", "<answer>7</answer>"]}
    states, _, _ = run("run_rounds", script)
    st = states[0]
    generated = len(st.partials[1]) + len(st.partials[3])
    assert st.response_len == generated
    # The hint tokens are in the tape but not on the meter.
    assert len(st.partials[2]) > 0


def test_malformed_request_is_terminal():
    states, _, _ = run("run_rounds", {0: ["<request>give me a hint</request>"]})
    st = states[0]
    assert st.malformed and st.completed and st.num_hints == 0


def test_self_authored_response_is_terminal():
    states, _, _ = run("run_rounds", {0: ["<response>I made this up</response>"]})
    st = states[0]
    assert st.malformed and st.completed


def test_max_hints_switches_to_the_refusal_message():
    asks = ["<request></request>"] * 5
    states, _, _ = run("run_rounds", {0: asks}, max_hints=2, max_turns=6)
    st = states[0]
    injected = [TOKENIZER.decode(p) for p in st.partials[2::2]]
    assert injected[0].startswith("\n<response>hint0-0")
    assert injected[1].startswith("\n<response>hint0-1")
    # Past the cap the model is told no and left to answer; note the refusal
    # carries no reopening <think>, unlike a real hint.
    assert injected[2] == "\n<response>No more hints available.</response>\n"
    assert st.last_given == 1
    # num_hints counts requests made, not hints delivered -- the reward
    # function needs the ask count, so refused asks still register.
    assert st.num_hints == 5


def test_hints_run_out_before_max_hints():
    asks = ["<request></request>"] * 6
    states, _, _ = run("run_rounds", {0: asks}, max_hints=6, max_turns=7)
    st = states[0]
    injected = [TOKENIZER.decode(p) for p in st.partials[2::2]]
    # Four scripted hints, then the refusal text with last_given pinned.
    assert injected[4] == "\n<response>No more hints available.</response>\n<think>\n"
    assert states[0].last_given == 3


def test_turn_cap_leaves_the_rollout_truncated():
    script = {0: ["<request></request>"] * 8}
    states, _, _ = run("run_rounds", script, max_hints=8, max_turns=3)
    st = states[0]
    assert st.turns == 3
    assert not st.completed and st.truncated == 1.0


def test_response_budget_exhaustion_truncates():
    # 119 decoded tokens on the first turn, leaving 131 of the 250 budget; the
    # second turn is capped at exactly that and so lands on the limit.
    script = {0: ["x" * 100 + "<request></request>", "y" * 200]}
    states, _, _ = run("run_rounds", script, max_response_len=250)
    st = states[0]
    assert st.truncated == 1.0 and st.completed
    assert st.response_len == 250


def test_generation_is_capped_at_the_remaining_budget():
    script = {0: ["a" * 100 + "<request></request>", "b" * 5000]}
    states, _, _ = run("run_rounds", script, max_response_len=300)
    st = states[0]
    # partials = [prompt, turn 1, injected hint, turn 2]; the second turn gets
    # 300 - 119 tokens and not one more, whatever the model wanted to say.
    assert len(st.partials[3]) == 300 - 119


def test_prompt_longer_than_context_never_generates():
    states, loop, _ = run("run_rounds", {0: ["<answer>7</answer>"]}, max_model_len=2)
    st = states[0]
    assert st.truncated == 1.0 and not st.generated_any
    # Nothing to force an answer onto.
    assert st.forced == 0.0


# ------------------------------------------------------------ forced answers


def test_forced_answer_closes_a_truncated_think_block():
    script = {0: ["reasoning that never ends" + "z" * 300]}
    states, _, _ = run(
        "run_rounds", script, answers={0: "7"},
        max_response_len=120, answer_budget=32,
    )
    st = states[0]
    assert st.forced == 1.0
    tape = TOKENIZER.decode([t for seg in st.partials[1:] for t in seg])
    assert tape.endswith("\n</think>\n\n<answer>7</answer>")
    # Scaffold forced, answer sampled, closing tag forced.
    assert st.masks[-3][0] == 0 and st.masks[-2][0] == 1 and st.masks[-1][0] == 0


def test_forced_answer_does_not_double_the_closing_tag():
    script = {0: ["reasoning" + "z" * 300]}
    states, _, _ = run(
        "run_rounds", script, answers={0: "7</answer>"},
        max_response_len=120, answer_budget=32,
    )
    tape = TOKENIZER.decode([t for seg in states[0].partials[1:] for t in seg])
    assert tape.count("</answer>") == 1


def test_forcing_is_off_without_an_answer_budget():
    script = {0: ["reasoning" + "z" * 300]}
    states, _, _ = run("run_rounds", script, max_response_len=120, answer_budget=0)
    assert states[0].forced == 0.0
    # Nothing rescued it, so the answer is missing: that is answer-truncation.
    assert (states[0].think_truncated, states[0].answer_truncated) == (0.0, 1.0)


# ---------------------------------------------------------- truncation kinds


def test_forced_answer_that_closes_itself_is_think_truncation():
    script = {0: ["reasoning" + "z" * 300]}
    states, _, _ = run(
        "run_rounds", script, answers={0: "7</answer>"},
        max_response_len=120, answer_budget=32,
    )
    st = states[0]
    assert st.forced == 1.0
    assert (st.think_truncated, st.answer_truncated) == (1.0, 0.0)


def test_forced_answer_that_runs_out_is_answer_truncation():
    script = {0: ["reasoning" + "z" * 300]}
    states, _, _ = run(
        "run_rounds", script, answers={0: "7" * 40},  # never reaches </answer>
        max_response_len=120, answer_budget=32,
    )
    st = states[0]
    assert st.forced == 1.0
    assert (st.think_truncated, st.answer_truncated) == (0.0, 1.0)
    tape = TOKENIZER.decode([t for seg in st.partials[1:] for t in seg])
    assert tape.endswith("7" * 32 + "</answer>")


def test_unrescuable_rollout_is_answer_truncation():
    script = {0: ["<request>help" + "z" * 300]}
    states, _, _ = run("run_rounds", script, max_response_len=120, answer_budget=32)
    assert (states[0].think_truncated, states[0].answer_truncated) == (0.0, 1.0)


def test_completed_rollout_is_not_truncated():
    states, _, _ = run("run_rounds", {0: ["<answer>7</answer>"]}, answer_budget=32)
    st = states[0]
    assert (st.truncated, st.think_truncated, st.answer_truncated) == (0.0, 0.0, 0.0)


def test_unrescuable_rollout_is_left_alone():
    # An open <request> with text inside fails the validator's tightness check.
    script = {0: ["<request>help" + "z" * 300]}
    states, _, _ = run(
        "run_rounds", script, max_response_len=120, answer_budget=32,
    )
    assert states[0].forced == 0.0


class RecordingEngine(ScriptedEngine):
    """Remembers the prompt lengths the forced-answer pass actually submitted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_prompts = []

    def generate_batch(self, prompts, max_tokens, mode="turn"):
        if mode == "answer":
            self.answer_prompts.extend(len(p) for p in prompts)
        return super().generate_batch(prompts, max_tokens, mode)


def test_forced_answer_skips_a_tape_that_fills_the_context_window():
    """A tape truncated by the response budget still carries the prompt and any
    injected hints, so tape + scaffold can reach max_model_len. Submitting that
    makes vLLM reject the whole batch, so the rollout has to be left alone.
    """
    engine = RecordingEngine({0: ["reasoning that never ends" + "z" * 300]},
                             answers={0: "7"})
    # 3-token prompt + 120 generated + a 19-token scaffold = 142 >= 140.
    loop = make_loop(engine, max_model_len=140, max_response_len=120,
                     answer_budget=32)
    states = loop.make_states([prompt_for(0)], [["hint"]])
    loop.run_rounds(states)
    loop.force_answers(states)

    st = states[0]
    assert st.truncated == 1.0, "the rollout must be a forcing candidate"
    assert st.forced == 0.0, "no answer can be forced onto an over-length tape"
    assert engine.answer_prompts == [], "nothing over-length was submitted"


def test_forced_answer_still_fires_when_the_window_has_room():
    """The guard above must not switch forcing off in the normal case."""
    script = {0: ["reasoning that never ends" + "z" * 300]}
    states, _, _ = run(
        "run_rounds", script, answers={0: "7"},
        max_response_len=120, answer_budget=32, max_model_len=4096,
    )
    assert states[0].forced == 1.0


def test_nested_scaffold_does_not_reopen_the_think_block():
    script = {0: ["<request></request>", "more thought" + "z" * 400]}
    states, _, _ = run(
        "run_rounds", script, answers={0: "7"},
        max_response_len=200, answer_budget=32, nested_request=True,
    )
    tape = TOKENIZER.decode([t for seg in states[0].partials[1:] for t in seg])
    assert tape.count("<think>") == 0
    assert tape.endswith("\n</think>\n\n<answer>7</answer>")


# ---------------------------------------------------- subpart requests (composed)


def run_subparts(script, answers=None, **config):
    engine = ScriptedEngine(script, answers=answers)
    loop = make_loop(engine, nested_request=True, subpart_requests=2, **config)
    states = loop.make_states([prompt_for(0)], [["sol-1", "sol-2"]])
    loop.run_rounds(states)
    loop.force_answers(states)
    return states[0]


def test_subpart_request_serves_that_subparts_solution():
    st = run_subparts({0: ["<request>2</request>", "<request>1</request>", "<request>2</request>",
                           "</think><answer_1>1</answer_1><answer_2>2</answer_2>"]}, max_turns=5)
    injected = [TOKENIZER.decode(p) for p in st.partials[2::2]]
    assert injected[:2] == ["\n<response>sol-2</response>\n", "\n<response>sol-1</response>\n"]
    # A subpart already served is an exhausted request.
    assert injected[2] == "\n<response>No more hints available.</response>\n"
    assert st.num_hints == 3 and st.completed and not st.malformed


@pytest.mark.parametrize("request_text", ["<request>3</request>", "<request>0</request>",
                                          "<request></request>", "<request>two</request>"])
def test_subpart_request_needs_a_valid_id(request_text):
    st = run_subparts({0: [request_text]})
    assert st.malformed and st.completed and st.num_hints == 0


def test_composed_forced_answer_opens_the_first_answer_tag():
    st = run_subparts({0: ["<request>1</request>", "more thought" + "z" * 400]},
                      answers={0: "4</answer_1><answer_2>5</answer_2>"},
                      max_response_len=200, answer_budget=64)
    tape = TOKENIZER.decode([t for seg in st.partials[1:] for t in seg])
    assert st.forced == 1.0 and st.think_truncated == 1.0
    assert tape.endswith("\n</think>\n\n<answer_1>4</answer_1><answer_2>5</answer_2>")


def scripted_population(n, seed):
    """A population that exercises every exit: answers, asks, malformed
    requests, silent failures and rollouts that run past the turn cap."""
    rng = random.Random(seed)
    script = {}
    for i in range(n):
        turns = []
        for _ in range(rng.randint(1, 6)):
            roll = rng.random()
            filler = "t" * rng.randint(5, 220)
            if roll < 0.40:
                turns.append(filler + "<request></request>")
                continue
            if roll < 0.70:
                turns.append(filler + "</think><answer>" + str(rng.randint(0, 99)) + "</answer>")
                break
            if roll < 0.80:
                turns.append(filler + "<request>tell me</request>")
                break
            if roll < 0.88:
                turns.append(filler + "<response>fake</response>")
                break
            if roll < 0.94:
                turns.append(filler)  # neither a request nor an answer
                break
            turns.append("q" * 900)  # eats the budget
            break
        script[i] = turns
    return script


@pytest.mark.parametrize("seed", range(12))
def test_continuous_matches_rounds(seed):
    script = scripted_population(40, seed)
    answers = {i: "42" for i in range(40)}
    config = dict(max_hints=3, max_turns=4, max_response_len=700, answer_budget=48)

    rounds, _, _ = run("run_rounds", script, answers=answers, **config)
    continuous, _, _ = run(
        "run_continuous", script, answers=answers,
        engine_kwargs=dict(step_size=3, shuffle=True, seed=seed), **config,
    )
    assert snapshot(rounds) == snapshot(continuous)


@pytest.mark.parametrize("step_size,shuffle", [(1, False), (1, True), (7, True), (1000, False)])
def test_continuous_is_insensitive_to_completion_order(step_size, shuffle):
    script = scripted_population(40, 99)
    answers = {i: "42" for i in range(40)}
    config = dict(max_hints=3, max_turns=4, max_response_len=700, answer_budget=48)

    baseline, _, _ = run("run_rounds", script, answers=answers, **config)
    observed, _, _ = run(
        "run_continuous", script, answers=answers,
        engine_kwargs=dict(step_size=step_size, shuffle=shuffle, seed=7), **config,
    )
    assert snapshot(baseline) == snapshot(observed)


def test_continuous_keeps_more_work_in_flight_than_rounds():
    """The whole point: the round driver's batch drains as rollouts finish,
    the continuous driver's does not."""
    script = scripted_population(64, 3)
    config = dict(max_hints=3, max_turns=5, max_response_len=700)

    _, _, round_engine = run("run_rounds", script, **config)
    _, _, cont_engine = run(
        "run_continuous", script, engine_kwargs=dict(step_size=1), **config
    )
    # Rounds: first batch is everyone, later batches are the survivors.
    assert round_engine.batch_sizes[0] == 64
    assert round_engine.batch_sizes[-1] < 64 // 4
    # Continuous: every rollout is submitted before the first step returns.
    assert cont_engine.max_concurrent == 64
