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
"""Multi-turn hint rollout: one set of turn rules, two ways to drive them.

A hint rollout is a conversation the environment joins: the policy thinks, may
emit ``<request></request>``, and the loop answers with ``<response>hint</response>``
before the policy resumes. Deciding what happens after a generation stops --
whether it asked, answered, malformed itself, or ran out of budget -- is the
same work no matter how the generations are scheduled, so it lives once here in
:meth:`HintLoop.ingest`.

Two drivers consume those rules:

``run_rounds``
    The original schedule. Every unfinished rollout generates, the driver waits
    for the whole batch, then all hints are injected together. Simple, but a
    round costs as long as its slowest member and the batch shrinks with each
    round -- by turn 4 a handful of rollouts hold the GPU alone.

``run_continuous``
    The same rules with no barrier. Rollouts are submitted to the engine
    individually and re-submitted the moment they come back needing a hint, so
    the engine always has the full population of unfinished rollouts to batch
    over. ``LLM.generate`` is itself a ``while has_unfinished_requests(): step()``
    loop, so driving ``step()`` by hand costs nothing extra -- it only removes
    the barrier that ``generate`` puts at the end of each round.

The module deliberately imports neither vLLM nor torch: the drivers talk to a
:class:`LoopEngine`, which the rollout worker implements over vLLM and the tests
implement over a script of canned generations.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Generation",
    "HintLoop",
    "HintLoopConfig",
    "LoopEngine",
    "RolloutState",
    "hint_injection",
]


# Below this many tokens of headroom a generation is not worth launching: it
# cannot reach an answer, and a zero-length request is an error in vLLM.
MIN_GENERATION_BUFFER = 64

NO_MORE_HINTS = "\n<response>No more hints available.</response>\n"

# Structural tags recognised by the task's reward validator. Kept in sync with
# has_malformed_structure in recipe/*/reward_function.py -- a forced completion
# is only worth anything if it lands somewhere that accepts.
TAG_PATTERN = r"(</think>|<think>|<request>|</request>|<response>|</response>|<answer>|</answer>|<abstain>)"
HINT_CYCLE = ["<request>", "</request>", "<response>", "</response>", "<think>"]
# Inline requests never close the think block, so no reopening tag follows.
NESTED_HINT_CYCLE = ["<request>", "</request>", "<response>", "</response>"]

VALID_REQUEST = "<request></request>"
_VALID_REQUEST_RE = re.compile(re.escape(VALID_REQUEST))
# Composed hint methods name the subpart: <request>i</request>, 1 <= i <= K.
_SUBPART_REQUEST_RE = re.compile(r"<request>(\d+)</request>")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def hint_injection(hint_text: str, nested: bool) -> str:
    """The environment's turn: the hint, plus a reopened <think> when the request
    was made outside the block. Inline requests never left it, so reopening one
    there would nest a second block and fail the structure validator."""
    if nested:
        return "\n<response>" + hint_text + "</response>\n"
    return "\n<response>" + hint_text + "</response>\n<think>\n"


@dataclass
class Generation:
    """What a driver needs back from the engine: the text and the exact tokens.

    The tokens are kept rather than re-tokenized because the response tensor is
    trained on directly; re-tokenizing the text would shift boundaries between
    what was sampled and what is learned from.
    """

    text: str
    token_ids: List[int]


class LoopEngine:
    """The generation calls the drivers make. Implemented over vLLM in the
    rollout worker and over a script in the tests."""

    def generate_batch(self, prompts: Sequence[Sequence[int]], max_tokens: Sequence[int],
                       mode: str = "turn") -> List[Generation]:
        """Block until every prompt has generated. ``mode`` is "turn" or "answer"."""
        raise NotImplementedError

    def submit(self, request_id: int, prompt: Sequence[int], max_tokens: int) -> None:
        """Queue one generation. Must not block."""
        raise NotImplementedError

    def step(self) -> List[Tuple[int, Generation]]:
        """Advance the engine one step; return whatever finished, possibly none."""
        raise NotImplementedError

    def busy(self) -> bool:
        """True while any submitted generation is unfinished."""
        raise NotImplementedError


@dataclass
class HintLoopConfig:
    max_hints: int = 6
    max_turns: int = 7
    # The real context window. Injected hint text counts against this.
    max_model_len: int = 8192
    # Decode budget for the policy's own tokens. Injected hint text is exempt.
    max_response_len: int = 2048
    # Held back from max_response_len and spent only on a forced answer, so a
    # rollout that thinks right up to the limit still gets the same room to
    # answer in as one that stopped early.
    answer_budget: int = 0
    # Requests live inside the <think> block rather than after it.
    nested_request: bool = False
    # Composed hint methods: K > 0 makes requests name a subpart,
    # <request>i</request>, served with hints[i-1] (that subpart's solution).
    # A subpart already served gets "No more hints available." instead.
    subpart_requests: int = 0

    @property
    def answer_open(self) -> str:
        """Tag a forced answer starts with: composed answers run <answer_1>..<answer_K>."""
        return "<answer_1>" if self.subpart_requests else "<answer>"

    @property
    def answer_close(self) -> str:
        return f"</answer_{self.subpart_requests}>" if self.subpart_requests else "</answer>"


@dataclass
class RolloutState:
    """One rollout's tape. ``partials[0]`` is the prompt; every later segment is
    either sampled (mask 1) or injected by the environment (mask 0)."""

    index: int
    prompt_ids: List[int]
    hints: Sequence[str]
    partials: List[List[int]] = field(default_factory=list)
    masks: List[List[int]] = field(default_factory=list)
    num_hints: float = 0.0
    last_given: int = -1
    # Subparts already served (subpart_requests mode).
    given_subparts: set = field(default_factory=set)
    response_len: int = 0
    truncated: float = 0.0
    forced: float = 0.0
    # Set by force_answers, mutually exclusive. ``think_truncated``: the
    # response budget ran out before the answer closed and the forced answer
    # then closed </answer> on its own -- benign, the tape carries a complete
    # answer. ``answer_truncated``: the answer never closed by itself -- the
    # forced answer also ran out (closing tag appended) or forcing was
    # impossible (unrescuable tape, no room, no answer budget). The answer
    # text is cut, and nothing downstream repairs that.
    think_truncated: float = 0.0
    answer_truncated: float = 0.0
    malformed: bool = False
    completed: bool = False
    # Generations spent. Capped at max_turns, matching one generation per
    # rollout per round in the round-based driver.
    turns: int = 0

    def __post_init__(self):
        if not self.partials:
            self.partials = [list(self.prompt_ids)]

    @property
    def generated_any(self) -> bool:
        """False for a rollout cut before it ever generated -- there is nothing
        to force an answer onto."""
        return len(self.partials) > 1

    def tokens(self) -> List[int]:
        out: List[int] = []
        for seg in self.partials:
            out.extend(seg)
        return out

    def total_len(self) -> int:
        return sum(len(seg) for seg in self.partials)

    def append(self, tokens: Sequence[int], trainable: bool) -> None:
        tokens = list(tokens)
        self.partials.append(tokens)
        self.masks.append([1 if trainable else 0] * len(tokens))


class HintLoop:
    """Turn rules plus the two drivers that schedule them."""

    def __init__(self, engine: LoopEngine, tokenizer, config: HintLoopConfig,
                 log: Optional[Callable[[str], None]] = None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config
        self._log = log if log is not None else (lambda msg: None)
        self._closing_tokens = self._encode(config.answer_close)
        self._no_more_hints_tokens = self._encode(NO_MORE_HINTS)

    # ---------------------------------------------------------------- helpers

    def _encode(self, text: str) -> List[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _sample_log(self, one_in: int, message: Callable[[], str]) -> None:
        """Log roughly one event in ``one_in``. The message is built lazily so
        the decode it usually needs is not paid for on every rollout."""
        if random.randint(1, one_in) == 1:
            self._log(message())

    def make_states(self, prompt_ids: Sequence[Sequence[int]],
                    hints: Sequence[Sequence[str]]) -> List[RolloutState]:
        return [
            RolloutState(index=i, prompt_ids=list(p), hints=h)
            for i, (p, h) in enumerate(zip(prompt_ids, hints))
        ]

    # ------------------------------------------------------------ turn rules

    def remaining_budget(self, st: RolloutState) -> int:
        return self.config.max_response_len - st.response_len

    def can_generate(self, st: RolloutState) -> bool:
        """Whether a rollout has room for another generation. Marks it complete
        and truncated when it does not, which is what earns it a forced answer."""
        input_len = st.total_len()
        if input_len > self.config.max_model_len - MIN_GENERATION_BUFFER:
            st.completed = True
            st.truncated = 1.0
            self._sample_log(32, lambda: (
                f"Early truncation: prompt length {input_len} exceeds "
                f"max_model_len {self.config.max_model_len}"))
            return False
        if self.remaining_budget(st) < MIN_GENERATION_BUFFER:
            st.completed = True
            st.truncated = 1.0
            self._sample_log(32, lambda: (
                f"Early truncation: response budget exhausted "
                f"({st.response_len}/{self.config.max_response_len})"))
            return False
        return True

    def ingest(self, st: RolloutState, gen: Generation) -> bool:
        """Fold one generation into a rollout and decide what happens next.

        Returns True when the rollout needs another generation. Any hint it
        earned has already been injected by the time this returns, so a caller
        can re-submit ``st`` immediately.
        """
        st.turns += 1
        st.append(gen.token_ids, trainable=True)
        st.response_len += len(gen.token_ids)

        if st.response_len >= self.config.max_response_len:
            st.completed = True
            st.truncated = 1.0
            self._sample_log(32, lambda: (
                f"Response length {st.response_len} exceeds max "
                f"{self.config.max_response_len}, marking complete"))
            return False

        text = gen.text

        # The model writing its own <response> would be fabricating a hint, so
        # </response> is a stop string; seeing it means the model tried.
        if "</response>" in text:
            st.malformed = True
            st.completed = True
            self._sample_log(8, lambda: (
                f"\n!!! MODEL GENERATED </response> !!!\n"
                f"Output text (last 500 chars): {text[-500:]!r}"))
            return False

        if "</request>" in text:
            subpart = None
            if self.config.subpart_requests:
                match = _SUBPART_REQUEST_RE.search(text)
                if match is not None and 1 <= int(match.group(1)) <= self.config.subpart_requests:
                    subpart = int(match.group(1))
                valid = subpart is not None
            else:
                valid = _VALID_REQUEST_RE.search(text) is not None
            if not valid:
                # A request with anything inside fails the validator's tightness
                # check, so there is nothing to rescue.
                st.malformed = True
                st.completed = True
                self._sample_log(64, lambda: (
                    f"Malformed <request> tag detected in output: {text[:200]}..."))
                return False
            st.num_hints += 1
            if subpart is not None:
                self._inject_subpart_hint(st, subpart)
            elif st.num_hints > self.config.max_hints:
                self._append_environment_turn(st, self._no_more_hints_tokens)
            else:
                self._inject_hint(st)
            return not st.completed

        if _ANSWER_RE.search(text) is not None:
            st.completed = True
            return False

        # Neither a request nor an answer: the rollout produced nothing usable
        # and the reward function will score it accordingly.
        st.completed = True
        return False

    def _inject_subpart_hint(self, st: RolloutState, subpart: int) -> None:
        if subpart in st.given_subparts or subpart > len(st.hints):
            self._append_environment_turn(st, self._no_more_hints_tokens)
            return
        st.given_subparts.add(subpart)
        hint_text = st.hints[subpart - 1]
        injection = hint_injection(hint_text, self.config.nested_request)
        tokens = self._encode(injection)
        self._sample_log(8, lambda: (
            f"\n+++ INJECTING SUBPART {subpart} SOLUTION +++\nHint: {hint_text!r}\n"
            f"Num tokens: {len(tokens)}"))
        self._append_environment_turn(st, tokens)

    def _inject_hint(self, st: RolloutState) -> None:
        next_idx = st.last_given + 1
        if next_idx < len(st.hints):
            hint_text = st.hints[next_idx]
            st.last_given = next_idx
        else:
            hint_text = "No more hints available."
        injection = hint_injection(hint_text, self.config.nested_request)
        tokens = self._encode(injection)
        self._sample_log(8, lambda: (
            f"\n+++ INJECTING HINT +++\nHint: {hint_text!r}\n"
            f"Injection string: {injection!r}\nNum tokens: {len(tokens)}"))
        self._append_environment_turn(st, tokens)

    def _append_environment_turn(self, st: RolloutState, tokens: Sequence[int]) -> None:
        """Append environment-authored tokens and re-check both limits.

        Injected text is not charged to the response budget: charging it made
        requesting a hint cost budget twice over -- the hint text plus the
        reasoning it prompts -- and pushed hint-using rollouts into truncation.
        The context window below still bounds the real sequence.
        """
        st.append(tokens, trainable=False)
        if st.total_len() >= self.config.max_model_len:
            st.completed = True
            st.truncated = 1.0
        elif st.response_len >= self.config.max_response_len:
            st.completed = True
            st.truncated = 1.0

    def finalize_unfinished(self, states: Iterable[RolloutState]) -> None:
        """A rollout still unfinished when the turn cap is reached spent its last
        turn on a hint and had none left to answer in. That exhausts a budget
        just as running out of tokens does, so it is flagged for forcing too:
        leaving it unflagged scored every max_turns-hint rollout 0 for being
        malformed -- a structural penalty falling only on the heaviest hint
        users, which is the same bias the forced answers exist to remove.
        """
        for st in states:
            if not st.completed:
                st.truncated = 1.0

    # --------------------------------------------------------------- drivers

    def run_rounds(self, states: Sequence[RolloutState]) -> None:
        """Barrier per turn: everyone generates, everyone waits, hints go in."""
        for _ in range(self.config.max_turns):
            live = [st for st in states if not st.completed and self.can_generate(st)]
            if not live:
                break
            prompts = [st.tokens() for st in live]
            budgets = [max(1, self.remaining_budget(st)) for st in live]
            generations = self.engine.generate_batch(prompts, budgets, mode="turn")
            for st, gen in zip(live, generations):
                self.ingest(st, gen)
        self.finalize_unfinished(states)

    def run_continuous(self, states: Sequence[RolloutState]) -> None:
        """No barrier: a rollout is re-submitted the moment its hint is in.

        The engine therefore never drains down to the few slow rollouts of a
        round -- every rollout that still has work is in the running batch.
        """
        pending = {}
        next_request_id = 0

        def submit(st: RolloutState) -> bool:
            nonlocal next_request_id
            if st.completed or st.turns >= self.config.max_turns:
                return False
            if not self.can_generate(st):
                return False
            request_id = next_request_id
            next_request_id += 1
            pending[request_id] = st
            self.engine.submit(request_id, st.tokens(), max(1, self.remaining_budget(st)))
            return True

        for st in states:
            submit(st)

        while pending:
            if not self.engine.busy():
                # Rollouts are pending but nothing is running: the engine
                # dropped a request. Stop rather than block on a step that will
                # never return; the sweep below flags whatever never finished
                # as truncated. Say so, because those rollouts are scored zero
                # and nothing downstream would show where the zeros came from.
                self._log(
                    f"!!! ENGINE DROPPED REQUESTS !!! engine idle with "
                    f"{len(pending)} rollout(s) still pending; flagging them "
                    f"truncated")
                break
            for request_id, gen in self.engine.step():
                st = pending.pop(request_id, None)
                if st is not None and self.ingest(st, gen):
                    submit(st)

        self.finalize_unfinished(states)

    # -------------------------------------------------------- forced answers

    def force_answers(self, states: Sequence[RolloutState]) -> int:
        """Give each truncated rollout a short, separate budget to answer in.

        Without this, running out of room mid-thought scores 0 from the
        structural validator -- a cliff rather than a gradient, charging the
        same whether the model was one token or a thousand from an answer.
        Asking for a hint pushes a rollout toward that cliff, so the cliff shows
        up in training as a penalty on hint-seeking even when the configured
        hint_penalty is zero.
        """
        forced = self._force_answers(states) if self.config.answer_budget > 0 else 0
        # Whatever is still truncated after forcing carries no complete answer.
        for st in states:
            if st.truncated and not st.forced:
                st.answer_truncated = 1.0
        return forced

    def _force_answers(self, states: Sequence[RolloutState]) -> int:
        targets: List[Tuple[RolloutState, List[int]]] = []
        for st in states:
            if not st.truncated or not st.generated_any:
                continue
            generated = []
            for seg in st.partials[1:]:
                generated.extend(seg)
            scaffold = self.answer_scaffold(self.tokenizer.decode(generated))
            if scaffold is None:
                continue
            scaffold_tokens = self._encode(scaffold) if scaffold else []
            # The tape plus the scaffold is what gets submitted, and vLLM
            # rejects a prompt at max_model_len for the whole batch. Nothing
            # bounds this sum: generations are capped by the response budget
            # and injected hints are exempt from it, so both grow the tape
            # while only the context window says stop. Skipping leaves the
            # rollout scored as truncated, exactly as it is with no forcing.
            if st.total_len() + len(scaffold_tokens) >= self.config.max_model_len:
                self._sample_log(4, lambda: (
                    f"No room to force an answer: {st.total_len()} tokens plus "
                    f"a {len(scaffold_tokens)}-token scaffold reaches "
                    f"max_model_len {self.config.max_model_len}"))
                continue
            targets.append((st, scaffold_tokens))

        if not targets:
            return 0

        for st, scaffold_tokens in targets:
            if scaffold_tokens:
                st.append(scaffold_tokens, trainable=False)

        prompts = [st.tokens() for st, _ in targets]
        budgets = [self.config.answer_budget] * len(targets)
        generations = self.engine.generate_batch(prompts, budgets, mode="answer")

        for (st, _), gen in zip(targets, generations):
            # Only the sampled answer tokens are trained on; the scaffold and
            # the closing tag were forced, not chosen by the policy.
            if gen.token_ids:
                st.append(gen.token_ids, trainable=True)
            # include_stop_str_in_output=False trims the stop string from the
            # text but leaves it in token_ids, so a model that hit </answer> on
            # its own already carries the tag. Appending unconditionally
            # produced "</answer></answer>", which the structure validator
            # rejects -- the forced answers were scored 0 for being malformed,
            # the exact failure this forcing exists to prevent.
            if self.tokenizer.decode(gen.token_ids).rstrip().endswith(self.config.answer_close):
                st.think_truncated = 1.0
            else:
                st.append(self._closing_tokens, trainable=False)
                st.answer_truncated = 1.0
            st.forced = 1.0

        self._sample_log(8, lambda: f"Forced an answer for {len(targets)} truncated rollout(s)")
        return len(targets)

    def answer_scaffold(self, text: str) -> Optional[str]:
        """Text to append so a truncated rollout can open a valid <answer>.

        Returns None when the rollout cannot be rescued, in which case it is
        left truncated and scores zero exactly as it did before.
        """
        # A <request> with anything inside is unrecoverable: the validator
        # rejects the response on a tightness check before it parses any tags.
        if self.config.subpart_requests:
            k = self.config.subpart_requests
            valid = sum(1 <= int(m) <= k for m in _SUBPART_REQUEST_RE.findall(text))
            if text.count("<request>") != valid:
                return None
            return self._answer_scaffold_composed(text)
        if text.count("<request>") != len(_VALID_REQUEST_RE.findall(text)):
            return None

        if self.config.nested_request:
            return self._answer_scaffold_nested(text)

        tags = re.findall(TAG_PATTERN, text)
        i, n = 0, len(tags)
        while True:
            if i == n:
                # Still inside a think block -- the assistant prefix opens one,
                # and this is where the great majority of truncations land.
                return "\n</think>\n\n<answer>"
            if tags[i] != "</think>":
                return None
            i += 1
            if i == n:
                return "\n<answer>"  # think closed, branch not chosen yet
            if tags[i] == "<answer>":
                # Mid-answer: let it finish; the caller appends </answer>.
                return "" if i + 1 == n else None
            if tags[i] != "<request>":
                return None  # <abstain>, or already terminal
            j = 0
            while j < len(HINT_CYCLE) and i < n and tags[i] == HINT_CYCLE[j]:
                i += 1
                j += 1
            if j == len(HINT_CYCLE):
                continue  # a whole hint cycle, keep walking
            if i == n and j == 2:
                # Out of budget between asking and being answered. Close the
                # cycle so the tag sequence stays parseable.
                return "<response>No more hints available.</response>\n<think>\n</think>\n\n<answer>"
            return None

    def _answer_scaffold_composed(self, text: str) -> Optional[str]:
        """Nested-grammar scaffold opening <answer_1> instead of <answer>.

        A truncation after </think> is past the point of rescue: the numbered
        answer tags are not in TAG_PATTERN, so a partial answer block cannot be
        walked, and only an empty one can be opened.
        """
        cut = text.rfind("</think>")
        if cut >= 0:
            if text[cut + len("</think>"):].strip():
                return None
            text = text[:cut + len("</think>")]
        scaffold = self._answer_scaffold_nested(text)
        if scaffold is None or not scaffold.endswith("<answer>"):
            return None
        return scaffold[:-len("<answer>")] + self.config.answer_open

    def _answer_scaffold_nested(self, text: str) -> Optional[str]:
        """answer_scaffold for inline requests: one think block, closed once.

        The tightness check on <request> has already run in the caller.
        """
        tags = re.findall(TAG_PATTERN, text)
        i, n = 0, len(tags)

        while i < n and tags[i] == "<request>":
            j = 0
            while j < len(NESTED_HINT_CYCLE) and i < n and tags[i] == NESTED_HINT_CYCLE[j]:
                i += 1
                j += 1
            if j == len(NESTED_HINT_CYCLE):
                continue  # a whole hint exchange, keep walking
            if i == n and j == 2:
                # Out of budget between asking and being answered. Close the
                # exchange so the tag sequence stays parseable.
                return "<response>No more hints available.</response>\n</think>\n\n<answer>"
            return None

        if i == n:
            # Still inside the think block -- where nearly every truncation lands.
            return "\n</think>\n\n<answer>"
        if tags[i] != "</think>":
            return None
        i += 1
        if i == n:
            return "\n<answer>"  # think closed, branch not chosen yet
        if tags[i] == "<answer>":
            # Mid-answer: let it finish; the caller appends </answer>.
            return "" if i + 1 == n else None
        return None
