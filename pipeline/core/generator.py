"""VLLM-based text generation."""

import asyncio
import re
from dataclasses import dataclass

_RESPONSE_TAG_RE = re.compile(r'<response>.*?</response>', re.DOTALL)


def _count_model_tokens(text: str, tokenizer) -> int:
    """Count tokens in text, excluding <response>...</response> segments.

    This gives the number of model-generated tokens only, since
    <response> content is system-injected hint text.
    """
    stripped = _RESPONSE_TAG_RE.sub('', text)
    return len(tokenizer.encode(stripped))


HINT_TRANSITION_PHRASES = [
    "\n\nI'm not sure how to proceed from here. Let me request a hint.",
    "\n\nI'm stuck on this step. Maybe a hint would help me move forward.",
    "\n\nI don't know how to continue with this approach. Let me ask for a hint.",
    "\n\nI'm having trouble figuring out the next step. Let me request a hint to help guide my thinking.",
    "\n\nHmm, I'm unsure about how to proceed. Let me request a hint.",
    "\n\nI need some guidance on how to approach this next part. Let me ask for a hint.",
    "\n\nI'm not confident in where to go from here. Let me request a hint.",
]


def _pick_transition(rng, enabled: bool) -> str | None:
    """A canned lead-in phrase, or None to request with no lead-in at all."""
    return rng.choice(HINT_TRANSITION_PHRASES) if enabled else None


def _close_think_for_request(text: str, transition: str | None, nested: bool = False) -> str:
    """Prepare the CoT so a forced <request></request> can follow it.

    nested=False puts the request outside the think block, so the block is
    closed first: with a canned transition phrase when one is given, otherwise
    silently after trimming the reasoning back to its last sentence boundary,
    so nothing before the request signals that a hint is coming.

    nested=True leaves the block open and the request is written inline. Any
    </think> or <answer> the model already produced this turn is stripped --
    it chose to answer, and the forced request replaces that choice -- and the
    reasoning is trimmed the same way. No phrase is spliced in: inline, there
    is no boundary left for a lead-in to announce.

    Only the turn currently being generated is rewritten. Everything up to the
    last injected <response> is committed transcript: rewriting into it would
    delete earlier hint exchanges.
    """
    split = 0
    resp = text.rfind("</response>")
    if resp >= 0:
        split = resp + len("</response>")
    if not nested:
        # Flat transcripts reopen reasoning with <think> after each hint; that
        # tag is the real start of the current turn. Inline ones never close.
        open_think = text.find("<think>", split)
        if open_think >= 0:
            split = open_think + len("<think>")
    head, tail = text[:split], text[split:]

    had_think = "</think>" in tail
    for tag in ("<answer>", "</think>"):
        pos = tail.rfind(tag)
        if pos >= 0:
            tail = tail[:pos]

    # A rollout stopped mid-sentence by the token cap never closed its thinking,
    # so back it up to the last clean boundary rather than cutting mid-word.
    # Without a transition this applies to every path: a request that follows a
    # *concluded* thought is its own giveaway, so the thought is left unfinished.
    if nested or not had_think or transition is None:
        period = tail.rfind(". ")
        paragraph = tail.rfind("\n\n")
        cut = max(period + 1 if period >= 0 else -1, paragraph if paragraph >= 0 else -1)
        if cut > 0:
            tail = tail[:cut]

    if nested:
        return head + tail.rstrip()
    if transition is not None:
        return head + tail + transition + "</think>"
    return head + tail.rstrip() + "\n</think>"


# A request tag no model can emit. vLLM's bad_words bans a token *id*, but BPE
# lets one string be spelled by more than one token sequence, so a decoding-level
# ban leaks: 6 of 1764 generations reached </request> and were served real hints.
# Detection routed through a tag containing NUL -- which no tokenizer produces --
# makes serving a hint impossible by construction rather than by luck.
_UNMATCHABLE_REQUEST_TAG = "\x00hint-requests-banned\x00"


def _request_tag_for(config, request_tag: str, any_forcing: bool) -> str:
    """The tag hint injection triggers on, honouring ban_hint_requests.

    Swapping the tag rather than guarding each of the seventeen injection sites
    keeps the ban in one place, and it also fixes what the model does next: the
    tag doubles as a stop string, so an unmatchable one lets a model that wrote
    a stray <request> keep reasoning to its own answer instead of halting on a
    request that will never be answered.
    """
    if not config.ban_hint_requests:
        return request_tag
    if any_forcing:
        raise ValueError(
            "ban_hint_requests and force_hints are contradictory: one forbids "
            "hints, the other injects them unconditionally"
        )
    return _UNMATCHABLE_REQUEST_TAG


def _hint_injection(hint: str, nested: bool) -> str:
    """The environment's turn: the hint, plus a reopened think block if the
    request lives outside one. Inline requests never left the block."""
    return f"\n<response>{hint}</response>\n" if nested else f"\n<response>{hint}</response>\n<think>\n"


DEFAULT_STOP_STRINGS = ["</answer>"]

# The tags a single-turn generation can legally emit. The assistant prefix opens
# <think>, so a generation that never closes it carries no tags at all.
_PLAIN_TAG_PATTERN = r"(</think>|<think>|<answer>|</answer>)"


def _answer_scaffold(text: str) -> str | None:
    """Text to append so a truncated generation can open a valid <answer>.

    This is HintLoop.answer_scaffold from verl with the hint grammar removed --
    the two must agree on what a rescuable truncation looks like, because RL
    forces answers at rollout time and SFT data generated here is what teaches
    the model that shape. Returns None when the generation cannot be rescued, in
    which case it stays truncated and is filtered out exactly as before.
    """
    tags = re.findall(_PLAIN_TAG_PATTERN, text)
    i, n = 0, len(tags)
    if i == n:
        # Still inside the think block the prefix opened -- where nearly every
        # truncation lands.
        return "\n</think>\n\n<answer>"
    if tags[i] != "</think>":
        return None
    i += 1
    if i == n:
        return "\n<answer>"  # think closed, answer not opened yet
    if tags[i] == "<answer>":
        # Mid-answer: let it finish; the caller appends </answer>.
        return "" if i + 1 == n else None
    return None



@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model_name: str
    batch_size: int = 16
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1  # Independent model replicas; prompts are sharded
                                 # across them. The cluster floor is 4 GPUs, so a
                                 # small model wants DP=4 (4x throughput) rather
                                 # than TP=4 (one replica, poor scaling).
    gpu_memory_utilization: float = 0.9
    verbose: bool = False
    seed: int | None = 42  # Set seed for reproducibility
    hint_transition: bool = True  # Splice a canned phrase before forced hint requests
    nested_request: bool = False  # Keep <request></request> inside the <think> block
    ban_hint_requests: bool = False  # Counterfactual eval: make asking for a hint
                                     # impossible, so the model has to answer alone
    answer_budget: int = 0  # Tokens held back for a forced answer when a generation
                            # runs out of room mid-thought. 0 disables forcing.


class _SamplingParams:
    """Mixin: builds SamplingParams from a GenerationConfig."""

    def _sampling_params(self, **kwargs):
        """SamplingParams with the generator's global constraints applied.

        ban_hint_requests blocks the token "request" rather than the string
        "<request>": vLLM's bad_words bans the *last* token of a phrase only
        when the preceding tokens match exactly, and "<request>" tokenises as
        "<", "request", ">" while generation actually emits "></" as one token
        there -- so the phrase form matches nothing and silently never fires.

        This suppresses request attempts; it does not eliminate them. bad_words
        bans a token id, and BPE can spell the same string another way, so a
        determined model still reaches "<request>" a fraction of a percent of
        the time. What guarantees no hint is *served* is _request_tag_for,
        which makes the injection trigger unmatchable; this ban is the cheaper
        first layer that keeps the attempt rate near zero.
        """
        from vllm import SamplingParams
        if self.config.ban_hint_requests:
            kwargs["bad_words"] = ["request"]
        return SamplingParams(**kwargs)


class Generator(_SamplingParams):
    """VLLM-based batched text generator."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from vllm import LLM
            print(f"Loading model: {self.config.model_name}")
            # vLLM refuses data parallelism on the offline LLM class:
            # "LLM(data_parallel_size=N) is not supported for single-process
            # usage and may hang." The async engine spawns one EngineCore per
            # replica and handles it fine, so send people there rather than
            # letting vLLM raise mid-run after a model load.
            if self.config.data_parallel_size > 1:
                raise ValueError(
                    f"data_parallel_size={self.config.data_parallel_size} is not "
                    f"supported by the synchronous generator: vLLM only allows data "
                    f"parallelism through the async engine. Add --async to use it "
                    f"(the standard mode for this pipeline), or drop "
                    f"--data-parallel-size and use --tensor-parallel-size instead."
                )
            self._model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
        return self._model

    def generate(self, prompts: list[list[dict]]) -> list[list[dict]]:
        """
        Generate completions for a list of prompts.

        Args:
            prompts: List of chat message lists

        Returns:
            List of lists of dicts with 'text' and 'finish_reason'
            (each prompt can have multiple samples if num_samples > 1)
        """

        stop = DEFAULT_STOP_STRINGS
        sampling_params = self._sampling_params(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=self.config.num_samples,
            seed=self.config.seed,
            stop=stop,
            include_stop_str_in_output=True,
        )

        # Apply chat template
        tokenizer = self.model.get_tokenizer()
        formatted_prompts = []
        for messages in prompts:
            # Separate assistant prefix from conversation
            if messages and messages[-1]["role"] == "assistant":
                conversation = messages[:-1]
                assistant_prefix = messages[-1]["content"]
            else:
                conversation = messages
                assistant_prefix = ""

            # Apply template to conversation, then append assistant prefix
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,  # Adds <|im_start|>assistant\n
            )
            text += assistant_prefix
            formatted_prompts.append(text)

        # Print sample prompt for debugging (only if verbose)
        if self.config.verbose and formatted_prompts:
            print("\n=== SAMPLE FORMATTED PROMPT ===")
            print(formatted_prompts[0])
            print("=== END SAMPLE ===\n")

        # Generate
        outputs = self.model.generate(formatted_prompts, sampling_params)

        results = []
        for output in outputs:
            # Each output can have multiple completions (if n > 1)
            samples = []
            for completion in output.outputs:
                samples.append({
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "token_count": len(completion.token_ids),
                })
            results.append(samples)

        return results

    def generate_batched(
        self,
        prompts: list[list[dict]],
        callback: callable = None,
    ) -> list[dict]:
        """
        Generate completions in batches with optional progress callback.

        Args:
            prompts: List of chat message lists
            callback: Optional function called after each batch with (batch_idx, results)

        Returns:
            List of generation results
        """
        all_results = []
        batch_size = self.config.batch_size

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self.generate(batch)
            all_results.extend(batch_results)

            if callback:
                callback(i // batch_size, batch_results)

        return all_results

    def generate_with_hints(
        self,
        prompts: list[list[dict]],
        ground_truths: list[dict],
        request_tag: str = "</request>",
        answer_tag: str = "</answer>",
        force_hints: int | list[int] = 0,
        force_hint_token_range: tuple[int, int] = (100, 500),
    ) -> list[list[dict]]:
        """
        Multi-turn generation with hint injection using turn-synchronized batching.

        Uses token-based accumulation (splicing) to match RL rollout behavior:
        each segment (model generation, hint injection) is tokenized independently
        and concatenated as raw token IDs. This avoids tokenization boundary
        mismatches between eval and RL.

        When force_hints is active, falls back to text-based accumulation for
        compatibility with the text manipulation required by forced hint injection.

        Args:
            prompts: List of chat message lists
            ground_truths: List of dicts containing 'hint_exprs' for each prompt
            request_tag: Tag that triggers hint injection (default: "</request>")
            answer_tag: Tag that signals completion (default: "</answer>")
            force_hints: Force at least this many hints per example by intercepting
                </think> tags and injecting hint requests. Can be a single int
                (applied to all) or a per-prompt list. (default: 0, disabled)
            force_hint_token_range: (min, max) token cap for forced-hint turns.
                On forced turns, max_tokens is sampled from this range so the model
                is cut off mid-reasoning before it can complete. (default: (100, 500))

        Returns:
            List of lists of dicts with 'text', 'finish_reason', 'token_count', 'num_hints'
        """
        import ast

        # Normalize force_hints to per-prompt list
        num_prompts = len(prompts)
        if isinstance(force_hints, int):
            force_hints_per = [force_hints] * num_prompts
        else:
            force_hints_per = force_hints
        any_forcing = any(f > 0 for f in force_hints_per)

        # Parse hint expressions
        hints_list = []
        for ground_truth in ground_truths:
            hints_expr = ground_truth.get("hint_exprs", ground_truth.get("hints_expr", []))
            if isinstance(hints_expr, str):
                try:
                    hints_expr = ast.literal_eval(hints_expr)
                except (ValueError, SyntaxError):
                    hints_expr = []
            hints_list.append(hints_expr)

        request_tag = _request_tag_for(self.config, request_tag, any_forcing)

        if any_forcing:
            # Forced hints require text manipulation — use legacy text-based path
            return self._generate_with_hints_text_based(
                prompts, hints_list, force_hints_per,
                request_tag, answer_tag, force_hint_token_range,
            )
        else:
            # Token-based path matching RL rollout splicing
            return self._generate_with_hints_token_spliced(
                prompts, hints_list,
                request_tag, answer_tag,
            )

    def _generate_with_hints_token_spliced(
        self,
        prompts: list[list[dict]],
        hints_list: list[list[str]],
        request_tag: str,
        answer_tag: str,
    ) -> list[list[dict]]:
        """Token-spliced multi-turn generation matching RL rollout behavior.

        Accumulates raw token IDs from each segment (generation, hint injection)
        and concatenates them, exactly like the RL rollout's agentic_loop.
        """

        num_prompts = len(prompts)
        tokenizer = self.model.get_tokenizer()
        base_seed = self.config.seed

        # Tokenize initial prompts (once, stored as first token segment)
        initial_token_ids = []
        for messages in prompts:
            if messages and messages[-1]["role"] == "assistant":
                conversation = messages[:-1]
                assistant_prefix = messages[-1]["content"]
            else:
                conversation = messages
                assistant_prefix = ""

            formatted = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            formatted += assistant_prefix
            tokens = tokenizer.encode(formatted, add_special_tokens=False)
            initial_token_ids.append(tokens)

        # Per-prompt state
        # accumulated_tokens: list of token segments (first is prompt, rest are gen/hint)
        # model_token_count: count of model-generated tokens only (for budget)
        state = {
            "accumulated_tokens": [[ids] for ids in initial_token_ids],
            "accumulated_text": [""] * num_prompts,
            "model_token_count": [0] * num_prompts,
            "last_given_index": [-1] * num_prompts,
            "num_hints_used": [0] * num_prompts,
            "turns": [0] * num_prompts,
            "finish_reason": ["length"] * num_prompts,
            "completed": [False] * num_prompts,
            "hint_selections": [[] for _ in range(num_prompts)],
        }

        max_budget = self.config.max_new_tokens
        turn = 0

        while True:
            # Find incomplete prompts with remaining budget
            incomplete_indices = []
            for i in range(num_prompts):
                if state["completed"][i]:
                    continue
                if state["model_token_count"][i] >= max_budget:
                    state["completed"][i] = True
                    state["finish_reason"][i] = "length"
                    continue
                incomplete_indices.append(i)

            if not incomplete_indices:
                break

            # Build prompts and per-prompt sampling params (different seed per prompt)
            formatted_prompts = []
            params_list = []
            for idx in incomplete_indices:
                all_tokens = []
                for segment in state["accumulated_tokens"][idx]:
                    all_tokens.extend(segment)
                formatted_prompts.append({"prompt_token_ids": all_tokens})
                remaining = max(max_budget - state["model_token_count"][idx], 1)
                params_list.append(self._sampling_params(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=remaining,
                    n=1,
                    stop=[request_tag, answer_tag],
                    include_stop_str_in_output=True,
                    seed=(base_seed + idx) if base_seed is not None else None,
                ))

            # Batch generate
            outputs = self.model.generate(formatted_prompts, params_list)

            # Process outputs
            for batch_idx, idx in enumerate(incomplete_indices):
                output = outputs[batch_idx].outputs[0]
                generated_text = output.text
                gen_tokens = list(output.token_ids)

                state["turns"][idx] += 1
                state["accumulated_text"][idx] += generated_text
                state["accumulated_tokens"][idx].append(gen_tokens)
                state["model_token_count"][idx] += len(gen_tokens)

                # Check if answer tag found
                if answer_tag in state["accumulated_text"][idx]:
                    state["finish_reason"][idx] = "stop"
                    state["completed"][idx] = True
                    continue

                # Check if hint was requested
                if request_tag in generated_text:
                    hints = hints_list[idx]
                    last_given = state["last_given_index"][idx]

                    if last_given + 1 < len(hints):
                        next_idx = last_given + 1
                        hint_text, new_last = hints[next_idx], next_idx

                        if hint_text is not None:
                            state["hint_selections"][idx].append({
                                "turn": state["turns"][idx],
                                "type": "sequential",
                                "prev_last_given": last_given,
                                "new_last_given": new_last,
                            })
                            state["last_given_index"][idx] = new_last
                            state["num_hints_used"][idx] += 1

                            # Inject hint — tokenize independently (matches RL splicing)
                            hint_response = _hint_injection(hint_text, self.config.nested_request)
                            hint_tokens = tokenizer.encode(hint_response, add_special_tokens=False)
                            state["accumulated_text"][idx] += hint_response
                            state["accumulated_tokens"][idx].append(hint_tokens)
                        else:
                            warning = _hint_injection("No more hints available.", self.config.nested_request)
                            warning_tokens = tokenizer.encode(warning, add_special_tokens=False)
                            state["accumulated_text"][idx] += warning
                            state["accumulated_tokens"][idx].append(warning_tokens)
                            state["num_hints_used"][idx] += 1
                    else:
                        warning = _hint_injection("No more hints available.", self.config.nested_request)
                        warning_tokens = tokenizer.encode(warning, add_special_tokens=False)
                        state["accumulated_text"][idx] += warning
                        state["accumulated_tokens"][idx].append(warning_tokens)
                        state["num_hints_used"][idx] += 1
                else:
                    # No hint requested and no answer
                    if output.finish_reason == "stop" and answer_tag not in state["accumulated_text"][idx]:
                        # Model generated EOS without <answer> — append </think> and continue
                        close_text = "</think>\n"
                        close_tokens = tokenizer.encode(close_text, add_special_tokens=False)
                        state["accumulated_text"][idx] += close_text
                        state["accumulated_tokens"][idx].append(close_tokens)
                    else:
                        state["finish_reason"][idx] = output.finish_reason or "length"
                        state["completed"][idx] = True

            if self.config.verbose:
                completed_count = sum(state["completed"])
                print(f"Turn {turn + 1}: {completed_count}/{num_prompts} completed")

            turn += 1

        # Build results
        all_results = []
        for idx in range(num_prompts):
            text = state["accumulated_text"][idx]
            result = {
                "text": text,
                "finish_reason": state["finish_reason"][idx],
                "token_count": state["model_token_count"][idx],
                "total_token_count": sum(len(seg) for seg in state["accumulated_tokens"][idx][1:]),
                "num_hints": state["num_hints_used"][idx],
                "turns": state["turns"][idx],
                "hint_selections": state["hint_selections"][idx],
            }
            all_results.append([result])

            if self.config.verbose:
                print(f"Prompt {idx + 1}: {result['turns']} turns, {result['num_hints']} hints, {result['finish_reason']}")

        return all_results

    def _generate_with_hints_text_based(
        self,
        prompts: list[list[dict]],
        hints_list: list[list[str]],
        force_hints_per: list[int],
        request_tag: str,
        answer_tag: str,
        force_hint_token_range: tuple[int, int],
    ) -> list[list[dict]]:
        """Text-based multi-turn generation for forced hint injection.

        Uses chat template re-application each turn. This path is only used
        when force_hints is active (SFT data generation), not during eval.
        """
        import copy
        import random as _random

        num_prompts = len(prompts)
        tokenizer = self.model.get_tokenizer()
        force_rng = _random.Random(42)

        state = {
            "accumulated_text": [""] * num_prompts,
            "current_messages": [copy.deepcopy(m) for m in prompts],
            "last_given_index": [-1] * num_prompts,
            "num_hints_used": [0] * num_prompts,
            "turns": [0] * num_prompts,
            "finish_reason": ["length"] * num_prompts,
            "completed": [False] * num_prompts,
            "hint_selections": [[] for _ in range(num_prompts)],
        }

        think_tag = "</think>"
        max_budget = self.config.max_new_tokens

        turn = 0
        while True:
            incomplete_indices = []
            for i in range(num_prompts):
                if state["completed"][i]:
                    continue
                model_tokens = _count_model_tokens(state["accumulated_text"][i], tokenizer)
                if model_tokens >= max_budget:
                    state["completed"][i] = True
                    state["finish_reason"][i] = "length"
                    continue
                incomplete_indices.append(i)

            if not incomplete_indices:
                break

            forcing_indices = []
            normal_indices = []
            for idx in incomplete_indices:
                if force_hints_per[idx] > 0 and state["num_hints_used"][idx] < force_hints_per[idx] and state["last_given_index"][idx] + 1 < len(hints_list[idx]):
                    forcing_indices.append(idx)
                else:
                    normal_indices.append(idx)

            if forcing_indices:
                cap = force_rng.randint(force_hint_token_range[0], force_hint_token_range[1])
                min_remaining = min(
                    max_budget - _count_model_tokens(state["accumulated_text"][idx], tokenizer)
                    for idx in forcing_indices
                )
                forcing_params = self._sampling_params(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=min(cap, max(min_remaining, 1)),
                    n=1,
                    stop=[request_tag, think_tag],
                    include_stop_str_in_output=True,
                    seed=self.config.seed,
                )
            else:
                forcing_params = None

            if normal_indices:
                min_remaining = min(
                    max_budget - _count_model_tokens(state["accumulated_text"][idx], tokenizer)
                    for idx in normal_indices
                )
                normal_params = self._sampling_params(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=max(min_remaining, 1),
                    n=1,
                    stop=[request_tag, answer_tag],
                    include_stop_str_in_output=True,
                    seed=self.config.seed,
                )
            else:
                normal_params = None

            for group_indices, params in [(forcing_indices, forcing_params), (normal_indices, normal_params)]:
                if not group_indices:
                    continue

                formatted_prompts = []
                for idx in group_indices:
                    messages = state["current_messages"][idx]
                    if messages and messages[-1]["role"] == "assistant":
                        conversation = messages[:-1]
                        assistant_content = messages[-1]["content"]
                    else:
                        conversation = messages
                        assistant_content = ""

                    formatted = tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True,
                    )
                    formatted += assistant_content
                    formatted_prompts.append(formatted)

                outputs = self.model.generate(formatted_prompts, params)

                for batch_idx, idx in enumerate(group_indices):
                    output = outputs[batch_idx].outputs[0]
                    generated_text = output.text

                    state["turns"][idx] += 1
                    state["accumulated_text"][idx] += generated_text

                    if answer_tag in state["accumulated_text"][idx]:
                        if force_hints_per[idx] > 0 and state["num_hints_used"][idx] < force_hints_per[idx] and state["last_given_index"][idx] + 1 < len(hints_list[idx]):
                            ans_pos = state["accumulated_text"][idx].rfind("<answer>")
                            if ans_pos >= 0:
                                state["accumulated_text"][idx] = state["accumulated_text"][idx][:ans_pos]
                            self._inject_forced_hint(state, idx, hints_list[idx], rng=force_rng)
                            self._rebuild_messages(state, idx, prompts[idx])
                            continue
                        state["finish_reason"][idx] = "stop"
                        state["completed"][idx] = True
                        continue

                    if force_hints_per[idx] > 0 and think_tag in generated_text and request_tag not in generated_text:
                        hints = hints_list[idx]
                        if state["num_hints_used"][idx] < force_hints_per[idx] and state["last_given_index"][idx] + 1 < len(hints):
                            self._inject_forced_hint(state, idx, hints, rng=force_rng)
                            self._rebuild_messages(state, idx, prompts[idx])
                            continue

                    if force_hints_per[idx] > 0 and output.finish_reason == "length":
                        if state["num_hints_used"][idx] < force_hints_per[idx] and state["last_given_index"][idx] + 1 < len(hints_list[idx]):
                            text = state["accumulated_text"][idx]
                            period_cut = text.rfind('. ')
                            newline_cut = text.rfind('\n\n')
                            cut = max(period_cut + 1 if period_cut >= 0 else -1,
                                      newline_cut if newline_cut >= 0 else -1)
                            if cut > 0:
                                text = text[:cut]
                            state["accumulated_text"][idx] = text + "</think>\n"
                            self._inject_forced_hint(state, idx, hints_list[idx], rng=force_rng)
                            self._rebuild_messages(state, idx, prompts[idx])
                            continue

                    if request_tag in generated_text:
                        hints = hints_list[idx]
                        last_given = state["last_given_index"][idx]

                        if last_given + 1 < len(hints):
                            next_idx = last_given + 1
                            hint_text, new_last = hints[next_idx], next_idx

                            if hint_text is not None:
                                state["hint_selections"][idx].append({
                                    "turn": state["turns"][idx],
                                    "type": "sequential",
                                    "prev_last_given": last_given,
                                    "new_last_given": new_last,
                                })
                                state["last_given_index"][idx] = new_last
                                state["num_hints_used"][idx] += 1

                                hint_response = _hint_injection(hint_text, self.config.nested_request)
                                state["accumulated_text"][idx] += hint_response

                                messages = state["current_messages"][idx]
                                if messages and messages[-1]["role"] == "assistant":
                                    messages[-1]["content"] += generated_text + hint_response
                                else:
                                    messages.append({
                                        "role": "assistant",
                                        "content": generated_text + hint_response,
                                    })
                            else:
                                warning = _hint_injection("No more hints available.", self.config.nested_request)
                                state["accumulated_text"][idx] += warning
                                state["num_hints_used"][idx] += 1

                                messages = state["current_messages"][idx]
                                if messages and messages[-1]["role"] == "assistant":
                                    messages[-1]["content"] += generated_text + warning
                                else:
                                    messages.append({
                                        "role": "assistant",
                                        "content": generated_text + warning,
                                    })
                        else:
                            warning = _hint_injection("No more hints available.", self.config.nested_request)
                            state["accumulated_text"][idx] += warning
                            state["num_hints_used"][idx] += 1

                            messages = state["current_messages"][idx]
                            if messages and messages[-1]["role"] == "assistant":
                                messages[-1]["content"] += generated_text + warning
                            else:
                                messages.append({
                                    "role": "assistant",
                                    "content": generated_text + warning,
                                })
                    else:
                        if output.finish_reason == "stop" and answer_tag not in state["accumulated_text"][idx]:
                            state["accumulated_text"][idx] += "</think>\n"
                            self._rebuild_messages(state, idx, prompts[idx])
                        else:
                            state["finish_reason"][idx] = output.finish_reason or "length"
                            state["completed"][idx] = True

            if self.config.verbose:
                completed_count = sum(state["completed"])
                print(f"Turn {turn + 1}: {completed_count}/{num_prompts} completed")

            turn += 1

        # Build results
        all_results = []
        for idx in range(num_prompts):
            text = state["accumulated_text"][idx]
            result = {
                "text": text,
                "finish_reason": state["finish_reason"][idx],
                "token_count": _count_model_tokens(text, tokenizer),
                "total_token_count": len(tokenizer.encode(text)),
                "num_hints": state["num_hints_used"][idx],
                "turns": state["turns"][idx],
                "hint_selections": state["hint_selections"][idx],
            }
            all_results.append([result])

            if self.config.verbose:
                print(f"Prompt {idx + 1}: {result['turns']} turns, {result['num_hints']} hints, {result['finish_reason']}")

        return all_results

    def _inject_forced_hint(self, state: dict, idx: int, hints: list[str], rng=None):
        """Inject a forced hint request + response into the accumulated text.

        Forced hints always use sequential selection (no reasoning to analyze).
        How the CoT is closed before the request depends on config.hint_transition;
        see _close_think_for_request.
        """
        prev_last = state["last_given_index"][idx]
        hint_idx = prev_last + 1
        hint = hints[hint_idx]
        state["hint_selections"][idx].append({
            "turn": state["turns"][idx],
            "type": "forced",
            "prev_last_given": prev_last,
            "new_last_given": hint_idx,
        })
        state["last_given_index"][idx] = hint_idx
        state["num_hints_used"][idx] += 1

        if rng is None:
            import random as _random
            rng = _random.Random(42 + idx + hint_idx)
        state["accumulated_text"][idx] = _close_think_for_request(
            state["accumulated_text"][idx],
            _pick_transition(rng, self.config.hint_transition),
            nested=self.config.nested_request,
        )

        # Append hint request + response
        forced = "\n<request></request>" + _hint_injection(hint, self.config.nested_request)
        state["accumulated_text"][idx] += forced

    def _rebuild_messages(self, state: dict, idx: int, original_prompt: list[dict]):
        """Rebuild current_messages from original prompt + accumulated text."""
        import copy
        messages = copy.deepcopy(original_prompt)
        if messages and messages[-1]["role"] == "assistant":
            messages[-1]["content"] += state["accumulated_text"][idx]
        else:
            messages.append({
                "role": "assistant",
                "content": state["accumulated_text"][idx],
            })
        state["current_messages"][idx] = messages

    def generate_with_hints_batched(
        self,
        prompts: list[list[dict]],
        ground_truths: list[dict],
        callback: callable = None,
        force_hints: int | list[int] = 0,
        force_hint_token_range: tuple[int, int] = (100, 500),
    ) -> list[list[dict]]:
        """
        Multi-turn generation with hint injection, split into macro-batches.

        The inner generate_with_hints already uses turn-synchronized batching.
        This outer method splits into macro-batches for memory management when
        processing very large prompt sets.

        Args:
            prompts: List of chat message lists
            ground_truths: List of dicts with 'hint_exprs'
            callback: Progress callback(batch_idx, results)
            force_hints: Force at least this many hints per example.
                Can be int (applied to all) or per-prompt list. (default: 0)
            force_hint_token_range: (min, max) token cap for forced-hint turns. (default: (100, 500))

        Returns:
            List of generation results
        """
        all_results = []
        batch_size = self.config.batch_size

        # Normalize to per-prompt list for slicing
        if isinstance(force_hints, int):
            force_hints_list = [force_hints] * len(prompts)
        else:
            force_hints_list = force_hints

        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_ground_truths = ground_truths[i:i + batch_size]
            batch_force_hints = force_hints_list[i:i + batch_size]

            if self.config.verbose:
                batch_num = i // batch_size + 1
                print(f"\n=== Macro-batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts) ===")

            batch_results = self.generate_with_hints(
                batch_prompts,
                batch_ground_truths,
                force_hints=batch_force_hints,
                force_hint_token_range=force_hint_token_range,
            )
            all_results.extend(batch_results)

            if callback:
                callback(i // batch_size, batch_results)

        return all_results


class AsyncGenerator(_SamplingParams):
    """
    Async VLLM-based generator for multi-turn hint generation.

    Uses AsyncLLMEngine for optimal throughput - completions are processed
    as they arrive and immediately resubmitted if hints are needed.

    Each prompt is handled by its own async task, allowing prompts to
    progress independently. When a prompt needs a hint, it immediately
    continues generation without waiting for other prompts.

    Note: Requires VLLM with AsyncLLMEngine support. If you encounter
    import errors, ensure you have a compatible VLLM version installed.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._engine = None
        self._tokenizer = None

    async def _get_engine(self):
        """Lazy load the async engine."""
        if self._engine is None:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs

            print(f"Loading async engine: {self.config.model_name}")
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                data_parallel_size=self.config.data_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        return self._engine

    def close(self):
        """Release the engine and its GPU memory.

        vLLM holds the KV cache for the life of the engine, and with
        data_parallel_size>1 that is one EngineCore process per replica. Anything
        that builds a second engine in the same process -- the oversampling loop
        rerunning generation -- otherwise dies at startup with "Free memory on
        device cuda:N is less than desired GPU memory utilization". Safe to call
        more than once.
        """
        engine, self._engine = self._engine, None
        self._tokenizer = None
        if engine is None:
            return
        try:
            engine.shutdown()
        except Exception as e:  # teardown must not mask the real error
            print(f"Warning: engine shutdown raised {e!r}")
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def _get_tokenizer(self):
        """Get tokenizer from engine."""
        if self._tokenizer is None:
            engine = await self._get_engine()
            # Handle both sync and async get_tokenizer methods across VLLM versions
            tokenizer_result = engine.get_tokenizer()
            if asyncio.iscoroutine(tokenizer_result):
                self._tokenizer = await tokenizer_result
            else:
                self._tokenizer = tokenizer_result
        return self._tokenizer

    def _format_prompt(self, messages: list[dict], tokenizer) -> str:
        """Format messages into a prompt string."""
        if messages and messages[-1]["role"] == "assistant":
            conversation = messages[:-1]
            assistant_content = messages[-1]["content"]
        else:
            conversation = messages
            assistant_content = ""

        formatted = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted += assistant_content
        return formatted

    async def generate_async(
        self,
        prompts: list[list[dict]],
        num_samples: int = 1,
    ) -> list[list[dict]]:
        """
        Async generation for regular (non-multi-turn) prompts.

        Each prompt is processed independently as an async task.
        Much faster than sync generation due to better GPU utilization.

        Args:
            prompts: List of chat message lists
            num_samples: Number of samples per prompt

        Returns:
            List of lists of dicts with 'text', 'finish_reason', 'token_count'
        """
        import uuid
        from tqdm.auto import tqdm

        engine = await self._get_engine()
        tokenizer = await self._get_tokenizer()

        num_prompts = len(prompts)

        # Sampling params
        stop = DEFAULT_STOP_STRINGS
        sampling_params = self._sampling_params(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=num_samples,
            seed=self.config.seed,
            stop=stop,
            include_stop_str_in_output=True,
        )

        # Results storage
        results = [None] * num_prompts

        async def process_single_prompt(idx: int):
            """Process a single prompt."""
            formatted = self._format_prompt(prompts[idx], tokenizer)
            request_id = f"req_{idx}_{uuid.uuid4().hex[:8]}"

            # Generate — collect all outputs (v1 engine streams one per sample)
            samples = []
            async for output in engine.generate(formatted, sampling_params, request_id):
                for completion in output.outputs:
                    if completion.finish_reason is not None:
                        samples.append({
                            "text": completion.text,
                            "finish_reason": completion.finish_reason,
                            "token_count": len(completion.token_ids),
                        })
            if not samples:
                results[idx] = [{"text": "", "finish_reason": "error", "token_count": 0}]
                return
            results[idx] = samples

        # Run all prompts concurrently
        if self.config.verbose:
            print(f"Starting async generation for {num_prompts} prompts")

        progress = tqdm(
            total=num_prompts,
            desc=f"Generating ({num_samples} sample{'s' if num_samples != 1 else ''}/prompt)",
            unit="prompt",
        )
        tasks = [
            asyncio.create_task(process_single_prompt(idx))
            for idx in range(num_prompts)
        ]
        for task in tasks:
            task.add_done_callback(lambda _: progress.update())
        try:
            await asyncio.gather(*tasks)
        finally:
            progress.close()

        if self.config.answer_budget > 0:
            await self._force_answers(engine, tokenizer, prompts, results)

        if self.config.verbose:
            print(f"Async generation complete: {num_prompts} prompts")

        return results

    async def _force_answers(self, engine, tokenizer, prompts, results) -> int:
        """Give each truncated sample a short, separate budget to answer in.

        Running out of room mid-thought otherwise costs the sample entirely: the
        SFT filter keeps only correct generations, and a truncated one cannot be
        correct. That silently biases the training set toward the problems the
        teacher happened to solve briefly, and biases it hardest on exactly the
        hard problems the model most needs examples of.

        RL already does this at rollout time (HintLoop.force_answers). Doing it
        here as well means the SFT model has seen the shape RL will impose on it,
        rather than meeting a forced </think> for the first time under GRPO.
        """
        import uuid

        targets = []
        for pi, samples in enumerate(results):
            for si, sample in enumerate(samples):
                if sample.get("finish_reason") != "length":
                    continue
                scaffold = _answer_scaffold(sample["text"])
                if scaffold is None:
                    continue
                targets.append((pi, si, scaffold))

        if not targets:
            return 0

        params = self._sampling_params(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.answer_budget,
            n=1,
            seed=self.config.seed,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

        async def finish(pi: int, si: int, scaffold: str):
            sample = results[pi][si]
            prefix = self._format_prompt(prompts[pi], tokenizer) + sample["text"] + scaffold
            request_id = f"force_{pi}_{si}_{uuid.uuid4().hex[:8]}"
            tail, tail_tokens = "", 0
            async for output in engine.generate(prefix, params, request_id):
                for completion in output.outputs:
                    if completion.finish_reason is not None:
                        tail = completion.text
                        tail_tokens = len(completion.token_ids)

            text = sample["text"] + scaffold + tail
            # include_stop_str_in_output keeps </answer> when the model emitted it
            # itself; appending unconditionally yields "</answer></answer>", which
            # the structure check rejects -- the exact failure forcing prevents.
            # If the tag had to be appended, the answer budget ran out too:
            # the answer is cut, not merely forced (see classify_truncation).
            sample["answer_truncated"] = not text.rstrip().endswith("</answer>")
            if sample["answer_truncated"]:
                text += "</answer>"
            sample["text"] = text
            sample["token_count"] += tail_tokens
            # The sample is complete now, so downstream truncation handling --
            # --retry-truncated above all -- must stop treating it as unfinished.
            # forced_answer is what keeps the fact recoverable.
            sample["finish_reason"] = "stop"
            sample["forced_answer"] = True

        await asyncio.gather(*[finish(*t) for t in targets])
        print(f"Forced an answer for {len(targets)} truncated generation(s)")
        return len(targets)

    async def generate_with_hints_async(
        self,
        prompts: list[list[dict]],
        ground_truths: list[dict],
        request_tag: str = "</request>",
        answer_tag: str = "</answer>",
        force_hints: int | list[int] = 0,
        force_hint_token_range: tuple[int, int] = (100, 500),
    ) -> list[list[dict]]:
        """
        Async multi-turn generation with hint injection.

        Uses token-based accumulation (splicing) to match RL rollout behavior.
        When force_hints is active, falls back to text-based accumulation.

        Args:
            prompts: List of chat message lists
            ground_truths: List of dicts containing 'hint_exprs' for each prompt
            request_tag: Tag that triggers hint injection
            answer_tag: Tag that signals completion
            force_hints: Force at least this many hints per example by intercepting
                </think> tags and injecting hint requests (default: 0, disabled)
            force_hint_token_range: (min, max) token cap for forced-hint turns.

        Returns:
            List of lists of dicts with 'text', 'finish_reason', 'token_count', 'num_hints'
        """
        import ast

        num_prompts = len(prompts)

        # Parse hint expressions
        hints_list = []
        for ground_truth in ground_truths:
            hints_expr = ground_truth.get("hint_exprs", ground_truth.get("hints_expr", []))
            if isinstance(hints_expr, str):
                try:
                    hints_expr = ast.literal_eval(hints_expr)
                except (ValueError, SyntaxError):
                    hints_expr = []
            hints_list.append(hints_expr)

        # Normalize force_hints
        if isinstance(force_hints, int):
            force_hints_per = [force_hints] * num_prompts
        else:
            force_hints_per = force_hints
        any_forcing = any(f > 0 for f in force_hints_per)

        request_tag = _request_tag_for(self.config, request_tag, any_forcing)

        if any_forcing:
            return await self._generate_with_hints_async_text_based(
                prompts, hints_list, force_hints_per,
                request_tag, answer_tag, force_hint_token_range,
            )
        else:
            return await self._generate_with_hints_async_token_spliced(
                prompts, hints_list,
                request_tag, answer_tag,
            )

    async def _generate_with_hints_async_token_spliced(
        self,
        prompts: list[list[dict]],
        hints_list: list[list[str]],
        request_tag: str,
        answer_tag: str,
    ) -> list[list[dict]]:
        """Async token-spliced multi-turn generation matching RL rollout behavior."""
        import uuid
        from vllm import TokensPrompt

        engine = await self._get_engine()
        tokenizer = await self._get_tokenizer()

        num_prompts = len(prompts)
        max_budget = self.config.max_new_tokens
        base_seed = self.config.seed

        results = [None] * num_prompts
        completed_count = [0]

        async def process_single_prompt(idx: int):
            hints = hints_list[idx]
            last_given_index = -1
            accumulated_text = ""
            num_hints_used = 0
            model_token_count = 0
            turns = 0
            finish_reason = "length"

            # Tokenize initial prompt once
            messages = prompts[idx]
            if messages and messages[-1]["role"] == "assistant":
                conversation = messages[:-1]
                assistant_prefix = messages[-1]["content"]
            else:
                conversation = messages
                assistant_prefix = ""

            formatted = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            formatted += assistant_prefix

            # accumulated_tokens: list of token segments
            accumulated_tokens = [tokenizer.encode(formatted, add_special_tokens=False)]

            while True:
                remaining = max_budget - model_token_count
                if remaining <= 0:
                    finish_reason = "length"
                    break

                params = self._sampling_params(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=remaining,
                    n=1,
                    stop=[request_tag, answer_tag],
                    include_stop_str_in_output=True,
                    seed=(base_seed + idx) if base_seed is not None else None,
                )

                # Flatten accumulated tokens
                all_tokens = []
                for segment in accumulated_tokens:
                    all_tokens.extend(segment)

                request_id = f"req_{idx}_{turns}_{uuid.uuid4().hex[:8]}"

                final_output = None
                async for output in engine.generate(
                    TokensPrompt(prompt_token_ids=all_tokens), params, request_id
                ):
                    if output.finished:
                        final_output = output
                        break

                if final_output is None:
                    finish_reason = "error"
                    break

                generated_text = final_output.outputs[0].text
                gen_tokens = list(final_output.outputs[0].token_ids)

                turns += 1
                accumulated_text += generated_text
                accumulated_tokens.append(gen_tokens)
                model_token_count += len(gen_tokens)

                # Check if answer tag found
                if answer_tag in accumulated_text:
                    finish_reason = "stop"
                    break

                # Check if hint was requested
                if request_tag in generated_text:
                    if last_given_index + 1 < len(hints):
                        next_idx = last_given_index + 1
                        hint_text, new_last = hints[next_idx], next_idx

                        if hint_text is not None:
                            last_given_index = new_last
                            num_hints_used += 1

                            # Inject hint — tokenize independently (matches RL splicing)
                            hint_response = _hint_injection(hint_text, self.config.nested_request)
                            hint_tokens = tokenizer.encode(hint_response, add_special_tokens=False)
                            accumulated_text += hint_response
                            accumulated_tokens.append(hint_tokens)

                            if self.config.verbose:
                                print(f"Prompt {idx + 1} got hint (last_given={new_last}), continuing...")
                        else:
                            warning = _hint_injection("No more hints available.", self.config.nested_request)
                            warning_tokens = tokenizer.encode(warning, add_special_tokens=False)
                            accumulated_text += warning
                            accumulated_tokens.append(warning_tokens)
                            num_hints_used += 1
                    else:
                        warning = _hint_injection("No more hints available.", self.config.nested_request)
                        warning_tokens = tokenizer.encode(warning, add_special_tokens=False)
                        accumulated_text += warning
                        accumulated_tokens.append(warning_tokens)
                        num_hints_used += 1
                else:
                    # No hint requested and no answer
                    if final_output.outputs[0].finish_reason == "stop" and answer_tag not in accumulated_text:
                        close_text = "</think>\n"
                        close_tokens = tokenizer.encode(close_text, add_special_tokens=False)
                        accumulated_text += close_text
                        accumulated_tokens.append(close_tokens)
                    else:
                        finish_reason = final_output.outputs[0].finish_reason or "length"
                        break

            results[idx] = [{
                "text": accumulated_text,
                "finish_reason": finish_reason,
                "token_count": model_token_count,
                "total_token_count": sum(len(seg) for seg in accumulated_tokens[1:]),
                "num_hints": num_hints_used,
                "turns": turns,
            }]

            completed_count[0] += 1
            if self.config.verbose:
                print(f"Prompt {idx + 1} completed ({finish_reason}) - {completed_count[0]}/{num_prompts}")

        if self.config.verbose:
            print(f"Starting async generation for {num_prompts} prompts")

        tasks = [process_single_prompt(idx) for idx in range(num_prompts)]
        await asyncio.gather(*tasks)

        if self.config.verbose:
            total_hints = sum(r[0]["num_hints"] for r in results if r)
            avg_turns = sum(r[0]["turns"] for r in results if r) / num_prompts
            print(f"\nAsync generation complete: {num_prompts} prompts, {total_hints} total hints, {avg_turns:.1f} avg turns")

        return results

    async def _generate_with_hints_async_text_based(
        self,
        prompts: list[list[dict]],
        hints_list: list[list[str]],
        force_hints_per: list[int],
        request_tag: str,
        answer_tag: str,
        force_hint_token_range: tuple[int, int],
    ) -> list[list[dict]]:
        """Text-based async multi-turn generation for forced hint injection.

        Uses chat template re-application each turn. Only used when force_hints
        is active (SFT data generation), not during eval.
        """
        import copy
        import uuid

        engine = await self._get_engine()
        tokenizer = await self._get_tokenizer()

        num_prompts = len(prompts)
        think_tag = "</think>"
        max_budget = self.config.max_new_tokens

        results = [None] * num_prompts
        completed_count = [0]

        async def process_single_prompt(idx: int):
            import random as _random
            messages = copy.deepcopy(prompts[idx])
            hints = hints_list[idx]
            last_given_index = -1
            accumulated_text = ""
            num_hints_used = 0
            turns = 0
            finish_reason = "length"

            prompt_force = force_hints_per[idx]
            force_rng = _random.Random(42 + idx)

            while True:
                model_tokens = _count_model_tokens(accumulated_text, tokenizer)
                remaining = max_budget - model_tokens
                if remaining <= 0:
                    finish_reason = "length"
                    break

                needs_forcing = prompt_force > 0 and num_hints_used < prompt_force and last_given_index + 1 < len(hints)
                if needs_forcing:
                    cap = force_rng.randint(force_hint_token_range[0], force_hint_token_range[1])
                    params = self._sampling_params(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=min(cap, remaining),
                        n=1,
                        stop=[request_tag, think_tag],
                        include_stop_str_in_output=True,
                        seed=self.config.seed,
                    )
                else:
                    params = self._sampling_params(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=remaining,
                        n=1,
                        stop=[request_tag, answer_tag],
                        include_stop_str_in_output=True,
                        seed=self.config.seed,
                    )

                formatted = self._format_prompt(messages, tokenizer)
                request_id = f"req_{idx}_{turns}_{uuid.uuid4().hex[:8]}"

                final_output = None
                async for output in engine.generate(formatted, params, request_id):
                    if output.finished:
                        final_output = output
                        break

                if final_output is None:
                    finish_reason = "error"
                    break

                generated_text = final_output.outputs[0].text
                turns += 1
                accumulated_text += generated_text

                if answer_tag in accumulated_text:
                    if prompt_force > 0 and num_hints_used < prompt_force and last_given_index + 1 < len(hints):
                        accumulated_text = _close_think_for_request(
                            accumulated_text,
                            _pick_transition(force_rng, self.config.hint_transition),
                            nested=self.config.nested_request,
                        )
                        next_idx = last_given_index + 1
                        hint = hints[next_idx]
                        last_given_index = next_idx
                        num_hints_used += 1
                        forced = "\n<request></request>" + _hint_injection(hint, self.config.nested_request)
                        accumulated_text += forced
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                        continue
                    finish_reason = "stop"
                    break

                if prompt_force > 0 and think_tag in generated_text and request_tag not in generated_text:
                    if num_hints_used < prompt_force and last_given_index + 1 < len(hints):
                        accumulated_text = _close_think_for_request(
                            accumulated_text,
                            _pick_transition(force_rng, self.config.hint_transition),
                            nested=self.config.nested_request,
                        )
                        next_idx = last_given_index + 1
                        hint = hints[next_idx]
                        last_given_index = next_idx
                        num_hints_used += 1
                        forced = "\n<request></request>" + _hint_injection(hint, self.config.nested_request)
                        accumulated_text += forced
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                        if self.config.verbose:
                            print(f"Prompt {idx + 1} forced hint {last_given_index + 1}, continuing...")
                        continue

                if prompt_force > 0 and final_output.outputs[0].finish_reason == "length":
                    if num_hints_used < prompt_force and last_given_index + 1 < len(hints):
                        accumulated_text = _close_think_for_request(
                            accumulated_text,
                            _pick_transition(force_rng, self.config.hint_transition),
                            nested=self.config.nested_request,
                        )
                        next_idx = last_given_index + 1
                        hint = hints[next_idx]
                        last_given_index = next_idx
                        num_hints_used += 1
                        forced = "\n<request></request>" + _hint_injection(hint, self.config.nested_request)
                        accumulated_text += forced
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                        if self.config.verbose:
                            print(f"Prompt {idx + 1} token-capped, forced hint {last_given_index + 1}, continuing...")
                        continue

                if request_tag in generated_text:
                    if last_given_index + 1 < len(hints):
                        next_idx = last_given_index + 1
                        hint_text, new_last = hints[next_idx], next_idx

                        if hint_text is not None:
                            last_given_index = new_last
                            num_hints_used += 1

                            hint_response = _hint_injection(hint_text, self.config.nested_request)
                            accumulated_text += hint_response

                            if messages and messages[-1]["role"] == "assistant":
                                messages[-1]["content"] += generated_text + hint_response
                            else:
                                messages.append({
                                    "role": "assistant",
                                    "content": generated_text + hint_response,
                                })

                            if self.config.verbose:
                                print(f"Prompt {idx + 1} got hint (last_given={new_last}), continuing...")
                        else:
                            warning = _hint_injection("No more hints available.", self.config.nested_request)
                            accumulated_text += warning
                            num_hints_used += 1
                            messages = copy.deepcopy(prompts[idx])
                            if messages and messages[-1]["role"] == "assistant":
                                messages[-1]["content"] += accumulated_text
                            else:
                                messages.append({"role": "assistant", "content": accumulated_text})
                    else:
                        warning = _hint_injection("No more hints available.", self.config.nested_request)
                        accumulated_text += warning
                        num_hints_used += 1
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                else:
                    if final_output.outputs[0].finish_reason == "stop" and answer_tag not in accumulated_text:
                        accumulated_text += "</think>\n"
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                    else:
                        finish_reason = final_output.outputs[0].finish_reason or "length"
                        break

            results[idx] = [{
                "text": accumulated_text,
                "finish_reason": finish_reason,
                "token_count": _count_model_tokens(accumulated_text, tokenizer),
                "total_token_count": len(tokenizer.encode(accumulated_text)),
                "num_hints": num_hints_used,
                "turns": turns,
            }]

            completed_count[0] += 1
            if self.config.verbose:
                print(f"Prompt {idx + 1} completed ({finish_reason}) - {completed_count[0]}/{num_prompts}")

        if self.config.verbose:
            print(f"Starting async generation for {num_prompts} prompts")

        tasks = [process_single_prompt(idx) for idx in range(num_prompts)]
        await asyncio.gather(*tasks)

        if self.config.verbose:
            total_hints = sum(r[0]["num_hints"] for r in results if r)
            avg_turns = sum(r[0]["turns"] for r in results if r) / num_prompts
            print(f"\nAsync generation complete: {num_prompts} prompts, {total_hints} total hints, {avg_turns:.1f} avg turns")

        return results
