"""Reward functions for competition_math task."""

import re


def extract_answer(solution_str: str) -> str | None:
    """Extract answer from <answer>...</answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def check_answer(predicted: str, correct: str) -> bool:
    """Check if predicted answer matches correct answer using math-verify."""
    if not predicted or not correct:
        return False

    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

    # Primary: parse both as LaTeX (wrapped in $...$)
    try:
        gold_parsed = parse(f"${correct}$", extraction_config=[LatexExtractionConfig()])
    except Exception:
        gold_parsed = []

    try:
        pred_parsed = parse(f"${predicted}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    except Exception:
        pred_parsed = []

    if gold_parsed and pred_parsed:
        try:
            if verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass

    # Fallback: try both as plain expressions
    try:
        gold_plain = parse(correct, extraction_config=[ExprExtractionConfig()])
        pred_plain = parse(predicted, extraction_config=[ExprExtractionConfig()])
        if gold_plain and pred_plain:
            return verify(gold_plain, pred_plain)
    except Exception:
        pass

    return False


def get_num_hints(solution_str: str) -> int:
    """Count all hint request/response exchanges (including exhausted ones)."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return len(responses)


# What the rollout loop answers a request with once the hints are used up
# (verl/verl/workers/rollout/hint_loop.py). Kept literal here because this
# module is loaded by file path and must not import from verl.
NO_MORE_HINTS = "No more hints available."


def get_num_exhausted_requests(solution_str: str) -> int:
    """Count requests made after the hints ran out."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return sum(r.strip() == NO_MORE_HINTS for r in responses)


def apply_exhausted_penalty(
    score: float,
    num_exhausted: int,
    exhausted_penalty: float,
    max_exhausted_requests: int | None,
) -> float:
    """Charge exhausted_penalty per request made after the hints ran out;
    more than max_exhausted_requests of them scores 0."""
    if max_exhausted_requests is not None and num_exhausted > max_exhausted_requests:
        return 0
    return max(score - exhausted_penalty * num_exhausted, 0)


def hint_cost(hints_used: int, hint_penalty: float, shape: str = "linear",
              alpha: float = 1.0) -> float:
    """Fraction of the base score forfeited for using ``hints_used`` hints.

    ``linear``     charges alpha*hint_penalty per hint.
    ``quadratic``  charges alpha*hint_penalty*k for the k-th hint, so with the
                   default hint_penalty=0.1 the marginal costs run
                   -0.1a, -0.2a, -0.3a, -0.4a, -0.5a and the cumulative cost is
                   alpha*hint_penalty*k(k+1)/2.

    alpha is the sweep knob; it scales both shapes and defaults to 1.0, so a
    linear run with alpha unset scores exactly as it did before.

    The cost is not clamped here; the caller clamps the final score at 0. With
    every problem carrying 5 hints, quadratic at hint_penalty=0.1 goes
    degenerate at alpha >= 0.6, where a *correct* full-ladder answer scores at
    or below the format_score paid for a *wrong* one.
    """
    if shape == "linear":
        return alpha * hint_penalty * hints_used
    if shape == "quadratic":
        return alpha * hint_penalty * hints_used * (hints_used + 1) / 2
    raise ValueError(f"unknown hint_penalty_shape {shape!r}; expected linear or quadratic")


def has_malformed_structure(solution_str: str) -> bool:
    """Validate the overall tag structure of the response.

    The response starts inside an open <think> block (from assistant prefix).
    Valid structure:
        ([text]</think><request></request><response>...</response><think>)*
        [text]</think>\\n\\n<answer>...</answer>

    Validates:
    - Correct tag sequence (state machine)
    - <request> tags are tight (no content inside)
    - No spurious/duplicate tags (e.g. double </think>)

    Returns:
        True if the structure is malformed, False if valid.
    """
    # Content check: <request> tags must be tight (no content inside)
    if solution_str.count('<request>') != len(re.findall(r'<request></request>', solution_str)):
        return True

    # Extract all structural tags in order
    tag_pattern = r'(</think>|<think>|<request>|</request>|<response>|</response>|<answer>|</answer>)'
    tags = re.findall(tag_pattern, solution_str)

    if not tags:
        return True

    i = 0
    while i < len(tags):
        if tags[i] != '</think>':
            return True
        i += 1

        if i >= len(tags):
            return True

        if tags[i] == '<request>':
            expected = ['<request>', '</request>', '<response>', '</response>', '<think>']
            for expected_tag in expected:
                if i >= len(tags) or tags[i] != expected_tag:
                    return True
                i += 1
        elif tags[i] == '<answer>':
            if i + 1 >= len(tags) or tags[i + 1] != '</answer>':
                return True
            i += 2
            return i != len(tags)
        else:
            return True

    return True


def has_malformed_structure_nested(solution_str: str) -> bool:
    """Validate the tag structure of an inline-request response.

    Used by methods with nested_request: the response starts inside an open
    <think> block (from the assistant prefix) and stays there until the very
    end, so </think> is reached exactly once, immediately before the answer.

        ([text]<request></request><response>...</response>)*
        [text]</think><answer>...</answer>

    A <think> tag anywhere in the response is malformed: the block is never
    reopened because it is never closed. Contrast has_malformed_structure,
    where the model leaves the think block to ask and re-enters it.

    Returns:
        True if the structure is malformed, False if valid.
    """
    # <request> tags must be tight (no content inside)
    if solution_str.count('<request>') != len(re.findall(r'<request></request>', solution_str)):
        return True

    tag_pattern = r'(</think>|<think>|<request>|</request>|<response>|</response>|<answer>|</answer>)'
    tags = re.findall(tag_pattern, solution_str)

    if not tags:
        return True

    i, n = 0, len(tags)
    while i < n and tags[i] == '<request>':
        for expected_tag in ('<request>', '</request>', '<response>', '</response>'):
            if i >= n or tags[i] != expected_tag:
                return True
            i += 1

    if i >= n or tags[i] != '</think>':
        return True
    i += 1

    if i >= n:
        return True

    if tags[i] == '<answer>':
        if i + 1 >= n or tags[i + 1] != '</answer>':
            return True
        i += 2
    else:
        return True

    return i != n


def compute_score(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_score: float = 0.1,
    score: float = 1.0,
    penalize_hint: bool = False,
    hint_penalty: float = 0.1,
    hint_penalty_shape: str = "linear",
    hint_penalty_alpha: float = 1.0,
    hint_bonus: float = 0.0,
    nested_request: bool = False,
    exhausted_penalty: float = 0.0,
    max_exhausted_requests: int | None = None,
    **kwargs,
) -> dict:
    """
    Compute reward score for competition_math task.

    Args:
        data_source: Data source identifier
        solution_str: Model's complete response
        ground_truth: Dict with 'answer' key containing correct answer
        extra_info: Additional context
        format_score: Partial credit for well-formatted but wrong answer
        score: Full score for correct answer
        penalize_hint: Whether to penalize hint usage
        hint_penalty: Penalty per hint used (multiplicative)
        hint_penalty_shape: "linear" (cost = hint_penalty*k) or "quadratic"
            (cost = hint_penalty*k*(k+1)/2, i.e. the k-th hint costs
            hint_penalty*k). See hint_cost.
        hint_bonus: Bonus added to format_score when hints were used and
            answer is wrong but formatted (default 0.0, no bonus)
        nested_request: Score against the inline-request grammar, where
            <request> sits inside <think> (default False)
        exhausted_penalty: Subtracted from the score for each request made
            after the hints ran out, independent of hint_penalty (default 0.0)
        max_exhausted_requests: More exhausted requests than this scores 0
            (default None, no limit)

    Returns:
        Dict with score and metadata
    """
    correct_answer = ground_truth.get("answer", "")
    verifier_label = ground_truth.get("correct") if "answer" not in ground_truth else None
    do_print = False

    if do_print:
        print(f"--------------------------------")
        print(f"Correct answer: {correct_answer}")
        print(f"Solution: {solution_str[:500]}...")

    # Counted before the malformed check: a truncated rollout still consumed
    # whatever hints it was given, and reporting 0 for it makes the logged
    # hint-usage rate track the malformed rate instead of actual hint use.
    num_hints = get_num_hints(solution_str)
    num_exhausted = get_num_exhausted_requests(solution_str)
    hints_used = num_hints - num_exhausted

    # Structural validation: verify entire tag sequence is well-formed.
    # nested_request methods keep the request inside <think>, a different and
    # incompatible grammar, so the validator is selected rather than patched.
    validator = (has_malformed_structure_nested if nested_request
                 else has_malformed_structure)
    if validator(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {
            "score": 0,
            "score_wo_hint_penalty": 0,
            "num_hints": num_hints,
            "num_exhausted": num_exhausted,
            "abstained": False,
            "correct": False,
            "malformed": True,
        }

    # Extract predicted answer
    predicted = extract_answer(solution_str)

    if predicted is None:
        if do_print:
            print(f"No answer found")
        return {
            "score": 0,
            "score_wo_hint_penalty": 0,
            "num_hints": num_hints,
            "num_exhausted": num_exhausted,
            "abstained": False,
            "correct": False,
            "malformed": True,
        }

    # Check correctness. method_c's verifier ground truth carries a "correct"
    # label (0/1) instead of a math "answer" -- the predicted <answer>0/1</answer>
    # tag is compared directly against that label rather than math-verified.
    if verifier_label is not None:
        is_correct = predicted.strip() in ("0", "1") and int(predicted.strip()) == int(verifier_label)
    else:
        is_correct = check_answer(predicted, correct_answer)

    if is_correct:
        if do_print:
            print(f"Correct! Predicted: {predicted}")
        base_score = score
        final_score = base_score
        if penalize_hint:
            final_score = max(
                base_score * (1 - hint_cost(hints_used, hint_penalty, hint_penalty_shape, hint_penalty_alpha)), 0)
        final_score = apply_exhausted_penalty(
            final_score, num_exhausted, exhausted_penalty, max_exhausted_requests)
        return {
            "score": final_score,
            "score_wo_hint_penalty": base_score,
            "num_hints": num_hints,
            "num_exhausted": num_exhausted,
            "abstained": False,
            "correct": True,
            "malformed": False,
        }
    else:
        final_format_score = format_score
        if hint_bonus > 0:
            final_format_score = format_score + hint_bonus * hints_used
        final_score = apply_exhausted_penalty(
            final_format_score, num_exhausted, exhausted_penalty, max_exhausted_requests)
        if do_print:
            print(f"Wrong. Predicted: {predicted}, Expected: {correct_answer}")
            if hint_bonus > 0:
                print(f"  Hint bonus applied: {format_score} + {hint_bonus}*{hints_used} = {final_format_score}")
        return {
            "score": final_score,
            "score_wo_hint_penalty": final_format_score,
            "num_hints": num_hints,
            "num_exhausted": num_exhausted,
            "abstained": False,
            "correct": False,
            "malformed": False,
        }


def compute_score_hint(
    data_source,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    hint_penalty: float = 0.1,
    hint_bonus: float = 0.0,
    format_score: float = 0.1,
    **kwargs,
) -> dict:
    """
    Reward function that penalizes hint usage.

    Each hint used reduces the score by hint_penalty (multiplicative).
    Final score = base_score * (1 - hint_cost(num_hints, hint_penalty, shape)),
    where shape is hint_penalty_shape, passed through kwargs: "linear" (the
    default, cost hint_penalty*k) or "quadratic" (cost hint_penalty*k(k+1)/2).

    If hint_bonus > 0, wrong-but-formatted answers that used hints
    get format_score + hint_bonus (encourages hint exploration).
    """
    return compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        penalize_hint=True,
        hint_penalty=hint_penalty,
        hint_bonus=hint_bonus,
        format_score=format_score,
        **kwargs,
    )
