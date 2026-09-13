import re
import importlib.util
from collections import Counter
from pathlib import Path

# The inline-request grammar is defined once, in verl/recipe/shared, because the
# SFT filter and this scorer have to agree on what a valid nested response looks
# like. Loaded by path: recipes are standalone files, not a package.
_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "shared" / "nested_grammar.py"
_spec = importlib.util.spec_from_file_location("nested_grammar", _GRAMMAR_PATH)
_grammar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grammar)
has_malformed_structure_nested = _grammar.has_malformed_structure_nested


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    """if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]"""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def validate_equation(equation_str, available_numbers):
    """Validate that equation uses all and only the numbers in available_numbers, with exact counts."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]

        # Cast to Python ints (numpy int64 from parquet can cause Counter mismatch)
        available_numbers = [int(n) for n in available_numbers]

        # Compare exact counts
        return Counter(numbers_in_eq) == Counter(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().=\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        
        def strip_trailing_result(equation_str):
            """
            If the equation ends with '= <number>' (optionally with whitespace), remove it.
            """
            return re.sub(r'\s*=\s*\d+\s*$', '', equation_str)

        # Evaluate the equation with restricted globals and locals
        result = eval(strip_trailing_result(equation_str), {"__builtins__": None}, {})
        return result
    except Exception:
        return None
    
def get_num_hints(solution_str):
    """Count all hint request/response exchanges (including exhausted ones)."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return len(responses)


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

    Duplicated from recipe/competition_math/reward_function.py: the two recipes
    have no cross-import path today and this runs inside the rollout workers,
    where a new import is the wrong thing to discover at step 1. Keep in step.

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


def has_malformed_structure(solution_str):
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
        return True  # No tags at all

    # Validate tag sequence with a state machine
    # Expected: (</think> <request> </request> <response> </response> <think>)* </think> <answer> </answer>
    i = 0
    while i < len(tags):
        # Must start each cycle with </think> (closing current think block)
        if tags[i] != '</think>':
            return True
        i += 1

        if i >= len(tags):
            return True  # Ended after </think> without terminal tag

        if tags[i] == '<request>':
            # Hint exchange: <request> </request> <response> </response> <think>
            expected = ['<request>', '</request>', '<response>', '</response>', '<think>']
            for expected_tag in expected:
                if i >= len(tags) or tags[i] != expected_tag:
                    return True
                i += 1
            # Loop back to expect next </think>
        elif tags[i] == '<answer>':
            # Terminal: <answer> </answer>
            if i + 1 >= len(tags) or tags[i + 1] != '</answer>':
                return True
            i += 2
            return i != len(tags)  # Malformed if extra tags after
        else:
            return True  # Unexpected tag after </think>

    return True  # Ran out of tags without proper termination


def compute_score(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., penalize_hint=False, hint_penalty=0.2, hint_penalty_shape='linear', hint_penalty_alpha=1.0, hint_bonus=0.0, nested_request=False, **kwargs):
    """The scoring function for countdown
    """
    #format_score = 0
    verifier_label = ground_truth.get("correct") if "target" not in ground_truth else None
    target = ground_truth.get('target')
    numbers = ground_truth.get('numbers')

    equation = extract_solution(solution_str=solution_str)
    do_print = False

    """if len(solution_str.split()) < 200:
        return 0"""

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    # Structural validation: verify entire tag sequence is well-formed.
    # nested_request methods keep the request inside <think>, a different and
    # incompatible grammar, so the validator is selected rather than patched.
    validator = (has_malformed_structure_nested if nested_request
                 else has_malformed_structure)
    if validator(solution_str):
        if do_print:
            print(f"Malformed structure detected - awarding 0")
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": 0, "abstained": False, "malformed": True, "correct": False}

    num_hints = get_num_hints(solution_str)

    if equation is None:
        if do_print:
            print(f"No equation found or length too short")
        return {"score": 0, "score_wo_hint_penalty": 0, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

    # method_c's verifier ground truth carries a "correct" label (0/1) instead
    # of "target"/"numbers" -- the predicted <answer>0/1</answer> tag is
    # compared directly against that label rather than evaluated as an
    # expression.
    if verifier_label is not None:
        predicted_label = equation.strip()
        is_correct = predicted_label in ("0", "1") and int(predicted_label) == int(verifier_label)
        final_score = score if is_correct else format_score
        return {"score": final_score, "score_wo_hint_penalty": final_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": is_correct}

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            if penalize_hint:
                penalized_hints = min(num_hints, 5)
                final_score = max(
                    score * (1 - hint_cost(penalized_hints, hint_penalty, hint_penalty_shape, hint_penalty_alpha)), 0)
            else:
                final_score = score
            return {"score": final_score, "score_wo_hint_penalty": score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": True}
        else:
            final_format_score = format_score
            if hint_bonus > 0 and num_hints > 0:
                penalized_hints = min(num_hints, 5)
                final_format_score = format_score + hint_bonus * penalized_hints
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
                if hint_bonus > 0 and num_hints > 0:
                    print(f"  Hint bonus applied: {format_score} + {hint_bonus}*{num_hints} = {final_format_score}")
            return {"score": final_format_score, "score_wo_hint_penalty": final_format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}
    except:
        if do_print:
            print(f"Error evaluating equation")
        return {"score": format_score, "score_wo_hint_penalty": format_score, "num_hints": num_hints, "abstained": False, "malformed": False, "correct": False}


def compute_score_hint(data_source, solution_str, ground_truth, extra_info, method='strict', format_score=0.1, score=1., hint_penalty=0.1, hint_bonus=0.0, **kwargs):
    """The scoring function for countdown that penalizes hint usage.

    Args:
        hint_penalty: Multiplicative penalty per hint (default 0.1).
            Final score = accuracy * (1 - hint_penalty * num_hints)
        hint_bonus: Bonus added to format_score when hints were used and
            answer is wrong but formatted (default 0.0, no bonus).
    """
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        method=method, format_score=format_score, score=score,
        penalize_hint=True, hint_penalty=hint_penalty, hint_bonus=hint_bonus, **kwargs
    )
