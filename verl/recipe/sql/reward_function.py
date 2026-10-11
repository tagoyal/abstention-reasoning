"""Reward function for Text-to-SQL task across Spider, BIRD, SParC, and CoSQL.

Scores model predictions by executing predicted and ground-truth SQL queries
against SQLite databases in read-only mode, with result normalization,
opcode-level timeouts, connection caching, and gold query caching.
"""

import importlib.util
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Load nested tag grammar from shared
_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "shared" / "nested_grammar.py"
if _GRAMMAR_PATH.exists():
    _spec = importlib.util.spec_from_file_location("nested_grammar", _GRAMMAR_PATH)
    _grammar = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_grammar)
    has_malformed_structure_nested = _grammar.has_malformed_structure_nested
else:
    def has_malformed_structure_nested(s: str) -> bool:
        return False

# Database path search location.
# DATABASE_PATH points directly at a `databases/` folder containing every DB
# the active dataset variant's primitives.json needs (e.g.
# data/sql_conceptual/databases or data/sql_partial_sql/databases). 
#
# Read lazily (per-call) rather than cached at import time: this module gets
# imported once (and cached) by pipeline.tasks.sql.task the first time any
# command calls get_task("sql"), which can happen before DATABASE_PATH is set
# in the environment. Re-reading os.environ on every lookup means it always
# sees whatever DATABASE_PATH is current, regardless of when/whether it was
# set relative to import time.
def _get_db_search_paths() -> list[Path]:
    database_path = os.environ.get("DATABASE_PATH")
    return [Path(database_path)] if database_path else []


# Global cache for connections and gold execution results. Keyed by
# (DATABASE_PATH, db_id) so a process that switches DATABASE_PATH mid-run
# (e.g. scoring two dataset variants back to back) never serves a stale path
# cached under a different DATABASE_PATH.
_DB_PATH_CACHE: dict[tuple[str | None, str], Path | None] = {}
_CONN_CACHE: dict[str, sqlite3.Connection] = {}
_GOLD_CACHE: dict[tuple[str, str], tuple[bool, list[tuple] | None, str | None]] = {}

# Guards the small cache dicts above only (cheap, in-memory lookups) -- never
# held across a query execution.
_CACHE_LOCK = threading.Lock()

# Per-db_id locks guarding actual query execution on that db's cached
# connection. A single global lock here would serialize every query across
# every database process-wide, which is the real bottleneck when callers try
# to parallelize verification (e.g. a ThreadPoolExecutor in the caller):
# with one lock, only one query runs at a time no matter how many threads are
# used. Per-connection locks still serialize concurrent use of the *same*
# SQLite connection (needed because set_progress_handler mutates shared
# connection state) while letting queries against different databases run
# truly concurrently -- which is exactly the I/O-bound, many-small-files
# workload this verifier sees (Spider/BIRD/CoSQL split across hundreds of
# separate .sqlite files).
_CONN_LOCKS: dict[str, threading.Lock] = {}


def _get_conn_lock(db_id: str) -> threading.Lock:
    with _CACHE_LOCK:
        lock = _CONN_LOCKS.get(db_id)
        if lock is None:
            lock = threading.Lock()
            _CONN_LOCKS[db_id] = lock
        return lock

NO_MORE_HINTS = "No more hints available."


def _find_db_path(db_id: str) -> Path | None:
    """Find the SQLite database file for a given db_id with caching."""
    if not db_id:
        return None
    database_path = os.environ.get("DATABASE_PATH")
    cache_key = (database_path, db_id)
    with _CACHE_LOCK:
        if cache_key in _DB_PATH_CACHE:
            return _DB_PATH_CACHE[cache_key]

    found = None
    for base_dir in _get_db_search_paths():
        if not base_dir.exists():
            continue
        candidates = [
            base_dir / db_id / f"{db_id}.sqlite",
            base_dir / f"{db_id}.sqlite",
            base_dir / db_id / f"{db_id}.db",
            base_dir / f"{db_id}.db",
        ]
        for cand in candidates:
            if cand.is_file():
                found = cand
                break
        if found:
            break

    with _CACHE_LOCK:
        _DB_PATH_CACHE[cache_key] = found
    return found


def _get_connection(db_id: str, db_path: Path) -> sqlite3.Connection:
    """Get or open a cached read-only SQLite connection."""
    with _CACHE_LOCK:
        if db_id in _CONN_CACHE:
            return _CONN_CACHE[db_id]

        uri = f"file:{db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        _CONN_CACHE[db_id] = conn
        return conn


def _normalize_cell(val: Any) -> Any:
    """Normalize cell values for robust table comparison."""
    if val is None:
        return None
    if isinstance(val, float):
        return round(val, 3)
    if isinstance(val, (int, bool)):
        return val
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, bytes):
        return val
    return str(val).strip().lower()


def _normalize_row(row: tuple) -> tuple:
    return tuple(_normalize_cell(x) for x in row)


def _execute_query(
    db_id: str,
    db_path: Path,
    sql: str,
    timeout: float = 3.0,
) -> tuple[bool, list[tuple] | None, str | None]:
    """Execute SQL query using cached connection with timeout progress handler."""
    conn = _get_connection(db_id, db_path)
    start_time = time.time()

    def _timeout_handler():
        if time.time() - start_time > timeout:
            return 1  # Abort query
        return 0

    with _get_conn_lock(db_id):
        try:
            conn.set_progress_handler(_timeout_handler, 1000)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            normalized = [_normalize_row(r) for r in rows]
            return True, normalized, None
        except Exception as e:
            return False, None, str(e)
        finally:
            conn.set_progress_handler(None, 0)


def _verify_sql(
    db_id: str | None,
    predicted_sql: str,
    gold_sql: str,
) -> tuple[bool, dict]:
    """Verify execution equivalence of predicted and gold queries."""
    db_path = _find_db_path(db_id) if db_id else None

    if db_path is None:
        raise FileNotFoundError(
            f"Could not locate SQLite database for db_id={db_id!r}. "
            f"Ensure DATABASE_PATH is set and points at a databases/ folder "
            f"containing this db_id."
        )

    cache_key = (db_id, gold_sql)
    with _CACHE_LOCK:
        cached_gold = _GOLD_CACHE.get(cache_key)

    if cached_gold is not None:
        gold_ok, gold_res, gold_err = cached_gold
    else:
        gold_ok, gold_res, gold_err = _execute_query(db_id, db_path, gold_sql)
        with _CACHE_LOCK:
            _GOLD_CACHE[cache_key] = (gold_ok, gold_res, gold_err)

    if not gold_ok:
        return False, {
            "eval_method": "execution",
            "db_found": True,
            "error": f"gold_execution_error: {gold_err}",
        }

    pred_ok, pred_res, pred_err = _execute_query(db_id, db_path, predicted_sql)
    if not pred_ok:
        return False, {
            "eval_method": "execution",
            "db_found": True,
            "error": f"pred_execution_error: {pred_err}",
        }

    has_order_by = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))
    if has_order_by:
        is_correct = (gold_res == pred_res)
    else:
        is_correct = (Counter(gold_res) == Counter(pred_res))

    return is_correct, {
        "eval_method": "execution",
        "db_found": True,
        "has_order_by": has_order_by,
        "gold_rows": len(gold_res) if gold_res is not None else 0,
        "pred_rows": len(pred_res) if pred_res is not None else 0,
    }


def _extract_boxed_answer(text: str) -> str | None:
    """Extract contents of \\boxed{...}, handling nested braces."""
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None

    start = match.end()
    depth = 1
    pos = start

    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1

    if depth == 0:
        return text[start:pos - 1]
    return None


def extract_answer(solution_str: str) -> str | None:
    """
    Extract SQL query from solution string.
    Checks inside <answer>...</answer> first, then \\boxed{...},
    and strips markdown code fences if present.
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    raw_answer = matches[-1].group(1).strip() if matches else solution_str

    boxed = _extract_boxed_answer(raw_answer)
    ans = boxed if boxed is not None else (raw_answer if matches else None)
    if ans is None:
        boxed_fallback = _extract_boxed_answer(solution_str)
        if boxed_fallback is not None:
            ans = boxed_fallback

    if ans is None:
        return None

    ans = ans.strip()
    code_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', ans, re.DOTALL | re.IGNORECASE)
    if code_match:
        ans = code_match.group(1).strip()

    ans = ans.rstrip(";")
    return ans


def get_num_hints(solution_str: str) -> int:
    """Count all hint request/response exchanges."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return len(responses)


def get_num_exhausted_requests(solution_str: str) -> int:
    """Count requests made after hints ran out."""
    responses = re.findall(r'<response>(.*?)</response>', solution_str, re.DOTALL)
    return sum(r.strip() == NO_MORE_HINTS for r in responses)


def apply_exhausted_penalty(
    score: float,
    num_exhausted: int,
    exhausted_penalty: float,
    max_exhausted_requests: int | None,
) -> float:
    """Apply penalty for hint requests made after hints were exhausted."""
    if max_exhausted_requests is not None and num_exhausted > max_exhausted_requests:
        return 0.0
    return max(score - exhausted_penalty * num_exhausted, 0.0)


def hint_cost(
    hints_used: int,
    hint_penalty: float,
    shape: str = "linear",
    alpha: float = 1.0,
    base: float = 0.0,
) -> float:
    """Fraction of base score forfeited for using hints.

    base is a fixed cost charged once if any hint was used (not scaled by alpha).
    """
    if hints_used <= 0:
        return 0.0
    if shape == "linear":
        return base + alpha * hint_penalty * hints_used
    if shape == "quadratic":
        return base + alpha * hint_penalty * hints_used * (hints_used + 1) / 2
    raise ValueError(f"unknown hint_penalty_shape {shape!r}; expected linear or quadratic")


def has_malformed_structure(solution_str: str) -> bool:
    """Validate overall tag structure of the response."""
    if solution_str.count('<request>') != len(re.findall(r'<request></request>', solution_str)):
        return True

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
    hint_penalty_base: float = 0.0,
    hint_bonus: float = 0.0,
    nested_request: bool = False,
    exhausted_penalty: float = 0.0,
    max_exhausted_requests: int | None = None,
    **kwargs,
) -> dict:
    """Compute reward score for Text-to-SQL rollouts."""
    correct_answer = ground_truth.get("answer", "")
    verifier_label = ground_truth.get("correct") if "answer" not in ground_truth else None
    db_id = (
        ground_truth.get("db_id")
        or ground_truth.get("variant")
        or (extra_info.get("interaction_kwargs", {}).get("ground_truth", {}).get("db_id") if extra_info else None)
    )

    num_hints = get_num_hints(solution_str)
    num_exhausted = get_num_exhausted_requests(solution_str)
    hints_used = num_hints - num_exhausted

    # Structural validation
    validator = (has_malformed_structure_nested if nested_request else has_malformed_structure)
    if validator(solution_str):
        return {
            "score": 0.0,
            "score_wo_hint_penalty": 0.0,
            "num_hints": num_hints,
            "num_exhausted": num_exhausted,
            "abstained": False,
            "correct": False,
            "malformed": True,
        }

    # Extract predicted SQL answer
    predicted = extract_answer(solution_str)
    if predicted is None:
        return {
            "score": 0.0,
            "score_wo_hint_penalty": 0.0,
            "num_hints": num_hints,
            "num_exhausted": num_exhausted,
            "abstained": False,
            "correct": False,
            "malformed": True,
        }

    # Check correctness
    if verifier_label is not None:
        is_correct = predicted.strip() in ("0", "1") and int(predicted.strip()) == int(verifier_label)
    else:
        is_correct, _ = _verify_sql(db_id, predicted, correct_answer)

    if is_correct:
        base_score = score
        final_score = base_score
        if penalize_hint:
            cost = hint_cost(hints_used, hint_penalty, hint_penalty_shape, hint_penalty_alpha, hint_penalty_base)
            final_score = max(base_score * (1.0 - cost), 0.0)
        final_score = apply_exhausted_penalty(
            final_score, num_exhausted, exhausted_penalty, max_exhausted_requests
        )
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
            final_format_score, num_exhausted, exhausted_penalty, max_exhausted_requests
        )
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
    """Reward function variant that penalizes hint usage."""
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
