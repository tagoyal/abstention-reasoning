"""Generate incremental solution hints for Text-to-SQL problems using TRAPI.

Divides the reasoning and query formulation into 5 sequential hint segments:
- hint_1: High-level table/schema identification and query strategy
- hint_2: Table relationships, join paths, or initial query framing
- hint_3: Filtering conditions or predicate logic (may include partial SQL syntax like WHERE clause fragments)
- hint_4: Aggregations, groupings, or subquery construction (may include partial SQL expressions)
- hint_5: High-level guidance on the final output projection/order without revealing the full query or complete SQL

Each hint is stored INDEPENDENTLY (not cumulative) for token efficiency.
At runtime, concatenate hint_1 + ... + hint_k to reveal the first k/5 of the hints.

Usage:
    python -m pipeline.tasks.sql.rewrite_solutions_prefix_hints \
        --input data/sql/primitives_spider_dev.json \
        --output data/sql/primitives_spider_dev_hints.json \
        --batch-size 20
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import tiktoken
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AsyncAzureOpenAI, RateLimitError
from tqdm.asyncio import tqdm_asyncio

# TRAPI configuration
TRAPI_SCOPE = "api://trapi/.default"
TRAPI_API_VERSION = "2024-10-21"
TRAPI_INSTANCE = "redmond/interactive"
TRAPI_ENDPOINT = f"https://trapi.research.microsoft.com/{TRAPI_INSTANCE}"

# Initialize tokenizer (cl100k_base used by GPT-4 / GPT-4o / GPT-5)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_ENCODING.encode(text))


CONCEPTUAL_HINTS_SYSTEM_PROMPT = """\
You are an expert SQL tutor guiding students to solve Text-to-SQL problems step-by-step using progressive hints.

Your task: Given the database schema, question (and optional evidence or intermediate turns), along with the target SQL answer, divide the solution reasoning path into exactly 5 sequential, independent hints (hint_1 to hint_5).

## Guidelines for 5-Step Hint Progression:
1. **Exactly 5 Hints**:
   - You MUST generate exactly 5 hints (`hint_1`, `hint_2`, `hint_3`, `hint_4`, `hint_5`).
   - Break the reasoning into 5 distinct, meaningful logical steps (e.g., conceptual strategy / schema linking, join conditions, filtering logic, aggregation / grouping / subquery formulation, final projection / ordering).
2. **Constructive Actions ONLY (No Negative Hints)**:
   - Every hint must tell the student what TO DO.
   - NEVER output negative statements about what is not needed. Do NOT say "no joins are needed", "no WHERE clause is needed", "no grouping required", "do not add a filter", etc. Focus 100% on constructive steps toward building the query.
3. **Table & Column Mentions Only When Non-Trivial**:
   - Only mention table or column names when it is non-trivial, tricky, requires disambiguation between multiple candidate tables, or involves translating domain evidence/formulas.
   - Do not waste an entire hint merely stating the obvious if a table directly matches the entity in the question.
4. **Use Partial SQL Where Helpful**:
   - Intermediate hints can and should include relevant partial SQL snippets or clauses when it helps clarity (for example: `JOIN ... ON ...`, `WHERE ...`, `GROUP BY ...`, `HAVING ...`, subquery fragments, or arithmetic/aggregate expressions).
   - Do not include SQL in every hint; keep conceptual framing natural.
5. **CRITICAL Rule for the Final Hint (hint_5)**:
   - The final hint (`hint_5`) must NEVER give the complete SQL statement. It should guide the final step (e.g. specifying the projection, ordering, limit, or formatting), leaving the final query assembly for the student.
6. **Leverage Intermediate Turns / Evidence**:
   - If intermediate conversational turns or domain evidence are provided, use them to guide the natural progression of steps.

## Example 1 (Multi-Table Join & Filtering)

**Question:** What are the names of car makers who produced cars in 1970?
**Target SQL:** `SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker JOIN CAR_NAMES AS T3 ON T2.model = T3.model JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id WHERE T4.year = '1970';`

**Output:**
{
  "hint_1": "Recognize that maker names and production years reside in separate tables, requiring a join chain from `CAR_MAKERS` through `MODEL_LIST` and `CAR_NAMES` to `CARS_DATA`.",
  "hint_2": "Connect `CAR_MAKERS` to `MODEL_LIST` on `T1.Id = T2.Maker`.",
  "hint_3": "Complete the path to `CARS_DATA` by joining `MODEL_LIST.model = CAR_NAMES.model` and `CAR_NAMES.MakeId = CARS_DATA.id`.",
  "hint_4": "Filter for the target production year using `WHERE T4.year = '1970'`.",
  "hint_5": "Project the `Maker` column from `CAR_MAKERS`, applying `DISTINCT` to avoid duplicate names for makers with multiple models."
}

## Example 2 (Aggregations, Ratios & Calculations)

**Question:** What percentage of clients who opened their accounts in districts with an average salary over 10000 are women?
**Evidence:** Female refers to gender = 'F'; Average salary is in column A11
**Target SQL:** `SELECT CAST(SUM(T2.gender = 'F') AS REAL) * 100 / COUNT(T2.client_id) FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.A11 > 10000`

**Output:**
{
  "hint_1": "Link client demographics to district economic data using `district.district_id = client.district_id`.",
  "hint_2": "Restrict the records to high-salary regions using the condition `WHERE T1.A11 > 10000`.",
  "hint_3": "Count the female clients in this subset using conditional aggregation `SUM(gender = 'F')`.",
  "hint_4": "Compute the ratio by dividing the female count by total matched clients `COUNT(T2.client_id)` and multiplying by 100.",
  "hint_5": "Cast the numerator to REAL in the final SELECT expression to avoid integer truncation in SQLite."
}

## Example 3 (Subqueries & Nested Aggregation)

**Question:** Find the names of all courses that have enrolled more students than the average course enrollment.
**Target SQL:** `SELECT T1.course_name FROM courses AS T1 JOIN enrollments AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id HAVING count(T2.student_id) > (SELECT avg(c) FROM (SELECT count(student_id) AS c FROM enrollments GROUP BY course_id))`

**Output:**
{
  "hint_1": "Construct an inner subquery to calculate the enrollment count per course: `SELECT count(student_id) AS c FROM enrollments GROUP BY course_id`.",
  "hint_2": "Wrap that subquery with `avg(c)` to compute the benchmark average enrollment across all courses.",
  "hint_3": "Join `courses` with `enrollments` on `course_id` and group records by course to aggregate individual enrollments.",
  "hint_4": "Add a `HAVING count(T2.student_id) > (...)` clause comparing each course's enrollment against the benchmark subquery.",
  "hint_5": "Project `course_name` in the final SELECT to list the qualifying courses."
}

## Output Format:
Return ONLY a JSON object with exactly 5 keys `hint_1`, `hint_2`, `hint_3`, `hint_4`, `hint_5` (no markdown code blocks):
{"hint_1": "...", "hint_2": "...", "hint_3": "...", "hint_4": "...", "hint_5": "..."}
"""

PARTIAL_SQL_HINTS_SYSTEM_PROMPT = """\
You are an expert SQL tutor guiding students to solve Text-to-SQL problems step-by-step using progressive solution hints that provide accompanying partial SQL.

Your task: Given the database schema, question (and optional evidence or intermediate turns), along with the target SQL answer, divide the solution into exactly 5 sequential, independent hint segments (hint_1 to hint_5).

## Core Philosophy:
Drawing a direct parallel with mathematical problem solving (where each step provides both a clear explanation and the concrete equation/work that solves that step), each hint must provide a rich, clear explanation accompanied by the EXACT partial SQL clause, expression, or snippet that solves that step. This ensures the student understands the reasoning and has the exact SQL syntax for that sub-goal, preventing syntax or join errors on previously solved steps.

## Guidelines for 5-Step Progression:
1. **Exactly 5 Hints**:
   - You MUST generate exactly 5 hints (`hint_1`, `hint_2`, `hint_3`, `hint_4`, `hint_5`).
2. **Mix of Explanation and Accompanying Partial SQL**:
   - Every single hint must blend a 1-2 sentence explanation of the logical step with the exact accompanying partial SQL code in backticks (e.g., `FROM ... JOIN ... ON ...`, `WHERE ...`, `GROUP BY ...`, `HAVING ...`, `ORDER BY ... LIMIT ...`, or `SELECT ...`).
   - The explanation provides context (which entities are being related, what condition is being tested, what calculation is being performed), while the partial SQL provides the precise syntax, table aliases, and column names.
3. **Constructive Actions ONLY (No Negative Hints)**:
   - Focus 100% on constructive steps toward building the query. NEVER include negative statements like "no joins needed", "no WHERE clause required", or "do not filter".
4. **Accurate Schema Names and Aliases**:
   - The partial SQL snippet must use the exact table names, column names, aliases, and SQL dialect functions from the target SQL answer.
5. **CRITICAL Rule for the Final Hint (hint_5)**:
   - hint_5 must NOT output the entire complete SQL query as a single finished line.
   - It should guide the final step (e.g. specifying the projection/SELECT clause, sorting/limit, or query assembly), leaving the final full query assembly to the student.
6. **Cumulative Solution Walkthrough**:
   - When hints 1 through 5 are read together, they provide both the complete conceptual walkthrough and all constituent partial SQL building blocks necessary to compose the target query.

## Example 1 (Multi-Table Join & Filtering)

**Question:** What are the names of car makers who produced cars in 1970?
**Target SQL:** `SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker JOIN CAR_NAMES AS T3 ON T2.model = T3.model JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id WHERE T4.year = '1970';`

**Output:**
{
  "hint_1": "Recognize that maker names and production years reside in separate tables, and start by linking `CAR_MAKERS` to `MODEL_LIST`: `FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker`.",
  "hint_2": "Complete the join chain through model names to car production records on matching keys: `JOIN CAR_NAMES AS T3 ON T2.model = T3.model JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id`.",
  "hint_3": "Filter the combined records to restrict specifically to cars manufactured in 1970: `WHERE T4.year = '1970'`.",
  "hint_4": "Select maker names with deduplication to avoid repeated entries for manufacturers with multiple 1970 models: `SELECT DISTINCT T1.Maker`.",
  "hint_5": "Assemble the final query by placing the `SELECT DISTINCT` projection before the 4-table join and year filter."
}

## Example 2 (Aggregations, Ratios & Calculations)

**Question:** What percentage of clients who opened their accounts in districts with an average salary over 10000 are women?
**Evidence:** Female refers to gender = 'F'; Average salary is in column A11
**Target SQL:** `SELECT CAST(SUM(T2.gender = 'F') AS REAL) * 100 / COUNT(T2.client_id) FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.A11 > 10000`

**Output:**
{
  "hint_1": "Link district economic records to client demographic data using their shared district identifier: `FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id`.",
  "hint_2": "Restrict the joined rows to high-salary regions where column A11 exceeds 10,000: `WHERE T1.A11 > 10000`.",
  "hint_3": "Count the number of female clients in this subset using conditional summation: `SUM(T2.gender = 'F')`.",
  "hint_4": "Count the total number of matched clients in these districts to form the denominator: `COUNT(T2.client_id)`.",
  "hint_5": "Compute the female percentage in the SELECT clause, casting the numerator to REAL to avoid integer truncation: `SELECT CAST(SUM(T2.gender = 'F') AS REAL) * 100 / COUNT(T2.client_id)`."
}

## Example 3 (Subqueries & Nested Aggregation)

**Question:** Find the names of all courses that have enrolled more students than the average course enrollment.
**Target SQL:** `SELECT T1.course_name FROM courses AS T1 JOIN enrollments AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id HAVING count(T2.student_id) > (SELECT avg(c) FROM (SELECT count(student_id) AS c FROM enrollments GROUP BY course_id))`

**Output:**
{
  "hint_1": "Construct an inner subquery to calculate the enrollment count per course: `SELECT count(student_id) AS c FROM enrollments GROUP BY course_id`.",
  "hint_2": "Compute the benchmark average enrollment across all courses by averaging the inner subquery counts: `SELECT avg(c) FROM (SELECT count(student_id) AS c FROM enrollments GROUP BY course_id)`.",
  "hint_3": "Join `courses` and `enrollments` on `course_id` and group records by course: `FROM courses AS T1 JOIN enrollments AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id`.",
  "hint_4": "Filter for courses exceeding the benchmark average using a HAVING comparison: `HAVING count(T2.student_id) > (...)`.",
  "hint_5": "Project the qualifying course names in the outer SELECT clause: `SELECT T1.course_name`."
}

## Output Format:
Return ONLY a JSON object with exactly 5 keys `hint_1`, `hint_2`, `hint_3`, `hint_4`, `hint_5` (no markdown code blocks):
{"hint_1": "...", "hint_2": "...", "hint_3": "...", "hint_4": "...", "hint_5": "..."}
"""

PARTIAL_SOLS_SYSTEM_PROMPT = """\
You are a SQL author solving a Text-to-SQL problem out loud, writing the query as you reason through it. Your task: given the database schema, question (and optional evidence), along with the target SQL answer, write out the derivation as a continuous first-person reasoning trace, then split that trace into exactly 5 sequential segments (sol_1 to sol_5).

## Core Philosophy:
Drawing a direct parallel with mathematical problem solving, each segment should read as an actual step of writing the query — the explanation plus the exact SQL fragment it produces — not as an instruction telling a student what to do. Crucially, each SQL fragment is CUMULATIVE: it is the entire query built so far, not an isolated clause. So a model given sol_1..sol_k must be able to continue writing sol_(k+1) directly, extending the same SQL string.

## What You Must Do

1. **Write the query directly, in first person**: state each step the way you'd actually write it while solving live ("I join...", "This gives...", "Now I filter...", "Putting this together..."), including the concrete SQL fragment, not a description of the step ("Use X because Y").

2. **Generate 5 segments** where:
   - **sol_1**: Opens the derivation (identifies the relevant table(s) and writes the first fragment of the query).
   - **sol_2**: Extends sol_1's fragment with the next piece, continuing directly from its result.
   - **sol_3**: Extends sol_2's fragment.
   - **sol_4**: Extends sol_3's fragment.
   - **sol_5**: Finishes the derivation with the complete, final query.

   At runtime, the final SQL fragment in sol_5 is the full target answer; each prior segment's fragment is a cumulative prefix (in construction order, not necessarily keyword order) of that same query.

## Critical Rules

1. **Fragments are cumulative, not isolated**: each segment's SQL snippet must contain everything built in previous segments plus its new addition (e.g. if sol_2's fragment is `FROM a JOIN b ON ...`, sol_3's fragment is `FROM a JOIN b ON ... WHERE ...`, not just `WHERE ...` on its own).

2. **Each segment explains and extends**: pair a short first-person explanation of the reasoning (which entities relate, what condition is needed, what's being aggregated) with the exact accompanying cumulative SQL fragment in backticks, using the exact table names, column names, aliases, and SQL dialect functions from the target answer.

3. **No repetition of explanation**: each segment's prose picks up from where the previous one left off; don't re-explain earlier steps, just state the new reasoning and the new cumulative fragment.

4. **Match the query's actual complexity**: if the target query only has 2-3 real construction steps, keep segments correspondingly short and some can simply confirm the query is already complete, rather than inventing busywork to fill 5 distinct segments. Never split one atomic step into multiple segments just to reach 5.

5. **First-person, no meta-commentary**: write as the query author, not as a tutor. Do not use instructive or pedagogical phrasing ("use the X table because...", "apply a filter...", "project the columns..."); just do the step ("I use X because...", "I filter on...", "I project...").

6. **Constructive only**: never include negative statements like "no joins needed" or "do not filter".

## Example 1 (Multi-Table Join & Filtering)

**Question:** What are the names of car makers who produced cars in 1970?
**Target SQL:** `SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker JOIN CAR_NAMES AS T3 ON T2.model = T3.model JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id WHERE T4.year = '1970';`

**Output:**
{
  "sol_1": "Maker names and production years live in different tables, so I start by linking `CAR_MAKERS` to `MODEL_LIST` on the maker id: `FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker`.",
  "sol_2": "I continue the join chain through model names into the production data so each car's year becomes available: `FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker JOIN CAR_NAMES AS T3 ON T2.model = T3.model JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id`.",
  "sol_3": "Now I restrict to cars produced in 1970: `FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker JOIN CAR_NAMES AS T3 ON T2.model = T3.model JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id WHERE T4.year = '1970'`.",
  "sol_4": "A maker can have several 1970 models, so I project maker names with DISTINCT to avoid duplicates: `SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker JOIN CAR_NAMES AS T3 ON T2.model = T3.model JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id WHERE T4.year = '1970'`.",
  "sol_5": "This is already the complete query: `SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker JOIN CAR_NAMES AS T3 ON T2.model = T3.model JOIN CARS_DATA AS T4 ON T3.MakeId = T4.id WHERE T4.year = '1970'`."
}

## Example 2 (Aggregations, Ratios & Calculations)

**Question:** What percentage of clients who opened their accounts in districts with an average salary over 10000 are women?
**Evidence:** Female refers to gender = 'F'; Average salary is in column A11
**Target SQL:** `SELECT CAST(SUM(T2.gender = 'F') AS REAL) * 100 / COUNT(T2.client_id) FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.A11 > 10000`

**Output:**
{
  "sol_1": "Client demographics and district salary figures are in separate tables, so I join them on the shared district id: `FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id`.",
  "sol_2": "I keep only the high-salary districts where A11 exceeds 10000: `FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.A11 > 10000`.",
  "sol_3": "For the percentage I need a count of female clients, which I compute with a conditional sum: `SUM(T2.gender = 'F')`.",
  "sol_4": "I also need the total matched client count as the denominator: `COUNT(T2.client_id)`.",
  "sol_5": "Putting these together and casting the numerator to REAL to avoid integer division, the full query is: `SELECT CAST(SUM(T2.gender = 'F') AS REAL) * 100 / COUNT(T2.client_id) FROM district AS T1 INNER JOIN client AS T2 ON T1.district_id = T2.district_id WHERE T1.A11 > 10000`."
}

## Example 3 (Subqueries & Nested Aggregation)

**Question:** Find the names of all courses that have enrolled more students than the average course enrollment.
**Target SQL:** `SELECT T1.course_name FROM courses AS T1 JOIN enrollments AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id HAVING count(T2.student_id) > (SELECT avg(c) FROM (SELECT count(student_id) AS c FROM enrollments GROUP BY course_id))`

**Output:**
{
  "sol_1": "To find the average enrollment I first need the per-course enrollment counts, via an inner subquery: `SELECT count(student_id) AS c FROM enrollments GROUP BY course_id`.",
  "sol_2": "Averaging those gives the benchmark I'll compare each course against: `SELECT avg(c) FROM (SELECT count(student_id) AS c FROM enrollments GROUP BY course_id)`.",
  "sol_3": "For the outer query, I join courses to enrollments and group by course so I can count enrollments per course again: `FROM courses AS T1 JOIN enrollments AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id`.",
  "sol_4": "I keep only the course groups whose enrollment count exceeds the benchmark average: `FROM courses AS T1 JOIN enrollments AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id HAVING count(T2.student_id) > (SELECT avg(c) FROM (SELECT count(student_id) AS c FROM enrollments GROUP BY course_id))`.",
  "sol_5": "Finally, I project the course names for the qualifying groups: `SELECT T1.course_name FROM courses AS T1 JOIN enrollments AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id HAVING count(T2.student_id) > (SELECT avg(c) FROM (SELECT count(student_id) AS c FROM enrollments GROUP BY course_id))`."
}

## Output Format:
Return ONLY a JSON object with exactly 5 keys `sol_1`, `sol_2`, `sol_3`, `sol_4`, `sol_5` (no markdown code blocks):
{"sol_1": "...", "sol_2": "...", "sol_3": "...", "sol_4": "...", "sol_5": "..."}
"""

SYSTEM_PROMPTS = {
    "conceptual": CONCEPTUAL_HINTS_SYSTEM_PROMPT,
    "partial_sql": PARTIAL_SQL_HINTS_SYSTEM_PROMPT,
    "partial_sols": PARTIAL_SOLS_SYSTEM_PROMPT,
}

CONCEPTUAL_USER_TEMPLATE = """\
Problem (Schema & Question):
{problem}

Target SQL Answer:
{answer}
{metadata_context}
Break down the path to construct this SQL query into exactly 5 sequential, independent conceptual hints (1-2 sentences each): hint_1, hint_2, hint_3, hint_4, hint_5:"""

PARTIAL_SQL_USER_TEMPLATE = """\
Problem (Schema & Question):
{problem}

Target SQL Answer:
{answer}
{metadata_context}
Break down the path to construct this SQL query into exactly 5 sequential, independent steps (hint_1, hint_2, hint_3, hint_4, hint_5).
Each step must contain a clear 1-2 sentence explanation of the reasoning paired with the exact accompanying partial SQL snippet/clause that implements that step:"""

PARTIAL_SOLS_USER_TEMPLATE = """\
Problem (Schema & Question):
{problem}

Target SQL Answer:
{answer}
{metadata_context}
Write out the construction of this SQL query as a continuous first-person reasoning trace, then split it into exactly 5 sequential segments (sol_1, sol_2, sol_3, sol_4, sol_5). Each segment's SQL fragment must be cumulative (the full query built so far), ending with sol_5 containing the complete target query:"""

USER_TEMPLATES = {
    "conceptual": CONCEPTUAL_USER_TEMPLATE,
    "partial_sql": PARTIAL_SQL_USER_TEMPLATE,
    "partial_sols": PARTIAL_SOLS_USER_TEMPLATE,
}


# Keys the model is asked to return per style. Output is normalized to
# hint_1..hint_5 in parse_hints_response.
KEY_PREFIXES = {
    "conceptual": "hint",
    "partial_sql": "hint",
    "partial_sols": "sol",
}


def detect_repetition(text: str, min_pattern_len: int = 2, min_repeats: int = 10) -> bool:
    """Detect if text contains excessive repetition (sign of model breakdown)."""
    if len(text) < min_pattern_len * min_repeats:
        return False
    for pattern_len in range(min_pattern_len, 6):
        if len(text) < pattern_len * min_repeats:
            continue
        tail = text[-pattern_len * min_repeats:]
        pattern = tail[:pattern_len]
        if tail == pattern * min_repeats:
            return True
    return False


def validate_hints(
    hints: dict,
    key_prefix: str = "hint",
    max_tokens_per_hint: int = 250,
    max_total_tokens: int = 650,
) -> tuple[bool, str]:
    """Validate hints for quality and formatting issues (must be exactly 5 hints)."""
    num_hints = len(hints)
    if num_hints != 5:
        return False, f"Expected exactly 5 hints, got {num_hints}"

    expected_keys = [f"{key_prefix}_{i}" for i in range(1, 6)]
    if set(hints.keys()) != set(expected_keys):
        return False, f"Keys {list(hints.keys())} do not match expected sequence {expected_keys}"

    total_tokens = 0
    for key in expected_keys:
        hint = hints[key]
        if not isinstance(hint, str):
            return False, f"{key} is not a string"
        if detect_repetition(hint):
            return False, f"{key} contains repetitive pattern"
        tokens = count_tokens(hint)
        total_tokens += tokens
        if tokens > max_tokens_per_hint:
            return False, f"{key} too long ({tokens} > {max_tokens_per_hint})"
        if tokens < 3:
            return False, f"{key} too short ({tokens} tokens)"
    if total_tokens > max_total_tokens:
        return False, f"Total tokens too long ({total_tokens} > {max_total_tokens})"
    return True, ""


def parse_hints_response(content: str, key_prefix: str = "hint") -> dict | None:
    """Parse JSON hints from LLM response."""
    if not content:
        return None
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        content = "\n".join(json_lines)
    try:
        hints = json.loads(content)
        is_valid, _ = validate_hints(hints, key_prefix=key_prefix)
        if not is_valid:
            return None
        # The model may answer with style-specific keys (e.g. sol_1..sol_5);
        # always store them as hint_1..hint_5 so the output schema is identical.
        return {f"hint_{i}": hints[f"{key_prefix}_{i}"] for i in range(1, 6)}
    except Exception:
        return None


def validate_hint_structure(hints: dict, hint_style: str = "conceptual") -> dict:
    """Compute token statistics for generated hints."""
    hint_keys = sorted(hints.keys(), key=lambda k: int(k.split("_")[1]))
    num_hints = len(hint_keys)
    tokens = [count_tokens(hints[key]) for key in hint_keys]
    total = sum(tokens)
    return {
        "num_hints": num_hints,
        "hint_style": hint_style,
        "tokens": tokens,
        "total_tokens": total,
        "avg_tokens": total / num_hints if num_hints > 0 else 0,
        "max_tokens": max(tokens) if tokens else 0,
        "distribution": [round(t / total * 100, 1) if total > 0 else 0 for t in tokens],
    }


def parse_rate_limit_delay(error_message: str) -> float:
    """Extract retry delay from rate limit error message."""
    match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(error_message))
    if match:
        value, unit = float(match.group(1)), match.group(2)
        seconds = value / 1000 if unit == "ms" else value
        return seconds + 0.5
    return 2.0


def format_metadata_context(metadata: dict | None) -> str:
    """Extract conversational context or notes from metadata to assist the model."""
    if not metadata:
        return ""
    lines = []
    if metadata.get("intermediate_turns"):
        lines.append("\nReference Conversational Decomposition Turns:")
        for t in metadata["intermediate_turns"]:
            lines.append(f"  - Turn {t.get('turn_idx')}: {t.get('question')} => SQL: {t.get('sql')}")
    if lines:
        return "\n".join(lines)
    return ""


async def generate_prefix_hints(
    client: AsyncAzureOpenAI,
    problem: str,
    answer: str,
    metadata: dict | None = None,
    model: str = "gpt-5.6-sol_2026-07-09",
    hint_style: str = "conceptual",
    max_retries: int = 5,
    semaphore: asyncio.Semaphore | None = None,
    verbose: bool = False,
) -> dict | None:
    """Generate 5 progressive hints for a Text-to-SQL problem."""
    meta_context = format_metadata_context(metadata)
    system_prompt = SYSTEM_PROMPTS.get(hint_style, CONCEPTUAL_HINTS_SYSTEM_PROMPT)
    user_template = USER_TEMPLATES.get(hint_style, CONCEPTUAL_USER_TEMPLATE)
    key_prefix = KEY_PREFIXES.get(hint_style, "hint")
    user_content = user_template.format(
        problem=problem,
        answer=answer,
        metadata_context=meta_context,
    )

    async def _call():
        attempt = 0
        while attempt < max_retries:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_completion_tokens=4096,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                hints = parse_hints_response(content, key_prefix=key_prefix)
                if hints is None:
                    attempt += 1
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                    continue
                return hints
            except RateLimitError as e:
                delay = parse_rate_limit_delay(str(e))
                if verbose:
                    print(f"Rate limited, sleeping {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                if verbose:
                    print(f"API error (attempt {attempt + 1}): {e}")
                attempt += 1
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                continue
        return None

    if semaphore:
        async with semaphore:
            return await _call()
    return await _call()


class IncrementalSaver:
    """Thread-safe incremental saving and resumption for hints generation."""

    def __init__(self, output_path: Path, primitives: list[dict], save_every: int = 30):
        self.output_path = output_path
        self.save_every = save_every
        self.unsaved_count = 0
        self.lock = asyncio.Lock()
        self.index_to_pos = {p["index"]: i for i, p in enumerate(primitives)}

        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            if len(self.results) < len(primitives):
                self.results.extend([None] * (len(primitives) - len(self.results)))
        else:
            self.results = [None] * len(primitives)
            self._save()

    def get_pending_indices(self) -> set[int]:
        pending = set()
        for idx, pos in self.index_to_pos.items():
            if pos >= len(self.results) or self.results[pos] is None:
                pending.add(idx)
            elif not self.results[pos].get("prefix_hints"):
                pending.add(idx)
        return pending

    async def save_result(self, result: dict) -> None:
        async with self.lock:
            idx = result["index"]
            pos = self.index_to_pos[idx]
            self.results[pos] = result
            self.unsaved_count += 1
            if self.unsaved_count >= self.save_every:
                self._save()
                self.unsaved_count = 0

    def _save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def finalize(self) -> list[dict]:
        self._save()
        return [r for r in self.results if r is not None]


async def process_and_save(
    client: AsyncAzureOpenAI,
    primitive: dict,
    model: str,
    hint_style: str,
    semaphore: asyncio.Semaphore,
    saver: IncrementalSaver,
    failed_queue: asyncio.Queue,
    verbose: bool = False,
) -> None:
    """Process a single primitive and save."""
    hints = await generate_prefix_hints(
        client=client,
        problem=primitive["problem"],
        answer=primitive["answer"],
        metadata=primitive.get("metadata"),
        model=model,
        hint_style=hint_style,
        semaphore=semaphore,
        verbose=verbose,
    )
    new_p = primitive.copy()
    if hints is not None:
        new_p["prefix_hints"] = hints
        new_p["hint_stats"] = validate_hint_structure(hints, hint_style=hint_style)
        await saver.save_result(new_p)
    else:
        await failed_queue.put(primitive)


async def generate_all_prefix_hints(
    input_path: Path,
    output_path: Path,
    model: str = "gpt-5.6-sol_2026-07-09",
    hint_style: str = "conceptual",
    filter_mode: str = "all",
    max_samples: int | None = None,
    batch_size: int = 20,
    save_every: int = 30,
    max_retries: int = 3,
    resume: bool = True,
    verbose: bool = False,
) -> None:
    """Generate prefix hints for all primitives in an input JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        primitives = json.load(f)

    print(f"Loaded {len(primitives)} primitives from {input_path}")
    if not resume and output_path.exists():
        output_path.unlink()

    saver = IncrementalSaver(output_path, primitives, save_every=save_every)

    if filter_mode == "complex":
        filtered_primitives = [p for p in primitives if p.get("metadata", {}).get("is_complex")]
        print(f"Filtered to {len(filtered_primitives)}/{len(primitives)} complex primitives")
    elif filter_mode == "multi_step":
        filtered_primitives = [p for p in primitives if p.get("metadata", {}).get("multi_step")]
        print(f"Filtered to {len(filtered_primitives)}/{len(primitives)} multi-step primitives")
    else:
        filtered_primitives = primitives

    primitives_to_process = filtered_primitives if max_samples is None else filtered_primitives[:max_samples]

    pending_indices = saver.get_pending_indices()
    pending = [p for p in primitives_to_process if p["index"] in pending_indices]

    total_in_range = len(primitives_to_process)
    completed_in_range = total_in_range - len(pending)
    print(f"Completed: {completed_in_range}/{total_in_range}")
    print(f"Pending: {len(pending)} to process (style={hint_style}, batch_size={batch_size}, save_every={save_every})")

    if not pending:
        print("Nothing to do!")
        return

    credential = get_bearer_token_provider(
        ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
        TRAPI_SCOPE,
    )
    client = AsyncAzureOpenAI(
        azure_endpoint=TRAPI_ENDPOINT,
        azure_ad_token_provider=credential,
        api_version=TRAPI_API_VERSION,
    )
    semaphore = asyncio.Semaphore(batch_size)

    current_batch = pending
    for retry_round in range(max_retries):
        if not current_batch:
            break
        round_label = "Generating" if retry_round == 0 else f"Retry {retry_round}"
        failed_queue: asyncio.Queue = asyncio.Queue()

        tasks = [
            process_and_save(client, p, model, hint_style, semaphore, saver, failed_queue, verbose)
            for p in current_batch
        ]
        await tqdm_asyncio.gather(*tasks, desc=f"{round_label} ({len(current_batch)} items)")

        failed_items = []
        while not failed_queue.empty():
            failed_items.append(await failed_queue.get())
        current_batch = failed_items

    if current_batch:
        print(f"{len(current_batch)} items failed after {max_retries} attempts.")
        for p in current_batch:
            failed_p = p.copy()
            failed_p["prefix_hints"] = None
            failed_p["hint_stats"] = None
            await saver.save_result(failed_p)

    all_results = saver.finalize()
    success = sum(1 for r in all_results if r.get("prefix_hints"))
    print(f"\nDone! {success}/{len(all_results)} successfully processed.")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 5 incremental hints for Text-to-SQL problems via TRAPI.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input primitives JSON")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output JSON with prefix hints")
    parser.add_argument("--model", "-m", type=str, default="gpt-5.6-sol_2026-07-09", help="TRAPI model name")
    parser.add_argument(
        "--hint-style",
        type=str,
        default="conceptual",
        choices=["conceptual", "partial_sql", "partial_sols"],
        help=(
            "Prompt style: 'conceptual' (strategy and natural reasoning), "
            "'partial_sql' (reasoning paired with concrete but non-cumulative partial SQL code), "
            "or 'partial_sols' (first-person reasoning with cumulative partial SQL, like a partial solution being written live)"
        ),
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="all",
        choices=["all", "complex", "multi_step"],
        help="Filter primitives to process (all, complex, or multi_step)",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="Concurrency limit")
    parser.add_argument("--save-every", "-s", type=int, default=30, help="Save interval")
    parser.add_argument("--max-samples", "-n", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--max-retries", "-r", type=int, default=3, help="Max retry rounds")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()
    asyncio.run(generate_all_prefix_hints(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        hint_style=args.hint_style,
        filter_mode=args.filter,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        save_every=args.save_every,
        max_retries=args.max_retries,
        resume=not args.no_resume,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
