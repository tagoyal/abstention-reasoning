"""Difficulty-scheduled hint generation.

The warm-start problem this solves: sampling hint requests from a fixed global
policy produced an SFT set where 89% of examples never asked for a hint, and the
difficulty signal that did exist was weak and non-monotone (Level 2 asked *less*
often than Level 1). A model trained on that learns "don't ask", leaving RL with
almost no hint-requesting behaviour to shape.

The fix is to decide how many hints each problem gets *before* generating, from
that problem's measured difficulty rather than its nominal label:

  Phase 1  probe    generate with NO hints, k samples per problem. The empirical
                    pass rate (correct / k) is the difficulty measure. A problem
                    the teacher already solves 5/5 times unaided does not need a
                    hint; one it never solves needs several.
  Phase 2  schedule bucket problems by pass rate and map each bucket to a hint
                    count. Lower pass rate -> more hints.
  Phase 3  generate run with the per-problem schedule, resampling problems that
                    come back incorrect until a target fraction is answered
                    correctly (see `generate_until_target` in
                    pipeline/commands/inference.py, which owns the loop).

Phase 1 output is a "difficulty profile" keyed by prompt index, so it is computed
once per (model, split) and reused across schedules.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Pass-rate upper bound -> hints to schedule. Read as "solved at most X of the
# time unaided -> give it N hints". Ordered easiest-last so the first match wins.
DEFAULT_BUCKETS: list[tuple[float, int]] = [
    (0.00, 4),   # never solved unaided
    (0.25, 3),
    (0.50, 2),
    (0.75, 1),
    (1.00, 0),   # reliably solved unaided -> no hint
]


@dataclass
class DifficultyProfile:
    """Per-problem empirical difficulty from a no-hint probe.

    Built in memory from the probe's own records and used immediately. It used
    to be written out and read back as its own JSON, which bought nothing: it
    is a pure function of the probe dataset, so the file could only ever go
    stale against the records it summarized, and it needed a name in a scheme
    that has no room for a keyed-by-index summary.
    """
    num_samples: int
    pass_rates: dict[int, float] = field(default_factory=dict)
    # Reasoning tokens spent per problem. Secondary difficulty signal, needed
    # because pass rate saturates: a strong teacher solves most problems 8/8, so
    # a large block ties at 1.0 and rank order inside it would otherwise be
    # arbitrary. Longer reasoning on an equally-solved problem means harder.
    effort: dict[int, float] = field(default_factory=dict)


def parse_buckets(spec: str) -> list[tuple[float, int]]:
    """Parse "0.0:4,0.25:3,0.5:2,0.75:1,1.0:0" into bucket pairs.

    Each entry is `max_pass_rate:num_hints`. Sorted ascending so the first
    bucket whose bound the pass rate falls at or below decides the hint count.
    """
    buckets = []
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        bound, hints = pair.split(":")
        buckets.append((float(bound.strip()), int(hints.strip())))
    if not buckets:
        raise ValueError(f"--hint-buckets parsed to nothing: {spec!r}")
    buckets.sort(key=lambda b: b[0])
    for bound, hints in buckets:
        if not 0.0 <= bound <= 1.0:
            raise ValueError(f"bucket bound {bound} outside [0, 1] in {spec!r}")
        if hints < 0:
            raise ValueError(f"negative hint count {hints} in {spec!r}")
    return buckets


def hints_for_pass_rate(pass_rate: float, buckets: list[tuple[float, int]]) -> int:
    """Hint count for one problem's pass rate."""
    for bound, hints in buckets:
        if pass_rate <= bound:
            return hints
    return buckets[-1][1]


def build_profile(records: list[dict]) -> DifficultyProfile:
    """Turn no-hint probe records into a difficulty profile.

    Expects each record to carry `index` and either `pass_rate` (already
    aggregated over samples) or `correct` (a single sample). num_samples comes
    from the records themselves rather than being passed alongside them, so it
    cannot disagree with the data it describes.
    """
    rates: dict[int, float] = {}
    effort: dict[int, float] = {}
    num_samples = 1
    for r in records:
        idx = r["index"]
        if "pass_rate" in r:
            rates[idx] = float(r["pass_rate"])
        else:
            rates[idx] = 1.0 if r.get("correct") else 0.0
        if r.get("token_count"):
            effort[idx] = float(r["token_count"])
        num_samples = max(num_samples, int(r.get("num_samples") or 1))
    if not rates:
        raise ValueError("hint schedule probe has no usable records")
    return DifficultyProfile(num_samples=num_samples, pass_rates=rates, effort=effort)


def schedule_from_profile(
    prompts: list[dict],
    profile: DifficultyProfile,
    buckets: list[tuple[float, int]],
    default_hints: int = 0,
) -> dict[int, int]:
    """Map prompt index -> scheduled hint count.

    Prompts missing from the profile (never probed) fall back to
    `default_hints` rather than silently becoming 0, so a partial probe is
    visible in the summary instead of quietly skewing the mix.
    """
    schedule = {}
    for p in prompts:
        idx = p["index"]
        if idx in profile.pass_rates:
            schedule[idx] = hints_for_pass_rate(profile.pass_rates[idx], buckets)
        else:
            schedule[idx] = default_hints
    return schedule


def summarize_schedule(schedule: dict[int, int], profile: DifficultyProfile | None = None) -> str:
    """Human-readable breakdown of the scheduled hint mix."""
    if not schedule:
        return "  (empty schedule)"
    total = len(schedule)
    counts: dict[int, int] = {}
    for n in schedule.values():
        counts[n] = counts.get(n, 0) + 1
    lines = [f"  scheduled {total} problems:"]
    for n in sorted(counts):
        lines.append(f"    {n} hint(s): {counts[n]:5d} ({counts[n] / total:5.1%})")
    hinted = sum(c for n, c in counts.items() if n > 0)
    lines.append(f"  hint-requesting fraction: {hinted}/{total} = {hinted / total:.1%}")
    mean = sum(n * c for n, c in counts.items()) / total
    lines.append(f"  mean hints per problem:   {mean:.2f}")
    if profile is not None:
        missing = total - sum(1 for i in schedule if i in profile.pass_rates)
        if missing:
            lines.append(f"  WARNING: {missing} problems absent from profile, defaulted")
    return "\n".join(lines)


def schedule_by_rank(
    prompts: list[dict],
    profile: "DifficultyProfile",
    target_hint_fraction: float,
    max_hints: int = 4,
) -> dict[int, int]:
    """Schedule hint counts by difficulty *rank*, hitting the target fraction exactly.

    Why not pass-rate thresholds: a k-sample probe yields only k+1 distinct pass
    rates, and one value routinely carries a large share of the mass (with a weak
    teacher, "solved 0 of 8" can be 30%+ of the set). Any value threshold either
    includes that whole block or excludes it, so the achieved hint fraction jumps
    in coarse steps and two different targets can produce identical buckets.

    Ranking sidesteps that: sort hardest-first, give the hardest
    `target_hint_fraction` of problems hints, and split those into `max_hints`
    equal-sized bands (hardest band gets `max_hints`, easiest gets 1). Ties are
    broken by index so the assignment is deterministic across runs.
    """
    if not 0.0 <= target_hint_fraction <= 1.0:
        raise ValueError(f"target_hint_fraction must be in [0, 1], got {target_hint_fraction}")
    if max_hints < 1:
        raise ValueError(f"max_hints must be >= 1, got {max_hints}")

    indices = [p["index"] for p in prompts]
    # Sort hardest-first. Unprobed problems sort as easiest (pass rate 1.0) so
    # they land in the no-hint tail rather than silently absorbing the largest
    # hint budget. Ties on pass rate -- and with a strong teacher most of the set
    # ties at 1.0 -- are broken by reasoning effort, descending, so reaching into
    # the tie block still picks the harder problems rather than an arbitrary
    # slice. Index last, to keep the order deterministic.
    ranked = sorted(
        indices,
        key=lambda i: (profile.pass_rates.get(i, 1.0), -profile.effort.get(i, 0.0), i),
    )

    n_hinted = round(len(ranked) * target_hint_fraction)
    schedule = {i: 0 for i in ranked}
    if n_hinted == 0:
        return schedule

    for pos, idx in enumerate(ranked[:n_hinted]):
        # band 0 = hardest slice -> max_hints hints.
        band = min(max_hints - 1, pos * max_hints // n_hinted)
        schedule[idx] = max_hints - band
    return schedule


def hint_fractions_from_schedule(target_hint_fraction: float, max_hints: int) -> dict[int, float]:
    """Hint-count shares that `schedule_by_rank` aims for: 1 - target_hint_fraction
    with no hints, and the rest split evenly over 1..max_hints."""
    if not 0.0 <= target_hint_fraction <= 1.0:
        raise ValueError(f"target_hint_fraction must be in [0, 1], got {target_hint_fraction}")
    if max_hints < 1:
        raise ValueError(f"max_hints must be >= 1, got {max_hints}")
    fractions = {0: 1.0 - target_hint_fraction}
    for n in range(1, max_hints + 1):
        fractions[n] = target_hint_fraction / max_hints
    return fractions


def parse_hint_mix(spec: str) -> dict[int, float]:
    """Parse "0:0.5,1:0.1,2:0.1,3:0.1,4:0.1,5:0.1" into {hint_count: share}.

    Shares are normalised, so "0:5,1:1,2:1" is as good as "0:0.71,...".
    """
    mix: dict[int, float] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, share = pair.split(":")
        k, share = int(k.strip()), float(share.strip())
        if k < 0 or share < 0:
            raise ValueError(f"negative hint count or share in {spec!r}")
        mix[k] = share
    total = sum(mix.values())
    if not mix or total <= 0:
        raise ValueError(f"--hint-mix parsed to nothing usable: {spec!r}")
    return {k: v / total for k, v in sorted(mix.items())}


def schedule_by_mix(
    prompts: list[dict],
    profile: "DifficultyProfile",
    mix: dict[int, float],
) -> dict[int, int]:
    """Assign every problem a hint count so the counts follow `mix` exactly.

    Same difficulty ranking as `schedule_by_rank` (hardest first, ties broken by
    reasoning effort, then index): the hardest slice of problems gets the largest
    hint count in the mix, the easiest slice the smallest. Slice sizes come from
    largest-remainder rounding, so they sum to len(prompts).

    Pass the FULL prompt list, not just the unfinished ones: ranks are relative,
    so scheduling a retry subset alone would re-spread it over the whole mix.
    """
    indices = [p["index"] for p in prompts]
    ranked = sorted(
        indices,
        key=lambda i: (profile.pass_rates.get(i, 1.0), -profile.effort.get(i, 0.0), i),
    )
    n = len(ranked)
    ks = sorted(mix, reverse=True)
    exact = {k: mix[k] * n for k in ks}
    counts = {k: int(exact[k]) for k in ks}
    leftover = n - sum(counts.values())
    for k in sorted(ks, key=lambda k: exact[k] - counts[k], reverse=True)[:leftover]:
        counts[k] += 1

    schedule: dict[int, int] = {}
    pos = 0
    for k in ks:
        for idx in ranked[pos:pos + counts[k]]:
            schedule[idx] = k
        pos += counts[k]
    return schedule


def rebalance_by_hints(
    records: list[dict],
    fractions: dict[int, float],
    total: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    """Resample correct records so hint counts follow `fractions`.

    Only correct records are used (they are the only ones SFT trains on), and
    records whose hint count is not a key of `fractions` are dropped. Each
    bucket is downsampled without replacement or upsampled by repeating its
    records, every record being used floor(target/n) times and the remainder
    drawn at random, so no record is repeated more than once more than any other.
    Repeats are tagged `rebalance_copy` > 0.

    `total` defaults to the number of eligible correct records, which keeps the
    dataset size and moves examples between buckets instead of inflating it.

    Returns (records, {hint_count: (available, used)}).
    """
    import random

    eligible = [r for r in records if r.get("correct") and r.get("num_hints") in fractions]
    if not eligible:
        raise ValueError("no correct records with a scheduled hint count to rebalance")
    total = total if total is not None else len(eligible)
    norm = sum(fractions.values())
    rng = random.Random(seed)

    by_hints: dict[int, list[dict]] = {}
    for r in sorted(eligible, key=lambda r: r["index"]):
        by_hints.setdefault(r["num_hints"], []).append(r)

    out: list[dict] = []
    report: dict[int, tuple[int, int]] = {}
    for n in sorted(fractions):
        pool = by_hints.get(n, [])
        target = round(total * fractions[n] / norm)
        report[n] = (len(pool), target if pool else 0)
        if not pool or target == 0:
            continue
        full, rest = divmod(target, len(pool))
        for copy in range(full):
            out.extend({**r, "rebalance_copy": copy} for r in pool)
        out.extend({**r, "rebalance_copy": full} for r in rng.sample(pool, rest))
    out.sort(key=lambda r: (r["index"], r["rebalance_copy"]))
    return out, report


def summarize_rebalance(report: dict[int, tuple[int, int]]) -> str:
    total = sum(used for _, used in report.values())
    lines = ["  hints | available -> used | share"]
    for n, (avail, used) in sorted(report.items()):
        lines.append(f"    {n:>3} | {avail:>9} -> {used:<5} | {used / total:5.1%}" if total else f"    {n}")
        if avail == 0:
            lines[-1] += "  (EMPTY bucket: nothing to sample)"
    return "\n".join(lines)
