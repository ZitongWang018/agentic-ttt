from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _finite(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    number = float(value)
    return number if math.isfinite(number) else fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=200)
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = json.loads(
        args.summary.read_text(encoding="utf-8")
    )
    complete = [row for row in rows if int(row.get("steps", 0)) == args.expected_steps]
    baselines = [row for row in complete if row.get("novelty_mode") == "off"]
    candidates = [
        row
        for row in complete
        if row.get("novelty_mode") == "learned_fixed_cosine"
    ]
    if len(baselines) != 1:
        raise ValueError(f"Expected one complete F2P baseline, got {len(baselines)}")
    if not candidates:
        raise ValueError("No complete learned fixed-cosine pilot candidates")
    baseline = baselines[0]
    invalid_limit = _finite(baseline.get("invalid_rate"), 1.0) + 0.02
    baseline_late_reward = _finite(
        baseline.get("late_productive_reward"), 0.0
    )
    late_reward_floor = 0.95 * baseline_late_reward
    eligible = [
        row
        for row in candidates
        if _finite(row.get("invalid_rate"), 1.0) <= invalid_limit
        and _finite(row.get("late_productive_reward"), -math.inf)
        >= late_reward_floor
    ]

    def selection_key(row: Dict[str, Any]) -> tuple[float, ...]:
        negative_change = _finite(
            row.get("mean_negative_logprob_change"), math.inf
        )
        return (
            _finite(row.get("early_unique_actions"), -math.inf),
            -_finite(row.get("early_exact_repeat_rate"), 1.0),
            _finite(row.get("productive_reward_total"), -math.inf),
            -_finite(row.get("invalid_rate"), 1.0),
            -negative_change,
            -float(row["novelty_lr"]),
            -float(row["lambda0"]),
        )

    pool = eligible if eligible else candidates
    selected = max(pool, key=selection_key)
    output = {
        "selection_rule": (
            "invalid<=f2p+0.02 and late_productive_reward>=0.95*f2p; "
            "then maximize early_unique_actions, minimize early repeat, "
            "maximize productive reward, minimize invalid rate, and maximize "
            "negative-logprob suppression"
        ),
        "constraints_satisfied": bool(eligible),
        "invalid_limit": invalid_limit,
        "late_productive_reward_floor": late_reward_floor,
        "baseline": baseline,
        "selected": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_tsv.write_text(
        f"{selected['novelty_lr']}\t{selected['lambda0']}\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
