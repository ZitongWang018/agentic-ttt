from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-steps", type=int, default=30)
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    eligible = [
        row for row in rows
        if row["loss_mode"] == "normalized_logp_l2"
        and row["finite"]
        and row["steps"] >= args.required_steps
        and row["updates"] > 0
    ]
    # Short-horizon selection: task reward first, then fewer invalid actions,
    # then greater action exploration, with lower gradient norm as a final
    # stability tiebreaker.
    eligible.sort(
        key=lambda row: (
            -row["reward_total"],
            row["invalid_rate"],
            -row["unique_actions"],
            row["mean_grad_norm"] if row["mean_grad_norm"] is not None else float("inf"),
        )
    )
    if len(eligible) < 3:
        raise RuntimeError(f"Only {len(eligible)} eligible pilot settings; need 3")
    selected = eligible[:3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("name\tloss_mode\talpha\tbeta\n")
        handle.write("original\toriginal\t1.0\t0.0\n")
        handle.write("no_w\tno_w\t1.0\t0.0\n")
        for index, row in enumerate(selected, start=1):
            handle.write(
                f"normalized_top{index}_a{row['alpha']}_b{row['beta']}\t"
                f"normalized_logp_l2\t{row['alpha']}\t{row['beta']}\n"
            )
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
