from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


SCRIPT = Path(__file__).with_name("select_pilot.py")


def _row(
    name: str,
    *,
    mode: str,
    unique: int,
    invalid: float,
    late_reward: float,
    lr: float,
    lambda0: float,
) -> dict:
    return {
        "run_dir": f"/tmp/{name}",
        "novelty_mode": mode,
        "novelty_lr": lr,
        "lambda0": lambda0,
        "steps": 200,
        "early_unique_actions": unique,
        "early_exact_repeat_rate": 0.2,
        "productive_reward_total": 5.0,
        "late_productive_reward": late_reward,
        "invalid_rate": invalid,
        "mean_negative_logprob_change": -0.1,
    }


def test_constraints_precede_unique_action_count() -> None:
    rows = [
        _row(
            "baseline",
            mode="off",
            unique=20,
            invalid=0.02,
            late_reward=4.0,
            lr=5e-6,
            lambda0=0.0,
        ),
        _row(
            "eligible",
            mode="learned_fixed_cosine",
            unique=25,
            invalid=0.03,
            late_reward=4.0,
            lr=2.5e-6,
            lambda0=0.5,
        ),
        _row(
            "invalid",
            mode="learned_fixed_cosine",
            unique=30,
            invalid=0.10,
            late_reward=4.0,
            lr=5e-6,
            lambda0=1.0,
        ),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        summary = root / "pilot.json"
        output_json = root / "selected.json"
        output_tsv = root / "selected.tsv"
        summary.write_text(json.dumps(rows), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--summary",
                str(summary),
                "--output-json",
                str(output_json),
                "--output-tsv",
                str(output_tsv),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        selected = json.loads(output_json.read_text(encoding="utf-8"))
        assert selected["selected"]["run_dir"] == "/tmp/eligible"
        assert output_tsv.read_text(encoding="utf-8") == "2.5e-06\t0.5\n"


if __name__ == "__main__":
    test_constraints_precede_unique_action_count()
    print("pilot selection tests passed")
