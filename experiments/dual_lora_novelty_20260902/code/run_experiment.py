from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0,
    str(REPO_ROOT / "experiments" / "f2p_loss_ablation_20260901" / "code"),
)

from loss_patch import install_loss_patch  # noqa: E402
from novelty_patch import (  # noqa: E402
    VALID_NEGATIVE_SELECTIONS,
    VALID_NOVELTY_MODES,
    VALID_STRENGTH_MODES,
    install_dual_lora_novelty_patch,
)


def _install_cluster_runtime_compatibility() -> None:
    import providers.huggingface as huggingface_provider

    original_from_pretrained = (
        huggingface_provider.AutoModelForCausalLM.from_pretrained
    )

    class ClusterAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs["attn_implementation"] = "sdpa"
            return original_from_pretrained(*args, **kwargs)

    huggingface_provider.AutoModelForCausalLM = ClusterAutoModelForCausalLM
    hf_language_model = huggingface_provider.hfLanguageModel
    original_init = hf_language_model.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.disable_compile = True
        print(
            "[DualLoRANovelty] generation compile disabled; using SDPA.",
            flush=True,
        )

    hf_language_model.__init__ = patched_init


def _argument_value(arguments: list[str], name: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--novelty_mode", required=True, choices=sorted(VALID_NOVELTY_MODES)
    )
    parser.add_argument("--novelty_rank", type=int, default=4)
    parser.add_argument("--novelty_alpha", type=int, default=8)
    parser.add_argument("--novelty_lr", type=float, default=5e-6)
    parser.add_argument("--novelty_update_frequency", type=int, default=5)
    parser.add_argument("--novelty_window_size", type=int, default=25)
    parser.add_argument("--novelty_batch_size", type=int, default=5)
    parser.add_argument("--novelty_lambda0", type=float, default=0.5)
    parser.add_argument("--novelty_decay_end", type=int, default=350)
    parser.add_argument(
        "--novelty_negative_selection",
        choices=sorted(VALID_NEGATIVE_SELECTIONS),
        default="uniform",
    )
    parser.add_argument(
        "--novelty_strength_mode",
        choices=sorted(VALID_STRENGTH_MODES),
        default="absolute",
    )
    parser.add_argument("--novelty_score_batch_size", type=int, default=5)
    parser.add_argument("--novelty_train_microbatch_size", type=int, default=1)
    novelty_args, eval_args = parser.parse_known_args()

    run_dir_value = _argument_value(eval_args, "--run_dir")
    seed_value = _argument_value(eval_args, "--seed")
    agents_config_value = _argument_value(eval_args, "--agents_config")
    max_steps_value = _argument_value(eval_args, "--max_steps")
    memory_frequency_value = _argument_value(
        eval_args, "--agent_memory_save_frequency"
    )
    snapshot_steps_value = _argument_value(
        eval_args, "--agent_lora_snapshot_steps"
    )
    if not run_dir_value:
        raise ValueError("The experiment requires an explicit --run_dir")
    novelty_seed = int(seed_value) if seed_value is not None else 42
    run_dir = Path(run_dir_value).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "experiment_config.json"
    metadata = {
        "f2p_source_run": str(
            REPO_ROOT
            / "experiments"
            / "f2p_loss_ablation_20260901"
            / "formal"
            / "normalized_top1_a0.25_b0.25"
        ),
        "f2p_loss_mode": "normalized_logp_l2",
        "f2p_alpha": 0.25,
        "f2p_beta": 0.25,
        "novelty_mode": novelty_args.novelty_mode,
        "novelty_rank": novelty_args.novelty_rank,
        "novelty_alpha": novelty_args.novelty_alpha,
        "novelty_lr": novelty_args.novelty_lr,
        "novelty_update_frequency": novelty_args.novelty_update_frequency,
        "novelty_window_size": novelty_args.novelty_window_size,
        "novelty_batch_size": novelty_args.novelty_batch_size,
        "novelty_lambda0": novelty_args.novelty_lambda0,
        "novelty_decay_end": novelty_args.novelty_decay_end,
        "novelty_seed": novelty_seed,
        "negative_selection": novelty_args.novelty_negative_selection,
        "novelty_strength_mode": novelty_args.novelty_strength_mode,
        "novelty_score_batch_size": novelty_args.novelty_score_batch_size,
        "novelty_train_microbatch_size": (
            novelty_args.novelty_train_microbatch_size
        ),
        "novelty_loss": "mean_token_logprob(historical_action|current_context)",
        "agents_config": agents_config_value,
        "max_steps": int(max_steps_value) if max_steps_value else None,
        "agent_memory_save_frequency": (
            int(memory_frequency_value) if memory_frequency_value else None
        ),
        "runtime_optimizations": [
            "batched_no_grad_action_scoring",
            "microbatched_novelty_backward",
        ],
        "runner": str(Path(__file__).resolve()),
    }
    if snapshot_steps_value and snapshot_steps_value.strip():
        metadata["lora_snapshot_steps"] = [
            int(value.strip())
            for value in snapshot_steps_value.split(",")
            if value.strip()
        ]
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(
                f"Refusing to reuse {run_dir}: experiment configuration differs"
            )
    else:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    _install_cluster_runtime_compatibility()
    install_loss_patch(
        loss_mode="normalized_logp_l2",
        alpha=0.25,
        beta=0.25,
    )
    install_dual_lora_novelty_patch(
        novelty_mode=novelty_args.novelty_mode,
        novelty_rank=novelty_args.novelty_rank,
        novelty_alpha=novelty_args.novelty_alpha,
        novelty_lr=novelty_args.novelty_lr,
        novelty_update_frequency=novelty_args.novelty_update_frequency,
        novelty_window_size=novelty_args.novelty_window_size,
        novelty_batch_size=novelty_args.novelty_batch_size,
        novelty_lambda0=novelty_args.novelty_lambda0,
        novelty_decay_end=novelty_args.novelty_decay_end,
        novelty_seed=novelty_seed,
        novelty_negative_selection=novelty_args.novelty_negative_selection,
        novelty_strength_mode=novelty_args.novelty_strength_mode,
        novelty_score_batch_size=novelty_args.novelty_score_batch_size,
        novelty_train_microbatch_size=(
            novelty_args.novelty_train_microbatch_size
        ),
    )
    sys.argv = [sys.argv[0], *eval_args]
    runpy.run_path(str(REPO_ROOT / "eval.py"), run_name="__main__")


if __name__ == "__main__":
    main()
