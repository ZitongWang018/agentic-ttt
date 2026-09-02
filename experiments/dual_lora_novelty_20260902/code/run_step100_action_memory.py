from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from typing import Any, Dict, List


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN = EXPERIMENT_ROOT / "f2p_checkpoint_rerun" / "f2p_only_seed42"
AGENT_ID = "qwen3_4b_dual_lora_novelty"
SOURCE_AGENT_DIR = SOURCE_RUN / AGENT_ID
SOURCE_AGENT_LOG = SOURCE_AGENT_DIR / "agent_log.jsonl"
SOURCE_CONFIG = SOURCE_RUN / "config.jsonl"
SOURCE_LORA = SOURCE_AGENT_DIR / "lora_checkpoints" / "step_0100"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0,
    str(REPO_ROOT / "experiments" / "f2p_loss_ablation_20260901" / "code"),
)

from action_memory_patch import (  # noqa: E402
    VALID_ACTION_MEMORY_MODES,
    install_action_memory_patch,
)
from loss_patch import install_loss_patch  # noqa: E402
from utils import atomic_write  # noqa: E402


def _install_cluster_runtime_compatibility() -> None:
    import providers.huggingface as huggingface_provider

    original_from_pretrained = huggingface_provider.AutoModelForCausalLM.from_pretrained

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
        print("[ActionMemory] generation compile disabled; using SDPA.", flush=True)

    hf_language_model.__init__ = patched_init


def _read_source_rows(branch_step: int) -> List[Dict[str, Any]]:
    rows = []
    with SOURCE_AGENT_LOG.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("step", -1)) < branch_step:
                rows.append(row)
    if len(rows) < branch_step:
        raise ValueError(
            f"Source log has only {len(rows)} rows before step {branch_step}"
        )
    return rows


def _read_environment_checkpoint(branch_step: int) -> Dict[str, Any]:
    found = None
    with SOURCE_CONFIG.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # The seed JSON at the beginning is pretty-printed; all
                # cumulative runtime checkpoints are one JSON object per line.
                continue
            if isinstance(value, dict) and int(value.get("step", -10**9)) == branch_step:
                found = value
    if found is None:
        raise ValueError(f"Environment step {branch_step} not found in {SOURCE_CONFIG}")
    return found


def _verified_transition(row: Dict[str, Any]) -> str:
    trace = row.get("f2p_trace", {}) or {}
    return (
        "[Verified transition]\n"
        "Observation before action:\n"
        + str(trace.get("previous_observation", ""))
        + "\nAction actually taken:\n"
        + str(row.get("action", ""))
        + "\nActual environment change:\n"
        + str(trace.get("real_observation", ""))
        + "\nStructured feedback:\n"
        + str(trace.get("real_outcome_text", ""))
    )


def _last_training_update(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for row in reversed(rows):
        update = (row.get("f2p_trace", {}) or {}).get("training_update")
        if isinstance(update, dict) and update.get("updated"):
            return update
    raise ValueError("No completed F2P update found before branch point")


def prepare_step100_branch(run_dir: Path, *, branch_step: int, window_size: int) -> None:
    if branch_step != 100:
        raise ValueError("This controlled experiment is fixed to the step100 checkpoint")
    for required in (SOURCE_AGENT_LOG, SOURCE_CONFIG, SOURCE_LORA / "adapter_model.safetensors"):
        if not required.exists():
            raise FileNotFoundError(f"Missing branch source artifact: {required}")

    rows = _read_source_rows(branch_step)
    env_checkpoint = _read_environment_checkpoint(branch_step)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.jsonl"
    if not config_path.exists():
        atomic_write(str(config_path), json.dumps(env_checkpoint, ensure_ascii=False) + "\n")

    memory_dir = run_dir / AGENT_ID / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = memory_dir / "lora"
    if not lora_dir.exists():
        shutil.copytree(SOURCE_LORA, lora_dir)

    memory_path = memory_dir / "memory.json"
    if memory_path.exists():
        return
    last_update = _last_training_update(rows)
    completed_updates = sum(
        bool(((row.get("f2p_trace", {}) or {}).get("training_update") or {}).get("updated"))
        for row in rows
    )
    history_rows = rows[-window_size:]
    memory = {
        "agent_id": AGENT_ID,
        "agent_name": "Qwen3-4B F2P step100 action-memory branch",
        "cfg": {
            "llm_name": "/home/sunward/models/Qwen3-4B",
            "enable_reflection": False,
            "enable_summarization": False,
            "enable_short_term_memory": True,
            "max_seq_len": 4096,
            "lr": 5e-6,
            "epochs": 2,
            "batch_size": 2,
            "grad_accum": 1,
            "fp16": True,
            "lora_config": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
        },
        "memory": {
            "short_term_memory": [_verified_transition(row) for row in rows[-5:]],
            "memories_since_train": 0,
            "steps_trained_total": completed_updates,
        },
        "adapter_subdir": "lora",
        "feedback_to_policy": {
            "beta": 1.0,
            "update_frequency": 5,
            "buffer": [],
            "last_trace": rows[-1].get("f2p_trace", {}),
        },
        "f2p_loss_ablation": {
            "loss_mode": "normalized_logp_l2",
            "alpha": 0.25,
            "beta": 0.25,
            "nll_scale": last_update.get("nll_scale"),
            "l2_scale": last_update.get("l2_scale"),
        },
        "action_memory_branch": {
            "mode": "uninitialized",
            "action_history": [str(row.get("action", "")) for row in history_rows],
            "steps_seen": branch_step,
            "last_update_step": -1,
            "updates_total": 0,
            "source_agent_log": str(SOURCE_AGENT_LOG.resolve()),
            "branch_step": branch_step,
            "last_trace": {},
        },
    }
    atomic_write(str(memory_path), json.dumps(memory, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--action_memory_mode",
        required=True,
        choices=sorted(VALID_ACTION_MEMORY_MODES),
    )
    parser.add_argument("--action_memory_rank", type=int, default=4)
    parser.add_argument("--action_memory_lr", type=float, default=1e-3)
    parser.add_argument("--action_memory_update_frequency", type=int, default=5)
    parser.add_argument("--action_memory_window_size", type=int, default=25)
    parser.add_argument("--action_memory_optimization_steps", type=int, default=10)
    parser.add_argument("--action_memory_reference_beta", type=float, default=1.0)
    parser.add_argument("--action_memory_apply_scale", type=float, default=1.0)
    parser.add_argument("--action_memory_target_logprob_drop", type=float, default=0.0)
    parser.add_argument("--action_memory_semantic_threshold", type=float, default=0.85)
    parser.add_argument("--action_memory_microbatch_tokens", type=int, default=8)
    parser.add_argument("--branch_step", type=int, default=100)
    args, eval_args = parser.parse_known_args()

    def value(name: str) -> str | None:
        for index, item in enumerate(eval_args):
            if item == name and index + 1 < len(eval_args):
                return eval_args[index + 1]
            if item.startswith(name + "="):
                return item.split("=", 1)[1]
        return None

    run_dir_value = value("--run_dir")
    if not run_dir_value:
        raise ValueError("An explicit --run_dir is required")
    run_dir = Path(run_dir_value).resolve()
    seed = int(value("--seed") or 42)
    max_steps = int(value("--max_steps") or 150)
    if max_steps <= args.branch_step:
        raise ValueError("--max_steps must exceed --branch_step")

    prepare_step100_branch(
        run_dir,
        branch_step=args.branch_step,
        window_size=args.action_memory_window_size,
    )
    metadata = {
        "experiment": "step100_action_memory_branch",
        "setting": args.action_memory_mode,
        "source_run": str(SOURCE_RUN.resolve()),
        "source_lora": str(SOURCE_LORA.resolve()),
        "branch_environment_step": args.branch_step,
        "evaluated_environment_steps": [args.branch_step, max_steps - 1],
        "new_step_count": max_steps - args.branch_step,
        "seed": seed,
        "game": value("--game_name") or "remnant",
        "f2p": {
            "loss_mode": "normalized_logp_l2",
            "alpha": 0.25,
            "beta": 0.25,
            "lr": 5e-6,
            "update_frequency": 5,
            "short_term_memory": 5,
        },
        "action_memory": {
            "rank": args.action_memory_rank,
            "lr": args.action_memory_lr,
            "update_frequency": args.action_memory_update_frequency,
            "window_size": args.action_memory_window_size,
            "optimization_steps": args.action_memory_optimization_steps,
            "reference_beta": args.action_memory_reference_beta,
            "apply_scale": args.action_memory_apply_scale,
            "target_logprob_drop": args.action_memory_target_logprob_drop,
            "strength_selection": (
                "minimum_scale_meeting_target_logprob_drop"
                if args.action_memory_target_logprob_drop > 0
                else "fixed_apply_scale"
            ),
            "semantic_threshold": args.action_memory_semantic_threshold,
            "microbatch_tokens": args.action_memory_microbatch_tokens,
            "adapter_formula": "delta_z = W_up(SiLU(W_down(h_action_slot)))",
            "loss": "negative unlikelihood + beta * KL(base || adapted)",
            "reset_before_each_update": True,
        },
        "prompt_history": {
            "enabled": args.action_memory_mode == "prompt_history",
            "window_size": args.action_memory_window_size,
        },
        "runner": str(Path(__file__).resolve()),
    }
    metadata_path = run_dir / "experiment_config.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(f"Refusing to reuse {run_dir}: metadata differs")
    else:
        atomic_write(
            str(metadata_path),
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )

    _install_cluster_runtime_compatibility()
    install_loss_patch(loss_mode="normalized_logp_l2", alpha=0.25, beta=0.25)
    install_action_memory_patch(
        mode=args.action_memory_mode,
        rank=args.action_memory_rank,
        lr=args.action_memory_lr,
        update_frequency=args.action_memory_update_frequency,
        window_size=args.action_memory_window_size,
        optimization_steps=args.action_memory_optimization_steps,
        reference_beta=args.action_memory_reference_beta,
        apply_scale=args.action_memory_apply_scale,
        target_logprob_drop=args.action_memory_target_logprob_drop,
        semantic_threshold=args.action_memory_semantic_threshold,
        microbatch_tokens=args.action_memory_microbatch_tokens,
        seed=seed,
    )
    if value("--resume_from_step") is None:
        eval_args.extend(["--resume_from_step", str(args.branch_step)])
    sys.argv = [sys.argv[0], *eval_args]
    runpy.run_path(str(REPO_ROOT / "eval.py"), run_name="__main__")


if __name__ == "__main__":
    main()
