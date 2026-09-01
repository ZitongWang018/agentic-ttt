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

from loss_patch import install_loss_patch  # noqa: E402


def _install_cluster_runtime_compatibility() -> None:
    """Use memory-efficient attention and avoid generation graph compilation."""
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
    hfLanguageModel = huggingface_provider.hfLanguageModel

    original_init = hfLanguageModel.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.disable_compile = True
        print(
            "[F2PLossAblation] Transformers generation auto-compile disabled; "
            "using SDPA attention/CUDA on this cluster.",
            flush=True,
        )

    hfLanguageModel.__init__ = patched_init


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
        "--loss_mode",
        required=True,
        choices=["original", "no_w", "normalized_logp_l2"],
    )
    parser.add_argument("--loss_alpha", type=float, default=1.0)
    parser.add_argument("--loss_beta", type=float, default=0.0)
    ablation_args, eval_args = parser.parse_known_args()

    run_dir_value = _argument_value(eval_args, "--run_dir")
    if not run_dir_value:
        raise ValueError("The isolated experiment requires an explicit --run_dir")
    run_dir = Path(run_dir_value).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "ablation_config.json"
    metadata = {
        "loss_mode": ablation_args.loss_mode,
        "alpha": ablation_args.loss_alpha,
        "beta": ablation_args.loss_beta,
        "attention_implementation": "sdpa",
        "score_logits_to_keep": True,
        "runner": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("loss_mode", "alpha", "beta"):
            if existing.get(key) != metadata[key]:
                raise ValueError(
                    f"Refusing to reuse {run_dir}: {key} differs "
                    f"({existing.get(key)!r} != {metadata[key]!r})"
                )
    else:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    _install_cluster_runtime_compatibility()
    install_loss_patch(
        loss_mode=ablation_args.loss_mode,
        alpha=ablation_args.loss_alpha,
        beta=ablation_args.loss_beta,
    )
    sys.argv = [sys.argv[0], *eval_args]
    runpy.run_path(str(REPO_ROOT / "eval.py"), run_name="__main__")


if __name__ == "__main__":
    main()
