from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from novelty_patch import (  # noqa: E402
    NOVELTY_ADAPTER,
    TASK_ADAPTER,
    _lora_layers,
    _set_active_adapters,
)


def _tiny_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def _config(rank: int, alpha: int) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def test_dual_adapter_routing_and_checkpoint() -> None:
    torch.manual_seed(7)
    peft = get_peft_model(_tiny_model(), _config(2, 4))
    peft.add_adapter(NOVELTY_ADAPTER, _config(1, 2))
    with torch.no_grad():
        for layer in _lora_layers(peft):
            layer.lora_B[TASK_ADAPTER].weight.fill_(0.02)
            layer.lora_B[NOVELTY_ADAPTER].weight.fill_(0.03)

    tokens = torch.tensor([[1, 2, 3, 4]])
    peft.eval()
    _set_active_adapters(
        peft,
        [TASK_ADAPTER],
        trainable_adapter=None,
        novelty_scale=0.0,
    )
    task_logits = peft(input_ids=tokens).logits

    _set_active_adapters(
        peft,
        [TASK_ADAPTER, NOVELTY_ADAPTER],
        trainable_adapter=None,
        novelty_scale=0.0,
    )
    zero_novelty_logits = peft(input_ids=tokens).logits
    assert torch.equal(task_logits, zero_novelty_logits)

    _set_active_adapters(
        peft,
        [TASK_ADAPTER, NOVELTY_ADAPTER],
        trainable_adapter=None,
        novelty_scale=0.5,
    )
    dual_logits = peft(input_ids=tokens).logits
    assert not torch.equal(task_logits, dual_logits)

    with tempfile.TemporaryDirectory() as directory:
        peft.save_pretrained(directory)
        root = Path(directory)
        assert (root / "adapter_config.json").is_file()
        assert (root / NOVELTY_ADAPTER / "adapter_config.json").is_file()

        loaded = PeftModel.from_pretrained(
            _tiny_model(), directory, is_trainable=True
        )
        loaded.load_adapter(
            str(root / NOVELTY_ADAPTER),
            adapter_name=NOVELTY_ADAPTER,
            is_trainable=True,
        )
        assert set(loaded.peft_config) == {TASK_ADAPTER, NOVELTY_ADAPTER}


if __name__ == "__main__":
    test_dual_adapter_routing_and_checkpoint()
    print("PEFT dual-adapter tests passed")
