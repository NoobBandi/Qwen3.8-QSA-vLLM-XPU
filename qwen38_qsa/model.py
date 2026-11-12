# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B-QSA: vLLM's Qwen3.5 with QSA in its 16 full-attention layers."""

from collections.abc import Iterable

import torch

import vllm.model_executor.models.qwen3_5 as qwen3_5
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

import os

from . import pins
from .attention import QSAQwen3_5Attention, indexer_config
from .indexer_qsa import QSAIndexer

# checkpoint leaf under ``self_attn._qsa_indexer`` -> (vLLM leaf, row slice)
_INDEXER_WEIGHTS = {
    "q_proj.weight": "index_qk_proj.weight",
    "k_proj.weight": "index_qk_proj.weight",
    "q_norm": "q_layernorm.weight",
    "k_norm": "k_token_norm.weight",
    "k_block_norm": "k_layernorm.weight",
}


class QSAQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Qwen3_5DecoderLayer builds its full attention by this module-level
        # name; swap it only while this model is built, so the MTP draft built
        # afterwards keeps dense attention (its layer has no indexer).
        dense = qwen3_5.Qwen3NextAttention
        qwen3_5.Qwen3NextAttention = QSAQwen3_5Attention
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_5.Qwen3NextAttention = dense
        pins.setup(vllm_config, vllm_config.device_config.device)

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        # vLLM embeds each step's token ids here, outside the XPU graph.
        pins.mark_tokens(input_ids)
        return super().embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if input_ids is not None and inputs_embeds is None:
            pins.mark_tokens(input_ids)     # text-only path: ids reach the forward
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        indexer: list[tuple[str, torch.Tensor]] = []

        def backbone():
            for name, tensor in weights:
                if "._qsa_indexer." in name:
                    indexer.append((name, tensor))
                else:
                    yield name, tensor

        loaded = super().load_weights(backbone())
        params = dict(self.named_parameters())
        for name, tensor in indexer:
            layer = int(name.split(".layers.")[1].split(".")[0])
            leaf = name.split("._qsa_indexer.")[1]
            target = next(
                p
                for p in params
                if p.endswith(f".layers.{layer}.self_attn.indexer.{_INDEXER_WEIGHTS[leaf]}")
            )
            param = params[target]
            q_rows = param.shape[0] - tensor.shape[0] if leaf == "k_proj.weight" else 0
            with torch.no_grad():
                if leaf == "q_proj.weight":
                    param.data[: tensor.shape[0]].copy_(tensor)
                elif leaf == "k_proj.weight":
                    param.data[q_rows:].copy_(tensor)
                else:
                    param.data.copy_(tensor)
            loaded.add(target)
        expected = {p for p in params if ".self_attn.indexer." in p}
        missing = expected - loaded
        if missing:
            raise ValueError(f"QSA indexer weights missing from the checkpoint: {sorted(missing)[:4]}")
        return loaded


def _mtp_indexer_source(config) -> int:
    """The target layer whose indexer the draft borrows: the last full-attention one."""
    types = config.layer_types
    return max(i for i, kind in enumerate(types) if kind == "full_attention")


def patch_mtp() -> None:
    """Give the MTP draft a QSA layer when the target is QSA.

    The draft's dense attention reads the whole context on every draft step:
    on the B70 that is the XPU flash kernel, which misreads QSA's packed
    (BLNHC) cache, or Triton, which is slow at long context; either way the
    draft falls far behind the target (33 tok/s at 66K against dense's 68).
    The checkpoint has no indexer for its MTP layer, so it borrows the last
    full-attention layer's: a draft that selects worse only costs acceptance,
    since the target verifies every token. Same idea as zhaohb/vllm#4.
    """
    from vllm.model_executor.models import qwen3_5_mtp

    mtp = qwen3_5_mtp.Qwen3_5MTP
    if getattr(mtp, "_qwen38_qsa", False):
        return
    original_init = mtp.__init__
    original_load = mtp.load_weights

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if getattr(vllm_config.model_config.hf_config, "qsa", None) is None:
            original_init(self, vllm_config=vllm_config, prefix=prefix)
            return
        # QWEN38_QSA_DRAFT_BUDGET: tokens the draft attends. A wider budget
        # than the target's 2048 is cheap -- one layer -- and brings the
        # borrowed indexer's misses back in. 4096 with the newest 1024 forced
        # (below) measured best on the B70: accepted 1.89 of 3 at 230K against
        # 1.73 with the target's settings (dense draft: 2.08), 66 tok/s.
        budget = int(os.environ.get("QWEN38_QSA_DRAFT_BUDGET", "4096"))

        def draft_attention(*args, **kwargs):
            config = indexer_config(args[0] if args else kwargs["config"])
            if budget:
                config.indexer_budget = budget
            return QSAQwen3_5Attention(*args, qsa_config=config, **kwargs)

        dense = qwen3_5.Qwen3NextAttention
        qwen3_5.Qwen3NextAttention = draft_attention
        try:
            original_init(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_5.Qwen3NextAttention = dense
        self._qwen38_qsa_source = _mtp_indexer_source(self.config)
        # QWEN38_QSA_DRAFT_RECENT: newest tokens the draft always attends,
        # the rest of its budget chosen by the borrowed indexer.
        recent = int(os.environ.get("QWEN38_QSA_DRAFT_RECENT", "1024"))
        for module in self.modules():
            if isinstance(module, QSAIndexer):
                module.recent_blocks = recent // module.compress_ratio

    def load_weights(self, weights):
        source = getattr(self, "_qwen38_qsa_source", None)
        if source is None:
            return original_load(self, weights)
        indexer: list[tuple[str, torch.Tensor]] = []
        marker = f".layers.{source}.self_attn._qsa_indexer."

        def rest():
            for name, tensor in weights:
                if marker in name:
                    indexer.append((name, tensor))
                elif "._qsa_indexer." not in name:
                    yield name, tensor

        loaded = original_load(self, rest())
        params = dict(self.named_parameters())
        for name, tensor in indexer:
            leaf = name.split("._qsa_indexer.")[1]
            target = next(
                p for p in params
                if ".self_attn.indexer." in p and p.endswith(_INDEXER_WEIGHTS[leaf])
            )
            param = params[target]
            with torch.no_grad():
                if leaf == "q_proj.weight":
                    param.data[: tensor.shape[0]].copy_(tensor)
                elif leaf == "k_proj.weight":
                    param.data[param.shape[0] - tensor.shape[0]:].copy_(tensor)
                else:
                    param.data.copy_(tensor)
            loaded.add(target)
        missing = {p for p in params if ".self_attn.indexer." in p} - loaded
        if missing:
            raise ValueError(f"MTP indexer weights not loaded: {sorted(missing)[:4]}")
        return loaded

    mtp.__init__ = __init__
    mtp.load_weights = load_weights
    mtp._qwen38_qsa = True
