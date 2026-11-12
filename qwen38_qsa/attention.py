# SPDX-License-Identifier: Apache-2.0
"""QSA full attention for Qwen3.8-27B-QSA on vLLM, Triton kernels on XPU.

Adapted from vLLM's Qwen4Exp QSA owner (vllm/models/qwen4_exp/nvidia/qsa.py):
the layer owns its paged K/V cache and two side caches, selects 2048 tokens
per query with the indexer and attends only those, all inside one custom op so
the decode step stays one captured XPU graph. The projections, norms, RoPE and
output gate are Qwen3.5's own (Qwen3NextAttention), so the backbone weights
load unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, get_kv_quant_mode

from . import pins, profile
from .indexer_qsa import QSAIndexer
from .qsa_cache import QSAForwardMetadata

logger = init_logger(__name__)


class QSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class QSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
    # with the layer's per-tensor scales and dequantized on load inside the QSA
    # Triton kernel. flash-attn never runs over this cache, so its fp8 probe
    # does not apply (see supports_kv_cache_dtype and the impl constructor).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # QSA dequantizes the fp8 KV in its own Triton kernel and never runs
        # flash-attn over the quantized cache, so the parent's fp8-KV rejection
        # does not apply and every combination it is handed is accepted here.
        return None

    @staticmethod
    def get_name() -> str:
        return "QWEN38_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[QSAFlashAttentionImpl]:
        return QSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[QSAMetadataBuilder]:
        return QSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class QSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # The parent constructor probes flash-attn for quantized-KV support and
        # raises where it is unavailable (sm120), but QSA dequantizes fp8 inside
        # its own Triton kernel and never runs flash-attn over the cache. Hand
        # the parent "auto" for that probe and restore the real dtype afterwards:
        # the parent only uses it there, and do_kv_cache_update reads the
        # attribute at call time.
        real_kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            kv_cache_dtype = "auto"
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        self.kv_cache_dtype = real_kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        k_scale = v_scale = None
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            # The cache is allocated as uint8; reinterpret the e4m3 bytes
            # (same itemsize, so shape and strides are preserved).
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            # Host-side per-tensor dequant scales (Python floats), as used by
            # other host-scale backends; folded into the kernel's scales.
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float
        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise NotImplementedError(
                "QSA requires BF16 Q and BF16 or FP8-e4m3 K/V"
            )

        from .ops.qsa_attn2 import qsa_sparse_paged_attention_attn2

        # attn2 (Xe2-shaped Triton) reads the selection and every bound from
        # device memory with a fixed grid, so the step can be captured.
        group_size = query.shape[1] // key_cache.shape[2]
        qsa_sparse_paged_attention_attn2(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            output_gate[:num_tokens].view(output[:num_tokens].shape),
            self.scale,
            1.0,
            triton.next_power_of_2(group_size),
        )
        return output

    def forward_dense(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Full causal attention over the paged cache, then the output gate.

        Triton's unified attention rather than the XPU flash kernel: QSA packs
        its side caches beside the K/V pages (BLNHC), so a block's K/V is
        followed by a gap, and the flash kernel ignores the block stride and
        reads the wrong rows (about 50 % relative error). Triton follows the
        strides.
        """
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention

        num_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        unified_attention(
            q=query[:num_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=(-1, -1),
            block_table=attn_metadata.block_table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )
        gate = output_gate[:num_tokens].view(output[:num_tokens].shape)
        output[:num_tokens].mul_(torch.sigmoid(gate))
        return output




# QWEN38_QSA_DENSE=1 attends in full everywhere (the indexer still runs):
# the same model without sparsity, for comparisons.
_DENSE = __import__("os").environ.get("QWEN38_QSA_DENSE") == "1"
# QWEN38_QSA_SYNC=1 synchronises around each stage of an eager QSA step, so
# a device fault is reported by the stage that caused it (debugging only).
_SYNC = __import__("os").environ.get("QWEN38_QSA_SYNC") == "1"


def _checkpoint(stage: str, layer_name: str) -> None:
    if not _SYNC or torch.xpu.is_current_stream_capturing():
        return
    try:
        torch.xpu.synchronize()
    except Exception as error:
        raise RuntimeError(f"QSA device fault surfaced at {stage} of {layer_name}") from error
def indexer_config(config) -> SimpleNamespace:
    """The indexer shape from the checkpoint's ``qsa`` block, in Qwen4Exp terms."""
    # ``qsa`` sits at the top of config.json, not in text_config.
    qsa = getattr(config, "qsa", None) or getattr(
        get_current_vllm_config().model_config.hf_config, "qsa", None
    ) or {}
    return SimpleNamespace(
        hidden_size=config.hidden_size,
        rms_norm_eps=config.rms_norm_eps,
        indexer_n_heads=int(qsa.get("indexer_heads", 4)),
        indexer_kv_heads=1,
        indexer_head_dim=int(qsa.get("indexer_head_dim", 128)),
        indexer_budget=int(qsa.get("budget", 2048)),
        indexer_compress_ratio=int(qsa.get("block_size", 4)),
    )


class QSAQwen3_5Attention(Qwen3NextAttention, AttentionLayerBase):
    """Qwen3.5 full attention that owns its K/V cache and attends sparsely.

    Built with Qwen3NextAttention's signature so it can stand in for it while
    Qwen3_5DecoderLayer is constructed.
    """

    supports_dcp = False

    def __init__(
        self,
        config,
        model_config=None,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        qsa_config=None,
    ) -> None:
        nn.Module.__init__(self)
        vllm_config = get_current_vllm_config()
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("QSA requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("QSA here is tested with a BF16 KV cache only")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = None
        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        assert self.attn_output_gate, "Qwen3.5 full attention carries an output gate"

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            getattr(self.rotary_emb, "is_neox_style", False)
            and (current_platform.is_cuda() or current_platform.is_xpu())
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = QSAFlashAttentionBackend
        self.impl = QSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        layer_id = int(prefix.split("layers.")[-1].split(".")[0])
        # The indexer stays in BF16: online FP8 of its projection would move
        # the scores the selection is ranked by.
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=qsa_config or indexer_config(config),
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=None,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens, self.indexer.packed_output_width, dtype=torch.int32
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")
        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata, metadata[self.indexer.raw_key_cache.prefix]
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        # Prefill attends sparsely too, as the indexer was trained (and the
        # dense kernels are no option here: XPU flash misreads the packed
        # cache, Triton prefill runs at ~100 tok/s at long context).
        sparse = not _DENSE
        _checkpoint("entry", self.layer_name)
        profile.start()
        if sparse and main_metadata.max_seq_len > self.indexer.output_width:
            logger.info_once(
                "QSA: sparse decode engaged (%d-token context, %d-token budget)",
                main_metadata.max_seq_len,
                self.indexer.output_width,
            )
        if self.indexer.pin and self.indexer.pin_first:
            # Once per step (every layer shares the main KV slots), and before
            # selection, so a <tool_call> earlier in this chunk counts.
            pins.write_slots(main_metadata.slot_mapping[:num_tokens])
        self.indexer.skip_topk = not sparse
        self.indexer(projected_qk, positions, self.topk_indices_buffer[:num_tokens])
        self.indexer.skip_topk = False
        _checkpoint("indexer", self.layer_name)
        profile.mark("expand")
        impl = cast(QSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self, key, value, self.kv_cache, main_metadata.slot_mapping
        )
        _checkpoint("kv update", self.layer_name)
        profile.mark("kv_update")
        if not sparse:
            impl.forward_dense(
                self, query, key, value, self.kv_cache, main_metadata, output,
                output_gate,
            )
            _checkpoint("dense attention", self.layer_name)
            return
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len
            > self._max_decode_query_len,
            output_gate=output_gate,
        )
        profile.mark("attention")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        torch.ops.vllm.qwen38_qsa_with_output(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
            gate.contiguous(),
            _encode_layer_name(self.layer_name),
        )
        output, _ = self.o_proj(attn_output.view(num_tokens, -1))
        return output


def qwen38_qsa_with_output(
    projected_qk: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """The whole QSA transaction: side caches, selection, K/V write, attend."""
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    layer._run_qsa(projected_qk, positions, query, key, value, output, output_gate)


def qwen38_qsa_with_output_fake(
    projected_qk, positions, query, key, value, output, output_gate, layer_name
) -> None:
    return None


direct_register_custom_op(
    op_name="qwen38_qsa_with_output",
    op_func=qwen38_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen38_qsa_with_output_fake,
)
