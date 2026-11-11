# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin: Qwen3.8-27B-QSA sparse attention on Intel XPU.

Loaded by vLLM's general-plugin hook in every process (API server, engine,
workers) through the `vllm.general_plugins` entry point in pyproject.toml.
"""

ARCH = "QSAQwen3_5ForConditionalGeneration"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models import config as model_configs

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "qwen38_qsa.model:" + ARCH)
    # Qwen3.5's hybrid-cache contract (fp32 GDN state from mamba_ssm_dtype).
    model_configs.MODELS_CONFIG_MAP.setdefault(
        ARCH, model_configs.Qwen3_5ForConditionalGenerationConfig
    )
    _skip_side_caches_in_zeroer()
    from .model import patch_mtp

    patch_mtp()


def _skip_side_caches_in_zeroer() -> None:
    """Leave the QSA side caches out of vLLM's new-block zeroing.

    Nothing reads a side-cache row before writing it: the compressed cache is
    read only below each query's visible-block count, the ring only at
    committed positions. Zeroing them on every new block is wasted work.
    """
    import vllm.v1.worker.utils as worker_utils

    zeroer = worker_utils.KVBlockZeroer
    if getattr(zeroer, "_qwen38_qsa", False):
        return
    original = zeroer.__init__

    def __init__(self, device, attn_groups_iter, kernel_block_sizes,
                 static_forward_context, num_blocks, runner_only_attn_layers=None):
        skipped = set(runner_only_attn_layers or ())
        skipped |= {
            name for name in static_forward_context
            if name.endswith((".raw_key_cache", ".compressed_key_cache"))
        }
        original(self, device, attn_groups_iter, kernel_block_sizes,
                 static_forward_context, num_blocks, skipped)

    zeroer.__init__ = __init__
    zeroer._qwen38_qsa = True
