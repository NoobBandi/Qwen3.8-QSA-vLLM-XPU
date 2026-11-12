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
    from .model import patch_mtp

    patch_mtp()
