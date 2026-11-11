#!/usr/bin/env bash
# Serve Qwen3.8-27B-QSA on two Intel cards with the settings measured in the
# README. Extra arguments go to `vllm serve`.
#
#   scripts/serve.sh /models/Qwen3.8-27B-QSA-BF16/vllm-qsa [--port 8000 ...]
#
# Env: IMAGE (default qwen3.8-qsa-vllm-xpu:v0.31.0), CACHE_DIR (compiled kernels,
# default ~/.cache/qwen3.8-qsa-vllm-xpu), CARDS (default level_zero:0,1),
# CONTAINER (default qwen38-qsa).
set -euo pipefail

MODEL="$(readlink -f "${1:?usage: serve.sh MODEL_OVERLAY_DIR [vllm serve args...]}")"
shift
# The overlay's symlinks point into the release directory beside it.
MOUNT="$(dirname "$MODEL")"
IMAGE="${IMAGE:-qwen3.8-qsa-vllm-xpu:v0.31.0}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/qwen3.8-qsa-vllm-xpu}"
mkdir -p "$CACHE_DIR"

# Every op that must stay outside an inductor partition: vLLM's defaults
# plus this plugin's attention. Inside a partition it loses the device.
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward","vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update","vllm::all_reduce","vllm::qwen38_qsa_with_output"'

exec docker run --rm --init --name "${CONTAINER:-qwen38-qsa}" \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --ipc host --shm-size 16g --network host \
  -e ONEAPI_DEVICE_SELECTOR="${CARDS:-level_zero:0,1}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/root/.cache/vllm/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/inductor \
  -v "$CACHE_DIR:/root/.cache/vllm" \
  -v "$MOUNT:$MOUNT:ro" \
  "$IMAGE" \
  --model "$MODEL" \
  --served-model-name qwen38-27b-qsa \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.935 \
  --quantization fp8_per_tensor \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"splitting_ops\":[$SPLIT]}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
