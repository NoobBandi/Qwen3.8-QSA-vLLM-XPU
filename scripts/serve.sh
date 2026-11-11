#!/usr/bin/env bash
# vllm serve for Qwen3.8-27B-QSA on two Intel cards, with the settings the
# README's results were measured with. Extra arguments go to `vllm serve`.
#
#   scripts/serve.sh /models/Qwen3.8-27B-QSA-BF16/vllm-qsa [--port 8000 ...]
set -euo pipefail
MODEL="${1:?usage: serve.sh MODEL_OVERLAY_DIR [vllm serve args...]}"
shift

# The sparse indexer's logits buffer; vLLM's default leaves too little memory
# for 256K of KV cache on two 32 GB cards.
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-128}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0,1}"

# Every op that must stay outside an inductor partition: vLLM's defaults
# plus this plugin's attention, which loses the device inside one.
SPLIT='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward","vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update","vllm::all_reduce","vllm::qwen38_qsa_with_output"'

exec vllm serve "$MODEL" \
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
