# vLLM 0.31.0 for Intel XPU with Qwen3.8-27B-QSA sparse attention:
#  - vllm-xpu-kernels 0.1.17.1, which addresses GDN state in 64 bits. The
#    0.1.15 in the stock image overflows int32 offsets with QSA's block-major
#    KV layout and loses the device on a new request
#    (vllm-project/vllm-xpu-kernels#658).
#  - this plugin, found by vLLM through its general-plugin entry point.
#
#   docker build -t qwen3.8-qsa-vllm-xpu:v0.31.0 .
FROM vllm/vllm-openai-xpu:v0.31.0
RUN uv pip install --python /opt/venv/bin/python "vllm_xpu_kernels==0.1.17.1" \
    --extra-index-url https://vllm-project.github.io/vllm-xpu-kernels/whl
COPY . /src/qwen3.8-qsa-vllm-xpu
RUN uv pip install --python /opt/venv/bin/python --no-deps /src/qwen3.8-qsa-vllm-xpu
# The sparse indexer's logits buffer; the default leaves too little room for
# 256K of KV cache on two 32 GB cards.
ENV VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
