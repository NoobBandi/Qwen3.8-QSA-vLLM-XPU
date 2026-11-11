# Qwen3.8-QSA-vLLM-XPU

Run [Qwen3.8-27B-QSA](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16) on Intel GPUs with vLLM,
including its sparse attention.

QSA ([paper](https://arxiv.org/abs/2608.30320)) adds a small learned indexer to each of the model's 16
full-attention layers. For each query, the indexer picks 2,048 earlier tokens, and the layer attends only
those. As a result, speed hardly depends on context length. The reference engine,
[kuraneko1/Qwen3.8-27B-QSA-llama.cpp](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp), targets
NVIDIA GPUs. This repository is a vLLM plugin that brings the same sparse path to Intel XPU. It has been
tested on 2× Arc Pro B70.

![QSA vs dense prefill and decode throughput](docs/qsa-vs-dense.png)

Throughput in tokens/s:

| Context | Prefill, QSA | Prefill, dense | Decode, QSA | Decode, dense |
|--------:|-------------:|---------------:|------------:|--------------:|
| 0.5K | 1,517 | 1,617 | 74.2 | 71.2 |
| 4K | 2,038 | 2,221 | 70.5 | 74.4 |
| 16K | 2,053 | 2,324 | 71.2 | 70.0 |
| 64K | 2,014 | 2,064 | **70.6** | 66.1 |
| 128K | **1,962** | 1,741 | **69.5** | 59.2 |
| 256K | **1,877** | 1,299 | **62.7** | 51.6 |

At 256K, QSA prefills 1.44× and decodes 1.21× as fast as dense. Below 64K, prefill is 2–12% slower,
because the indexer has a fixed cost. Decode is within a few percent of dense up to 16K.

The two runs differ only in attention. Both use the same vLLM build and the same `scripts/serve.sh` settings:

- tensor parallel 2;
- online FP8 weights;
- MTP with 3 draft tokens;
- greedy decoding.

Dense loads the release as plain Qwen3.5, so its indexers are ignored. The prompts are vLLM's source
code. Prefill is measured cold, as prompt tokens per second until the first token. Raw results are in
[`docs/results`](docs/results).

A needle test (three passphrases at 15%, 50% and 85% of the context) found 3 of 3 from 20K to 230K.
Tool calls and image input work.

## Quick start

You need:

- vLLM 0.31.0 for Intel XPU. Use [vLLM's XPU install guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html), or work inside `vllm/vllm-openai-xpu:v0.31.0`.
- An Xe2 GPU. For 256K of context, two 32 GB cards.
- The [52 GB release](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16).

```bash
git clone https://github.com/NoobBandi/Qwen3.8-QSA-vLLM-XPU && cd Qwen3.8-QSA-vLLM-XPU

# 0.1.15, which ships with vLLM 0.31.0, overflows with QSA's KV layout (see Pitfalls)
pip install "vllm_xpu_kernels==0.1.17.1" --extra-index-url https://vllm-project.github.io/vllm-xpu-kernels/whl
pip install -e .        # vLLM finds the plugin through its entry point

python3 scripts/make_overlay.py /models/Qwen3.8-27B-QSA-BF16   # -> .../vllm-qsa (config only, weights are symlinked)
scripts/serve.sh /models/Qwen3.8-27B-QSA-BF16/vllm-qsa --port 8000
```

`serve.sh` runs `vllm serve` with the settings measured above. The first start takes a few minutes longer,
while kernels compile. If you write your own command, keep these settings:

- `vllm::qwen38_qsa_with_output` in `splitting_ops`. Otherwise the device is lost.
- `cudagraph_mode: FULL_DECODE_ONLY`. With PIECEWISE, decode drops to about 20 tok/s.
- `--max-num-batched-tokens 4096`.
- `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, which leaves room for 256K of KV cache.

## How it works

- **Model.** The plugin registers `QSAQwen3_5ForConditionalGeneration`: vLLM's Qwen3.5 with QSA
  attention, adapted from vLLM's Qwen4Exp QSA to match the release's `modeling_qsa.py`. The backbone
  weights load unchanged.
- **Kernels.** Scoring and attention are Triton kernels tuned for Xe2. Attention uses KaloyanNaumov's
  `attn2`. Top-k comes from vllm-xpu-kernels.
- **Prefill** is sparse as well as decode.
- **MTP draft.** The draft layer is sparse too, and borrows layer 63's indexer, because the checkpoint
  has none for it. It attends 4,096 tokens, and the newest 1,024 are always among them. Set
  `QWEN38_QSA_DRAFT_BUDGET` and `QWEN38_QSA_DRAFT_RECENT` to change those numbers.
- **XPU graphs.** Selection, the cache update and attention form one custom op, so a whole decode step
  is captured as one XPU graph.

## Pitfalls on XPU

- **XPU flash attention ignores the KV cache's block stride.** On a cache whose blocks are not
  contiguous, such as QSA's packed layout, it silently reads the wrong rows (about 50% error). See
  [`repro/xpu_flash_block_stride.py`](repro/xpu_flash_block_stride.py). This plugin never uses it on
  a QSA cache.
- **vllm-xpu-kernels 0.1.15 overflows int32 GDN offsets** with this layout and loses the device
  ([#658](https://github.com/vllm-project/vllm-xpu-kernels/issues/658)). Install 0.1.17.1 instead.
- **A custom op inside an inductor partition can lose the device.** Add it to `splitting_ops`.

## Limitations

- Tested only on 2× Arc Pro B70 with vLLM 0.31.0.
- Prefill below 64K is up to 12% slower than dense.
- At 256K, the borrowed draft indexer accepts 1.81 of 3 MTP tokens per step; dense accepts 2.27.
- Requests with images are attended sparsely. The reference runs them dense; long multimodal contexts
  were not tested.
- Without a per-card affinity patch, two-card runs pin about 64 GB of host RAM
  ([intel/compute-runtime#953](https://github.com/intel/compute-runtime/issues/953)).

## Development

```bash
python3 tests/test_kernels.py          # kernels against a torch reference, on XPU
python3 tests/test_prefill_select.py

# Benchmarks against a running server. corpus.txt is any long text; vLLM's source works:
find "$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')" -name "*.py" | sort | xargs cat > corpus.txt
python3 bench/needle.py --url http://127.0.0.1:8000    # also bench/decode.py, bench/prefill.py
python3 scripts/make_overlay.py /models/Qwen3.8-27B-QSA-BF16 --dense
bench/sweep.sh .../vllm-qsa .../vllm-dense               # data for the chart above
python3 bench/plot.py
```

These variables help with debugging: `QWEN38_QSA_DENSE`, `QWEN38_QSA_SYNC`, `QWEN38_QSA_PROFILE` and
`QWEN38_QSA_UNFUSED` (see the source).

## Credits

- Model and QSA training: higa2515.
- Reference llama.cpp engine: kuraneko1.
- Qwen4Exp QSA implementation: vLLM.
- Xe2 attention kernel, prefill launch and the sparse-draft idea: KaloyanNaumov
  ([zhaohb/vllm](https://github.com/zhaohb/vllm) #2, #8, #4).

Apache-2.0. See [NOTICE](NOTICE).
