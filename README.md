# qwen3.8-qsa-vllm-xpu

A vLLM plugin that serves [Qwen3.8-27B-QSA](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16)
with its sparse attention on Intel GPUs. It has been tested on two Arc Pro B70 cards (Xe2, 32 GB each).

QSA (Qwen Sparse Attention) gives each of the model's 16 full-attention layers a small learned indexer.
For each query, the indexer picks 2,048 of the earlier tokens, and the layer attends only those, instead
of the whole context. Stock vLLM cannot load the checkpoint. If you load it as a plain Qwen3.5, it runs,
but with dense attention: the indexers are ignored, and so is the speed-up. This plugin runs the sparse
path on XPU. The custom kernels are Triton, compiled for Intel GPUs; the rest of the model uses vLLM's
own SYCL kernels.

## Background

This project exists because of [kuraneko1/Qwen3.8-27B-QSA-llama.cpp](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp),
the reference inference engine for this model. It is a llama.cpp fork that runs QSA with MTP,
multimodal input and prefix-cache reuse, and its prebuilt runtime targets NVIDIA GPUs (CUDA). This
repository brings the same model, with its sparse attention, to Intel GPUs, served by vLLM. The
architecture is described in [the QSA paper](https://arxiv.org/abs/2608.30320).

## Results

These were measured on 2× Arc Pro B70 with tensor parallel 2, online FP8 weights, MTP with 3 draft tokens,
FULL_DECODE_ONLY XPU graphs and greedy decoding. "Dense" is the same checkpoint loaded by stock vLLM as
Qwen3.5.

| Context | Decode, dense | Decode, QSA | MTP tokens accepted per step (of 3), dense / QSA |
|--------:|--------------:|------------:|-------------------------------------------------:|
| 2K      | 82 tok/s      | 78 tok/s    | 2.44 / 2.33 |
| 66K     | 68 tok/s      | 67 tok/s    | 2.15 / 1.91 |
| 230K    | 51 tok/s      | **66 tok/s**| 2.08 / 1.89 |

| Context | Prefill, dense | Prefill, QSA |
|--------:|---------------:|-------------:|
| 16K     | 2,321 tok/s    | 2,132 tok/s  |
| 64K     | 1,972 tok/s    | 2,085 tok/s  |
| 132K    | 1,618 tok/s    | **2,029 tok/s** |

QSA's speed barely changes with context length, while dense keeps getting slower. Below about 64K, the
indexer costs 5–8%.

On correctness, a needle test (three passphrases at 15%, 50% and 85% of the context) found 3 of 3 at
every length from 20K to 230K tokens. Tool calls and image input work.

## Requirements

- An Intel GPU with Xe2 (the kernel launch settings were tuned on a B70) and the host's `xe` or `i915` driver.
- Docker. The image is built on `vllm/vllm-openai-xpu:v0.31.0`.
- The release itself, [higa2515/Qwen3.8-27B-QSA-BF16](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16) (52 GB).
- For 256K of context: two 32 GB cards at `--gpu-memory-utilization 0.935`.

## Quick start

```bash
# 1. The image: vLLM 0.31.0 for XPU, vllm-xpu-kernels 0.1.17.1, and this plugin
docker build -t qwen3.8-qsa-vllm-xpu:v0.31.0 .

# 2. A directory vLLM can load the checkpoint from: an edited config.json, plus
#    symlinks to the release's files (no weights are copied)
python3 scripts/make_overlay.py /models/Qwen3.8-27B-QSA-BF16
#    -> /models/Qwen3.8-27B-QSA-BF16/vllm-qsa

# 3. Serve on cards 0 and 1, on port 8000
scripts/serve.sh /models/Qwen3.8-27B-QSA-BF16/vllm-qsa --port 8000
```

`serve.sh` passes the settings the results above were measured with. The first start compiles the kernels
and graphs; they are cached in `~/.cache/qwen3.8-qsa-vllm-xpu`, and a start with a warm cache takes about two
minutes.

If you write your own `vllm serve` command, keep these settings:

- `--compilation-config` must list `vllm::qwen38_qsa_with_output` in `splitting_ops`. Inside an inductor
  partition, the op loses the device.
- `cudagraph_mode` should be `FULL_DECODE_ONLY`. With `PIECEWISE`, decode drops to about 20 tok/s.
- `--max-num-batched-tokens 4096` is the prefill chunk the scorer was tuned for.

### Two cards and host memory

With both cards visible to every tensor-parallel worker, Intel's compute runtime mirrors each worker's
device memory in pinned host RAM. For this model that is about 64 GB
([intel/compute-runtime#953](https://github.com/intel/compute-runtime/issues/953)). Our measurements
used a per-worker card affinity patch that avoids this, after
[SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook).
That patch is not part of this repository.

## Tests and benchmarks

The kernel tests compare against a torch reference written from the release's `modeling_qsa.py`.

```bash
docker run --rm --device /dev/dri -v "$PWD:/w" -w /w --entrypoint python3 \
    qwen3.8-qsa-vllm-xpu:v0.31.0 tests/test_kernels.py
docker run --rm --device /dev/dri -v "$PWD:/w" -w /w --entrypoint python3 \
    qwen3.8-qsa-vllm-xpu:v0.31.0 tests/test_prefill_select.py
```

The benchmarks talk to a running server. They need a long text to fill the context; vLLM's own source
code works:

```bash
docker run --rm --entrypoint sh qwen3.8-qsa-vllm-xpu:v0.31.0 -c \
    'find /opt/venv/lib/python3.12/site-packages/vllm -name "*.py" | sort | xargs cat' > corpus.txt
python3 bench/needle.py  --url http://127.0.0.1:8000
python3 bench/decode.py  --url http://127.0.0.1:8000 --lengths 2000,64000,230000
python3 bench/prefill.py --url http://127.0.0.1:8000 --lengths 16000,64000,128000
```

The tables above were measured on a different code corpus. With `serve.sh` and the vLLM-source corpus
from the command above, the same machine gives:

- decode: 70 / 71 / 68 tok/s at 2K / 64K / 211K, accepting 2.04 / 2.13 / 2.04 tokens per step;
- prefill: 2,074 / 2,018 / 1,968 tok/s at 17K / 64K / 124K;
- needle: 3 of 3 at 21K, 111K and 211K.

## How it works

- **Model.** The plugin registers `QSAQwen3_5ForConditionalGeneration`: vLLM's Qwen3.5 with QSA
  attention in the 16 full-attention layers. The backbone weights load unchanged. The indexer weights
  come from the release's `qsa-indexers.safetensors`.
- **Indexer.** It is adapted from vLLM's Qwen4Exp QSA and matches the release's reference semantics:
  - each raw key is RMS-normed before a group of 4 is pooled;
  - the pooled key has its own norm and is rotated by RoPE at the group's first position;
  - a group's score is Σ relu(q·k) over 4 heads;
  - the query always attends its own group (2,048 + 4 tokens).
- **Kernels.**
  - Scoring uses vLLM's Triton scorer with launch settings swept for Xe2. Decode uses
    BLOCK_N 32, 2 stages, 4 warps, which is 2.7× faster at 250K. Prefill uses KaloyanNaumov's settings.
  - Top-k uses vllm-xpu-kernels' `top_k_per_row_decode`.
  - Attention uses KaloyanNaumov's `attn2` kernel, shaped for Xe2's register file at head_dim 256.
  - Prefill is sparse too.
- **MTP draft.** The draft layer is sparse as well. Dense, it fell far behind at long context: 33 tok/s
  at 66K. The checkpoint has no indexer for the draft layer, so it borrows layer 63's. Its budget is
  4,096 tokens, with the newest 1,024 always included. Acceptance at 230K went from 1.73 to 1.89.
- **One op.** Indexing, the cache update and attention run inside one custom op, so the whole decode step
  stays a single captured XPU graph.

### Settings

| Variable | Default | |
|---|---|---|
| `QWEN38_QSA_DRAFT_BUDGET` | `4096` | Tokens the MTP draft attends. 8192 gave no gain and leaves too little KV memory for 256K. |
| `QWEN38_QSA_DRAFT_RECENT` | `1024` | Newest tokens the draft always attends. |
| `QWEN38_QSA_DECODE_LAUNCH` | `32,2,4` | Decode scorer BLOCK_N, stages, warps (for sweeps on other GPUs). |
| `QWEN38_QSA_DENSE` | off | `1` attends in full, for comparisons (slow at long context). |
| `QWEN38_QSA_SYNC` | off | `1` synchronises after each stage of an eager step, to locate device faults. |
| `QWEN38_QSA_PROFILE` | off | `1` prints per-stage times of eager steps. |
| `QWEN38_QSA_UNFUSED` | off | `1` disables the fused pre-indexer kernel. |

Pass them to the container with `-e`. `serve.sh` forwards nothing extra, so add them to it.

## Pitfalls found on XPU

Each of these cost hours, and some will bite other XPU work too.

- **XPU flash attention ignores the KV cache's block stride.** QSA packs its side caches beside each
  block's K/V, so blocks are not contiguous. `flash_attn_varlen_func` then reads the wrong rows, with
  about 50% error. Triton's `unified_attention` reads them correctly.
  [`repro/xpu_flash_block_stride.py`](repro/xpu_flash_block_stride.py) reproduces it on the stock image.
  This plugin never calls the flash kernel on a QSA cache.
- **vllm-xpu-kernels 0.1.15 (the stock v0.31.0 image) overflows int32 GDN state offsets** with QSA's
  block-major KV layout. The device is lost on a new request
  ([vllm-xpu-kernels#658](https://github.com/vllm-project/vllm-xpu-kernels/issues/658)). 0.1.17.1
  fixes it, which is why the image upgrades it.
- **A custom op inside an inductor partition can lose the device.** List it in `splitting_ops`.
- **Triton on XPU needs N ≥ 16 in `tl.dot`.** At decode with one query row, the scorer pads its rows.
- **Vendor split-K attention spills at head_dim 256 on Xe2**, at 2.3 ms per layer. attn2 takes
  0.13–0.24 ms.

## Limitations

- Tested only on 2× Arc Pro B70 with vLLM 0.31.0. Other Xe2 parts should work, but may want a new launch
  sweep. Xe-HPC (Max series) is untested.
- At short contexts, decode and prefill are 5–8% slower than dense: the indexer's fixed cost.
- The borrowed draft indexer accepts fewer tokens than a dense draft: 1.89 against 2.08 at 230K.
- The release's reference runs any request with an image or video densely (`multimodal_policy: dense`).
  This plugin attends those requests sparsely like any other. Below 2,048 tokens, sparse and dense are
  the same; image chat was checked, but long multimodal contexts were not.

## Credits

- The model and QSA training: higa2515.
- The reference llama.cpp engine that prompted this port: [kuraneko1](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp).
- The Qwen4Exp QSA implementation this plugin is adapted from: vLLM.
- The Xe2 attention kernel, the prefill launch and the sparse MTP idea: KaloyanNaumov, in
  [zhaohb/vllm](https://github.com/zhaohb/vllm) #2, #8 and #4.

See [NOTICE](NOTICE). Licensed under Apache-2.0.
