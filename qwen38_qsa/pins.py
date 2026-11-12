# SPDX-License-Identifier: Apache-2.0
"""Tool-call pinning: always attend the blocks that open earlier tool calls.

QSA's indexer picks 512 of the context's 4-token blocks per query. In long
agent sessions it rarely picks the earlier tool calls, and those are what a
dense model copies the call syntax from: right after ``<tool_call>`` the
release, sparse, wrote ``<|im_start|>`` or ``</tool_call>`` where ``<function=``
belongs, and the parser then dropped the whole call (136K-token Pi session:
P(correct '<') 0.2 sparse vs 1.0 dense; 3/36 calls valid). With the block of
each recent ``<tool_call>`` and the two after it forced in, the same position
gives 1.0 and 36/36 calls are valid.

How it works, without token ids inside the attention op:
- ``mark_tokens`` runs where vLLM embeds the step's token ids (outside the
  XPU graph) and flags the ``<tool_call>`` tokens in ``STEP``.
- every QSA layer writes ``STEP`` into ``SLOTS``, indexed like the main KV
  cache (``slot_mapping``), so a flag lives exactly as long as its token's KV
  and a rejected draft token is overwritten by the accepted one.
- ``block_bonus`` turns a request's page table into one value per block:
  1e6 for the newest ``CAP`` pinned blocks, so they outrank any score (sums of
  ReLUs, well below 1e4) and still rank among themselves by score.

QWEN38_QSA_PIN_CAP=0 turns it off.
"""

import json
import os

import torch

CAP = int(os.environ.get("QWEN38_QSA_PIN_CAP", "192"))   # blocks (3 per call)
WINDOW = 3                                             # the call's block and the two after
BONUS = 1.0e6
_SLOT_CAPACITY = 1 << 22                               # KV token slots covered

STEP: torch.Tensor | None = None      # [max tokens per step] uint8, this step's flags
SLOTS: torch.Tensor | None = None     # [_SLOT_CAPACITY + 1] uint8, per KV slot; last = sink
TOKEN_IDS: torch.Tensor | None = None
# The bonus is the same for every layer of a step (one page table, one set of
# flags): the step's first QSA layer computes it and the others reuse it. That
# holds under XPU graph replay too, since the layers always run in order.
_BONUS: dict[str, torch.Tensor] = {}


def enabled() -> bool:
    return CAP > 0 and STEP is not None


def setup(vllm_config, device) -> None:
    """Allocate the flag buffers and find ``<tool_call>``'s id in the tokenizer."""
    global STEP, SLOTS, TOKEN_IDS
    if CAP <= 0 or STEP is not None:
        return
    tokenizer = vllm_config.model_config.tokenizer
    ids = []
    try:
        cfg = json.load(open(os.path.join(tokenizer, "tokenizer_config.json")))
        for tid, tok in (cfg.get("added_tokens_decoder") or {}).items():
            if tok.get("content") == "<tool_call>":
                ids.append(int(tid))
    except OSError:
        pass
    if not ids:
        return
    max_tokens = 2 * int(vllm_config.scheduler_config.max_num_batched_tokens) + 1024
    STEP = torch.zeros(max_tokens, dtype=torch.uint8, device=device)
    SLOTS = torch.zeros(_SLOT_CAPACITY + 1, dtype=torch.uint8, device=device)
    TOKEN_IDS = torch.tensor(ids, dtype=torch.long, device=device)


def mark_tokens(input_ids: torch.Tensor) -> None:
    if not enabled():
        return
    n = input_ids.shape[0]
    STEP.zero_()
    STEP[:n] = (input_ids[:, None] == TOKEN_IDS[None, :]).any(-1)


def write_slots(slot_mapping: torch.Tensor) -> None:
    """Record this step's flags at its tokens' KV slots (padding goes to the sink)."""
    if not enabled():
        return
    n = slot_mapping.shape[0]
    slots = slot_mapping.long()
    slots = torch.where((slots >= 0) & (slots < _SLOT_CAPACITY), slots, _SLOT_CAPACITY)
    SLOTS.index_copy_(0, slots, STEP[:n])


def cached_bonus(kind: str, first: bool, page_table: torch.Tensor, rows_per_page: int,
                 columns: int, compress_ratio: int) -> torch.Tensor:
    """``block_bonus``, computed by the step's first layer (``first``) and reused."""
    if first or kind not in _BONUS:
        _BONUS[kind] = block_bonus(page_table, rows_per_page, columns, compress_ratio)
    return _BONUS[kind]


def block_bonus(page_table: torch.Tensor, rows_per_page: int, columns: int,
                compress_ratio: int) -> torch.Tensor:
    """[requests, columns] float32: BONUS on each request's newest pinned blocks."""
    device = page_table.device
    column = torch.arange(columns, device=device)
    page = page_table[:, column // rows_per_page].long().clamp(min=0)
    first = page * (rows_per_page * compress_ratio) + (column % rows_per_page) * compress_ratio
    offsets = torch.arange(compress_ratio, device=device)
    slots = (first[..., None] + offsets).clamp(max=_SLOT_CAPACITY)
    pinned = SLOTS[slots].amax(-1)                       # a <tool_call> in the block
    window = pinned.clone()
    for shift in range(1, WINDOW):                      # ... and the blocks after it
        window[:, shift:] = torch.maximum(window[:, shift:], pinned[:, :-shift])
    newer = window.flip(1).cumsum(1, dtype=torch.int32).flip(1)
    keep = window.bool() & (newer <= CAP)
    return keep.to(torch.float32) * BONUS
