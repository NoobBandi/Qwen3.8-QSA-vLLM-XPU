# SPDX-License-Identifier: Apache-2.0
"""A cheaper output head for the MTP draft: FP8, over the tokens it can use.

The draft computes logits three times per decode step, each time reading the
whole BF16 lm_head (248K tokens x 5120, 1.27 GB per card): 6.4 of a step's
~43 ms on 2x B70, the largest cost after the weights themselves. The draft
only proposes tokens, and the target verifies every one with its own BF16 head,
so what the draft head computes changes how many tokens are accepted, never
the output. This head keeps:

* the ``QWEN38_QSA_DRAFT_VOCAB`` most frequent tokens (default 147,456 of
  248,320; ``draft_vocab.txt`` ranks them on the model's own Chinese and
  English answers, agent traces, code, and jieba's word frequencies in
  Simplified and Taiwan Traditional) and every special token; ``full`` keeps
  all of them. What is left out is mostly other scripts (Korean, Arabic,
  Thai, Cyrillic...): those still decode correctly, with fewer drafts accepted;
* their rows of the target's head in FP8 with a scale per row
  (``QWEN38_QSA_DRAFT_HEAD=bf16`` keeps BF16; ``off`` leaves the head alone).

Tokens outside the kept set get -inf, so the draft never proposes them. The
kept rows are split evenly between the cards, whichever card's shard of the
target head they come from.
"""

import os
import types

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MODE = os.environ.get("QWEN38_QSA_DRAFT_HEAD", "fp8")
VOCAB = os.environ.get("QWEN38_QSA_DRAFT_VOCAB", "147456")
VOCAB_FILE = os.environ.get(
    "QWEN38_QSA_DRAFT_VOCAB_FILE", os.path.join(os.path.dirname(__file__), "draft_vocab.txt")
)
FP8_MAX = 448.0


def kept_tokens(vocab_size: int, special_from: int) -> torch.Tensor | None:
    """Sorted token ids the draft may propose, or None for all of them."""
    if VOCAB == "full":
        if MODE == "off":
            return None
        return torch.arange(vocab_size)
    ranked = [int(line) for line in open(VOCAB_FILE) if line.strip()]
    kept = set(ranked[: int(VOCAB)])
    kept.update(range(special_from, vocab_size))          # every special token (and padding)
    return torch.tensor(sorted(i for i in kept if i < vocab_size))


class DraftHead:
    def __init__(self, lm_head, logits_processor, kept: torch.Tensor):
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        rank, world = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        weight = lm_head.weight
        self.width = logits_processor.org_vocab_size
        per_rank = weight.shape[0]                # every rank's padded shard is this wide
        assert lm_head.shard_indices.org_vocab_start_index == rank * per_rank, "unexpected vocab partition"
        # The common tokens all sit in the first shard, so the kept rows are
        # dealt out evenly instead: rank r holds the r-th slice of the sorted
        # kept ids, whichever shard they come from.
        self.rows = -(-len(kept) // world)
        slices = [kept[r * self.rows:(r + 1) * self.rows] for r in range(world)]
        device = weight.device
        self.columns = torch.cat([torch.cat([ids, torch.full((self.rows - len(ids),), self.width)])
                                  for ids in slices]).to(device)          # padding goes to a sink column
        if MODE == "fp8":
            self.weight = torch.zeros((self.rows, weight.shape[1]), dtype=torch.float8_e4m3fn, device=device)
            self.scale = torch.ones((self.rows, 1), dtype=torch.float32, device=device)
        else:
            self.weight, self.scale = torch.zeros((self.rows, weight.shape[1]), dtype=weight.dtype, device=device), None
        # Fetch rows from their owners a piece at a time: every rank fills the
        # piece's rows it owns, zeros elsewhere, and a sum over ranks completes it.
        piece = 8192
        for begin in range(0, len(kept), piece):
            ids = kept[begin:begin + piece]
            rows = self._fetch(weight, ids, rank * per_rank, per_rank)
            lo, hi = max(begin, rank * self.rows), min(begin + len(ids), (rank + 1) * self.rows)
            if lo >= hi:
                continue
            take = rows[lo - begin:hi - begin]
            at = slice(lo - rank * self.rows, hi - rank * self.rows)
            if MODE == "fp8":
                take = take.float()
                scale = take.abs().amax(1, keepdim=True).clamp(min=1e-12) / FP8_MAX
                self.scale[at] = scale
                self.weight[at] = (take / scale).to(torch.float8_e4m3fn)
            else:
                self.weight[at] = take
            del take, rows
        if MODE == "fp8":
            self.scale = self.scale.t().contiguous()   # [1, rows], as fp8_gemm_w8a16 expects
        logger.info(
            "QSA draft head: %d of %d tokens, %s, %.0f MB per card (was %.0f MB read per draft step)",
            len(kept), self.width, MODE, self.weight.numel() * self.weight.element_size() / 2**20,
            weight.numel() * weight.element_size() / 2**20,
        )

    @staticmethod
    def owned_rows(weight: torch.Tensor, ids: torch.Tensor, start: int, per_rank: int) -> torch.Tensor:
        """Rows ``ids`` of the full head where this rank's shard holds them, zeros elsewhere."""
        ids = ids.to(weight.device)
        owned = (ids >= start) & (ids < start + per_rank)
        rows = torch.zeros((len(ids), weight.shape[1]), dtype=weight.dtype, device=weight.device)
        rows[owned] = weight.detach()[ids[owned] - start]
        return rows

    @classmethod
    def _fetch(cls, weight: torch.Tensor, ids: torch.Tensor, start: int, per_rank: int) -> torch.Tensor:
        """Rows ``ids`` of the full head, [len(ids), hidden], on every rank."""
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(cls.owned_rows(weight, ids, start, per_rank))

    def local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This card's kept logits, [tokens, rows]."""
        if self.scale is not None:
            return torch.ops._xpu_C.fp8_gemm_w8a16(hidden_states, self.weight.t(), self.scale, None)
        return torch.nn.functional.linear(hidden_states, self.weight)

    def assemble(self, gathered: torch.Tensor) -> torch.Tensor:
        """Every card's kept logits, [tokens, world * rows], placed in a -inf row per token."""
        logits = torch.full((gathered.shape[0], self.width + 1), float("-inf"),
                            dtype=gathered.dtype, device=gathered.device)
        logits.index_copy_(1, self.columns, gathered)
        return logits[:, : self.width]

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_gather

        return self.assemble(tensor_model_parallel_all_gather(self.local(hidden_states), dim=-1))


def install(draft_model) -> None:
    """Route the draft's logits through a DraftHead (after its lm_head is final)."""
    if MODE == "off" and VOCAB == "full":
        return
    lm_head = getattr(draft_model, "lm_head", None)
    processor = getattr(draft_model, "logits_processor", None)
    if lm_head is None or processor is None or not hasattr(lm_head, "shard_indices"):
        logger.warning("QSA draft head: no lm_head on %s; left as it is", type(draft_model).__name__)
        return
    if MODE == "fp8" and not hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):
        logger.warning("QSA draft head: no XPU FP8 GEMM; left as it is")
        return
    special_from = int(os.environ.get("QWEN38_QSA_DRAFT_SPECIAL_FROM", "248044"))
    kept = kept_tokens(processor.org_vocab_size, special_from)
    if kept is None:
        return
    head = DraftHead(lm_head, processor, kept)
    draft_model._qwen38_draft_head = head

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        return self._qwen38_draft_head(hidden_states)

    draft_model.compute_logits = types.MethodType(compute_logits, draft_model)


def patch_speculator() -> None:
    from vllm.v1.worker.gpu.spec_decode.mtp import speculator as mtp

    cls = mtp.MTPSpeculator
    if getattr(cls, "_qwen38_draft_head", False):
        return
    load = cls.load_draft_model

    def load_draft_model(self, *args, **kwargs):
        model = load(self, *args, **kwargs)
        install(model)
        return model

    cls.load_draft_model = load_draft_model
    cls._qwen38_draft_head = True
