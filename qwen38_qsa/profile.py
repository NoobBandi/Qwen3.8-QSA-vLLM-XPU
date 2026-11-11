# SPDX-License-Identifier: Apache-2.0
"""QWEN38_QSA_PROFILE=1: per-stage wall time of eager QSA steps (debugging).

Each stage is closed with a device synchronise, so the numbers are the stage's
own device time plus launch overhead; the totals are logged every 200 calls.
"""
import os
import time
from collections import defaultdict

import torch

ENABLED = os.environ.get("QWEN38_QSA_PROFILE") == "1"
_totals: dict[str, float] = defaultdict(float)
_calls: dict[str, int] = defaultdict(int)
_last = 0.0


def start() -> None:
    global _last
    if ENABLED and not torch.xpu.is_current_stream_capturing():
        torch.xpu.synchronize()
        _last = time.perf_counter()


def mark(stage: str) -> None:
    global _last
    if not ENABLED or torch.xpu.is_current_stream_capturing():
        return
    torch.xpu.synchronize()
    now = time.perf_counter()
    _totals[stage] += now - _last
    _calls[stage] += 1
    _last = now
    if stage == "attention" and _calls[stage] % 200 == 0:
        line = "  ".join(f"{k} {_totals[k] / _calls[k] * 1e3:.3f}ms" for k in _totals)
        print(f"[qwen38_qsa profile] per call: {line}", flush=True)
        _totals.clear(); _calls.clear()
