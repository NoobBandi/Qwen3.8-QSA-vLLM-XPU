# SPDX-License-Identifier: Apache-2.0
"""Dense attention for short contexts, sparse beyond a threshold.

QSA's selection (scoring, top-k and index expansion in every layer) costs
about the same at any context length, while dense attention grows with it.
On 2x B70 dense is faster up to about 48K tokens: decode steps cross near 40K,
and dense also accepts more MTP draft tokens. So steps whose longest context
is within ``THRESHOLD`` attend densely through XPU flash; the indexer still
updates its side caches, so a request that grows past the threshold switches
to sparse attention with everything in place. Dense attention is what the
indexer was distilled from, so the switch never trades accuracy for speed.

A decode step runs as one captured XPU graph, and a Python branch inside it
would be frozen at capture. So every graph manager captures each graph twice,
once per regime, and replays the one matching the step: the attention
metadata builder runs on the host before every replay and records whether the
step's longest context is within the threshold.

QWEN38_QSA_DENSE_BELOW sets the threshold in tokens (default 49152). 0 turns
the switch off; the indexer's output width (2052) keeps QSA's result exactly,
since up to there it selects every token anyway.
"""

import os

THRESHOLD = int(os.environ.get("QWEN38_QSA_DENSE_BELOW", "49152"))

DENSE = False                 # the current step's regime, for graph replay
CAPTURING: bool | None = None  # forced regime while graphs are captured


def enabled() -> bool:
    return bool(THRESHOLD)


def observe(max_seq_len: int) -> None:
    """Record the regime of the step whose metadata is being built."""
    global DENSE
    if CAPTURING is None:
        DENSE = enabled() and max_seq_len <= THRESHOLD


def dense_step(max_seq_len: int) -> bool:
    """Whether the step being run (or captured) attends densely."""
    if CAPTURING is not None:
        return CAPTURING
    return enabled() and max_seq_len <= THRESHOLD


def patch_graph_managers() -> None:
    """Capture every FULL graph once per regime and replay by the step's regime."""
    from vllm.v1.worker.gpu import cudagraph_utils as cg

    manager = cg.CudaGraphManager
    if getattr(manager, "_qwen38_regime", False):
        return
    capture, run_fullgraph = manager.capture, manager.run_fullgraph
    extrapolate = cg._extrapolate_full_graph_memory

    def capture_both(self, *args, **kwargs):
        global CAPTURING
        if not enabled():
            return capture(self, *args, **kwargs)
        try:
            CAPTURING = False
            capture(self, *args, **kwargs)
            sparse, self.graphs = self.graphs, {}
            CAPTURING = True
            capture(self, *args, **kwargs)
            self._qwen38_graphs = {False: sparse, True: self.graphs}
        finally:
            CAPTURING = None

    def replay(self, desc):
        graphs = getattr(self, "_qwen38_graphs", None)
        if graphs is not None:
            self.graphs = graphs[DENSE]
        return run_fullgraph(self, desc)

    def extrapolate_both(mem_samples, total_graphs):
        # Profiling samples both passes; reserve memory for both sets.
        return extrapolate(mem_samples, total_graphs * (2 if enabled() else 1))

    manager.capture = capture_both
    manager.run_fullgraph = replay
    cg._extrapolate_full_graph_memory = extrapolate_both
    manager._qwen38_regime = True
