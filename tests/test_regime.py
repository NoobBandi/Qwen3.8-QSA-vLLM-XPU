"""Dense/sparse regime switch (qwen38_qsa.regime) on CPU, against a stand-in graph manager.

Checks that every graph is captured once per regime, that replay picks the graph
matching the regime the metadata builder last observed, and that graph memory
is reserved for both sets.

    python3 tests/test_regime.py
"""
import sys
import types

# A stand-in for vllm.v1.worker.gpu.cudagraph_utils: capture() records, per
# descriptor, the regime the model's attention saw; run_fullgraph() replays it.
cg = types.ModuleType("vllm.v1.worker.gpu.cudagraph_utils")


class CudaGraphManager:
    def __init__(self):
        self.graphs = {}

    def capture(self, forward):
        for desc in ("decode-1", "decode-4"):
            forward()                                  # warm-up, as vLLM does
            assert desc not in self.graphs, "graph captured twice into one set"
            self.graphs[desc] = forward()

    def run_fullgraph(self, desc):
        return self.graphs[desc]


cg.CudaGraphManager = CudaGraphManager
cg._extrapolate_full_graph_memory = lambda samples, total: total * 10
for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["vllm.v1.worker.gpu.cudagraph_utils"] = cg
sys.modules["vllm.v1.worker.gpu"].cudagraph_utils = cg

from qwen38_qsa import regime  # noqa: E402

failed = 0


def check(name, got, want):
    global failed
    ok = got == want
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})"))


regime.THRESHOLD = 49152
regime.patch_graph_managers()
regime.patch_graph_managers()                          # idempotent
manager = CudaGraphManager()
seen = []


def forward():
    # What the attention op does while a graph is captured: the regime is forced.
    dense = regime.dense_step(max_seq_len=262144)      # capture uses max_model_len
    seen.append(dense)
    return "dense" if dense else "sparse"


manager.capture(forward)
check("captured both regimes, warm-ups included", seen, [False] * 4 + [True] * 4)
check("capture flag cleared", regime.CAPTURING, None)

regime.observe(1500)
check("replay at 1.5K", manager.run_fullgraph("decode-4"), "dense")
regime.observe(49152)
check("replay at the threshold", manager.run_fullgraph("decode-1"), "dense")
regime.observe(49153)
check("replay just past it", manager.run_fullgraph("decode-4"), "sparse")
check("eager step at 30K", regime.dense_step(30000), True)
check("eager step at 200K", regime.dense_step(200000), False)
check("graph memory for both sets", cg._extrapolate_full_graph_memory([], 3), 60)

regime.THRESHOLD = 0                                   # switch off
check("off: eager 1K is sparse", regime.dense_step(1000), False)
off = CudaGraphManager()
seen.clear()
off.capture(forward)
check("off: one capture, sparse", seen, [False] * 4)
regime.observe(1000)
check("off: replay", off.run_fullgraph("decode-1"), "sparse")
check("off: graph memory for one set", cg._extrapolate_full_graph_memory([], 3), 30)

print("all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
