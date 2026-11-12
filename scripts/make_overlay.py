#!/usr/bin/env python3
"""Make a directory vLLM can load Qwen3.8-27B-QSA from, without copying weights.

The release's config.json names its own Transformers classes (auto_map,
model_type qwen3_5_qsa). vLLM should instead see a Qwen3.5 config whose
architecture is the one this plugin registers. The overlay holds that edited
config.json and symlinks every other file of the release.

    python3 scripts/make_overlay.py /models/Qwen3.8-27B-QSA-BF16
    # -> /models/Qwen3.8-27B-QSA-BF16/vllm-qsa, pass it to --model

With --dense it makes vllm-dense instead: plain Qwen3.5 with the indexers
ignored, the baseline bench/sweep.sh compares against.
"""
import argparse
import json
import os
import sys

ARCH = "QSAQwen3_5ForConditionalGeneration"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the downloaded Qwen3.8-27B-QSA-BF16 directory")
    ap.add_argument("--dense", action="store_true", help="plain Qwen3.5, no sparse attention")
    ap.add_argument("--out", help="overlay directory (default: <model>/vllm-qsa or vllm-dense)")
    args = ap.parse_args()

    model = os.path.abspath(args.model)
    out = os.path.abspath(args.out or os.path.join(model, "vllm-dense" if args.dense else "vllm-qsa"))
    config = json.load(open(os.path.join(model, "config.json")))
    if "qsa" not in config:
        sys.exit(f"{model}/config.json has no 'qsa' section; is this the QSA release?")

    config.pop("auto_map", None)
    config["model_type"] = "qwen3_5"
    config["architectures"] = [ARCH]
    if args.dense:
        config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        del config["qsa"]

    os.makedirs(out, exist_ok=True)
    for name in sorted(os.listdir(model)):
        src = os.path.join(model, name)
        dst = os.path.join(out, name)
        if name == "config.json" or not os.path.isfile(src):
            continue
        # Plain Qwen3.5 has no module for the indexer weights; vLLM would
        # refuse to load them, so dense leaves them out of the overlay and
        # out of its copy of the weight index.
        if args.dense and name == "qsa-indexers.safetensors":
            continue
        if args.dense and name == "model.safetensors.index.json":
            index = json.load(open(src))
            index["weight_map"] = {k: v for k, v in index["weight_map"].items()
                                   if v != "qsa-indexers.safetensors"}
            if os.path.lexists(dst):
                os.remove(dst)
            with open(dst, "w") as f:
                json.dump(index, f, indent=2)
            continue
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(src, out), dst)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
