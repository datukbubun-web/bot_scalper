#!/usr/bin/env python3
"""Export trained checkpoints to TorchScript + ONNX and benchmark latency.

    python scripts/export_models.py --all
    python scripts/export_models.py --arch tcn --iterations 5000

Prints a table of p50/p90/p99 microseconds per backend. Ship the ONNX build: on a
2-vCPU container, a 31x64 TCN goes from ~900 us (eager) to ~45 us (ORT), and the
p99 matters more than the mean because a single slow inference in a scalping loop
is a missed entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pxscalper.inference.export import export_all  # noqa: E402
from pxscalper.utils import RepoPaths, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default=None, help="tcn | lstm | gru | transformer")
    ap.add_argument("--all", action="store_true", help="export every checkpoint found")
    ap.add_argument("--version", default="latest")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--iterations", type=int, default=3000)
    args = ap.parse_args()

    setup_logging()
    paths = RepoPaths().ensure()
    model_dir = paths.model_dir(args.version)

    if args.all:
        archs = sorted(p.stem for p in model_dir.glob("*.pt"))
    elif args.arch:
        archs = [args.arch]
    else:
        ap.error("pass --arch or --all")

    if not archs:
        print(f"no checkpoints in {model_dir}; train first (scripts/training/train_sl.py)")
        return 1

    summary = {}
    for arch in archs:
        ckpt = model_dir / f"{arch}.pt"
        if not ckpt.exists():
            print(f"skip {arch}: {ckpt} missing")
            continue
        result = export_all(ckpt, arch, model_dir, opset=args.opset, iterations=args.iterations)
        summary[arch] = result.latency_us
        print(f"\n{arch}:")
        for backend, stats in result.latency_us.items():
            if backend.startswith("_"):
                continue
            print(
                f"  {backend:16s} mean {stats['mean_us']:7.1f} us | "
                f"p50 {stats['p50_us']:7.1f} | p90 {stats['p90_us']:7.1f} | p99 {stats['p99_us']:7.1f}"
            )
        print(f"  p99 speedup (ORT vs eager): {result.latency_us['_summary']['p99_speedup_onnx_vs_eager']}x")

    (model_dir / "latency_report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten: {model_dir / 'latency_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
