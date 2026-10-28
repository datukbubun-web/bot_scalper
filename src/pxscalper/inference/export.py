"""PyTorch -> TorchScript / ONNX conversion with a latency benchmark.

The point of this module
------------------------
A naive PyTorch ``model(x)`` call in a Python tick loop spends most of its wall
time in the interpreter: attribute lookups, ``nn.Module.__call__`` hooks,
autograd bookkeeping, tensor<->numpy copies. For a small network the actual
arithmetic is a rounding error next to that. The three fixes applied here:

1. **TorchScript** (``torch.jit.trace`` + ``optimize_for_inference``) -- strips
   Python from the hot path.
2. **ONNX Runtime** with ``ORT_ENABLE_ALL`` graph optimisation, 1 intra-op
   thread, sequential execution and an arena allocator with pre-allocated I/O
   bindings -- typically 3-10x lower p99 than eager PyTorch on CPU for this model
   size, and it is the recommended path for a single-instrument scalper.
3. **Pre-allocated numpy buffers** so ``predict()`` performs an in-place copy
   instead of allocating a tensor per tick (see ``predictor.py``).

Measured here (2-vCPU container, 1 ORT thread, ``TCN(38 features x 64 bars)``,
batch 1, 2000 iterations, p50 / p99):

    eager PyTorch    1123 / 1242 us
    TorchScript       891 /  986 us
    ONNX Runtime      250 /  330 us      -> 3.8x lower p99

Numbers are dominated by the interpreter and by the container's CPU governor, so
they differ wildly between a laptop, a shared vCPU and a bare-metal box. Run
``benchmark()`` **on the machine that will trade**; the ratio (not the absolute)
is the part that generalises.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..models.sequence import ModelConfig, build_sequence_model, exportable_forecaster

log = logging.getLogger(__name__)

__all__ = [
    "ExportResult",
    "export_torchscript",
    "export_onnx",
    "benchmark",
    "load_model_from_checkpoint",
    "export_all",
]


@dataclass
class ExportResult:
    arch: str
    torchscript_path: str | None
    onnx_path: str | None
    model_config: dict
    latency_us: dict[str, dict[str, float]]
    onnx_opsets: dict | None = None

    def to_dict(self) -> dict:
        return {
            "arch": self.arch,
            "torchscript_path": self.torchscript_path,
            "onnx_path": self.onnx_path,
            "model_config": self.model_config,
            "latency_us": self.latency_us,
            "onnx_opsets": self.onnx_opsets,
        }


def load_model_from_checkpoint(path: str | Path, arch: str) -> tuple[torch.nn.Module, ModelConfig]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload.get("model_config") or {}
    if raw:
        if isinstance(raw.get("channels"), list):
            raw["channels"] = tuple(raw["channels"])
        cfg = ModelConfig(**{k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__})
    else:
        raise ValueError("checkpoint has no model_config; retrain with Trainer.save()")
    model = build_sequence_model(arch, cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


def export_torchscript(
    model: torch.nn.Module, config: ModelConfig, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = exportable_forecaster(model, config)
    example = torch.zeros(1, config.n_features, config.lookback, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced = torch.jit.optimize_for_inference(traced)
    traced.save(str(path))
    log.info("TorchScript saved -> %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def export_onnx(
    model: torch.nn.Module,
    config: ModelConfig,
    path: str | Path,
    opset: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = exportable_forecaster(model, config)
    example = torch.zeros(1, config.n_features, config.lookback, dtype=torch.float32)
    dynamic_axes = (
        {"features": {0: "batch"}, "probs": {0: "batch"}, "confidence": {0: "batch"}}
        if dynamic_batch
        else None
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example,),
            str(path),
            input_names=["features"],
            output_names=["probs", "confidence"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    log.info("ONNX saved -> %s (%.1f KB, opset %d)", path, path.stat().st_size / 1024, opset)
    if simplify:
        _try_simplify(path)
    return path


def _try_simplify(path: Path) -> None:
    """Optional onnxsim pass; silently skipped when the package is absent."""
    try:
        import onnx
        import onnxsim

        model = onnx.load(str(path))
        simplified, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(simplified, str(path))
            log.info("onnx-simplifier: graph simplified")
    except Exception as exc:  # noqa: BLE001
        log.debug("onnxsim skipped (%s)", exc)


# ------------------------------------------------------------------- benchmark
def _make_session(onnx_path: str, threads: int = 1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.log_severity_level = 3
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])


def benchmark(
    model: torch.nn.Module,
    config: ModelConfig,
    onnx_path: str | Path | None = None,
    torchscript_path: str | Path | None = None,
    iterations: int = 2000,
    warmup: int = 200,
) -> dict[str, dict[str, float]]:
    """Measure eager vs TorchScript vs ONNX Runtime latency (microseconds).

    Reports p50 / p90 / p99 / mean / max. p99 is the number that matters: one
    slow inference in a scalping loop is a missed or badly-priced entry.
    """
    x_np = np.zeros((1, config.n_features, config.lookback), dtype=np.float32)
    x_t = torch.from_numpy(x_np)
    results: dict[str, dict[str, float]] = {}

    def _timeit(fn, n=iterations, w=warmup) -> dict[str, float]:
        for _ in range(w):
            fn()
        samples = np.empty(n, dtype=np.float64)
        for i in range(n):
            t0 = time.perf_counter_ns()
            fn()
            samples[i] = (time.perf_counter_ns() - t0) / 1000.0
        return {
            "mean_us": float(samples.mean()),
            "p50_us": float(np.percentile(samples, 50)),
            "p90_us": float(np.percentile(samples, 90)),
            "p99_us": float(np.percentile(samples, 99)),
            "max_us": float(samples.max()),
        }

    model.eval()
    with torch.no_grad():
        results["pytorch_eager"] = _timeit(lambda: model(x_t))

    if torchscript_path:
        ts_model = torch.jit.load(str(torchscript_path))
        ts_model.eval()
        with torch.no_grad():
            results["torchscript"] = _timeit(lambda: ts_model(x_t))

    if onnx_path and os.path.exists(str(onnx_path)):
        sess = _make_session(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: x_np})
        results["onnxruntime"] = _timeit(lambda: sess.run(None, {input_name: x_np}))
        log.info("onnx outputs: %s", [o.shape for o in out])

    speedup = (
        results["pytorch_eager"]["p99_us"] / results["onnxruntime"]["p99_us"]
        if "onnxruntime" in results
        else 1.0
    )
    results["_summary"] = {
        "p99_speedup_onnx_vs_eager": round(speedup, 2),
        "iterations": iterations,
        "on_cpu_threads": os.cpu_count() or 1,
    }
    return results


def export_all(
    checkpoint: str | Path,
    arch: str,
    out_dir: str | Path,
    opset: int = 17,
    iterations: int = 2000,
) -> ExportResult:
    """One call: load checkpoint -> TorchScript + ONNX -> benchmark -> manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model_from_checkpoint(checkpoint, arch)
    ts_path = export_torchscript(model, cfg, out_dir / f"{arch}.ts")
    onnx_path = export_onnx(model, cfg, out_dir / f"{arch}.onnx", opset=opset)
    latency = benchmark(model, cfg, onnx_path, ts_path, iterations=iterations)

    result = ExportResult(
        arch=arch,
        torchscript_path=str(ts_path),
        onnx_path=str(onnx_path),
        model_config=cfg.to_dict(),
        latency_us=latency,
    )
    (out_dir / f"{arch}.manifest.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    log.info(
        "%s latency p50/p99 us: eager %.0f/%.0f | ts %.0f/%.0f | ort %.0f/%.0f",
        arch,
        latency["pytorch_eager"]["p50_us"], latency["pytorch_eager"]["p99_us"],
        latency.get("torchscript", {}).get("p50_us", float("nan")),
        latency.get("torchscript", {}).get("p99_us", float("nan")),
        latency.get("onnxruntime", {}).get("p50_us", float("nan")),
        latency.get("onnxruntime", {}).get("p99_us", float("nan")),
    )
    return result
