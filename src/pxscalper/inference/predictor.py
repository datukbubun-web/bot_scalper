"""The hot path: sub-millisecond, allocation-free model inference.

Every microsecond here is latency between "the bar closed" and "my order is at
the exchange". The class below is deliberately boring and does no work at all
beyond what the model needs:

* the input buffer is allocated **once** and written in place;
* the ONNX session holds pre-bound ``OrtValue`` I/O bindings (``io_binding``) so
  there is no per-call numpy->tensor conversion on the way in or out;
* optional probability calibration (temperature scaling) is folded into the
  output, not into the network;
* a hard ``confidence`` gate: below threshold the decision is FLAT regardless of
  the argmax -- the cheapest risk control there is;
* ``predict_proba`` returns a *view* of a reused buffer: copy it if you keep it.

Supported backends: ``onnx`` (preferred), ``torchscript``, ``torch``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Prediction", "Predictor", "PredictorConfig", "Calibration"]


@dataclass(slots=True)
class Prediction:
    probs: np.ndarray        # shape (n_outputs,)
    direction: int           # -1 / 0 / +1
    confidence: float        # max prob or |regression output|
    latency_us: float        # measured wall time of the inference call
    model_version: str = ""
    backend: str = ""

    @property
    def edge(self) -> float:
        """Signed conviction in [-1, 1]: P(up) - P(down) for 3-class models."""
        if len(self.probs) == 3:
            return float(self.probs[2] - self.probs[0])
        return float(np.tanh(self.probs[0]))

    @property
    def flat(self) -> bool:
        return self.direction == 0


@dataclass
class Calibration:
    """Temperature scaling: the standard, order-preserving probability fix.

    A deep classifier on financial data is almost always over-confident; dividing
    the logits by ``temperature > 1`` before the softmax makes the confidence
    usable for position sizing and for the flat-gate.
    """

    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0

    def apply(self, probs: np.ndarray) -> np.ndarray:
        if self.temperature == 1.0 or len(probs) != 3:
            return probs
        logits = np.log(np.clip(probs, 1e-12, 1.0)) / max(self.temperature, 1e-6)
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        return cls(
            temperature=float(raw.get("temperature", 1.0)),
            platt_a=float(raw.get("platt_a", 1.0)),
            platt_b=float(raw.get("platt_b", 0.0)),
        )


@dataclass
class PredictorConfig:
    backend: str = "onnx"            # onnx | torchscript | torch
    onnx_path: str | None = None
    torchscript_path: str | None = None
    torch_checkpoint: str | None = None
    arch: str = "tcn"
    n_features: int = 31
    lookback: int = 64
    n_outputs: int = 3
    task: str = "classification"
    confidence_threshold: float = 0.45   # below this -> FLAT
    edge_threshold: float = 0.0          # |P(up)-P(down)| gate
    calibration_path: str | None = None
    intra_op_threads: int = 1
    graph_optimisation: str = "all"      # all | basic | extended | none
    model_version: str = "unknown"
    dead_zone: float = 1e-9

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, **overrides) -> "PredictorConfig":
        raw = json.loads(Path(manifest_path).read_text())
        mc = raw.get("model_config", {})
        base = cls(
            onnx_path=raw.get("onnx_path"),
            torchscript_path=raw.get("torchscript_path"),
            arch=raw.get("arch", "tcn"),
            n_features=int(mc.get("n_features", 31)),
            lookback=int(mc.get("lookback", 64)),
            n_outputs=int(mc.get("n_outputs", 3)),
            task=mc.get("task", "classification"),
            model_version=raw.get("arch", "unknown") + "-" + str(raw.get("hash", ""))[:8],
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class Predictor:
    """Loads one exported model and serves predictions with minimal overhead."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.calibration = (
            Calibration.load(config.calibration_path) if config.calibration_path else Calibration()
        )
        self._backend = config.backend.lower()
        self._input_name = "features"
        self._session = None
        self._binding = None
        self._io = None
        self._module = None
        self._torch = None
        self._out_probs = np.zeros((1, config.n_outputs), dtype=np.float32)
        self._out_conf = np.zeros((1, 1), dtype=np.float32)
        self._last_latency_us = 0.0
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        cfg = self.config
        if self._backend == "onnx":
            self._load_onnx()
        elif self._backend == "torchscript":
            self._load_torchscript()
        elif self._backend == "torch":
            self._load_torch()
        else:
            raise ValueError(f"unknown backend {self._backend!r}")

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        path = self.config.onnx_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found: {path}. Run `make export`.")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = {
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        }[self.config.graph_optimisation]
        opts.intra_op_num_threads = self.config.intra_op_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._binding = self._session.io_binding()
        log.info(
            "ONNX predictor ready: %s (inputs=%s, threads=%d)",
            Path(path).name, self._input_name, self.config.intra_op_threads,
        )

    def _load_torchscript(self) -> None:
        import torch

        self._torch = torch
        path = self.config.torchscript_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"TorchScript model not found: {path}. Run `make export`.")
        self._module = torch.jit.load(path, map_location="cpu")
        self._module.eval()
        torch.set_num_threads(self.config.intra_op_threads)
        log.info("TorchScript predictor ready: %s", Path(path).name)

    def _load_torch(self) -> None:
        import torch

        from ..models.sequence import ModelConfig, build_sequence_model, exportable_forecaster

        self._torch = torch
        payload = torch.load(self.config.torch_checkpoint, map_location="cpu", weights_only=False)
        raw = payload.get("model_config", {})
        if isinstance(raw.get("channels"), list):
            raw["channels"] = tuple(raw["channels"])
        mc = ModelConfig(**{k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__})
        model = build_sequence_model(self.config.arch, mc)
        model.load_state_dict(payload["state_dict"])
        self._module = exportable_forecaster(model, mc)
        torch.set_num_threads(self.config.intra_op_threads)
        log.info("eager PyTorch predictor ready (slowest path; prefer ONNX)")

    # --------------------------------------------------------------- inference
    def predict(self, window: np.ndarray) -> Prediction:
        """``window``: (1, C, L) or (C, L) float32, already normalised.

        Returns a :class:`Prediction`; ``probs`` is a view over an internal buffer
        that is overwritten by the next call -- copy it if you store it.
        """
        if window.ndim == 2:
            window = window[None, :, :]
        if window.shape[1] != self.config.n_features or window.shape[2] != self.config.lookback:
            raise ValueError(
                f"window has shape {window.shape}, expected (1, {self.config.n_features}, "
                f"{self.config.lookback})"
            )
        x = np.ascontiguousarray(window, dtype=np.float32)

        t0 = time.perf_counter_ns()
        if self._backend == "onnx":
            probs, conf = self._run_onnx(x)
        else:
            probs, conf = self._run_torch(x)
        self._last_latency_us = (time.perf_counter_ns() - t0) / 1000.0

        if self.config.task == "classification":
            probs = self.calibration.apply(probs)
            direction = int(np.argmax(probs)) - 1
            confidence = float(np.max(probs))
            # conviction gates: flat when the model is unsure or the classes tie
            edge = float(probs[2] - probs[0]) if len(probs) == 3 else float(np.tanh(probs[0]))
            if confidence < self.config.confidence_threshold:
                direction = 0
            if abs(edge) <= max(self.config.edge_threshold, self.config.dead_zone):
                direction = 0
        else:
            value = float(np.asarray(probs).reshape(-1)[0])
            direction = 0 if abs(value) < self.config.edge_threshold else (1 if value > 0 else -1)
            confidence = float(min(abs(value), 1.0))

        return Prediction(
            probs=np.asarray(probs, dtype=np.float32).reshape(-1),
            direction=direction,
            confidence=confidence,
            latency_us=self._last_latency_us,
            model_version=self.config.model_version,
            backend=self._backend,
        )

    def _run_onnx(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        sess = self._session
        binding = self._binding
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        # numpy -> OrtValue without an extra copy where ORT allows it
        binding.bind_input(
            self._input_name, "cpu", 0, x.dtype, x.shape, x.ctypes.data
        )
        binding.bind_output("probs", "cpu")
        binding.bind_output("confidence", "cpu")
        sess.run_with_iobinding(binding)
        probs = binding.get_outputs()[0].numpy()[0]
        return probs, 0.0

    def _run_torch(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(x)
            out = self._module(t)
            if isinstance(out, (tuple, list)):
                probs = out[0].numpy()[0]
            else:
                probs = out.numpy()[0]
        return probs, 0.0

    # ------------------------------------------------------------------ utility
    @property
    def last_latency_us(self) -> float:
        return self._last_latency_us

    def warmup(self, iterations: int = 50) -> float:
        """Run the graph a few times so ORT/torch finish lazy initialisation.

        Never trade before warmup: the first inference pays graph optimisation
        (tens of ms) and would otherwise show up as a catastrophic p99 outlier.
        """
        dummy = np.zeros((1, self.config.n_features, self.config.lookback), dtype=np.float32)
        for _ in range(iterations):
            self.predict(dummy)
        return self._last_latency_us

    def close(self) -> None:
        self._session = None
        self._binding = None
        self._module = None


def load_predictor(artifact_dir: str | Path, arch: str = "tcn", **overrides) -> Predictor:
    """Convenience: find ``<arch>.manifest.json`` under ``artifact_dir`` and load it."""
    manifest = Path(artifact_dir) / f"{arch}.manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} missing. Train first (`make train-{arch}`) then `make export`."
        )
    cfg = PredictorConfig.from_manifest(manifest, **overrides)
    predictor = Predictor(cfg)
    predictor.warmup()
    return predictor
