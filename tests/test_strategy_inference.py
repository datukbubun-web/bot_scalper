"""Decision layer + inference hot path."""

from __future__ import annotations

import numpy as np
import pytest

from pxscalper.strategy.decision import Intent, ScalpingStrategy, StrategyConfig


class FakePredictor:
    """Stands in for the ONNX predictor with a scripted probability vector."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)
        self.config = type("C", (), {"backend": "fake"})()
        self.last_latency_us = 12.0

    def predict(self, window):
        from pxscalper.inference.predictor import Prediction

        direction = int(np.argmax(self.probs)) - 1
        return Prediction(
            probs=self.probs, direction=direction, confidence=float(self.probs.max()),
            latency_us=12.0, model_version="fake", backend="fake",
        )


class FakeRL:
    def __init__(self, action):
        self.action = action

    def act(self, obs):
        return self.action


WINDOW = np.zeros((1, 31, 64), dtype=np.float32)


def test_long_signal_passes():
    strategy = ScalpingStrategy(FakePredictor([0.05, 0.15, 0.80]), config=StrategyConfig())
    decision = strategy.decide(WINDOW)
    assert decision.intent is Intent.LONG
    assert decision.size_multiplier > 0


def test_flat_when_uncertain():
    strategy = ScalpingStrategy(FakePredictor([0.34, 0.33, 0.33]), config=StrategyConfig())
    decision = strategy.decide(WINDOW)
    assert decision.intent is Intent.FLAT


def test_rl_veto_blocks_trade():
    strategy = ScalpingStrategy(
        FakePredictor([0.05, 0.15, 0.80]), FakeRL(-0.9), StrategyConfig(require_rl_agreement=True)
    )
    decision = strategy.decide(WINDOW, rl_observation=np.zeros(38, dtype=np.float32))
    assert decision.intent is Intent.FLAT
    assert "RL veto" in decision.reason


def test_rl_agreement_sizes_position():
    strategy = ScalpingStrategy(
        FakePredictor([0.02, 0.08, 0.90]), FakeRL(0.5), StrategyConfig()
    )
    decision = strategy.decide(WINDOW, rl_observation=np.zeros(38, dtype=np.float32))
    assert decision.intent is Intent.LONG
    assert 0 < decision.size_multiplier <= 1.0


def test_loss_cooldown_blocks():
    strategy = ScalpingStrategy(FakePredictor([0.05, 0.15, 0.80]), config=StrategyConfig())
    decision = strategy.decide(WINDOW, now_s=1000.0, last_loss_s=990.0)
    assert decision.intent is Intent.FLAT
    assert "cooldown" in decision.reason


def test_predictor_confidence_gate_returns_flat():
    from pxscalper.inference.predictor import Predictor, PredictorConfig

    class Dummy(Predictor):
        def __init__(self):  # noqa: D107 - bypass file loading
            self.config = PredictorConfig(confidence_threshold=0.9, n_outputs=3)
            self.calibration = __import__(
                "pxscalper.inference.predictor", fromlist=["Calibration"]
            ).Calibration()
            self._backend = "onnx"
            self._last_latency_us = 0.0

        def _run_onnx(self, x):
            return np.array([0.30, 0.40, 0.30], dtype=np.float32), 0.0

        def _run_torch(self, x):  # pragma: no cover
            raise NotImplementedError

    predictor = Dummy()
    prediction = predictor.predict(WINDOW)
    assert prediction.direction == 0  # 0.40 < 0.90 threshold
    assert prediction.flat


def test_calibration_temperature_softens_confidence():
    from pxscalper.inference.predictor import Calibration

    raw = np.array([0.01, 0.04, 0.95], dtype=np.float32)
    softened = Calibration(temperature=3.0).apply(raw)
    assert softened.max() < raw.max()
    assert abs(softened.sum() - 1.0) < 1e-6
    assert int(np.argmax(softened)) == int(np.argmax(raw))  # order preserved
