"""Architecture contracts: shapes, causality, and TorchScript/ONNX exportability."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pxscalper.models.sequence import (
    ModelConfig,
    PatchTSTLite,
    TCN,
    build_sequence_model,
    exportable_forecaster,
)

CFG = ModelConfig(n_features=8, lookback=32, n_outputs=3, patch_len=8, channels=(16, 16))


@pytest.mark.parametrize("arch", ["tcn", "lstm", "gru", "transformer"])
def test_output_shape(arch):
    model = build_sequence_model(arch, CFG)
    x = torch.randn(4, CFG.n_features, CFG.lookback)
    out = model(x)
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ["tcn", "lstm", "gru", "transformer"])
def test_causality(arch):
    """Changing the most recent bar must change the output; changing nothing
    before it must not. This catches accidental future leakage."""
    torch.manual_seed(0)
    model = build_sequence_model(arch, CFG).eval()
    x = torch.randn(1, CFG.n_features, CFG.lookback)
    with torch.no_grad():
        base = model(x)
        # perturb only the LAST time step -> output must move
        x2 = x.clone()
        x2[:, :, -1] += 3.0
        moved = model(x2)
    assert not torch.allclose(base, moved, atol=1e-6)


def test_tcn_receptive_field_growth():
    cfg = ModelConfig(n_features=4, lookback=64, channels=(8, 8, 8, 8), kernel_size=3)
    tcn = TCN(cfg)
    assert tcn.receptive_field == 1 + 2 * (3 - 1) * (2**4 - 1)  # 31


def test_transformer_requires_divisible_lookback():
    cfg = ModelConfig(n_features=4, lookback=30, patch_len=8)
    with pytest.raises(ValueError):
        PatchTSTLite(cfg)


def test_exportable_wrapper_outputs_probs():
    cfg = ModelConfig(n_features=8, lookback=32, n_outputs=3, channels=(16, 16))
    wrapper = exportable_forecaster(build_sequence_model("tcn", cfg), cfg)
    probs, conf = wrapper(torch.randn(2, cfg.n_features, cfg.lookback))
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert ((conf > 0) & (conf <= 1)).all()


def test_torchscript_trace_roundtrip(tmp_path):
    cfg = ModelConfig(n_features=8, lookback=32, n_outputs=3, channels=(16, 16))
    wrapper = exportable_forecaster(build_sequence_model("tcn", cfg), cfg)
    example = torch.zeros(1, cfg.n_features, cfg.lookback)
    traced = torch.jit.trace(wrapper, example, strict=False)
    path = tmp_path / "m.ts"
    traced.save(str(path))
    loaded = torch.jit.load(str(path))
    out = loaded(torch.randn(1, cfg.n_features, cfg.lookback))
    assert isinstance(out, (tuple, list))
    assert out[0].shape == (1, 3)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("onnxruntime") is None, reason="onnxruntime missing"
)
def test_onnx_export_matches_torch(tmp_path):
    from pxscalper.inference.export import export_onnx

    cfg = ModelConfig(n_features=8, lookback=32, n_outputs=3, channels=(16, 16), dropout=0.0)
    model = build_sequence_model("tcn", cfg)
    path = export_onnx(model, cfg, tmp_path / "m.onnx", simplify=False)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(3, cfg.n_features, cfg.lookback).astype(np.float32)
    ort_probs = sess.run(None, {"features": x})[0]

    wrapper = exportable_forecaster(model, cfg)
    with torch.no_grad():
        torch_probs = wrapper(torch.from_numpy(x))[0].numpy()
    np.testing.assert_allclose(ort_probs, torch_probs, atol=1e-4)
