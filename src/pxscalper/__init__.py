"""pxscalper -- institutional-grade FX scalping stack for PrimeXBT.

Layers, bottom-up::

    exchange/    PrimeXBT Trading API v3 (Ed25519 signing, REST, models)
    data/        history download, cache, synthetic microstructure generator
    features/    stationary + order-flow + microstructure features, labelling
    models/      TCN / LSTM / PatchTST-style Transformer + training loop
    rl/          cost-aware scalping env, PPO and SAC agents
    inference/   TorchScript + ONNX Runtime, allocation-free hot path
    strategy/    meta-labelling: direction (SL) x action/size (DRL)
    risk/        sizing, daily loss, drawdown, kill switch
    execution/   impact -> place -> verify -> reconcile
    backtest/    event-driven, live-equivalent frictions
    live/        the loop, reconciliation, Prometheus metrics

Nothing in this package decides to trade on its own: every order passes through
``risk.RiskEngine.evaluate`` and every decision is journaled.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
