"""Event-driven backtesting with live-equivalent frictions."""

from .engine import BacktestConfig, Backtester, BacktestResult, Trade

__all__ = ["BacktestConfig", "BacktestResult", "Backtester", "Trade"]
