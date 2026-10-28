"""Execution: impact-checked, idempotent order routing."""

from .router import ExecutionConfig, ExecutionResult, OrderRouter, make_client_order_id

__all__ = ["ExecutionConfig", "ExecutionResult", "OrderRouter", "make_client_order_id"]
