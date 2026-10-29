"""Command line interface.

    pxs doctor      verify credentials, clock skew and symbol metadata
    pxs data        download history to data/
    pxs smoke       full offline pipeline self-test (no keys, no network)
    pxs backtest    run the backtester on cached or synthetic data
    pxs trade       start the live/paper loop
    pxs reconcile   compare journal vs. exchange and fix drift
    pxs kill        engage the kill switch
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from . import __version__
from .utils import RepoPaths, load_dotenv, now_ms, setup_logging

app = typer.Typer(add_completion=False, help="pxscalper -- FX scalping stack for PrimeXBT")
console = Console()
log = logging.getLogger("pxscalper.cli")


def _bootstrap(verbose: bool = False) -> RepoPaths:
    load_dotenv(".env")
    setup_logging("DEBUG" if verbose else os.getenv("PX_LOG_LEVEL", "INFO"))
    return RepoPaths().ensure()


def _client(require_trade: bool = False):
    from .exchange.rest import PrimeXBTClient
    from .exchange.signing import ApiCredentials, SignatureError

    key = os.getenv("PX_API_KEY", "").strip()
    secret = os.getenv("PX_API_SECRET", "").strip()
    if not key or not secret:
        console.print("[red]PX_API_KEY / PX_API_SECRET not set.[/red] Copy .env.example to .env.")
        raise typer.Exit(2)
    try:
        creds = ApiCredentials(key, secret)
    except SignatureError as exc:
        console.print(f"[red]credential error:[/red] {exc}")
        raise typer.Exit(2) from exc
    return PrimeXBTClient(credentials=creds)


@app.command()
def version() -> None:
    """Print the version."""
    rprint(f"pxscalper {__version__}")


@app.command()
def doctor(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Verify credentials, clock, symbol metadata and installed extras."""
    _bootstrap(verbose)
    from .exchange.rest import ApiError

    table = Table(title="pxscalper doctor", show_lines=False)
    table.add_column("check")
    table.add_column("result")
    table.add_column("detail")

    # ---- package versions
    import numpy as np

    table.add_row("python", sys.version.split()[0], "")
    table.add_row("numpy", np.__version__, "")
    try:
        import torch

        table.add_row("torch", torch.__version__, "")
    except Exception:  # noqa: BLE001
        table.add_row("torch", "[yellow]missing[/yellow]", "pip install torch")
    try:
        import onnxruntime as ort

        table.add_row("onnxruntime", ort.__version__, ort.get_available_providers()[0])
    except Exception:  # noqa: BLE001
        table.add_row("onnxruntime", "[yellow]missing[/yellow]", "pip install onnxruntime")

    # ---- credentials
    key = os.getenv("PX_API_KEY", "")
    secret = os.getenv("PX_API_SECRET", "")
    table.add_row("PX_API_KEY", "set" if key else "[red]missing[/red]", key[:12] + "..." if key else "")
    if secret:
        table.add_row("PX_API_SECRET", "set", f"{len(secret)} chars")
    else:
        table.add_row(
            "PX_API_SECRET", "[red]missing[/red]",
            "the base64 PKCS#8 Ed25519 secret -- NOT the pxk_ key id",
        )

    client = None
    if key and secret:
        try:
            client = _client()
        except typer.Exit:
            client = None
    if client is not None:
        from .exchange.signing import ApiCredentials

        try:
            offset = client.sync_clock()
            table.add_row("clock offset", f"{offset:+.0f} ms",
                          "ok" if abs(offset) < 1000 else "clock drift -- sync NTP")
        except Exception as exc:  # noqa: BLE001
            table.add_row("clock", "[red]fail[/red]", str(exc))
        try:
            metrics = client.metrics()
            table.add_row(
                "account", metrics.account_id,
                f"{'LIVE' if metrics.real_money else 'DEMO'} | equity {metrics.equity:.2f} "
                f"{metrics.currency} | {metrics.position_mode}",
            )
            table.add_row("positions", str(metrics.open_positions), f"orders {metrics.open_orders}")
        except ApiError as exc:
            table.add_row("signed call", "[red]401/403[/red]", f"{exc.code}: {exc.description}")
        except Exception as exc:  # noqa: BLE001
            table.add_row("signed call", "[red]fail[/red]", str(exc)[:80])
        try:
            symbols = [s.strip() for s in os.getenv("PX_SYMBOLS", "EUR/USD").split(",") if s.strip()]
            for s in client.settings(symbols=symbols):
                table.add_row(
                    f"symbol {s.symbol}", "ok",
                    f"minQty {s.min_qty} step {s.qty_step} taker {s.taker_fee_pct}% "
                    f"maxLev {s.max_leverage} {s.trading_hours or ''}",
                )
        except Exception as exc:  # noqa: BLE001
            table.add_row("symbols", "[yellow]unavailable[/yellow]", str(exc)[:80])

    console.print(table)
    rprint("\n[dim]bars are public: `curl https://api.primexbt.com/v3/trading/bars?symbol=EUR%2FUSD&barSize=1M&limit=5`[/dim]")


@app.command()
def data(
    symbols: str = typer.Option(None, help="comma separated, default PX_SYMBOLS"),
    bars: str = typer.Option("1M"),
    limit: int = typer.Option(5000),
    update: bool = typer.Option(False, help="incremental update of the cache"),
) -> None:
    """Download historical bars into data/ (public endpoint, no key needed)."""
    paths = _bootstrap()
    from .data.pipeline import DataPipeline
    from .exchange.rest import PrimeXBTClient

    client = PrimeXBTClient()  # bars are unsigned
    pipeline = DataPipeline(cache_dir=paths.data, client=client)
    symbol_list = [s.strip() for s in (symbols or os.getenv("PX_SYMBOLS", "EUR/USD")).split(",") if s.strip()]
    table = Table(title=f"download {bars}")
    table.add_column("symbol")
    table.add_column("bars", justify="right")
    table.add_column("first")
    table.add_column("last")
    import pandas as pd

    for symbol in symbol_list:
        try:
            frame = (
                pipeline.update(symbol, bars, limit)
                if update
                else pipeline.download(symbol, bars, limit)
            )
            table.add_row(
                symbol, str(len(frame)),
                str(pd.to_datetime(frame["time"].iloc[0], unit="ms", utc=True)),
                str(pd.to_datetime(frame["time"].iloc[-1], unit="ms", utc=True)),
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row(symbol, "[red]fail[/red]", str(exc)[:60], "")
    console.print(table)


@app.command()
def smoke(
    arch: str = typer.Option("tcn", help="tcn | lstm | transformer"),
    steps: int = typer.Option(30, help="training epochs for the tiny demo"),
    rl_steps: int = typer.Option(4000, help="PPO environment steps"),
) -> None:
    """End-to-end self-test with synthetic data: features -> model -> ONNX -> RL -> backtest."""
    from .ops.smoke import run_smoke

    paths = _bootstrap()
    report = run_smoke(paths, arch=arch, epochs=steps, rl_steps=rl_steps)
    console.print_json(json.dumps(report, indent=2))


@app.command()
def backtest(
    config: str = typer.Option("configs/live.example.yaml"),
    synthetic: bool = typer.Option(False, help="use synthetic data instead of cache"),
) -> None:
    """Walk-forward backtest using cached (or synthetic) data."""
    _bootstrap()
    from .ops.backtest_runner import run_backtest_from_config

    result = run_backtest_from_config(Path(config), synthetic=synthetic)
    console.print(result.summary())
    console.print(f"[dim]trades={len(result.trades)} metrics written to artifacts/[/dim]")


@app.command()
def trade(
    config: str = typer.Option("configs/live.example.yaml"),
    i_understand_the_risk: bool = typer.Option(
        False, "--i-understand-the-risk", help="required to send real orders"
    ),
) -> None:
    """Start the live (or paper) scalping loop."""
    paths = _bootstrap()
    from .ops.live_runner import start_from_config

    allow_live = (
        os.getenv("PX_LIVE_TRADING", "0") == "1" and i_understand_the_risk
    )
    if os.getenv("PX_LIVE_TRADING", "0") == "1" and not i_understand_the_risk:
        console.print(
            "[yellow]PX_LIVE_TRADING=1 but --i-understand-the-risk was not passed -> paper mode[/yellow]"
        )
    start_from_config(Path(config), allow_live=allow_live, paths=paths)


@app.command()
def catalogue(
    fx_only: bool = typer.Option(True, help="only show FX-like instruments"),
    max_spread: float = typer.Option(3.0, help="max spread in bps"),
    limit: int = typer.Option(25),
    equity: float = typer.Option(10_000.0, help="account equity used for feasibility"),
    leverage: float = typer.Option(5.0, help="position cap used for feasibility"),
) -> None:
    """What the account can actually trade -- spread, size step, feasibility.

    Run this BEFORE choosing symbols. PrimeXBT FX is sized in whole base-currency
    units (EUR/USD: 1000 EUR per step, USD/JPY: 1000 JPY per step ≈ $157k of
    notional), so a small account usually cannot trade every pair.
    """
    _bootstrap()
    from .exchange.catalogue import SymbolResolver, resolve_units

    resolver = SymbolResolver(_client())
    resolver.refresh()
    pool = resolver.fx_instruments() if fx_only else list(resolver._instruments.values())
    rows = [i for i in pool if i.market_open and i.spread_bps <= max_spread]

    table = Table(title=f"kandidat trading (spread <= {max_spread} bps, open)")
    for col in ("symbol", "class", "price", "spread", "step", "$/step", "minRisk$", "tradable"):
        table.add_column(col, justify="right" if col not in ("symbol", "class", "tradable") else "left")

    room = equity * leverage
    for instrument in sorted(rows, key=lambda i: i.spread_bps)[:limit]:
        step_notional = resolver.notional_per_step(instrument)
        qty, notional, _note = resolve_units(instrument, room)
        feasible = qty > 0
        table.add_row(
            instrument.symbol, instrument.category, f"{instrument.price:,.5f}",
            f"{instrument.spread_bps:.2f}", f"{instrument.qty_step:g}",
            f"${step_notional:,.0f}" if step_notional else "n/a",
            f"${step_notional / leverage:,.0f}" if step_notional else "n/a",
            "[green]yes[/green]" if feasible else "[red]too coarse[/red]",
        )
    console.print(table)
    console.print(
        f"[dim]equity ${equity:,.0f} x {leverage:g} = ${room:,.0f} notional room. "
        "'too coarse' = one qtyStep already exceeds that room. "
        "minRisk$ = margin needed for a single step at this leverage.[/dim]"
    )


risk_app = typer.Typer(add_completion=False, help="Inspect and tune the risk guards.")
app.add_typer(risk_app, name="risk")


@risk_app.command("show")
def risk_show(
    config: str = typer.Option("configs/live.mini100.yaml", "--config", "-c"),
    env: bool = typer.Option(True, help="include PX_RISK_* environment overrides"),
) -> None:
    """Show the guards that decide whether the bot may trade.

    Read this before touching a number: the four loss limits do different
    things, and confusing them is how a small account dies.
    """
    _bootstrap()
    from .ops.config_edit import describe, read_effective

    if not Path(config).exists():
        console.print(f"[red]{config} not found[/red]")
        raise typer.Exit(2)
    cfg = read_effective(config, allow_env=env)
    table = Table(title=f"risk guards — {config}", show_header=True, header_style="bold")
    table.add_column("setting")
    table.add_column("value", justify="right")
    table.add_column("what it does")
    for name, value, meaning in describe(cfg):
        table.add_row(name, value, meaning)
    console.print(table)
    if cfg.max_intraday_loss_pct and cfg.risk_per_trade_pct:
        stops = cfg.max_intraday_loss_pct / cfg.risk_per_trade_pct
        style = "green" if stops >= 3 else "red"
        console.print(
            f"[{style}]intraday brake = {stops:.1f} stops[/{style}]"
            + ("" if stops >= 3 else "  <- less than 3 stops: one bad bar halts trading")
        )


@risk_app.command("set")
def risk_set(
    assignments: list[str] = typer.Argument(
        ..., help="FIELD=VALUE, e.g. max_intraday_loss_pct=2.5"
    ),
    config: str = typer.Option("configs/live.mini100.yaml", "--config", "-c"),
    dry_run: bool = typer.Option(False, help="print the result, do not write"),
) -> None:
    """Change a guard in the YAML, keeping every comment in the file.

        pxs risk set max_intraday_loss_pct=2.5
        pxs risk set max_total_loss_pct=10 max_daily_trades=6
    """
    _bootstrap()
    from .ops.config_edit import ConfigEditError, read_risk, set_values, summarise

    if not Path(config).exists():
        console.print(f"[red]{config} not found[/red]")
        raise typer.Exit(2)

    updates: dict[str, str] = {}
    for item in assignments:
        field, sep, value = item.partition("=")
        if not sep:
            console.print(f"[red]'{item}' is not FIELD=VALUE[/red]")
            raise typer.Exit(2)
        updates[field.strip()] = value.strip()

    try:
        if dry_run:
            from .ops.config_edit import coerce, format_value

            current = read_risk(config)
            console.print("[yellow]dry run — nothing written[/yellow]")
            for k, v in updates.items():
                console.print(f"  {k}: {current.get(k)!r} -> {format_value(coerce(k, v))}")
            raise typer.Exit(0)
        changes = set_values(config, updates)
    except ConfigEditError as exc:
        console.print(f"[red]refused:[/red] {exc}")
        raise typer.Exit(2) from exc

    console.print(f"[green]updated {config}[/green]\n{summarise(changes)}")
    cfg = read_risk(config)
    if "max_intraday_loss_pct" in updates and float(cfg.get("risk_per_trade_pct", 0) or 0) > 0:
        stops = float(cfg["max_intraday_loss_pct"]) / float(cfg["risk_per_trade_pct"])
        if stops < 3:
            console.print(
                f"[yellow]warning: the intraday brake is now {stops:.1f} stops — "
                f"a single normal loss can halt trading for the day[/yellow]"
            )
    console.print("check it with: pxs risk show --config " + config)


@app.command()
def reconcile() -> None:
    """Compare the local journal with the exchange and fix drift."""
    _bootstrap()
    from .live.reconcile import Reconciler

    report = Reconciler(_client()).run()
    console.print_json(json.dumps(report.to_dict(), indent=2))


@app.command()
def kill(reason: str = typer.Option("cli", help="reason recorded in the switch file")) -> None:
    """Engage the kill switch (halts the loop and flattens)."""
    boot = _bootstrap()
    from .risk.engine import KillSwitch

    KillSwitch(Path("state/KILL")).engage(reason)
    rprint("[red]kill switch engaged[/red] -- the running loop will flatten and stop.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
