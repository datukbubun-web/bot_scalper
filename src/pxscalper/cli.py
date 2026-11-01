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
    oos_frac: float = typer.Option(
        0.0, "--oos-frac", help="evaluate only the last fraction of windows (e.g. 0.40)"
    ),
) -> None:
    """Walk-forward backtest using cached (or synthetic) data."""
    _bootstrap()
    from .ops.backtest_runner import run_backtest_from_config

    result = run_backtest_from_config(Path(config), synthetic=synthetic, oos_frac=oos_frac)
    console.print(result.summary())
    console.print(f"[dim]trades={len(result.trades)} metrics written to artifacts/[/dim]")


@app.command()
def orderbook(
    symbol: str = typer.Option("XAU/USD", help="symbol to probe"),
    ladder: str = typer.Option("0.01,0.1,1,10,50", help="comma-separated size ladder"),
) -> None:
    """Reconstruct PrimeXBT's internal L2 order book via /quotes + /impact VWAP inversion."""
    _bootstrap()
    from .exchange.orderbook import probe_synthetic_orderbook

    sizes = [float(x.strip()) for x in ladder.split(",") if x.strip()]
    snap = probe_synthetic_orderbook(_client(), symbol, size_ladder=sizes)
    if snap is None:
        console.print(f"[red]could not probe order book for {symbol}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold]{snap.symbol}[/bold] bid={snap.bid:,.4f} | ask={snap.ask:,.4f} | "
        f"spread={snap.spread_bps:.2f} bps | L2 imbalance={snap.l2_imbalance:+.3f} | "
        f"slopes (bid/ask)={snap.bid_slope_bps:.4f}/{snap.ask_slope_bps:.4f} bps/unit"
    )
    table = Table(title=f"Synthetic L2 Slices — {snap.symbol}", show_header=True, header_style="bold")
    for col in ("side", "cum_qty", "slice_qty", "vwap_px", "marginal_px", "offset_bps"):
        table.add_column(col, justify="right" if col != "side" else "left")
    for lvl in reversed(snap.asks):
        table.add_row(
            "[red]ASK[/red]",
            f"{lvl.cumulative_qty:g}",
            f"{lvl.slice_qty:g}",
            f"{lvl.vwap_price:,.4f}",
            f"{lvl.marginal_price:,.4f}",
            f"{lvl.offset_bps:+.2f}",
        )
    for lvl in snap.bids:
        table.add_row(
            "[green]BID[/green]",
            f"{lvl.cumulative_qty:g}",
            f"{lvl.slice_qty:g}",
            f"{lvl.vwap_price:,.4f}",
            f"{lvl.marginal_price:,.4f}",
            f"{lvl.offset_bps:+.2f}",
        )
    console.print(table)


@app.command()
def bias(
    symbol: str = typer.Option("XAU/USD", help="primary symbol"),
    bars: str = typer.Option("5M", help="scalper execution bar size (1M | 5M)"),
    htf: str = typer.Option("15M", help="intraday bias higher timeframe"),
    context_symbol: str = typer.Option("XAG/USD", help="cross-asset fair-value driver"),
) -> None:
    """Inspect causal Hybrid Intraday Bias + Scalper Execution state on cached data."""
    paths = _bootstrap()
    import pandas as pd

    from .data.pipeline import DataPipeline
    from .features.mtf import resample_ohlcv_causal
    from .strategy.intraday_bias import IntradayBiasConfig, IntradayBiasEngine

    pipe = DataPipeline(cache_dir=paths.data)
    bs_up = bars.upper()
    htf_up = htf.upper()
    ltf_df = pipe.load(symbol, bs_up)
    if ltf_df is None or ltf_df.empty:
        console.print(f"[red]no cached {symbol} {bs_up} data in {paths.data}[/red]")
        raise typer.Exit(1)

    ctx_df = pipe.load(context_symbol, bs_up)
    if ctx_df is None and bs_up != "1M":
        raw1 = pipe.load(context_symbol, "1M")
        if raw1 is not None and not raw1.empty:
            ctx_df = resample_ohlcv_causal(raw1, bs_up, source_bar_size="1M", drop_incomplete=True)

    htf_p = pipe.load(symbol, htf_up)
    htf_c = pipe.load(context_symbol, htf_up)
    if (htf_c is None or htf_c.empty) and context_symbol:
        src5 = pipe.load(context_symbol, "5M")
        if src5 is not None and not src5.empty:
            htf_c = resample_ohlcv_causal(src5, htf_up, source_bar_size="5M", drop_incomplete=True)

    ltf_tail = ltf_df.tail(2500).reset_index(drop=True)
    ctx_map = {context_symbol: ctx_df} if ctx_df is not None and not ctx_df.empty else None
    htf_bars_map = {htf_up: htf_p} if htf_p is not None and not htf_p.empty else None
    htf_ctx_map = (
        {htf_up: {context_symbol: htf_c}} if htf_c is not None and not htf_c.empty else None
    )
    if bs_up == "1M":
        p5 = pipe.load(symbol, "5M")
        c5 = pipe.load(context_symbol, "5M")
        if p5 is not None and not p5.empty:
            htf_bars_map = {**(htf_bars_map or {}), "5M": p5}
        if c5 is not None and not c5.empty:
            htf_ctx_map = {**(htf_ctx_map or {}), "5M": {context_symbol: c5}}

    eng = IntradayBiasEngine(IntradayBiasConfig(htf_bar_size=htf_up))
    snap = eng.evaluate_bars(
        ltf_bars=ltf_tail,
        ltf_bar_size=bs_up,
        context=ctx_map,
        htf_bars=htf_bars_map,
        htf_context=htf_ctx_map,
    )
    last_ts = pd.to_datetime(int(ltf_tail["time"].iloc[-1]), unit="ms", utc=True)
    long_ok, long_al, long_why = snap.evaluate_trade_alignment(1, eng.config)
    short_ok, short_al, short_why = snap.evaluate_trade_alignment(-1, eng.config)

    table = Table(
        title=f"Hybrid Intraday Bias + Scalper State — {symbol} ({htf_up} Bias -> {bs_up} Exec)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("metric")
    table.add_column("value")
    table.add_column("detail")
    table.add_row("last closed bar", str(last_ts), f"close={float(ltf_tail['close'].iloc[-1]):,.2f}")
    table.add_row("intraday direction", snap.direction.value, f"composite bias={snap.bias_score:+.3f}")
    table.add_row("market regime", snap.regime.value, f"roll_blackout={snap.roll_blackout}")
    table.add_row(f"{htf_up} fair-value score", f"{snap.htf_fv_score:+.3f}", f"driver={context_symbol}")
    table.add_row(f"{bs_up} scalper FV score", f"{snap.ltf_fv_score:+.3f}", "short-horizon dislocation")
    table.add_row("session VWAP trend", f"{snap.session_trend:+.3f}", f"vwap_z={snap.session_vwap_z:+.2f}")
    table.add_row("volatility regime", f"{snap.vol_regime_ratio:.2f}x", f"2H ER={snap.efficiency_ratio:.2f}")
    table.add_row(
        "LONG scalp gate",
        "[green]ALLOW[/green]" if long_ok else "[red]VETO[/red]",
        f"align={long_al:+.2f} | {long_why}",
    )
    table.add_row(
        "SHORT scalp gate",
        "[green]ALLOW[/green]" if short_ok else "[red]VETO[/red]",
        f"align={short_al:+.2f} | {short_why}",
    )
    console.print(table)


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


@app.command()
def edge(
    symbol: str = typer.Option("XAU/USD", help="symbol to screen"),
    bars: str = typer.Option("1M", help="bar size: 1M / 5M / 15M"),
    horizon: int = typer.Option(1, help="hold the signal this many bars"),
    cost_bps: float = typer.Option(0.6, help="round-trip cost per trade"),
    top: int = typer.Option(8),
    reality_check: bool = typer.Option(True, "--reality-check/--no-reality-check"),
) -> None:
    """Search for an exploitable edge -- and report honestly when there is none.

    Every candidate is pre-registered, the best one is compared against the
    distribution of the best-of-N under the null (so luck is not mistaken for
    alpha), and the result is quoted **net of cost**. A signal that is real but
    smaller than its own cost is a curiosity, not a strategy.
    """
    _bootstrap()
    import pandas as pd

    from .research.edge_screen import reality_check as rc_fn
    from .research.edge_screen import screen

    key = symbol.replace("/", "")
    path = Path(f"data/{key}_{bars}.parquet")
    if not path.exists():
        path = Path(f"data/{key}_{bars}.csv.gz")
    if not path.exists():
        console.print(f"[red]no cached data at data/{key}_{bars}[/red] -- run `pxs data` first")
        raise typer.Exit(2)

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    ts = pd.to_datetime(df["time"], unit="ms", utc=True)
    console.print(
        f"[bold]{symbol}[/bold] {bars}: {len(df):,} bars "
        f"({ts.min():%Y-%m-%d} -> {ts.max():%Y-%m-%d}), hold {horizon} bar(s), "
        f"cost {cost_bps} bps"
    )

    results = screen(df, horizon=horizon, cost_bps=cost_bps)
    table = Table(show_header=True, header_style="bold")
    table.add_column("candidate")
    table.add_column("n", justify="right")
    table.add_column("gross bps", justify="right")
    table.add_column("t", justify="right")
    table.add_column("hit", justify="right")
    table.add_column("net bps", justify="right")
    table.add_column("p (Bonf.)", justify="right")
    for r in results[:top]:
        table.add_row(
            r.name,
            f"{r.n}",
            f"{r.mean_bps:+.3f}",
            f"{r.t_stat:+.2f}",
            f"{r.hit_rate:.1%}",
            f"[{'green' if r.net_bps > 0 else 'red'}]{r.net_bps:+.3f}[/]",
            f"{r.p_bonferroni:.3f}",
        )
    console.print(table)

    if reality_check:
        rc = rc_fn(df, horizon=horizon, n_boot=200, block=120)
        console.print(
            f"reality check: best t={rc['observed_max_t']:.2f} vs null median "
            f"{rc['null_max_t_median']:.2f} / p95 {rc['null_max_t_p95']:.2f} -> "
            f"p={rc['p_value']:.3f}"
        )

    winners = [r for r in results if r.net_bps > 0]
    if not winners:
        best = results[0]
        console.print(
            f"[yellow]no candidate beats its cost. Best gross {best.mean_bps:+.3f} bps "
            f"({best.name}) is {cost_bps / max(best.mean_bps, 1e-9):.0f}x too small for a "
            f"{cost_bps:.1f} bps round trip.[/yellow]"
        )
        raise typer.Exit(1)
    console.print(f"[green]{len(winners)} candidate(s) clear the cost hurdle[/green]")




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
    from .ops.config_edit import TOP_LEVEL_DEFAULTS, describe, read_effective, read_top_level

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
    top = read_top_level(config)
    table.add_row(
        "close_all_on_profit",
        f"${top['close_all_on_profit']:g}",
        "close ALL when basket profit hits this"
        + ("" if top["close_all_enforce"] else " (report only!)"),
    )
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
    from .ops.config_edit import (
        TOP_LEVEL_DEFAULTS,
        ConfigEditError,
        read_risk,
        set_top_level,
        set_values,
        summarise,
    )

    if not Path(config).exists():
        console.print(f"[red]{config} not found[/red]")
        raise typer.Exit(2)

    updates: dict[str, str] = {}
    top_updates: dict[str, str] = {}
    for item in assignments:
        field, sep, value = item.partition("=")
        if not sep:
            console.print(f"[red]'{item}' is not FIELD=VALUE[/red]")
            raise typer.Exit(2)
        name = field.strip()
        if name in TOP_LEVEL_DEFAULTS:
            top_updates[name] = value.strip()
        else:
            updates[name] = value.strip()

    try:
        if dry_run:
            from .ops.config_edit import coerce, format_value

            current = {**read_risk(config), **read_top_level(config)}
            console.print("[yellow]dry run — nothing written[/yellow]")
            for k, v in updates.items():
                console.print(f"  {k}: {current.get(k)!r} -> {format_value(coerce(k, v))}")
            raise typer.Exit(0)
        changes = set_values(config, updates) if updates else []
        if top_updates:
            changes += set_top_level(config, top_updates)
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
