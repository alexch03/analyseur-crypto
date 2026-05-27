"""Chart renderer: generates annotated candlestick PNG images with SMC overlays.

Produces TradingView-style dark theme charts showing:
- Candlesticks (OHLCV)
- Swing points (HH, HL, LH, LL markers)
- Support/resistance levels (horizontal lines)
- BOS / CHOCH annotations with connecting lines
- FVG zones (shaded rectangles)
- Order Block zones (shaded rectangles)
- Trade setup overlays (entry, SL, TP lines with labels)

Par défaut, **zoom** sur les dernières bougies (``focus_last_bars``) et **compact_overlays**
pour limiter le texte et ne dessiner que les structures dans cette fenêtre — meilleure
lisibilité quand l’historique est long.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from app.schemas.domain import (
    FVGType,
    FairValueGap,
    MarketContextDTO,
    OBType,
    OrderBlock,
    SRLevel,
    SRRole,
    Side,
    StructureEvent,
    StructureEventType,
    SwingKind,
    SwingPoint,
    TradeSetupDTO,
    Trend,
)

matplotlib.use("Agg")

DARK_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume="in",
        ohlc="i",
    ),
    figcolor="#131722",
    facecolor="#131722",
    gridcolor="#1e222d",
    gridstyle="--",
    gridaxis="both",
    y_on_right=True,
    rc={
        "axes.labelcolor": "#787b86",
        "xtick.color": "#787b86",
        "ytick.color": "#787b86",
        "font.size": 9,
    },
)


def render_chart(
    ctx: MarketContextDTO,
    setups: list[TradeSetupDTO] | None = None,
    *,
    width: int = 1200,
    height: int = 700,
    title: str | None = None,
    output_path: str | Path | None = None,
    focus_last_bars: int | None = 120,
    compact_overlays: bool = True,
) -> bytes:
    """Render a full SMC-annotated chart and return PNG bytes.

    Parameters
    ----------
    focus_last_bars:
        Si défini (>0), l'axe X est limité aux dernières bougies (zoom) et l'axe Y
        est calé sur les plus bas / plus hauts de cette zone (+ niveaux des setups).
        ``None`` = vue pleine largeur (comportement historique).
    compact_overlays:
        Réduit les étiquettes texte (FVG/OB, swings HH/LL, S/R) et ne trace les
        zones / swings / événements de structure que s'ils intersectent la fenêtre
        visible (quand ``focus_last_bars`` est utilisé).

    If output_path is provided, also saves to disk.
    """
    df = _prepare_df(ctx.ohlcv)
    n = len(df)
    setups = setups or []
    x0 = 0
    if focus_last_bars is not None and focus_last_bars > 0 and n > 0:
        x0 = max(0, n - min(int(focus_last_bars), n))
    # Sans zoom horizontal, le mode compact limite quand même FVG/OB/swings à la fin du graph.
    vis_x0 = x0
    if focus_last_bars is None and compact_overlays and n > 0:
        vis_x0 = max(0, n - min(220, n))

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=DARK_STYLE,
        volume=True,
        returnfig=True,
        figsize=(width / 100, height / 100),
        tight_layout=True,
        warn_too_much_data=9999,
    )

    ax_main = axes[0]

    if title:
        ax_main.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    else:
        ax_main.set_title(
            f"{ctx.symbol} | {ctx.timeframe}",
            color="white", fontsize=13, fontweight="bold", pad=12,
        )

    _draw_swings(ax_main, ctx.swings, df, x0=vis_x0, compact=compact_overlays)
    _draw_sr_levels(ax_main, ctx.sr_levels, n, df=df, x0=vis_x0, compact=compact_overlays)
    _draw_structure_events(ax_main, ctx.structure_events, df, x0=vis_x0, compact=compact_overlays)
    _draw_fvgs(ax_main, ctx.fvgs, n, x0=vis_x0, compact=compact_overlays)
    _draw_order_blocks(ax_main, ctx.order_blocks, n, x0=vis_x0, compact=compact_overlays)

    if setups:
        _draw_setups(ax_main, setups, n)

    _apply_price_zoom(ax_main, axes, df, setups, x0=x0, focus_last_bars=focus_last_bars, n=n)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    if output_path:
        Path(output_path).write_bytes(png_bytes)

    return png_bytes


def _prepare_df(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    if "timestamp" in df.columns:
        df.index = pd.DatetimeIndex(df["timestamp"])
    required = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    rename = {}
    for target, source in required.items():
        if source in df.columns and target not in df.columns:
            rename[source] = target
    if rename:
        df = df.rename(columns=rename)
    return df


def _apply_price_zoom(
    ax_main,
    axes,
    df: pd.DataFrame,
    setups: list[TradeSetupDTO],
    *,
    x0: int,
    focus_last_bars: int | None,
    n: int,
) -> None:
    """Zoom horizontal (dernières bougies) + vertical (prix utiles + setups)."""
    if n <= 0 or focus_last_bars is None or focus_last_bars <= 0:
        return
    ax_main.set_xlim(x0 - 0.5, n + 4)
    seg = df.iloc[x0:]
    if len(seg) == 0:
        return
    ymin = float(seg["Low"].min())
    ymax = float(seg["High"].max())
    for s in setups:
        prices = [float(s.entry), float(s.stop_loss), *map(float, s.take_profits)]
        ymin = min(ymin, *prices)
        ymax = max(ymax, *prices)
    span = max(ymax - ymin, ymax * 1e-6)
    pad = max(span * 0.06, ymax * 0.002)
    ax_main.set_ylim(ymin - pad, ymax + pad)
    if len(axes) > 1:
        axes[1].set_xlim(ax_main.get_xlim())


def _draw_swings(
    ax,
    swings: list[SwingPoint],
    df: pd.DataFrame,
    *,
    x0: int = 0,
    compact: bool = False,
) -> None:
    for sw in swings:
        if sw.index >= len(df):
            continue
        if compact and sw.index < x0 - 1:
            continue
        color = "#26a69a" if sw.kind == SwingKind.HIGH else "#ef5350"
        marker = "v" if sw.kind == SwingKind.HIGH else "^"
        y_offset = sw.price * 0.002 if sw.kind == SwingKind.HIGH else -sw.price * 0.002
        ms = 5 if compact else 6
        ax.plot(sw.index, sw.price, marker=marker, color=color, markersize=ms, zorder=5)
        if compact:
            continue
        label = "HH" if sw.kind == SwingKind.HIGH else "LL"
        ax.annotate(
            label, (sw.index, sw.price + y_offset),
            fontsize=7, color=color, ha="center", va="bottom" if sw.kind == SwingKind.HIGH else "top",
            fontweight="bold",
        )


def _draw_sr_levels(
    ax,
    levels: list[SRLevel],
    n_bars: int,
    *,
    df: pd.DataFrame | None = None,
    x0: int = 0,
    compact: bool = False,
) -> None:
    p_min, p_max = float("-inf"), float("inf")
    if df is not None and len(df) > x0:
        seg = df.iloc[x0:]
        y_lo = float(seg["Low"].min())
        y_hi = float(seg["High"].max())
        band = max((y_hi - y_lo) * 0.18, y_hi * 0.004)
        p_min, p_max = y_lo - band, y_hi + band
    last_close = float(df.iloc[-1]["Close"]) if df is not None and len(df) else 0.0
    filtered = [lv for lv in levels if p_min <= lv.price <= p_max]
    filtered.sort(key=lambda lv: (-lv.touches, abs(lv.price - last_close)))
    cap = 6 if compact else 22
    for lv in filtered[:cap]:
        color = "#2196f3" if lv.role == SRRole.SUPPORT else "#ff9800"
        ax.axhline(y=lv.price, color=color, linestyle="--", linewidth=0.8, alpha=0.55, zorder=2)
        if compact:
            continue
        ax.annotate(
            f"{'S' if lv.role == SRRole.SUPPORT else 'R'} {lv.price:.2f} ({lv.touches}t)",
            (n_bars - 1, lv.price), fontsize=7, color=color, ha="right", va="bottom", alpha=0.8,
        )


def _draw_structure_events(
    ax,
    events: list[StructureEvent],
    df: pd.DataFrame,
    *,
    x0: int = 0,
    compact: bool = False,
) -> None:
    visible: list[StructureEvent] = []
    for ev in events:
        if ev.index >= len(df):
            continue
        ref_idx = ev.swing_ref.index
        if ref_idx >= len(df):
            continue
        if compact and ev.index < x0 - 1:
            continue
        visible.append(ev)
    if compact and len(visible) > 14:
        visible = visible[-14:]
    for ev in visible:
        ref_idx = ev.swing_ref.index
        if ev.event_type == StructureEventType.BOS:
            color = "#26a69a" if ev.direction == Trend.BULLISH else "#ef5350"
            label = "BOS"
        else:
            color = "#ff9800" if ev.direction == Trend.BEARISH else "#2196f3"
            label = "CHOCH"

        y = ev.swing_ref.price
        ax.annotate(
            "", xy=(ev.index, y), xytext=(ref_idx, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2 if compact else 1.5, ls="--"),
            zorder=4,
        )
        mid_x = (ref_idx + ev.index) / 2
        fs = 7 if compact else 8
        alpha = 0.78 if compact else 0.9
        ax.annotate(
            label, (mid_x, y), fontsize=fs, color=color,
            ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#131722", edgecolor=color, alpha=alpha),
        )


def _draw_fvgs(
    ax,
    fvgs: list[FairValueGap],
    n_bars: int,
    *,
    x0: int = 0,
    compact: bool = False,
) -> None:
    for fvg in fvgs:
        if fvg.mitigated:
            continue
        if fvg.fvg_type == FVGType.BULLISH:
            color = "#26a69a"
        else:
            color = "#ef5350"

        right_edge = min(fvg.index + 15, n_bars - 1)
        if compact and right_edge < x0 - 1:
            continue
        alpha_fill = 0.11 if compact else 0.15
        rect = mpatches.FancyBboxPatch(
            (fvg.index - 0.4, fvg.bottom), right_edge - fvg.index + 0.8, fvg.top - fvg.bottom,
            boxstyle="round,pad=0", facecolor=color, alpha=alpha_fill, edgecolor=color,
            linewidth=0.45 if compact else 0.5, zorder=1,
        )
        ax.add_patch(rect)
        if compact:
            continue
        ax.annotate(
            "FVG", (fvg.index + 1, (fvg.top + fvg.bottom) / 2),
            fontsize=7, color=color, ha="left", va="center", fontweight="bold", alpha=0.8,
        )


def _draw_order_blocks(
    ax,
    obs: list[OrderBlock],
    n_bars: int,
    *,
    x0: int = 0,
    compact: bool = False,
) -> None:
    for ob in obs:
        if ob.mitigated:
            continue
        if ob.ob_type == OBType.BULLISH:
            color = "#2196f3"
        else:
            color = "#ff5722"

        right_edge = min(ob.index + 20, n_bars - 1)
        if compact and right_edge < x0 - 1:
            continue
        alpha_fill = 0.09 if compact else 0.12
        rect = mpatches.FancyBboxPatch(
            (ob.index - 0.4, ob.bottom), right_edge - ob.index + 0.8, ob.top - ob.bottom,
            boxstyle="round,pad=0", facecolor=color, alpha=alpha_fill, edgecolor=color,
            linewidth=0.65 if compact else 0.8, zorder=1,
        )
        ax.add_patch(rect)
        if compact:
            continue
        ax.annotate(
            "OB", (ob.index + 1, (ob.top + ob.bottom) / 2),
            fontsize=7, color=color, ha="left", va="center", fontweight="bold", alpha=0.8,
        )


def _draw_setups(ax, setups: list[TradeSetupDTO], n_bars: int) -> None:
    for setup in setups:
        is_long = setup.side == Side.LONG
        entry_color = "#ffffff"
        sl_color = "#ef5350"
        tp_color = "#26a69a"

        x_start = n_bars - 8
        x_end = n_bars + 2

        ax.hlines(setup.entry, x_start, x_end, colors=entry_color, linestyles="-", linewidth=1.2, zorder=6)
        ax.annotate(
            f"Entry {setup.entry:.2f}", (x_end, setup.entry),
            fontsize=7, color=entry_color, ha="left", va="center", fontweight="bold",
        )

        ax.hlines(setup.stop_loss, x_start, x_end, colors=sl_color, linestyles="--", linewidth=1.0, zorder=6)
        ax.annotate(
            f"SL {setup.stop_loss:.2f}", (x_end, setup.stop_loss),
            fontsize=7, color=sl_color, ha="left", va="center",
        )

        for i, tp in enumerate(setup.take_profits):
            ax.hlines(tp, x_start, x_end, colors=tp_color, linestyles="--", linewidth=0.8, zorder=6)
            ax.annotate(
                f"TP{i + 1} {tp:.2f}", (x_end, tp),
                fontsize=7, color=tp_color, ha="left", va="center",
            )

        direction_arrow = "↑" if is_long else "↓"
        ax.annotate(
            f"{direction_arrow} {setup.setup_type}\nR:R {setup.risk_reward:.1f} | Conf {setup.confidence:.0%}",
            (x_start - 2, setup.entry),
            fontsize=8, color="#ffd700", ha="right", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#131722", edgecolor="#ffd700", alpha=0.9),
        )
