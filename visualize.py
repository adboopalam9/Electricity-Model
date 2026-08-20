"""
Visualization for the CAISO Gas Dispatch Model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ── Design tokens ─────────────────────────────────────────────────────────────
C = {
    "gas":        "#E07B39",   # warm orange  — gas dispatch
    "load":       "#2C6FAC",   # steel blue   — total load
    "renewables": "#3D9A6A",   # green        — renewables offset
    "idle":       "#DEDEDE",   # light gray   — idle capacity
    "price":      "#6C3483",   # purple       — clearing price
    "clearing":   "#C0392B",   # red          — clearing price marker
    "text":       "#1C1C1C",
    "subtext":    "#777777",
    "grid":       "#EBEBEB",
    "bg":         "#F9F9F9",
}


def _style(ax):
    ax.set_facecolor(C["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D5D5D5")
    ax.tick_params(colors=C["subtext"], labelsize=9)
    ax.yaxis.grid(color=C["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _label(ax, title="", xlabel="", ylabel=""):
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=C["text"], pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=C["subtext"], labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=C["subtext"], labelpad=8)


# ── Supply curve ──────────────────────────────────────────────────────────────

def plot_supply_curve(
    merit_order: pd.DataFrame,
    net_load_mw: float,
    title: str = "",
) -> plt.Figure:
    """
    Merit order staircase: X = cumulative MW, Y = $/MWh.
    Orange = dispatched plants, gray = idle, red dashed = net load.
    """
    x, y = [0.0], [0.0]
    cum = 0.0
    for _, row in merit_order.iterrows():
        x += [cum, cum + row["capacity_mw"]]
        y += [row["marginal_cost"], row["marginal_cost"]]
        cum += row["capacity_mw"]
    x, y = np.array(x), np.array(y)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    _style(ax)

    ax.fill_between(x, y, where=(x <= net_load_mw),
                    color=C["gas"], alpha=0.88, label="Dispatched", zorder=2)
    ax.fill_between(x, y, where=(x > net_load_mw),
                    color=C["idle"], alpha=0.75, label="Idle", zorder=2)
    ax.plot(x, y, color="#444444", lw=0.5, zorder=3)

    ax.axvline(net_load_mw, color=C["clearing"], lw=2.0,
               ls="--", zorder=4, label=f"Net load  {net_load_mw:,.0f} MW")

    # Clearing price
    cumcap = merit_order["capacity_mw"].cumsum()
    hit    = merit_order[cumcap >= net_load_mw]
    clearing = float(hit["marginal_cost"].iloc[0]) if len(hit) > 0 and net_load_mw > 0 else 0.0

    ax.axhline(clearing, color=C["clearing"], lw=1.0, ls=":", alpha=0.7, zorder=3)
    ax.annotate(
        f"Clearing: ${clearing:.2f}/MWh",
        xy=(net_load_mw, clearing),
        xytext=(20, 12), textcoords="offset points",
        fontsize=9, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=C["clearing"],
                  edgecolor="none", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=C["clearing"], lw=1.2),
        zorder=5,
    )

    _label(ax,
           title=title or "CAISO Gas Fleet — Merit Order Stack",
           xlabel="Cumulative Capacity (MW)",
           ylabel="Marginal Cost ($/MWh)")

    leg = ax.legend(fontsize=9, frameon=True, framealpha=0.95,
                    edgecolor="#CCCCCC", loc="upper left")
    leg.get_frame().set_linewidth(0.6)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    return fig


# ── Dispatch heatmap ──────────────────────────────────────────────────────────

def plot_dispatch_heatmap(results: pd.DataFrame) -> plt.Figure:
    """
    Gas dispatch intensity (MW) by hour-of-day × date.
    Dark = heavy gas dispatch, light = solar squeezing gas off the grid.
    """
    df = results.copy()
    df["hour"] = df["datetime_utc"].dt.hour
    df["date"] = df["datetime_utc"].dt.date

    pivot  = df.pivot_table(index="hour", columns="date",
                             values="gas_dispatched_mw", aggfunc="mean").sort_index()
    n_days = len(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(11, n_days * 0.8), 7))
    fig.patch.set_facecolor("white")

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   origin="lower", interpolation="nearest", vmin=0)

    cbar = plt.colorbar(im, ax=ax, pad=0.015, shrink=0.88, aspect=30)
    cbar.set_label("Gas Dispatched (MW)", fontsize=9, color=C["subtext"])
    cbar.ax.tick_params(labelsize=8, colors=C["subtext"])
    cbar.outline.set_edgecolor("#CCCCCC")

    # White cell dividers
    ax.set_xticks(np.arange(-0.5, n_days, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    ax.set_yticks(range(0, 24, 3))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)],
                       fontsize=8, color=C["subtext"])
    ax.set_xticks(range(n_days))
    ax.set_xticklabels([str(d) for d in pivot.columns],
                       rotation=38, ha="right", fontsize=8, color=C["subtext"])

    # Cell value labels when the window is small enough to read
    if n_days <= 10:
        mx = np.nanmax(pivot.values)
        for r in range(24):
            for c in range(n_days):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    txt_col = "white" if val > mx * 0.55 else C["text"]
                    ax.text(c, r, f"{val:,.0f}", ha="center", va="center",
                            fontsize=6, color=txt_col, fontweight="bold")

    ax.spines[:].set_visible(False)
    _label(ax,
           title="Gas Dispatch Heatmap — MW by Hour × Day (UTC)",
           xlabel="Date",
           ylabel="Hour of Day (UTC)")

    plt.tight_layout()
    return fig


# ── Daily profile ─────────────────────────────────────────────────────────────

def plot_daily_profile(results: pd.DataFrame) -> plt.Figure:
    """
    Two-panel chart.
    Top:    Load decomposition — renewables offset + gas dispatch + total load line.
    Bottom: Gas clearing price timeline.
    """
    df = results.copy().sort_values("datetime_utc")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06},
        layout="constrained",
    )
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _style(ax)

    # ── Top: load decomposition ───────────────────────────────────────────────
    # Renewables + non-gas (bottom layer)
    ax1.fill_between(df["datetime_utc"], df["renewables_offset_mw"],
                     color=C["renewables"], alpha=0.60, zorder=2,
                     label="Renewables & Other")
    # Gas dispatch (stacked on top)
    ax1.fill_between(df["datetime_utc"],
                     df["renewables_offset_mw"],
                     df["renewables_offset_mw"] + df["gas_dispatched_mw"],
                     color=C["gas"], alpha=0.75, zorder=3,
                     label="Gas Dispatch")
    # Total load line
    ax1.plot(df["datetime_utc"], df["load_mw"],
             color=C["load"], lw=2.2, zorder=5, label="Total Load")

    _label(ax1,
           title="CAISO Gas Dispatch — Weekly Profile",
           ylabel="MW")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    leg = ax1.legend(fontsize=9, frameon=True, framealpha=0.95,
                     edgecolor="#CCCCCC", loc="upper right")
    leg.get_frame().set_linewidth(0.6)

    # ── Bottom: clearing price ────────────────────────────────────────────────
    ax2.fill_between(df["datetime_utc"], df["clearing_price"],
                     color=C["price"], alpha=0.20, zorder=1)
    ax2.plot(df["datetime_utc"], df["clearing_price"],
             color=C["price"], lw=1.8, zorder=2, label="Gas Clearing Price")

    _label(ax2, ylabel="$/MWh", xlabel="Date (UTC)")
    leg2 = ax2.legend(fontsize=9, frameon=True, framealpha=0.95,
                      edgecolor="#CCCCCC", loc="upper right")
    leg2.get_frame().set_linewidth(0.6)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    return fig


# ── Emissions profile ────────────────────────────────────────────────────────

def plot_emissions_profile(results: pd.DataFrame) -> plt.Figure:
    """
    Two-panel chart.
    Top:    Hourly CO2 emissions (tons) stacked with gas dispatch.
    Bottom: CO2 intensity (lb/MWh) over time.
    """
    df = results.copy().sort_values("datetime_utc")
    df["co2_lbs_per_mwh"] = np.where(
        df["gas_dispatched_mw"] > 0,
        df["co2_tons"] / df["gas_dispatched_mw"] * 2000,
        0,
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06},
        layout="constrained",
    )
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _style(ax)

    co2_color = "#8B4513"

    ax1.fill_between(df["datetime_utc"], df["co2_tons"],
                     color=co2_color, alpha=0.55, zorder=2, label="CO₂ (tons/hr)")
    ax1.plot(df["datetime_utc"], df["co2_tons"],
             color=co2_color, lw=1.5, zorder=3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(df["datetime_utc"], df["gas_dispatched_mw"],
                  color=C["gas"], lw=1.8, alpha=0.8, ls="--",
                  zorder=4, label="Gas Dispatch (MW)")
    ax1_twin.set_ylabel("Gas Dispatch (MW)", fontsize=9, color=C["subtext"])
    ax1_twin.tick_params(colors=C["subtext"], labelsize=9)
    ax1_twin.spines["right"].set_color("#D5D5D5")

    _label(ax1,
           title="CAISO Gas Fleet — CO₂ Emissions Profile",
           ylabel="CO₂ (short tons/hr)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, frameon=True, framealpha=0.95,
               edgecolor="#CCCCCC", loc="upper right").get_frame().set_linewidth(0.6)

    ax2.plot(df["datetime_utc"], df["co2_lbs_per_mwh"],
             color=co2_color, lw=1.8, zorder=2)
    ax2.fill_between(df["datetime_utc"], df["co2_lbs_per_mwh"],
                     color=co2_color, alpha=0.15, zorder=1)
    _label(ax2, ylabel="CO₂ Intensity (lb/MWh)", xlabel="Date (UTC)")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    return fig
