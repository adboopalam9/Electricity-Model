"""
CAISO Gas Plant Dispatch Model — entry point.

Data sources (all free):
  - EIA API v2  : load, fuel mix, gas price
  - EIA-860     : plant capacity + prime mover (~15 MB download, cached)
  - EPA CEMS    : plant-level hourly generation, heat rates, emissions (optional)

Run:
  python main.py

Output:
  data/dispatch_results.csv
  supply_curve_day.png
  supply_curve_night.png
  dispatch_heatmap.png
  daily_profile.png
  emissions_profile.png   (when CEMS data is available)
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from fetch_data import (
    fetch_caiso_load,
    fetch_caiso_fuel_mix,
    fetch_henry_hub,
    fetch_eia860_gas_plants,
)
from fetch_epa_cems import (
    fetch_cems_hourly,
    compute_plant_heat_rates,
    merge_cems_with_eia860,
    _most_recent_cems_quarter,
)
from dispatch_model import run_dispatch, compute_merit_order
from visualize import (
    plot_supply_curve,
    plot_dispatch_heatmap,
    plot_daily_profile,
    plot_emissions_profile,
)

# ── Config ────────────────────────────────────────────────────────────────────

# Rolling window: always use the most recent 7 days with confirmed EIA data.
# EIA electricity data has a ~2-day publication lag.
_today      = date.today()
END_DATE    = (_today - timedelta(days=2)).strftime("%Y-%m-%d")
START_DATE  = (_today - timedelta(days=8)).strftime("%Y-%m-%d")

# Free EIA API key: https://www.eia.gov/opendata/register.php
EIA_API_KEY = "21wTgf5yYw0VhkDncWFzuKKuq0q7bvzd0tVzA4yO"

# Free EPA CAMPD API key: https://www.epa.gov/power-sector/cam-api-portal
# Set to None to skip CEMS integration and use generic heat rates.
EPA_API_KEY = "dqNxsQjFjsJUksaUc8Y0VluWkEnNlbeDfWoFEa2L"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ── Caching helpers ───────────────────────────────────────────────────────────

def _cached_csv(path: Path, fetch_fn, *args, **kwargs) -> pd.DataFrame:
    if path.exists():
        print(f"  [cache] {path.name}")
        df = pd.read_csv(path)
        # Restore datetime column if present
        for col in df.columns:
            if "datetime" in col or col == "date":
                df[col] = pd.to_datetime(df[col])
        return df
    print(f"  [fetch] {path.name}")
    df = fetch_fn(*args, **kwargs)
    df.to_csv(path, index=False)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  CAISO Gas Plant Dispatch Model")
    print(f"  {START_DATE}  →  {END_DATE}")
    print("=" * 50)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────

    print("\n[1/5] CAISO hourly load  (EIA API v2)")
    load_df = _cached_csv(
        DATA_DIR / f"caiso_load_{START_DATE}_{END_DATE}.csv",
        fetch_caiso_load, START_DATE, END_DATE, EIA_API_KEY,
    )
    print(f"      {len(load_df)} hourly records | "
          f"avg {load_df['load_mw'].mean():,.0f} MW")

    print("\n[2/5] CAISO fuel mix  (EIA API v2)")
    fuel_df = _cached_csv(
        DATA_DIR / f"caiso_fuel_{START_DATE}_{END_DATE}.csv",
        fetch_caiso_fuel_mix, START_DATE, END_DATE, EIA_API_KEY,
    )
    fuel_cols = [c for c in fuel_df.columns if c != "datetime_utc"]
    print(f"      Fuels: {fuel_cols}")

    print("\n[3/5] Henry Hub gas prices")
    if EIA_API_KEY == "YOUR_EIA_API_KEY":
        raise SystemExit(
            "\n  ✗  Set EIA_API_KEY in main.py first.\n"
            "     Free key: https://www.eia.gov/opendata/register.php\n"
        )
    gas_df = _cached_csv(
        DATA_DIR / f"henry_hub_{START_DATE}_{END_DATE}.csv",
        fetch_henry_hub, START_DATE, END_DATE, EIA_API_KEY,
    )
    print(f"      ${gas_df['gas_price_per_mmbtu'].min():.2f} – "
          f"${gas_df['gas_price_per_mmbtu'].max():.2f} /MMBtu")

    print("\n[4/5] EIA-860 gas plant specs (CA)")
    plants_df = _cached_csv(
        DATA_DIR / "eia860_ca_gas_plants.csv",
        fetch_eia860_gas_plants,
    )
    print(f"      {len(plants_df)} generators | "
          f"{plants_df['capacity_mw'].sum():,.0f} MW total")

    # ── 5. EPA CEMS: plant-specific heat rates + emissions factors ────────────

    if EPA_API_KEY:
        print("\n[5/5] EPA CEMS plant-level data")
        cems_start, cems_end = _most_recent_cems_quarter()
        print(f"      Querying CEMS quarter: {cems_start} → {cems_end}")

        cems_df = _cached_csv(
            DATA_DIR / f"cems_ca_gas_{cems_start}_{cems_end}.csv",
            fetch_cems_hourly, "CA", cems_start, cems_end, EPA_API_KEY,
        )

        if len(cems_df) > 0:
            cems_rates = compute_plant_heat_rates(cems_df)
            cems_rates.to_csv(DATA_DIR / "cems_plant_heat_rates.csv", index=False)
            print(f"      {len(cems_rates)} facilities with valid heat rates")

            plants_df = merge_cems_with_eia860(plants_df, cems_rates)
            plants_df.to_csv(DATA_DIR / "plants_with_cems.csv", index=False)
        else:
            print("      No CEMS data returned — using generic heat rates")
    else:
        print("\n[5/5] EPA CEMS — skipped (no EPA_API_KEY)")
        print("      Using generic heat rates by prime mover type")
        print("      Get a free key: https://www.epa.gov/power-sector/cam-api-portal")
        if "co2_tons_per_mwh" not in plants_df.columns:
            plants_df["co2_tons_per_mwh"] = 0.053

    # ── 2. Run dispatch ───────────────────────────────────────────────────────

    print("\n[Dispatch] Running merit order dispatch...")
    results = run_dispatch(load_df, fuel_df, plants_df, gas_df)
    results.to_csv(DATA_DIR / "dispatch_results.csv", index=False)

    print("\n── Results summary ──────────────────────────────────────")
    summary = results[["load_mw", "net_load_mw", "gas_dispatched_mw",
                        "clearing_price", "renewables_offset_mw",
                        "co2_tons"]].describe().round(1)
    print(summary.to_string())

    avg_offset_pct = (results["renewables_offset_mw"] / results["load_mw"]).mean() * 100
    total_co2 = results["co2_tons"].sum()
    total_gen = results["gas_dispatched_mw"].sum()
    avg_co2_rate = total_co2 / total_gen * 1000 if total_gen > 0 else 0
    print(f"\n  Renewables offset avg: {avg_offset_pct:.1f}% of gross load")
    print(f"  Total CO2: {total_co2:,.0f} short tons ({avg_co2_rate:.0f} lb/MWh)")

    # ── 3. Charts ─────────────────────────────────────────────────────────────

    print("\n── Generating charts ───────────────────────────────────")

    # Supply curve: compare hour 13 (peak solar) vs hour 20 (evening ramp)
    for label, hour_offset in [("day_hr13", 13), ("night_hr20", 20)]:
        row = results[results["datetime_utc"].dt.hour == hour_offset].iloc[0]
        merit = compute_merit_order(plants_df, row["gas_price"])
        dt_str = str(row["datetime_utc"])
        fig = plot_supply_curve(
            merit, row["net_load_mw"],
            title=f"Merit Order Stack — {dt_str} UTC  (net load {row['net_load_mw']:,.0f} MW)"
        )
        out = f"supply_curve_{label}.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved {out}")

    fig = plot_dispatch_heatmap(results)
    fig.savefig("dispatch_heatmap.png", dpi=150)
    print("  Saved dispatch_heatmap.png")

    fig = plot_daily_profile(results)
    fig.savefig("daily_profile.png", dpi=150)
    print("  Saved daily_profile.png")

    if results["co2_tons"].sum() > 0:
        fig = plot_emissions_profile(results)
        fig.savefig("emissions_profile.png", dpi=150)
        print("  Saved emissions_profile.png")

    print("\nDone. Open the PNG files to see the charts.\n")


if __name__ == "__main__":
    main()
