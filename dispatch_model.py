"""
Merit order dispatch engine for CAISO gas fleet.

Logic:
  1. Compute each plant's marginal cost = (heat_rate × gas_price / 1000) + VOM
  2. Sort cheapest → most expensive (merit order)
  3. Per hour: subtract all non-gas generation from load → net load
  4. Walk merit order until net load is met; last plant called sets the clearing price
"""
import pandas as pd

# EIA API v2 fuel type codes that are NOT natural gas.
# These get subtracted from load to compute the gas residual (net load).
# EIA codes: NUC=nuclear, WAT=hydro, WND=wind, SUN=solar,
#            COL=coal, OIL=oil, OTH=other (geo/biomass/etc)
NON_GAS_FUELS = ["NUC", "WAT", "WND", "SUN", "COL", "OIL", "OTH", "UNK", "GEO"]


def compute_merit_order(plants: pd.DataFrame, gas_price: float) -> pd.DataFrame:
    """
    Marginal cost ($/MWh) = (heat_rate_btu_kwh × gas_price_$/mmbtu / 1000) + vom_$/mwh
    Unit check: [Btu/kWh × $/MMBtu] / [1000 kWh/MWh × Btu/kBtu] = $/MWh ✓
    Returns plants sorted cheapest → most expensive.
    """
    df = plants.copy()
    df["marginal_cost"] = (df["heat_rate_btu_kwh"] * gas_price / 1000) + df["vom_per_mwh"]
    return df.sort_values("marginal_cost").reset_index(drop=True)


def compute_net_load(load_mw: float, fuel_row: pd.Series) -> float:
    """
    Gas residual = total load − all non-gas generation.
    Negative fuel values (e.g. batteries charging, net exports) naturally add to net load.
    Clipped at 0 — renewables can't make gas demand negative.
    """
    non_gas = sum(float(fuel_row.get(f, 0) or 0) for f in NON_GAS_FUELS)
    return max(load_mw - non_gas, 0.0)


def dispatch_hour(merit_order: pd.DataFrame, net_load_mw: float) -> tuple[pd.DataFrame, float]:
    """
    Walk merit order cheapest-first until net load is met.
    Returns (dispatched_df, clearing_price_$/mwh).
    clearing_price = marginal cost of the last (most expensive) plant called.
    """
    df = merit_order.copy()
    df["dispatched_mw"] = 0.0
    df["on"] = False

    remaining = net_load_mw
    clearing_price = 0.0

    for i in df.index:
        if remaining <= 0:
            break
        cap = df.at[i, "capacity_mw"]
        dispatch = min(cap, remaining)
        df.at[i, "dispatched_mw"] = dispatch
        df.at[i, "on"] = True
        remaining -= dispatch
        clearing_price = df.at[i, "marginal_cost"]

    return df, clearing_price


def _hour_co2(dispatched: pd.DataFrame) -> float:
    """Total CO2 (short tons) from dispatched plants this hour."""
    if "co2_tons_per_mwh" not in dispatched.columns:
        return 0.0
    on = dispatched[dispatched["on"]]
    return (on["dispatched_mw"] * on["co2_tons_per_mwh"]).sum()


def run_dispatch(
    load_df: pd.DataFrame,
    fuel_mix_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    gas_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run hourly dispatch across the full date range.
    Gas price is forward-filled through weekends/holidays where EIA has no data.

    Returns hourly summary with columns:
      datetime_utc, load_mw, net_load_mw, gas_dispatched_mw,
      clearing_price, renewables_offset_mw, gas_price
    """
    load_series = load_df.set_index("datetime_utc")["load_mw"]
    fuel_indexed = fuel_mix_df.set_index("datetime_utc")
    price_by_date = gas_prices_df.set_index("date")["gas_price_per_mmbtu"]

    results = []
    for dt in load_series.index:
        day = dt.normalize()
        # Forward-fill gas price through non-trading days
        if day in price_by_date.index:
            gas_price = float(price_by_date[day])
        else:
            prior = price_by_date[price_by_date.index <= day]
            gas_price = float(prior.iloc[-1]) if len(prior) else 3.50

        fuel_row = fuel_indexed.loc[dt] if dt in fuel_indexed.index else pd.Series(dtype=float)
        load_mw  = float(load_series[dt])
        net_load = compute_net_load(load_mw, fuel_row)

        merit_order          = compute_merit_order(plants_df, gas_price)
        dispatched, clearing = dispatch_hour(merit_order, net_load)

        results.append({
            "datetime_utc":        dt,
            "load_mw":             load_mw,
            "net_load_mw":         net_load,
            "gas_dispatched_mw":   dispatched["dispatched_mw"].sum(),
            "clearing_price":      clearing,
            "renewables_offset_mw": load_mw - net_load,
            "gas_price":           gas_price,
            "co2_tons":            _hour_co2(dispatched),
        })

    return pd.DataFrame(results)
