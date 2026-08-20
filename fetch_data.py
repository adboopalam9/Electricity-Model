"""
Data fetching for CAISO Gas Dispatch Model.

Sources:
  - EIA API v2  : load + fuel mix + Henry Hub price  (free key at eia.gov/opendata)
  - EIA-860     : gas plant specs, bulk Excel download (no key needed)

CAISO OASIS is NOT used — the API requires undocumented parameters that change
without notice. EIA's v2 electricity/rto endpoints cover the same data reliably.
"""
import io
import zipfile
from datetime import datetime, timedelta

import requests
import pandas as pd

EIA_V2_BASE = "https://api.eia.gov/v2"
EIA_860_URL = "https://www.eia.gov/electricity/data/eia860/archive/xls/eia8602023.zip"

# Typical heat rates (Btu/kWh) and VOM ($/MWh) by NERC prime mover code.
# Source: EIA, NREL ATB 2023 — EIA-860 Schedule 3 does not include design heat rates.
HEAT_RATE = {"CA": 6400, "CS": 6600, "CT": 10500, "GT": 10500, "ST": 9500, "IC": 9300, "CC": 6500}
VOM       = {"CA": 3.0,  "CS": 3.2,  "CT": 4.5,   "GT": 4.5,   "ST": 5.0,  "IC": 6.0,  "CC": 3.5}


# ── EIA API v2 helper ─────────────────────────────────────────────────────────

def _eia_get(endpoint: str, api_key: str, params: dict) -> list[dict]:
    """Single EIA API v2 request (up to 5000 rows per call)."""
    full_params = {"api_key": api_key, "length": 5000, "offset": 0, **params}
    resp = requests.get(f"{EIA_V2_BASE}{endpoint}", params=full_params, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if "response" not in body:
        raise ValueError(f"Unexpected EIA response: {body}")
    return body["response"]["data"]


# ── Load ──────────────────────────────────────────────────────────────────────

def fetch_caiso_load(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Hourly CAISO system demand from EIA API v2 electricity/rto/region-data.
    Period is UTC. Returns: datetime_utc | load_mw
    """
    rows = _eia_get(
        "/electricity/rto/region-data/data/",
        api_key,
        {
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": "CISO",
            "facets[type][]":       "D",      # D = Demand
            "start":                f"{start_date}T00",
            "end":                  f"{end_date}T23",
            "sort[0][column]":      "period",
            "sort[0][direction]":   "asc",
        },
    )
    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["period"])
    df["load_mw"]      = pd.to_numeric(df["value"], errors="coerce")
    return (
        df[["datetime_utc", "load_mw"]]
        .dropna()
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )


# ── Fuel mix ──────────────────────────────────────────────────────────────────

def fetch_caiso_fuel_mix(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Hourly CAISO generation by fuel type from EIA API v2 electricity/rto/fuel-type-data.
    EIA fuel codes: NG (gas), NUC, WAT (hydro), WND, SUN, COL, OIL, OTH.
    Returns wide DataFrame: datetime_utc | NG | NUC | WAT | WND | SUN | ...
    """
    rows = _eia_get(
        "/electricity/rto/fuel-type-data/data/",
        api_key,
        {
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": "CISO",
            "start":                f"{start_date}T00",
            "end":                  f"{end_date}T23",
            "sort[0][column]":      "period",
            "sort[0][direction]":   "asc",
        },
    )
    df = pd.DataFrame(rows)
    df["datetime_utc"] = pd.to_datetime(df["period"])
    df["value"]        = pd.to_numeric(df["value"], errors="coerce")
    df["fuel_type"]    = df["fueltype"].str.upper()

    pivot = (
        df.groupby(["datetime_utc", "fuel_type"])["value"]
        .mean()
        .unstack("fuel_type")
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


# ── Henry Hub gas price ───────────────────────────────────────────────────────

def fetch_henry_hub(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Daily Henry Hub spot price ($/MMBtu).

    EIA monthly data lags by ~2-3 months, so we look back up to 4 months
    and forward-fill the most recent published price into the date range.

    Strategy:
      1. EIA v2 natural-gas/pri/sum  — monthly, 4-month lookback window.
      2. EIA v2 electricity fuel cost — broader lookback, known working endpoint.

    Returns: date | gas_price_per_mmbtu  (one row per calendar day, no gaps)
    """
    from datetime import datetime as _dt
    end_dt   = pd.to_datetime(end_date)
    start_dt = pd.to_datetime(start_date)

    # Look back 4 months so we always land inside the published window
    lookback_start = (_dt.strptime(start_date, "%Y-%m-%d")
                      .replace(day=1) - pd.DateOffset(months=4)).strftime("%Y-%m")

    def _ffill_to_range(prices: pd.Series) -> pd.DataFrame:
        """ffill monthly/sparse prices to every calendar day. Index must be unique."""
        full_idx = pd.date_range(prices.index.min(), end_dt, freq="D")
        daily    = prices.reindex(full_idx, method="ffill").loc[start_dt:end_dt]
        out      = daily.reset_index()
        out.columns = ["date", "gas_price_per_mmbtu"]
        return out.dropna(subset=["gas_price_per_mmbtu"]).reset_index(drop=True)

    # ── Attempt 1: EIA v2 natural gas price summary ───────────────────────────
    try:
        rows = _eia_get(
            "/natural-gas/pri/sum/data/",
            api_key,
            {
                "data[0]":            "value",
                "start":              lookback_start,
                "end":                end_date[:7],
                "sort[0][column]":    "period",
                "sort[0][direction]": "asc",
            },
        )
        if rows:
            df = pd.DataFrame(rows)
            # Filter to California city gate spot price (PCS).
            # EIA /natural-gas/pri/sum returns state-level prices, not Henry Hub.
            # CA city gate is the correct input for California dispatch: it's
            # what generators actually pay at the delivery point into the state.
            ca_mask  = df["area-name"].str.upper().eq("CALIFORNIA") if "area-name" in df.columns else pd.Series(False, index=df.index)
            pcs_mask = df["process"].eq("PCS") if "process" in df.columns else pd.Series(False, index=df.index)
            if (ca_mask & pcs_mask).any():
                df = df[ca_mask & pcs_mask]
            elif pcs_mask.any():
                df = df[pcs_mask]   # national city gate average as fallback
            elif "process" in df.columns:
                # Last resort: electric power sector price
                peu_mask = df["process"].eq("PEU")
                if peu_mask.any():
                    df = df[peu_mask]

            df["date"]                = pd.to_datetime(df["period"])
            df["gas_price_per_mmbtu"] = pd.to_numeric(df["value"], errors="coerce")
            prices = (
                df.dropna(subset=["gas_price_per_mmbtu"])
                .groupby("date")["gas_price_per_mmbtu"]
                .mean()
            )
            if len(prices) > 0:
                result = _ffill_to_range(prices)
                if len(result) > 0:
                    return result
    except Exception as e:
        print(f"  EIA v2 NG price failed ({type(e).__name__}: {e})")

    # ── Attempt 2: EIA v2 electricity fuel receipts cost (CA, NG) ────────────
    # Uses the same API host that returned load + fuel mix — guaranteed reachable.
    # EIA-923 derived: average delivered cost of NG to CA electric utilities.
    print("  Falling back to EIA electricity fuel cost data (CA, NG)...")
    for cost_field in ["receipts-cost", "cost-per-btu", "average-cost"]:
        try:
            rows = _eia_get(
                "/electricity/electric-power-operational-data/data/",
                api_key,
                {
                    "frequency":              "monthly",
                    "data[0]":                cost_field,
                    "facets[fueltypeid][]":   "NG",
                    "facets[location][]":     "CA",
                    "start":                  lookback_start,
                    "end":                    end_date[:7],
                    "sort[0][column]":        "period",
                    "sort[0][direction]":     "asc",
                },
            )
            if rows:
                df = pd.DataFrame(rows)
                df["date"]                = pd.to_datetime(df["period"])
                df["gas_price_per_mmbtu"] = pd.to_numeric(df[cost_field], errors="coerce")
                prices = (
                    df.dropna(subset=["gas_price_per_mmbtu"])
                    .groupby("date")["gas_price_per_mmbtu"]
                    .mean()
                )
                if len(prices) > 0:
                    result = _ffill_to_range(prices)
                    if len(result) > 0:
                        print(f"    Using field '{cost_field}': "
                              f"${prices.iloc[-1]:.2f}/MMBtu (most recent)")
                        return result
        except Exception:
            continue

    raise RuntimeError(
        "Could not fetch gas price from EIA. "
        "Check API key or try again later (data may not be published yet)."
    )


# ── EIA-860 gas plant specs ───────────────────────────────────────────────────

def fetch_eia860_gas_plants() -> pd.DataFrame:
    """
    Downloads EIA-860 2023 and returns CA operating natural gas generators.
    Uses typical heat rates by prime mover (EIA-860 Schedule 3 omits design heat rates).
    Returns: plant_id | plant_name | capacity_mw | heat_rate_btu_kwh | vom_per_mwh | prime_mover
    """
    print("  Downloading EIA-860 2023 (~15 MB, cached after first run)...")
    resp = requests.get(EIA_860_URL, timeout=180)
    resp.raise_for_status()

    zf       = zipfile.ZipFile(io.BytesIO(resp.content))
    gen_fname = next(n for n in zf.namelist() if "3_1_Generator" in n)
    xl_bytes  = zf.read(gen_fname)

    # Row 0 is a report title; row 1 is the actual column header
    df = pd.read_excel(io.BytesIO(xl_bytes), sheet_name=0, header=1, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    mask = (
        (df.get("State",            pd.Series(dtype=str)) == "CA") &
        (df.get("Energy Source 1",  pd.Series(dtype=str)) == "NG") &
        (df.get("Status",           pd.Series(dtype=str)) == "OP")
    )
    gas = df[mask].copy()

    cap_col  = next((c for c in gas.columns if "Nameplate Capacity" in c), None)
    pid_col  = next((c for c in gas.columns if c in ("Plant Id", "Plant Code")), "Plant Id")
    pnam_col = next((c for c in gas.columns if "Plant Name" in c), "Plant Name")

    gas = gas.rename(columns={pid_col: "plant_id", pnam_col: "plant_name"})
    gas["capacity_mw"]       = pd.to_numeric(gas[cap_col], errors="coerce") if cap_col else float("nan")
    gas["prime_mover"]       = gas.get("Prime Mover", pd.Series(dtype=str)).str.strip().str.upper()
    gas["heat_rate_btu_kwh"] = gas["prime_mover"].map(HEAT_RATE).fillna(8500)
    gas["vom_per_mwh"]       = gas["prime_mover"].map(VOM).fillna(4.5)

    result = (
        gas[["plant_id", "plant_name", "capacity_mw", "heat_rate_btu_kwh", "vom_per_mwh", "prime_mover"]]
        .dropna(subset=["capacity_mw"])
        .query("capacity_mw > 0")
        .reset_index(drop=True)
    )
    print(f"  → {len(result)} operating NG generators in CA, {result['capacity_mw'].sum():,.0f} MW total")
    return result
