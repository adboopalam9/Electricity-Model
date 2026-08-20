"""
EPA CEMS (Continuous Emissions Monitoring System) data fetcher.

Source: EPA Clean Air Markets Program Data (CAMPD) Streaming API
  - Base: https://api.epa.gov/easey
  - Free API key: https://www.epa.gov/power-sector/cam-api-portal

Provides what EIA-860 cannot: actual hourly generation by plant,
real heat rates (heat_input / gross_load), and emissions (CO2, NOx, SO2).

CEMS data lags ~3 months (quarterly reporting), so we fetch a recent quarter
to derive plant-specific heat rates and apply them to current dispatch.
"""

from datetime import date, timedelta

import pandas as pd
import requests

CAMPD_BASE = "https://api.epa.gov/easey"
STREAMING_HOURLY = f"{CAMPD_BASE}/streaming-services/emissions/apportioned/hourly"


def _most_recent_cems_quarter(today: date | None = None) -> tuple[str, str]:
    """
    CEMS data is reported quarterly with ~4-5 month lag.
    Returns (begin_date, end_date) for the most recent likely-complete quarter.
    Uses 150-day lag to be conservative (data trickles in over several months).
    """
    today = today or date.today()
    quarter_bounds = [
        (date(y, m, 1), date(y, m + 2, {3: 31, 6: 30, 9: 30, 12: 31}[m + 2]))
        for y in [today.year, today.year - 1]
        for m in [1, 4, 7, 10]
    ]
    for qs, qe in sorted(quarter_bounds, key=lambda x: x[1], reverse=True):
        if qe + timedelta(days=150) <= today:
            return qs.strftime("%Y-%m-%d"), qe.strftime("%Y-%m-%d")

    fallback_start = date(today.year - 1, 7, 1)
    fallback_end = date(today.year - 1, 9, 30)
    return fallback_start.strftime("%Y-%m-%d"), fallback_end.strftime("%Y-%m-%d")


def fetch_cems_hourly(
    state: str,
    begin_date: str,
    end_date: str,
    api_key: str,
    fuel_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Hourly CEMS data for gas plants from CAMPD streaming API.
    The facilityId (ORIS code) matches EIA plant_id for cross-referencing.

    Returns DataFrame with columns:
      facilityId, facilityName, unitId, date, hour, grossLoad, steamLoad,
      heatInput, co2Mass, noxMass, so2Mass, opTime, state, unitType,
      primaryFuelInfo, datetime_utc
    """
    if fuel_types is None:
        fuel_types = ["Pipeline Natural Gas", "Other Gas", "Natural Gas"]

    headers = {"x-api-key": api_key, "Accept": "application/json"}
    all_data = []

    for ft in fuel_types:
        params = {
            "stateCode": state,
            "beginDate": begin_date,
            "endDate": end_date,
            "unitFuelType": ft,
            "operatingHoursOnly": "true",
        }
        print(f"    Querying CAMPD API: {state} / {ft} / {begin_date} → {end_date} ...")
        resp = requests.get(STREAMING_HOURLY, headers=headers, params=params, timeout=300)
        if resp.status_code == 200:
            chunk = resp.json()
            if chunk:
                all_data.extend(chunk)
                print(f"      → {len(chunk):,} records")
            else:
                print("      → 0 records")
        else:
            print(f"      → HTTP {resp.status_code}, skipping")

    data = all_data
    if not data:
        print("    Warning: CAMPD returned no data for this query.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["datetime_utc"] = (
        pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int), unit="h")
    )

    for col in ["grossLoad", "heatInput", "co2Mass", "noxMass", "so2Mass", "opTime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_facilities = df["facilityId"].nunique()
    print(f"    → {len(df):,} hourly records from {n_facilities} facilities")
    return df


def compute_plant_heat_rates(cems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Plant-level average heat rates from CEMS data.
    heat_rate_btu_kwh = (total_heat_input_mmbtu / total_generation_mwh) × 1000

    Filters out unreasonable heat rates (<3,000 or >25,000 Btu/kWh).
    """
    operating = cems_df[
        (cems_df["grossLoad"] > 0) & (cems_df["heatInput"] > 0)
    ].copy()

    if operating.empty:
        return pd.DataFrame()

    agg = (
        operating.groupby("facilityId")
        .agg(
            facility_name=("facilityName", "first"),
            total_heat_input_mmbtu=("heatInput", "sum"),
            total_generation_mwh=("grossLoad", "sum"),
            total_co2_tons=("co2Mass", "sum"),
            total_nox_lbs=("noxMass", "sum"),
            operating_hours=("grossLoad", "count"),
            avg_gross_load_mw=("grossLoad", "mean"),
        )
        .reset_index()
        .rename(columns={"facilityId": "facility_id"})
    )

    agg["heat_rate_btu_kwh"] = (
        agg["total_heat_input_mmbtu"] / agg["total_generation_mwh"] * 1000
    )
    agg["co2_tons_per_mwh"] = agg["total_co2_tons"] / agg["total_generation_mwh"]
    agg["nox_lbs_per_mwh"] = agg["total_nox_lbs"] / agg["total_generation_mwh"]

    reasonable = agg[
        (agg["heat_rate_btu_kwh"] > 3000) & (agg["heat_rate_btu_kwh"] < 25000)
    ].reset_index(drop=True)

    dropped = len(agg) - len(reasonable)
    if dropped:
        print(f"    Dropped {dropped} facilities with unreasonable heat rates")

    return reasonable


def merge_cems_with_eia860(
    eia860_df: pd.DataFrame,
    cems_rates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace generic heat rates in EIA-860 data with CEMS-derived plant-specific
    heat rates where available. Adds co2_tons_per_mwh for emissions tracking.
    """
    merged = eia860_df.copy()

    cems_lookup = cems_rates.set_index("facility_id")
    hr_map = cems_lookup["heat_rate_btu_kwh"]
    co2_map = cems_lookup["co2_tons_per_mwh"]

    plant_ids = merged["plant_id"].astype(int)
    mask = plant_ids.isin(hr_map.index)
    matched = mask.sum()

    merged["heat_rate_source"] = "generic"
    merged.loc[mask, "heat_rate_btu_kwh"] = plant_ids[mask].map(hr_map).values
    merged.loc[mask, "heat_rate_source"] = "cems"

    # CO2: CEMS rate where available, EPA default for NG (117 lb/MMBtu ≈ 0.053 tons/MWh at 9000 Btu/kWh)
    merged["co2_tons_per_mwh"] = 0.053
    if matched > 0:
        merged.loc[mask, "co2_tons_per_mwh"] = plant_ids[mask].map(co2_map).values

    total = len(merged)
    print(f"    CEMS heat rates matched: {matched}/{total} generators "
          f"({matched / total * 100:.0f}%)")
    generic = merged[merged["heat_rate_source"] == "generic"]
    cems = merged[merged["heat_rate_source"] == "cems"]
    if matched > 0:
        print(f"    Avg heat rate — CEMS: {cems['heat_rate_btu_kwh'].mean():,.0f} "
              f"vs generic: {generic['heat_rate_btu_kwh'].mean():,.0f} Btu/kWh")

    return merged


def get_cems_hourly_generation(cems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CEMS hourly data to total gas generation per hour.
    Useful for comparing actual generation against dispatch model output.
    """
    hourly = (
        cems_df.groupby("datetime_utc")
        .agg(
            cems_gas_generation_mw=("grossLoad", "sum"),
            cems_co2_tons=("co2Mass", "sum"),
            cems_nox_lbs=("noxMass", "sum"),
            cems_facilities_online=("facilityId", "nunique"),
        )
        .reset_index()
    )
    return hourly
