"""
Ocean Shipment Transit Analyzer
================================
Streamlit app: ingest a raw shipment file, let the user pick a start and end
milestone (and aggregation level — container or master-shipment), compute
transit times, summarize by lane × carrier, and explore trends interactively.

Trends tab shows TWO charts:
  - CHART 1 (Transit time): per-carrier avg/median lines + Combined line + container-volume bars.
  - CHART 2 (Consistency):   per-carrier std-dev lines + Combined std-dev line + container-volume bars.

Run locally:  streamlit run app.py
Deploy:       push to GitHub, connect repo on Streamlit Community Cloud.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================================
# Constants
# ============================================================================

# Bump on any schema/column change. Folded into every cache key so old
# session_state / @st.cache_data values can't survive an upgrade.
CODE_VERSION = "2026-06-11-r7"

MILESTONES: List[Tuple[str, str, List[str]]] = [
    ("CEP", "Empty Container Pickup",      ["CEP"]),
    ("CGI", "Container Gate-In at POL",    ["CGI"]),
    ("CLL", "Container Loaded on Vessel",  ["CLL"]),
    ("VDL", "Vessel Departure from POL",   ["VDL", "VDL_P44"]),
    ("VAD", "Vessel Arrival at POD",       ["VAD", "VAD_P44"]),
    ("CDD", "Container Discharged at POD", ["CDD"]),
    ("CGO", "Container Gate-Out from POD", ["CGO"]),
    ("CER", "Empty Container Return",      ["CER"]),
]
MILESTONE_LABELS  = {c: f"{c} — {d}" for c, d, _ in MILESTONES}
MILESTONE_SOURCES = {c: s for c, _, s in MILESTONES}
MILESTONE_ORDER   = [c for c, _, _ in MILESTONES]

REQUIRED_COLUMNS = {
    "SHIPMENT_ID", "MASTER_SHIPMENT_ID", "CONTAINER_NUMBER",
    "CARRIER_NAME", "CARRIER_SCAC",
    "POL_LOCODE", "POL", "POL_COUNTRY",
    "POD_LOCODE", "POD", "POD_COUNTRY",
    "SHIPMENT_CREATED_DATE",
}

MAX_PREVIEW_ROWS = 10_000

# High-contrast palette for carrier lines (works in PNG export too).
CARRIER_PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Vivid
COMBINED_COLOR  = "#000000"
VOLUME_COLOR    = "#bcbcbc"

COL_TOTAL_HOURS    = "TOTAL_HOURS"
COL_TRANSIT_DAYS   = "TRANSIT_DAYS"
COL_TRANSIT_HOURS  = "TRANSIT_HOURS"
COMBINED_LABEL     = "Combined (all selected)"


# ============================================================================
# IO + caching
# ============================================================================

def _bytes_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@st.cache_data(show_spinner=False, max_entries=4)
def load_file(file_bytes: bytes, filename: str, _ver: str = CODE_VERSION) -> pd.DataFrame:
    name = filename.lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(bio)
    else:
        df = pd.read_csv(bio, encoding="utf-8-sig", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame) -> List[str]:
    return sorted(REQUIRED_COLUMNS - set(df.columns))


# ============================================================================
# Master-shipment aggregation
# ============================================================================

@st.cache_data(show_spinner=False, max_entries=4)
def aggregate_to_master_shipment(df: pd.DataFrame, _ver: str = CODE_VERSION) -> pd.DataFrame:
    if "MASTER_SHIPMENT_ID" not in df.columns:
        return df
    container_counts = (
        df.groupby("MASTER_SHIPMENT_ID")["CONTAINER_NUMBER"]
          .nunique().rename("CONTAINER_COUNT")
    )
    grouped = df.groupby("MASTER_SHIPMENT_ID", as_index=False, sort=False).first()
    grouped = grouped.merge(container_counts, on="MASTER_SHIPMENT_ID", how="left")
    grouped["CONTAINER_COUNT"] = grouped["CONTAINER_COUNT"].fillna(1).astype(int)
    return grouped


# ============================================================================
# Milestone resolution
# ============================================================================

def resolve_milestone(df: pd.DataFrame, code: str) -> pd.Series:
    sources = MILESTONE_SOURCES[code]
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for src in sources:
        if src not in df.columns:
            continue
        parsed = pd.to_datetime(df[src], errors="coerce", utc=True)
        result = result.where(result.notna(), parsed)
    return result


# ============================================================================
# Lane key
# ============================================================================

def build_lane_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _fmt(locode_col, city_col, ctry_col):
        loc  = out[locode_col].fillna("").astype(str).str.strip()
        city = out[city_col].fillna("").astype(str).str.strip()
        ctry = out[ctry_col].fillna("").astype(str).str.strip()
        place = np.where(
            (city != "") & (ctry != ""), city + ", " + ctry,
            np.where(city != "", city, ctry),
        )
        place = pd.Series(place, index=out.index)
        result = np.where(
            (loc != "") & (place != ""), loc + " - " + place,
            np.where(loc != "", loc, place),
        )
        result = pd.Series(result, index=out.index)
        return result.where(result != "", "Unknown")

    out["POL_FULL"] = _fmt("POL_LOCODE", "POL", "POL_COUNTRY")
    out["POD_FULL"] = _fmt("POD_LOCODE", "POD", "POD_COUNTRY")
    out["LANE"] = out["POL_FULL"] + " → " + out["POD_FULL"]
    return out


# ============================================================================
# Duration helpers — vectorized
# ============================================================================

def hours_to_components_vec(hours: pd.Series) -> Tuple[pd.Series, pd.Series]:
    total_minutes = (hours * 60).round()
    days = np.floor_divide(total_minutes, 24 * 60)
    rem_minutes = total_minutes - days * 24 * 60
    hours_part = (rem_minutes / 60).round()
    carry = hours_part == 24
    days = days.where(~carry, days + 1)
    hours_part = hours_part.where(~carry, 0)
    return days.astype("Int64"), hours_part.astype("Int64")


def hours_to_components(total_hours: float) -> Tuple[int, int]:
    if pd.isna(total_hours):
        return (np.nan, np.nan)
    total_minutes = int(round(total_hours * 60))
    days = total_minutes // (24 * 60)
    rem_minutes = total_minutes - days * 24 * 60
    hours = int(round(rem_minutes / 60))
    if hours == 24:
        days += 1; hours = 0
    return (int(days), int(hours))


# ============================================================================
# Transit calculation
# ============================================================================

@st.cache_data(show_spinner=False, max_entries=8)
def compute_transit(
    df: pd.DataFrame, start_code: str, end_code: str,
    aggregation_level: str, _ver: str = CODE_VERSION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = df
    if aggregation_level == "Master Shipment":
        base = aggregate_to_master_shipment(df)
    work = build_lane_keys(base).reset_index(drop=True)

    start_ts = resolve_milestone(work, start_code)
    end_ts   = resolve_milestone(work, end_code)
    work["START_TS"] = start_ts
    work["END_TS"]   = end_ts
    delta_hours = (end_ts - start_ts).dt.total_seconds() / 3600.0
    work["_DELTA_H"] = delta_hours

    reason = pd.Series("", index=work.index, dtype="object")
    reason = reason.mask(start_ts.isna() & end_ts.isna(), f"Missing both {start_code} and {end_code}")
    reason = reason.mask(start_ts.isna() & end_ts.notna(), f"Missing {start_code} (start)")
    reason = reason.mask(start_ts.notna() & end_ts.isna(), f"Missing {end_code} (end)")
    neg_mask = start_ts.notna() & end_ts.notna() & (delta_hours <= 0)
    reason = reason.mask(neg_mask, f"{end_code} not after {start_code} (non-positive transit)")
    work["MISSED_REASON"] = reason

    valid_mask = start_ts.notna() & end_ts.notna() & (delta_hours > 0)

    base_cols = ["SHIPMENT_ID", "MASTER_SHIPMENT_ID", "CONTAINER_NUMBER"]
    if aggregation_level == "Master Shipment" and "CONTAINER_COUNT" in work.columns:
        base_cols.append("CONTAINER_COUNT")
    base_cols += [
        "CARRIER_NAME", "CARRIER_SCAC",
        "POL_LOCODE", "POL", "POL_COUNTRY", "POL_FULL",
        "POD_LOCODE", "POD", "POD_COUNTRY", "POD_FULL",
        "LANE", "SHIPMENT_CREATED_DATE",
    ]

    calc = work.loc[valid_mask, base_cols + ["START_TS", "END_TS", "_DELTA_H"]].copy()
    calc.rename(columns={"START_TS": f"{start_code}_TS", "END_TS": f"{end_code}_TS"}, inplace=True)
    d_series, h_series = hours_to_components_vec(calc["_DELTA_H"])
    calc[COL_TRANSIT_DAYS]  = d_series
    calc[COL_TRANSIT_HOURS] = h_series
    calc[COL_TOTAL_HOURS]   = calc["_DELTA_H"].round(2)
    calc.drop(columns=["_DELTA_H"], inplace=True)
    calc.sort_values(["LANE", "CARRIER_NAME", "SHIPMENT_CREATED_DATE"], inplace=True)
    calc.reset_index(drop=True, inplace=True)

    missed = work.loc[~valid_mask, base_cols + ["MISSED_REASON"]].copy()
    missed.reset_index(drop=True, inplace=True)
    return calc, missed


# ============================================================================
# Lane × carrier summary
# ============================================================================

def _stats_dict(label_lane, cname, cscac, agg_row, vol):
    return {
        "LANE": label_lane, "CARRIER_NAME": cname, "CARRIER_SCAC": cscac,
        "VOLUME": int(vol),
        "MIN_DAYS":     hours_to_components(agg_row["min"])[0],
        "MIN_HOURS":    hours_to_components(agg_row["min"])[1],
        "MAX_DAYS":     hours_to_components(agg_row["max"])[0],
        "MAX_HOURS":    hours_to_components(agg_row["max"])[1],
        "AVG_DAYS":     hours_to_components(agg_row["mean"])[0],
        "AVG_HOURS":    hours_to_components(agg_row["mean"])[1],
        "MEDIAN_DAYS":  hours_to_components(agg_row["median"])[0],
        "MEDIAN_HOURS": hours_to_components(agg_row["median"])[1],
        "MIN_TOTAL_HOURS":    round(agg_row["min"], 2),
        "MAX_TOTAL_HOURS":    round(agg_row["max"], 2),
        "AVG_TOTAL_HOURS":    round(agg_row["mean"], 2),
        "MEDIAN_TOTAL_HOURS": round(agg_row["median"], 2),
    }


@st.cache_data(show_spinner=False, max_entries=8)
def build_lane_summary(calc: pd.DataFrame, _ver: str = CODE_VERSION) -> pd.DataFrame:
    if calc.empty or COL_TOTAL_HOURS not in calc.columns:
        return pd.DataFrame()
    lane_agg = (calc.groupby("LANE")[COL_TOTAL_HOURS]
                .agg(["count", "min", "max", "mean", "median"])
                .sort_values("count", ascending=False))
    carrier_agg = (calc.groupby(["LANE", "CARRIER_NAME", "CARRIER_SCAC"], dropna=False)[COL_TOTAL_HOURS]
                   .agg(["count", "min", "max", "mean", "median"]))
    rows: List[dict] = []
    for lane, lstats in lane_agg.iterrows():
        rows.append(_stats_dict(lane, "ALL CARRIERS", "ALL", lstats, lstats["count"]))
        try:
            sub = carrier_agg.loc[lane].sort_values("count", ascending=False)
        except KeyError:
            continue
        for (cname, cscac), cstats in sub.iterrows():
            rows.append(_stats_dict("", cname, cscac, cstats, cstats["count"]))
    return pd.DataFrame(rows)


# ============================================================================
# Trend data — transit (avg/median) + std dev
# ============================================================================

def _metric_prefix(metric: str) -> str:
    return "AVG" if metric == "Average" else "MEDIAN"


def _trend_col_names(metric: str) -> Tuple[str, str, str, str]:
    p = _metric_prefix(metric)
    return f"{p}_DAYS", f"{p}_HOURS", f"{p}_TOTAL_HOURS", f"{p}_DECIMAL_DAYS"


def _container_volume(df: pd.DataFrame) -> pd.Series:
    """One value per row representing how many containers it covers.
    Container mode: always 1. MSID mode: CONTAINER_COUNT."""
    if "CONTAINER_COUNT" in df.columns:
        return df["CONTAINER_COUNT"].fillna(1).astype(int)
    return pd.Series(1, index=df.index, dtype=int)


def _bucket(s: pd.Series, agg: str) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if agg == "Monthly":
        return s.dt.to_period("M").dt.to_timestamp()
    return s.dt.to_period("W").dt.start_time


def _build_note(shipment_count: int) -> str:
    if shipment_count == 0:
        return "No shipments in this bucket"
    if shipment_count == 1:
        return "1 shipment only — std dev N/A"
    return ""


@st.cache_data(show_spinner=False, max_entries=64)
def build_transit_frame(
    calc: pd.DataFrame, lane_selection: str,
    carrier_selection: Tuple[str, ...], metric: str, aggregation: str,
    _ver: str = CODE_VERSION,
) -> pd.DataFrame:
    """
    Long-form: one row per (bucket × series). SERIES is the carrier name OR
    'Combined (all selected)'. Columns include the metric value split into
    days+hours plus a single-number TOTAL_HOURS, decimal days, bucket-wide
    container volume, this-series shipment count, and a NOTES column.
    """
    if calc.empty or COL_TOTAL_HOURS not in calc.columns:
        return pd.DataFrame()

    df = calc.copy()
    df = df.dropna(subset=["SHIPMENT_CREATED_DATE"])
    if lane_selection != "All Lanes":
        df = df[df["LANE"] == lane_selection]
    carriers = list(carrier_selection)
    if "All Carriers" in carriers:
        carriers = sorted(df["CARRIER_NAME"].dropna().unique().tolist())
    df = df[df["CARRIER_NAME"].isin(carriers)]
    if df.empty:
        return pd.DataFrame()

    df["BUCKET"] = _bucket(df["SHIPMENT_CREATED_DATE"], aggregation)
    df["_VOL"]   = _container_volume(df)

    agg_func = "mean" if metric == "Average" else "median"
    days_col, hours_col, total_col, dec_col = _trend_col_names(metric)

    # Per-bucket total container volume (sum across all selected carriers)
    bucket_vol = df.groupby("BUCKET")["_VOL"].sum().rename("TOTAL_BUCKET_VOLUME")

    # Per-carrier rows
    per_c = (
        df.groupby(["BUCKET", "CARRIER_NAME"])
          .agg(SHIPMENT_COUNT=("_VOL", "sum"),
               _val=(COL_TOTAL_HOURS, agg_func))
          .reset_index()
          .rename(columns={"CARRIER_NAME": "SERIES", "_val": total_col})
    )
    per_c["IS_COMBINED"] = False

    # Combined rows
    comb = (
        df.groupby("BUCKET")
          .agg(SHIPMENT_COUNT=("_VOL", "sum"),
               _val=(COL_TOTAL_HOURS, agg_func))
          .reset_index()
          .rename(columns={"_val": total_col})
    )
    comb["SERIES"] = COMBINED_LABEL
    comb["IS_COMBINED"] = True

    out = pd.concat([per_c, comb], ignore_index=True)
    out = out.merge(bucket_vol, on="BUCKET", how="left")
    d_s, h_s = hours_to_components_vec(out[total_col])
    out[days_col]  = d_s
    out[hours_col] = h_s
    out[total_col] = out[total_col].round(2)
    out[dec_col]   = (out[total_col] / 24).round(3)
    out["NOTES"]   = [_build_note(int(c)) for c in out["SHIPMENT_COUNT"]]

    cols = ["BUCKET", "SERIES", "IS_COMBINED", "SHIPMENT_COUNT",
            "TOTAL_BUCKET_VOLUME", days_col, hours_col, total_col, dec_col, "NOTES"]
    out = out[cols].sort_values(["BUCKET", "IS_COMBINED", "SERIES"]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def build_stddev_frame(
    calc: pd.DataFrame, lane_selection: str,
    carrier_selection: Tuple[str, ...], aggregation: str,
    _ver: str = CODE_VERSION,
) -> pd.DataFrame:
    """
    Std dev frame, long form. STDEV_HOURS / STDEV_DAYS are NaN when
    SHIPMENT_COUNT < 2 (mathematically undefined). NOTES flags these.
    """
    if calc.empty or COL_TOTAL_HOURS not in calc.columns:
        return pd.DataFrame()

    df = calc.copy()
    df = df.dropna(subset=["SHIPMENT_CREATED_DATE"])
    if lane_selection != "All Lanes":
        df = df[df["LANE"] == lane_selection]
    carriers = list(carrier_selection)
    if "All Carriers" in carriers:
        carriers = sorted(df["CARRIER_NAME"].dropna().unique().tolist())
    df = df[df["CARRIER_NAME"].isin(carriers)]
    if df.empty:
        return pd.DataFrame()

    df["BUCKET"] = _bucket(df["SHIPMENT_CREATED_DATE"], aggregation)
    df["_VOL"]   = _container_volume(df)
    bucket_vol = df.groupby("BUCKET")["_VOL"].sum().rename("TOTAL_BUCKET_VOLUME")

    per_c = (
        df.groupby(["BUCKET", "CARRIER_NAME"])
          .agg(SHIPMENT_COUNT=("_VOL", "sum"),
               STDEV_HOURS=(COL_TOTAL_HOURS, "std"))
          .reset_index()
          .rename(columns={"CARRIER_NAME": "SERIES"})
    )
    per_c["IS_COMBINED"] = False

    comb = (
        df.groupby("BUCKET")
          .agg(SHIPMENT_COUNT=("_VOL", "sum"),
               STDEV_HOURS=(COL_TOTAL_HOURS, "std"))
          .reset_index()
    )
    comb["SERIES"] = COMBINED_LABEL
    comb["IS_COMBINED"] = True

    out = pd.concat([per_c, comb], ignore_index=True)
    out = out.merge(bucket_vol, on="BUCKET", how="left")
    out["STDEV_HOURS"]        = out["STDEV_HOURS"].round(2)
    out["STDEV_DECIMAL_DAYS"] = (out["STDEV_HOURS"] / 24).round(3)
    out["NOTES"]              = [_build_note(int(c)) for c in out["SHIPMENT_COUNT"]]

    out = out[["BUCKET", "SERIES", "IS_COMBINED", "SHIPMENT_COUNT",
               "TOTAL_BUCKET_VOLUME", "STDEV_HOURS", "STDEV_DECIMAL_DAYS", "NOTES"]]
    out.sort_values(["BUCKET", "IS_COMBINED", "SERIES"], inplace=True)
    return out.reset_index(drop=True)


# ============================================================================
# Figures
# ============================================================================

def _carrier_color_map(series_names: List[str]) -> dict:
    pure = [s for s in series_names if s != COMBINED_LABEL]
    return {name: CARRIER_PALETTE[i % len(CARRIER_PALETTE)] for i, name in enumerate(pure)}


def _add_volume_bars(fig, bucket_vol_df, name="Container volume"):
    fig.add_trace(
        go.Bar(
            x=bucket_vol_df["BUCKET"], y=bucket_vol_df["TOTAL_BUCKET_VOLUME"],
            name=name,
            marker=dict(color=VOLUME_COLOR, opacity=0.55),
            hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>%{y} containers<extra></extra>",
        ),
        secondary_y=True,
    )


def make_transit_figure(
    transit_df: pd.DataFrame, start_code: str, end_code: str,
    aggregation_level: str, metric: str, agg_period: str, lane_label: str,
    show_volume: bool, show_combined: bool, show_per_carrier: bool,
) -> go.Figure:
    days_col, hours_col, total_col, dec_col = _trend_col_names(metric)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bucket_vol = (
        transit_df.drop_duplicates("BUCKET")[["BUCKET", "TOTAL_BUCKET_VOLUME"]]
        .sort_values("BUCKET")
    )
    if show_volume:
        _add_volume_bars(fig, bucket_vol)

    color_map = _carrier_color_map(transit_df["SERIES"].unique().tolist())

    if show_per_carrier:
        for name in [s for s in transit_df["SERIES"].unique() if s != COMBINED_LABEL]:
            sub = transit_df[transit_df["SERIES"] == name].sort_values("BUCKET")
            fig.add_trace(
                go.Scatter(
                    x=sub["BUCKET"], y=sub[dec_col],
                    name=name, mode="lines+markers",
                    line=dict(color=color_map[name], width=2.5),
                    marker=dict(size=7),
                    customdata=sub[[days_col, hours_col, total_col, "SHIPMENT_COUNT"]].to_numpy(),
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        f"{metric} transit per shipment: %{{customdata[0]}}d %{{customdata[1]}}h "
                        "(%{customdata[2]:.2f} total hrs)<br>"
                        "n = %{customdata[3]} containers<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    if show_combined:
        comb = transit_df[transit_df["SERIES"] == COMBINED_LABEL].sort_values("BUCKET")
        if not comb.empty:
            fig.add_trace(
                go.Scatter(
                    x=comb["BUCKET"], y=comb[dec_col],
                    name=f"Combined {metric.lower()} (all selected)",
                    mode="lines+markers",
                    line=dict(color=COMBINED_COLOR, width=4),
                    marker=dict(size=10, symbol="diamond"),
                    customdata=comb[[days_col, hours_col, total_col, "SHIPMENT_COUNT"]].to_numpy(),
                    hovertemplate=(
                        f"<b>Combined {metric.lower()}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        f"{metric} transit per shipment: %{{customdata[0]}}d %{{customdata[1]}}h "
                        "(%{customdata[2]:.2f} total hrs)<br>"
                        "n = %{customdata[3]} containers in bucket<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    fig.update_layout(
        title=dict(
            text=(f"<b>Transit time ({metric.lower()} per shipment)</b><br>"
                  f"<sub>{lane_label} · {start_code} → {end_code} · "
                  f"{agg_period.lower()} · {aggregation_level.lower()}-level</sub>"),
            x=0.5, xanchor="center",
        ),
        height=560, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=70, r=70, t=110, b=110),
        bargap=0.4,
    )
    fig.update_xaxes(title_text=f"{agg_period} bucket (anchor: SHIPMENT_CREATED_DATE)",
                     showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text=f"{metric} transit (days)", secondary_y=False,
                     rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="Container volume (bars)", secondary_y=True,
                     rangemode="tozero", showgrid=False)
    return fig


def make_stddev_figure(
    stddev_df: pd.DataFrame, start_code: str, end_code: str,
    aggregation_level: str, agg_period: str, lane_label: str,
    show_volume: bool, show_combined: bool, show_per_carrier: bool,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bucket_vol = (
        stddev_df.drop_duplicates("BUCKET")[["BUCKET", "TOTAL_BUCKET_VOLUME"]]
        .sort_values("BUCKET")
    )
    if show_volume:
        _add_volume_bars(fig, bucket_vol)

    color_map = _carrier_color_map(stddev_df["SERIES"].unique().tolist())

    def _split_for_plot(sub: pd.DataFrame):
        """Return two frames: regular (count>=2) and singletons (count==1)."""
        regular = sub[sub["SHIPMENT_COUNT"] >= 2].copy()
        single  = sub[sub["SHIPMENT_COUNT"] == 1].copy()
        # For singletons, std dev is N/A — plot a hollow marker at y=0 so the
        # user sees the carrier was active but can read "1 ship" in the hover.
        single["STDEV_DECIMAL_DAYS"] = 0.0
        return regular, single

    if show_per_carrier:
        for name in [s for s in stddev_df["SERIES"].unique() if s != COMBINED_LABEL]:
            sub = stddev_df[stddev_df["SERIES"] == name].sort_values("BUCKET")
            regular, single = _split_for_plot(sub)
            if not regular.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regular["BUCKET"], y=regular["STDEV_DECIMAL_DAYS"],
                        name=f"{name} std dev", mode="lines+markers",
                        line=dict(color=color_map[name], width=2, dash="dot"),
                        marker=dict(size=6),
                        customdata=regular[["STDEV_HOURS", "SHIPMENT_COUNT"]].to_numpy(),
                        hovertemplate=(
                            f"<b>{name} std dev</b><br>"
                            "%{x|%Y-%m-%d}<br>"
                            "±%{y:.2f} days  (%{customdata[0]:.2f} hrs)<br>"
                            "n = %{customdata[1]} containers<extra></extra>"
                        ),
                    ),
                    secondary_y=False,
                )
            if not single.empty:
                fig.add_trace(
                    go.Scatter(
                        x=single["BUCKET"], y=single["STDEV_DECIMAL_DAYS"],
                        name=f"{name} (1 shipment — std dev N/A)",
                        mode="markers",
                        marker=dict(size=10, symbol="circle-open",
                                    color=color_map[name], line=dict(width=2)),
                        hovertemplate=(
                            f"<b>{name}</b><br>"
                            "%{x|%Y-%m-%d}<br>"
                            "1 shipment only — std dev N/A<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    secondary_y=False,
                )

    if show_combined:
        comb = stddev_df[stddev_df["SERIES"] == COMBINED_LABEL].sort_values("BUCKET")
        if not comb.empty:
            regular, single = _split_for_plot(comb)
            if not regular.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regular["BUCKET"], y=regular["STDEV_DECIMAL_DAYS"],
                        name="Combined std dev (all selected)",
                        mode="lines+markers",
                        line=dict(color=COMBINED_COLOR, width=4, dash="dash"),
                        marker=dict(size=10, symbol="x"),
                        customdata=regular[["STDEV_HOURS", "SHIPMENT_COUNT"]].to_numpy(),
                        hovertemplate=(
                            "<b>Combined std dev</b><br>"
                            "%{x|%Y-%m-%d}<br>"
                            "±%{y:.2f} days  (%{customdata[0]:.2f} hrs)<br>"
                            "n = %{customdata[1]} containers in bucket<extra></extra>"
                        ),
                    ),
                    secondary_y=False,
                )
            if not single.empty:
                fig.add_trace(
                    go.Scatter(
                        x=single["BUCKET"], y=single["STDEV_DECIMAL_DAYS"],
                        name="Combined (1 shipment — std dev N/A)",
                        mode="markers",
                        marker=dict(size=12, symbol="circle-open",
                                    color=COMBINED_COLOR, line=dict(width=2)),
                        hovertemplate=("<b>Combined</b><br>%{x|%Y-%m-%d}<br>"
                                       "1 shipment only — std dev N/A<extra></extra>"),
                        showlegend=False,
                    ),
                    secondary_y=False,
                )

    fig.update_layout(
        title=dict(
            text=(f"<b>Transit-time consistency (std dev)</b><br>"
                  f"<sub>Lower = more consistent · {lane_label} · "
                  f"{start_code} → {end_code} · {agg_period.lower()} · "
                  f"{aggregation_level.lower()}-level</sub>"),
            x=0.5, xanchor="center",
        ),
        height=560, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=70, r=70, t=110, b=110),
        bargap=0.4,
    )
    fig.update_xaxes(title_text=f"{agg_period} bucket (anchor: SHIPMENT_CREATED_DATE)",
                     showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="Std dev of transit (± days)", secondary_y=False,
                     rangemode="tozero", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="Container volume (bars)", secondary_y=True,
                     rangemode="tozero", showgrid=False)
    return fig


def fig_to_png_bytes(fig, width=1600, height=720, scale=2) -> bytes:
    return fig.to_image(format="png", width=width, height=height, scale=scale, engine="kaleido")


# ============================================================================
# Excel writers
# ============================================================================

def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            tz = getattr(out[col].dt, "tz", None)
            if tz is not None:
                out[col] = out[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def _autosize(writer, sheet_name: str, df: pd.DataFrame):
    """
    Column-width auto-sizer. Bulletproofed against any dtype/value weirdness —
    runs each value through str() inside the comprehension, and falls back to
    just the header length on any unexpected error so it can never crash the
    Excel write.
    """
    ws = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns):
        header_len = len(str(col))
        try:
            values = df[col].head(200).tolist()
            sample = [str(v) for v in values]
            max_len = max([header_len] + [len(s) for s in sample]) if sample else header_len
        except Exception:
            max_len = header_len
        ws.set_column(i, i, min(max_len + 2, 40))
    ws.freeze_panes(1, 0)


@st.cache_data(show_spinner=False, max_entries=8)
def write_excel(
    calc: pd.DataFrame, missed: pd.DataFrame, lane_sum: pd.DataFrame,
    start_code: str, end_code: str, aggregation_level: str, raw_row_count: int,
    _ver: str = CODE_VERSION,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        calc_out = _strip_tz(calc)
        calc_out.to_excel(writer, sheet_name="Calculated Transit", index=False)
        _autosize(writer, "Calculated Transit", calc_out)

        missed_out = _strip_tz(missed)
        missed_out.to_excel(writer, sheet_name="Missed Milestones", index=False)
        _autosize(writer, "Missed Milestones", missed_out)

        lane_sum.to_excel(writer, sheet_name="Lane x Carrier Summary", index=False)
        _autosize(writer, "Lane x Carrier Summary", lane_sum)

        meta = pd.DataFrame({
            "Field": [
                "Start milestone", "End milestone", "Aggregation level",
                "Raw input rows", "Calculated rows", "Missed rows",
                "Unique lanes", "Generated at (UTC)", "Code version",
            ],
            "Value": [
                MILESTONE_LABELS.get(start_code, start_code),
                MILESTONE_LABELS.get(end_code, end_code),
                aggregation_level,
                raw_row_count, len(calc), len(missed),
                calc["LANE"].nunique() if not calc.empty else 0,
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                CODE_VERSION,
            ],
        })
        meta.to_excel(writer, sheet_name="Run Info", index=False)
        _autosize(writer, "Run Info", meta)
    buf.seek(0)
    return buf.getvalue()


# ---- Per-lane trend exports (6 sheets + Key) ----

def _key_sheet_df() -> pd.DataFrame:
    rows = [
        ("HOW TO READ THIS FILE",
         "This Excel is the data behind the charts in the ZIP file sent with it. "
         "Open them side by side."),
        ("", ""),
        ("THE SHEETS IN THIS FILE", ""),
        ("Key", "This sheet. Quick guide to everything else."),
        ("Average-Weekly",   "Numbers behind transit_avg_weekly.png."),
        ("Average-Monthly",  "Numbers behind transit_avg_monthly.png."),
        ("Median-Weekly",    "Numbers behind transit_median_weekly.png."),
        ("Median-Monthly",   "Numbers behind transit_median_monthly.png."),
        ("StdDev-Weekly",    "Numbers behind stddev_weekly.png."),
        ("StdDev-Monthly",   "Numbers behind stddev_monthly.png."),
        ("", ""),
        ("HOW TO READ A TRANSIT SHEET (Average / Median)", ""),
        ("Each row", "One carrier (or 'Combined (all selected)') in one time bucket."),
        ("BUCKET", "Start date of the week or month."),
        ("SERIES", "Carrier name. 'Combined (all selected)' pools all carriers together."),
        ("SHIPMENT_COUNT", "Containers this carrier had in this bucket."),
        ("TOTAL_BUCKET_VOLUME", "Containers in the bucket across ALL selected carriers."),
        ("AVG_DAYS / AVG_HOURS", "The average shown as days + hours (e.g., 42d 16h). NOT a sum."),
        ("AVG_TOTAL_HOURS", "The same average as a single number in hours (for sorting). "
                            "This is the average PER SHIPMENT, not a total."),
        ("AVG_DECIMAL_DAYS", "Same average as decimal days. This is what the chart plots."),
        ("NOTES", "Flags for edge cases (e.g., '1 shipment only — std dev N/A')."),
        ("", ""),
        ("HOW TO READ A STD DEV SHEET", ""),
        ("STDEV_HOURS", "Standard deviation of transit times in this bucket, in hours."),
        ("STDEV_DECIMAL_DAYS", "Same in decimal days (this is what the chart plots)."),
        ("If STDEV_* is blank", "The bucket had 0 or 1 shipments — std dev is undefined."),
        ("", ""),
        ("HOW TO READ THE PNG CHARTS", ""),
        ("transit_*.png", "Average or median transit time over time. "
                          "Colored lines = each carrier. Bold black diamond line = combined "
                          "across all selected carriers. Gray bars (right axis) = container volume."),
        ("Use it for", "Setting SAP lead times. Comparing carrier speed against the lane average."),
        ("stddev_*.png", "Variability of transit times over time. "
                         "Colored dotted lines = each carrier's std dev. Bold dashed black line = "
                         "combined std dev. Gray bars (right axis) = container volume. "
                         "LOWER = MORE CONSISTENT."),
        ("Use it for", "Picking reliable carriers. A carrier with low std dev is predictable."),
        ("Hollow markers", "Mean the carrier had only 1 shipment that bucket — std dev is N/A."),
        ("", ""),
        ("CARE", "If SHIPMENT_COUNT in a bucket is small (say <5), treat both avg and "
                 "std dev as noisy signals — small sample, low confidence."),
    ]
    return pd.DataFrame(rows, columns=["Field", "Explanation"])


def _format_key_sheet(writer, ws):
    """Light styling on the Key sheet: wider columns + wrap."""
    wb = writer.book
    header_fmt = wb.add_format({"bold": True, "bg_color": "#f0f2f5", "border": 1})
    field_fmt  = wb.add_format({"bold": True, "valign": "top"})
    body_fmt   = wb.add_format({"text_wrap": True, "valign": "top"})
    section_fmt = wb.add_format({"bold": True, "bg_color": "#e8eef5",
                                 "valign": "top"})
    ws.set_column("A:A", 32, field_fmt)
    ws.set_column("B:B", 95, body_fmt)
    ws.set_row(0, None, header_fmt)
    # Bold rows whose Field is uppercase (section headers)
    df = _key_sheet_df()
    for idx, val in enumerate(df["Field"].tolist(), start=1):
        if val and val == val.upper() and len(val) > 3:
            ws.set_row(idx, None, section_fmt)


def write_trend_pack_excel(
    calc: pd.DataFrame, lane_selection: str,
    carrier_selection: Tuple[str, ...],
) -> bytes:
    """6 data sheets + Key. Used both for the lane-specific zip companion
    and for the in-app 'all 4 / 6 combinations' download."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        key_df = _key_sheet_df()
        key_df.to_excel(w, sheet_name="Key", index=False)
        _format_key_sheet(w, w.sheets["Key"])
        w.sheets["Key"].freeze_panes(1, 0)

        for metric in ("Average", "Median"):
            for agg in ("Weekly", "Monthly"):
                tf = build_transit_frame(
                    calc, lane_selection, carrier_selection, metric, agg
                )
                sheet = f"{metric}-{agg}"[:31]
                if tf.empty:
                    pd.DataFrame({"info": ["No data for this combination."]}) \
                        .to_excel(w, sheet_name=sheet, index=False)
                else:
                    view = tf.drop(columns=["IS_COMBINED"], errors="ignore")
                    view.to_excel(w, sheet_name=sheet, index=False)
                    _autosize(w, sheet, view)

        for agg in ("Weekly", "Monthly"):
            sf = build_stddev_frame(calc, lane_selection, carrier_selection, agg)
            sheet = f"StdDev-{agg}"[:31]
            if sf.empty:
                pd.DataFrame({"info": ["No data for this combination."]}) \
                    .to_excel(w, sheet_name=sheet, index=False)
            else:
                view = sf.drop(columns=["IS_COMBINED"], errors="ignore")
                view.to_excel(w, sheet_name=sheet, index=False)
                _autosize(w, sheet, view)
    return buf.getvalue()


def write_trend_pack_zip(
    calc: pd.DataFrame, lane_selection: str,
    carrier_selection: Tuple[str, ...],
    start_code: str, end_code: str, aggregation_level: str,
    show_volume=True, show_combined=True, show_per_carrier=True,
) -> bytes:
    """6 PNGs: 4 transit (avg/median × weekly/monthly) + 2 std-dev (weekly/monthly)."""
    lane_label = "All Lanes" if lane_selection == "All Lanes" else lane_selection
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for metric in ("Average", "Median"):
            for agg in ("Weekly", "Monthly"):
                tf = build_transit_frame(calc, lane_selection, carrier_selection, metric, agg)
                if tf.empty:
                    continue
                fig = make_transit_figure(
                    tf, start_code, end_code, aggregation_level,
                    metric, agg, lane_label,
                    show_volume, show_combined, show_per_carrier,
                )
                zf.writestr(f"transit_{metric.lower()}_{agg.lower()}.png", fig_to_png_bytes(fig))
        for agg in ("Weekly", "Monthly"):
            sf = build_stddev_frame(calc, lane_selection, carrier_selection, agg)
            if sf.empty:
                continue
            fig = make_stddev_figure(
                sf, start_code, end_code, aggregation_level,
                agg, lane_label,
                show_volume, show_combined, show_per_carrier,
            )
            zf.writestr(f"stddev_{agg.lower()}.png", fig_to_png_bytes(fig))
    return zbuf.getvalue()


def _safe_lane_name(lane: str) -> str:
    """
    Sanitize lane string into a filesystem-safe folder name.
    'VNHPH - Hai Phong, VN → USSAV - Savannah, US' → 'VNHPH_USSAV'
    Falls back to a sanitized full lane string if LOCODE parsing fails.
    """
    parts = lane.split(" → ")
    if len(parts) == 2:
        pol = parts[0].split(" - ")[0].strip()
        pod = parts[1].split(" - ")[0].strip()
        if pol and pod:
            return re.sub(r"[^A-Za-z0-9_-]", "_", f"{pol}_{pod}")
    return re.sub(r"[^A-Za-z0-9_-]", "_", lane)[:80]


def write_bulk_lane_pack_zip(
    calc: pd.DataFrame, lanes: List[str],
    start_code: str, end_code: str, aggregation_level: str,
    show_volume: bool = True, show_combined: bool = True, show_per_carrier: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    master_excel_bytes: Optional[bytes] = None,
) -> bytes:
    """
    Build ONE master ZIP containing per-lane folders. Each lane folder holds
    the 6 PNGs (4 transit + 2 std-dev) plus the data Excel (Key + 6 sheets).
    Lanes stay independent — each pack uses the carriers active on THAT lane.

    If `master_excel_bytes` is provided, it's dropped at the root of the ZIP
    as `00_master_report.xlsx` so the bulk pack carries the full deliverable.
    """
    zbuf = io.BytesIO()
    skipped: List[str] = []
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as outer:
        for i, lane in enumerate(lanes):
            if progress_callback:
                progress_callback(i, len(lanes), lane)

            lane_calc = calc[calc["LANE"] == lane]
            if lane_calc.empty:
                skipped.append(lane)
                continue
            lane_carriers = tuple(sorted(
                lane_calc["CARRIER_NAME"].dropna().unique().tolist()
            ))
            if not lane_carriers:
                skipped.append(lane)
                continue

            folder = _safe_lane_name(lane) + "/"

            # 4 transit PNGs
            for metric in ("Average", "Median"):
                for agg in ("Weekly", "Monthly"):
                    tf = build_transit_frame(calc, lane, lane_carriers, metric, agg)
                    if tf.empty:
                        continue
                    fig = make_transit_figure(
                        tf, start_code, end_code, aggregation_level,
                        metric, agg, lane,
                        show_volume, show_combined, show_per_carrier,
                    )
                    outer.writestr(
                        folder + f"transit_{metric.lower()}_{agg.lower()}.png",
                        fig_to_png_bytes(fig),
                    )

            # 2 std-dev PNGs
            for agg in ("Weekly", "Monthly"):
                sf = build_stddev_frame(calc, lane, lane_carriers, agg)
                if sf.empty:
                    continue
                fig = make_stddev_figure(
                    sf, start_code, end_code, aggregation_level,
                    agg, lane,
                    show_volume, show_combined, show_per_carrier,
                )
                outer.writestr(
                    folder + f"stddev_{agg.lower()}.png",
                    fig_to_png_bytes(fig),
                )

            # Excel (Key + 6 sheets)
            excel_bytes = write_trend_pack_excel(calc, lane, lane_carriers)
            outer.writestr(folder + "data_with_key.xlsx", excel_bytes)

        # Master Excel at root (if provided)
        if master_excel_bytes is not None:
            outer.writestr("00_master_report.xlsx", master_excel_bytes)

        # Root README.txt
        included = [l for l in lanes if l not in skipped]
        readme_lines = [
            "Bulk lane export",
            "================",
            f"Generated:        {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Milestone pair:   {start_code} → {end_code}",
            f"Aggregation:      {aggregation_level}",
            f"Lanes included:   {len(included)}",
            "",
            "Files at the root:",
        ]
        if master_excel_bytes is not None:
            readme_lines += [
                "  • 00_master_report.xlsx        — master report across ALL lanes",
                "                                   (Calculated, Missed, Lane × Carrier",
                "                                    Summary, Run Info)",
            ]
        readme_lines += [
            "  • README.txt                   — this file",
            "",
            "Each folder is one lane. Inside each folder:",
            "  • transit_average_weekly.png    — avg transit trend, weekly",
            "  • transit_average_monthly.png   — avg transit trend, monthly",
            "  • transit_median_weekly.png     — median transit trend, weekly",
            "  • transit_median_monthly.png    — median transit trend, monthly",
            "  • stddev_weekly.png             — transit consistency, weekly",
            "  • stddev_monthly.png            — transit consistency, monthly",
            "  • data_with_key.xlsx            — Key sheet + 6 data sheets",
            "",
            "Open data_with_key.xlsx first — the Key sheet explains how to read",
            "every chart and every column.",
            "",
            "Lanes:",
        ]
        for lane in included:
            readme_lines.append(f"  • {lane}  →  {_safe_lane_name(lane)}/")
        if skipped:
            readme_lines.append("")
            readme_lines.append("Skipped (no eligible shipments after filtering):")
            for lane in skipped:
                readme_lines.append(f"  • {lane}")
        outer.writestr("README.txt", "\n".join(readme_lines))
    return zbuf.getvalue()


def _filename(prefix, lane, carriers, ext, suffix=""):
    lane_part = "all-lanes" if lane == "All Lanes" else lane.split(" - ")[0].lower()
    if "All Carriers" in carriers:
        carr_part = "all-carriers"
    elif len(carriers) == 1:
        carr_part = carriers[0].lower().replace(" ", "-")
    else:
        carr_part = f"{len(carriers)}-carriers"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, lane_part, carr_part]
    if suffix:
        parts.append(suffix)
    parts.append(ts)
    return "_".join(parts) + "." + ext


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Ocean Transit Analyzer", layout="wide")
st.title("Ocean Shipment Transit Analyzer")
st.caption(
    "Upload a raw shipment file, pick start/end milestones and aggregation level — "
    "get per-shipment transit times, a lane × carrier breakdown, and interactive "
    "trend + consistency charts."
)

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader(
        "Raw shipment file (CSV or XLSX)",
        type=["csv", "xlsx", "xls"],
        help="Must contain SHIPMENT_ID, MASTER_SHIPMENT_ID, CONTAINER_NUMBER, "
             "CARRIER_NAME/SCAC, POL/POD fields, SHIPMENT_CREATED_DATE, and milestone "
             "columns (CEP, CGI, CLL, VDL, VAD, CDD, CGO, CER).",
    )

    st.header("2. Configure")
    agg_level = st.radio(
        "Aggregation level",
        options=["Container", "Master Shipment"],
        index=0,
        help=("**Container** = one row per CONTAINER_NUMBER.\n\n"
              "**Master Shipment** = collapse containers under the same "
              "MASTER_SHIPMENT_ID into a single row."),
    )
    start_code = st.selectbox("Transit START milestone", options=MILESTONE_ORDER, index=2,
                              format_func=lambda c: MILESTONE_LABELS[c])
    end_code = st.selectbox("Transit END milestone", options=MILESTONE_ORDER, index=4,
                            format_func=lambda c: MILESTONE_LABELS[c])
    st.caption("ℹ️ VDL/VAD auto-fall back to their _P44 variants when null.")

    run = st.button("Run analysis", type="primary", use_container_width=True)

    with st.expander("Troubleshooting"):
        st.caption("If you upgrade the app and see a KeyError on cached data, hit "
                   "Reset to clear caches + session and start fresh.")
        if st.button("🔄 Reset all caches & session", use_container_width=True):
            st.cache_data.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.caption(f"Code version: `{CODE_VERSION}`")

if uploaded is None:
    st.info("⬅️ Upload a CSV or XLSX file to begin.")
    st.stop()

try:
    file_bytes = uploaded.getvalue()
    raw = load_file(file_bytes, uploaded.name)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

missing = validate_columns(raw)
if missing:
    st.error(f"File is missing required columns: {', '.join(missing)}")
    st.stop()

if start_code == end_code:
    st.warning("Start and end milestones are the same — transit will be 0 for every shipment.")

current_params = (CODE_VERSION, _bytes_hash(file_bytes), start_code, end_code, agg_level)
prev_params = st.session_state.get("results", {}).get("params")

if not run and prev_params is None:
    st.success(f"Loaded {len(raw):,} rows. Configure in the sidebar, then click **Run analysis**.")
    with st.expander("Preview raw data (first 20 rows)"):
        st.dataframe(raw.head(20), use_container_width=True)
    st.stop()

needs_run = run or prev_params != current_params
if needs_run:
    with st.spinner("Computing transits…"):
        try:
            calc, missed = compute_transit(raw, start_code, end_code, agg_level)
            lane_sum = build_lane_summary(calc)
        except Exception as e:
            st.error(f"Computation failed: {e}")
            st.stop()
        if COL_TOTAL_HOURS not in calc.columns and not calc.empty:
            st.error("Internal error: missing TOTAL_HOURS column. Try Reset in the sidebar.")
            st.stop()
        st.session_state["results"] = {
            "calc": calc, "missed": missed, "lane_sum": lane_sum,
            "start_code": start_code, "end_code": end_code,
            "agg_level": agg_level, "raw_count": len(raw),
            "params": current_params,
        }

results    = st.session_state["results"]
calc       = results["calc"]
missed     = results["missed"]
lane_sum   = results["lane_sum"]
start_code = results["start_code"]
end_code   = results["end_code"]
agg_level  = results["agg_level"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Input rows ({agg_level.lower()}-level)",
          f"{(len(calc) + len(missed)):,}",
          delta=f"raw: {results['raw_count']:,}", delta_color="off")
c2.metric("Calculated", f"{len(calc):,}")
c3.metric("Missed / ineligible", f"{len(missed):,}")
c4.metric("Unique lanes", f"{calc['LANE'].nunique() if not calc.empty else 0:,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "✅ Calculated Transit",
    "⚠️ Missed Milestones",
    "🛣️ Lane × Carrier Summary",
    "📈 Trends & Consistency",
])

with tab1:
    st.subheader(f"Transit: {start_code} → {end_code}  ·  level: {agg_level}")
    st.caption("`TRANSIT_DAYS` + `TRANSIT_HOURS` are the d+h split. "
               "`TOTAL_HOURS` is the same transit as one numeric value in hours.")
    if len(calc) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing first {MAX_PREVIEW_ROWS:,} of {len(calc):,} rows.")
    st.dataframe(calc.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Shipments excluded from transit calculation")
    if not missed.empty:
        st.bar_chart(missed["MISSED_REASON"].value_counts())
    if len(missed) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing first {MAX_PREVIEW_ROWS:,} of {len(missed):,} rows.")
    st.dataframe(missed.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Per-lane breakdown")
    st.caption("First row per lane block = ALL CARRIERS combined. Then one row per "
               "carrier (descending volume). `AVG_TOTAL_HOURS` = average per shipment "
               "(NOT a sum across shipments).")
    st.dataframe(lane_sum, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Trends & Consistency")
    st.caption("Two charts: **Chart 1** = transit time (set SAP lead time, compare "
               "carriers). **Chart 2** = consistency / std dev (pick reliable carriers).")

    if calc.empty:
        st.info("No calculated transits to plot.")
    else:
        lanes_sorted = (
            calc.groupby("LANE").size().sort_values(ascending=False).index.tolist()
        )
        col_a, col_b, col_c, col_d = st.columns([2, 2, 1, 1])

        with col_a:
            lane_choice = st.selectbox("Lane", options=["All Lanes"] + lanes_sorted, index=0)

        if lane_choice == "All Lanes":
            carrier_pool = sorted(calc["CARRIER_NAME"].dropna().unique().tolist())
        else:
            carrier_pool = sorted(
                calc.loc[calc["LANE"] == lane_choice, "CARRIER_NAME"]
                    .dropna().unique().tolist()
            )

        with col_b:
            carrier_choice = st.multiselect(
                "Carrier(s)",
                options=["All Carriers"] + carrier_pool,
                default=["All Carriers"],
                help="‘All Carriers’ is mutually exclusive with specific carriers.",
            )
            if "All Carriers" in carrier_choice and len(carrier_choice) > 1:
                st.warning("‘All Carriers’ is mutually exclusive — using ‘All Carriers’ only.")
                carrier_choice = ["All Carriers"]
            if not carrier_choice:
                st.info("Select at least one carrier (or ‘All Carriers’).")
                st.stop()

        with col_c:
            metric_choice = st.radio("Metric", ["Average", "Median"])
        with col_d:
            agg_choice = st.radio("Aggregate", ["Weekly", "Monthly"])

        with st.expander("Chart overlays", expanded=True):
            ov_c1, ov_c2, ov_c3 = st.columns(3)
            with ov_c1:
                show_volume = st.checkbox("Container volume (bars)", value=True)
            with ov_c2:
                show_combined = st.checkbox("Combined line (all selected)", value=True)
            with ov_c3:
                show_per_carrier = st.checkbox("Per-carrier lines", value=True)

        # Carriers list — expand "All Carriers" to actual names so combined is meaningful
        if "All Carriers" in carrier_choice:
            effective_carriers = tuple(carrier_pool)
        else:
            effective_carriers = tuple(carrier_choice)

        lane_label = lane_choice if lane_choice != "All Lanes" else "All Lanes"

        transit_df = build_transit_frame(
            calc, lane_choice, effective_carriers, metric_choice, agg_choice
        )
        stddev_df = build_stddev_frame(
            calc, lane_choice, effective_carriers, agg_choice
        )

        if transit_df.empty:
            st.info("No data for this lane × carrier combination.")
        else:
            # CHART 1 — Transit time
            fig1 = make_transit_figure(
                transit_df, start_code, end_code, agg_level, metric_choice, agg_choice,
                lane_label, show_volume, show_combined, show_per_carrier,
            )
            st.plotly_chart(fig1, use_container_width=True)

            # CHART 2 — Std dev
            fig2 = make_stddev_figure(
                stddev_df, start_code, end_code, agg_level, agg_choice,
                lane_label, show_volume, show_combined, show_per_carrier,
            )
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Underlying data — how to read the columns", expanded=False):
                days_col, hours_col, total_col, dec_col = _trend_col_names(metric_choice)
                st.markdown(
                    f"**Transit sheet columns**\n"
                    f"- **BUCKET** — start date of the {agg_choice.lower()} period.\n"
                    f"- **SERIES** — carrier name OR `{COMBINED_LABEL}` (pooled across all selected).\n"
                    f"- **SHIPMENT_COUNT** — containers from THIS carrier in this bucket.\n"
                    f"- **TOTAL_BUCKET_VOLUME** — containers in the bucket across ALL selected carriers.\n"
                    f"- **{days_col} / {hours_col}** — {metric_choice.lower()} as days + hours (e.g., `42d 16h`). NOT a sum.\n"
                    f"- **{total_col}** — {metric_choice.lower()} as a single number in hours. "
                    f"**Average per shipment**, not a total.\n"
                    f"- **{dec_col}** — same as decimal days (the chart Y-axis).\n"
                    f"- **NOTES** — flags for edge cases (e.g., `1 shipment only — std dev N/A`).\n\n"
                    f"**Std-dev sheet columns**\n"
                    f"- **STDEV_HOURS / STDEV_DECIMAL_DAYS** — spread of transit times in that bucket. "
                    f"Lower = more consistent. Blank when SHIPMENT_COUNT < 2."
                )
                st.markdown("**Transit data (current view)**")
                st.dataframe(
                    transit_df.drop(columns=["IS_COMBINED"]),
                    use_container_width=True, hide_index=True,
                )
                st.markdown("**Std dev data (current view)**")
                st.dataframe(
                    stddev_df.drop(columns=["IS_COMBINED"]),
                    use_container_width=True, hide_index=True,
                )

            # -------- Exports --------
            st.markdown("**Export charts / data**")
            scope = st.radio(
                "Scope",
                ["Current view only", "Full pack (6 charts + Key sheet)"],
                horizontal=True, key="trend_scope",
                help="Full pack = 4 transit PNGs (avg/median × weekly/monthly) + "
                     "2 std-dev PNGs (weekly/monthly), plus the Excel with all 6 "
                     "data sheets and a Key sheet.",
            )

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if scope == "Current view only":
                    try:
                        single_zip = io.BytesIO()
                        with zipfile.ZipFile(single_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                            zf.writestr(
                                f"transit_{metric_choice.lower()}_{agg_choice.lower()}.png",
                                fig_to_png_bytes(fig1),
                            )
                            zf.writestr(
                                f"stddev_{agg_choice.lower()}.png",
                                fig_to_png_bytes(fig2),
                            )
                        st.download_button(
                            "⬇️ Download both charts (PNG, zipped)",
                            data=single_zip.getvalue(),
                            file_name=_filename(
                                "charts", lane_choice, carrier_choice, "zip",
                                suffix=f"{metric_choice.lower()}-{agg_choice.lower()}",
                            ),
                            mime="application/zip",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.caption(f"PNG export failed: {e}")
                else:
                    with st.spinner("Rendering 6 charts…"):
                        zip_bytes = write_trend_pack_zip(
                            calc, lane_choice, effective_carriers,
                            start_code, end_code, agg_level,
                            show_volume, show_combined, show_per_carrier,
                        )
                    st.download_button(
                        "⬇️ Download all 6 charts as ZIP (PNG)",
                        data=zip_bytes,
                        file_name=_filename("trend-pack", lane_choice, carrier_choice, "zip"),
                        mime="application/zip",
                        use_container_width=True,
                    )

            with col_dl2:
                if scope == "Current view only":
                    # Two-sheet excel: current transit metric + current std-dev
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                        key_df = _key_sheet_df()
                        key_df.to_excel(w, sheet_name="Key", index=False)
                        _format_key_sheet(w, w.sheets["Key"])
                        w.sheets["Key"].freeze_panes(1, 0)

                        t_view = transit_df.drop(columns=["IS_COMBINED"], errors="ignore")
                        t_sheet = f"{metric_choice}-{agg_choice}"[:31]
                        t_view.to_excel(w, sheet_name=t_sheet, index=False)
                        _autosize(w, t_sheet, t_view)

                        s_view = stddev_df.drop(columns=["IS_COMBINED"], errors="ignore")
                        s_sheet = f"StdDev-{agg_choice}"[:31]
                        s_view.to_excel(w, sheet_name=s_sheet, index=False)
                        _autosize(w, s_sheet, s_view)
                    st.download_button(
                        "⬇️ Download current data as Excel (Key + 2 sheets)",
                        data=buf.getvalue(),
                        file_name=_filename(
                            "data", lane_choice, carrier_choice, "xlsx",
                            suffix=f"{metric_choice.lower()}-{agg_choice.lower()}",
                        ),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                else:
                    with st.spinner("Building 6-sheet Excel + Key…"):
                        xlsx_bytes = write_trend_pack_excel(
                            calc, lane_choice, effective_carriers
                        )
                    st.download_button(
                        "⬇️ Download full pack as Excel (Key + 6 sheets)",
                        data=xlsx_bytes,
                        file_name=_filename("trend-pack", lane_choice, carrier_choice, "xlsx"),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

        # =================================================================
        # Bulk lane export
        # =================================================================
        st.divider()
        st.subheader("📦 Bulk lane export")
        st.caption(
            "Generate ONE master ZIP that bundles a full pack for every lane "
            "you pick. Each lane gets its own folder inside the ZIP with the 6 "
            "PNGs + the data Excel (Key + 6 sheets). Lanes stay independent — "
            "no combining or averaging across lanes. Useful when you need to "
            "send a customer the same deliverable for many lanes at once."
        )

        bulk_pool = lanes_sorted

        include_all = st.checkbox(
            f"Include ALL {len(bulk_pool)} lanes (otherwise pick below)",
            value=False, key="bulk_include_all",
        )
        if include_all:
            bulk_lanes = bulk_pool
            st.caption(f"All {len(bulk_pool)} lanes will be included.")
        else:
            bulk_lanes = st.multiselect(
                "Lanes to include in the bulk pack",
                options=bulk_pool,
                default=[],
                help="Each selected lane becomes its own folder in the master ZIP. "
                     "All carriers active on each lane are included automatically.",
                key="bulk_lanes_select",
            )

        if bulk_lanes:
            # Per-lane PNG render is ~0.5s; 1 lane = 6 renders + 1 Excel ≈ 4s.
            est_seconds = max(3, int(len(bulk_lanes) * 4))
            est_str = (f"~{est_seconds}s" if est_seconds < 60
                       else f"~{est_seconds // 60}m {est_seconds % 60}s")
            st.caption(
                f"**{len(bulk_lanes)}** lane(s) selected · {len(bulk_lanes) * 6} "
                f"PNG renders + {len(bulk_lanes)} Excel files · est. {est_str}."
            )

            if st.button("🚀 Generate bulk pack", type="primary",
                         use_container_width=True, key="bulk_generate"):
                progress = st.progress(0.0, text="Starting…")
                status_box = st.empty()

                def _cb(i, total, lane):
                    pct = (i + 1) / total
                    progress.progress(pct, text=f"Lane {i+1} of {total}: {lane}")

                try:
                    # Build the master report Excel once and include it at the
                    # root of the bulk ZIP, so the customer-facing deliverable
                    # is fully self-contained in one download.
                    master_xlsx = write_excel(
                        calc, missed, lane_sum,
                        start_code, end_code, agg_level, results["raw_count"],
                    )
                    bulk_bytes = write_bulk_lane_pack_zip(
                        calc, bulk_lanes,
                        start_code, end_code, agg_level,
                        show_volume=show_volume, show_combined=show_combined,
                        show_per_carrier=show_per_carrier,
                        progress_callback=_cb,
                        master_excel_bytes=master_xlsx,
                    )
                except Exception as e:
                    st.error(f"Bulk pack build failed: {e}")
                    st.stop()
                progress.progress(1.0, text=f"Done — packed {len(bulk_lanes)} lanes.")
                size_mb = len(bulk_bytes) / 1024 / 1024
                status_box.success(
                    f"Built bulk pack for **{len(bulk_lanes)} lanes** "
                    f"({size_mb:.1f} MB). Download below."
                )
                st.session_state["bulk_pack"] = {
                    "bytes": bulk_bytes,
                    "lanes": list(bulk_lanes),
                    "params": current_params,  # invalidate on config change
                }

        # Persist download button between reruns
        bulk_pack = st.session_state.get("bulk_pack")
        if bulk_pack and bulk_pack.get("params") == current_params:
            st.download_button(
                f"⬇️ Download bulk pack ({len(bulk_pack['lanes'])} lanes, "
                f"{len(bulk_pack['bytes']) / 1024 / 1024:.1f} MB)",
                data=bulk_pack["bytes"],
                file_name=(
                    f"bulk_lane_pack_{len(bulk_pack['lanes'])}lanes_"
                    f"{start_code}-{end_code}_"
                    f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
                ),
                mime="application/zip",
                type="primary",
                use_container_width=True,
                key="bulk_download_btn",
            )

# ---- Main Excel download ----
st.divider()
st.subheader("Download main report")
with st.spinner("Preparing Excel…"):
    main_xlsx = write_excel(
        calc, missed, lane_sum,
        start_code, end_code, agg_level, results["raw_count"],
    )
fname = (
    f"transit_{start_code}_to_{end_code}_{agg_level.lower().replace(' ', '-')}_"
    f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
)
st.download_button(
    "⬇️ Download Excel (Calculated + Missed + Lane Summary + Run Info)",
    data=main_xlsx,
    file_name=fname,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
    use_container_width=True,
)
