"""
Ocean Shipment Transit Analyzer
================================
Streamlit app: ingest a raw shipment file, let the user pick a start and end
milestone (and aggregation level — container or master-shipment), compute
transit times, summarize by lane × carrier, and explore trends interactively.

Run locally:  streamlit run app.py
Deploy:       push to GitHub, connect repo on Streamlit Community Cloud.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================================
# Constants
# ============================================================================

# 8 ocean milestones. VDL and VAD auto-fall back to their _P44 variants when
# the carrier-reported value is null.
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

# Vivid, high-contrast palette for trend lines (works well in PNG export too)
TREND_COLORS = px.colors.qualitative.Bold + px.colors.qualitative.Vivid


# ============================================================================
# IO + caching
# ============================================================================

def _bytes_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@st.cache_data(show_spinner=False, max_entries=4)
def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Cached read. Hashed on file bytes so re-runs are instant."""
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
def aggregate_to_master_shipment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per MASTER_SHIPMENT_ID. `groupby().first()` picks the
    first non-null value in each column per master shipment, so when one
    container has a milestone and another doesn't, we keep the populated one.
    Adds CONTAINER_COUNT (how many containers were rolled up).
    """
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
    """Vectorized: fractional hours → (days, remainder_hours) as nullable Int64."""
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
        days += 1
        hours = 0
    return (int(days), int(hours))


# ============================================================================
# Transit calculation
# ============================================================================

@st.cache_data(show_spinner=False, max_entries=8)
def compute_transit(
    df: pd.DataFrame,
    start_code: str,
    end_code: str,
    aggregation_level: str,
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
    work["DURATION_HOURS_INTERNAL"] = delta_hours

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

    # --- Calculated ---
    calc = work.loc[
        valid_mask,
        base_cols + ["START_TS", "END_TS", "DURATION_HOURS_INTERNAL"],
    ].copy()
    calc.rename(columns={
        "START_TS": f"{start_code}_TS",
        "END_TS":   f"{end_code}_TS",
    }, inplace=True)

    d_series, h_series = hours_to_components_vec(calc["DURATION_HOURS_INTERNAL"])
    calc["TRANSIT_DAYS"] = d_series
    calc["TRANSIT_HOURS"] = h_series         # the integer-hour part of the d/h split
    calc["DURATION_HOURS"] = calc["DURATION_HOURS_INTERNAL"].round(2)  # numeric single-value
    calc.drop(columns=["DURATION_HOURS_INTERNAL"], inplace=True)

    calc.sort_values(["LANE", "CARRIER_NAME", "SHIPMENT_CREATED_DATE"], inplace=True)
    calc.reset_index(drop=True, inplace=True)

    # --- Missed ---
    missed = work.loc[~valid_mask, base_cols + ["MISSED_REASON"]].copy()
    missed.reset_index(drop=True, inplace=True)
    return calc, missed


# ============================================================================
# Lane × carrier summary
# ============================================================================

def _stats_dict(label_lane, cname, cscac, agg_row, vol):
    return {
        "LANE": label_lane,
        "CARRIER_NAME": cname,
        "CARRIER_SCAC": cscac,
        "VOLUME": int(vol),
        "MIN_DAYS":     hours_to_components(agg_row["min"])[0],
        "MIN_HOURS":    hours_to_components(agg_row["min"])[1],
        "MAX_DAYS":     hours_to_components(agg_row["max"])[0],
        "MAX_HOURS":    hours_to_components(agg_row["max"])[1],
        "AVG_DAYS":     hours_to_components(agg_row["mean"])[0],
        "AVG_HOURS":    hours_to_components(agg_row["mean"])[1],
        "MEDIAN_DAYS":  hours_to_components(agg_row["median"])[0],
        "MEDIAN_HOURS": hours_to_components(agg_row["median"])[1],
        "MIN_DURATION_HOURS":    round(agg_row["min"], 2),
        "MAX_DURATION_HOURS":    round(agg_row["max"], 2),
        "AVG_DURATION_HOURS":    round(agg_row["mean"], 2),
        "MEDIAN_DURATION_HOURS": round(agg_row["median"], 2),
    }


@st.cache_data(show_spinner=False, max_entries=8)
def build_lane_summary(calc: pd.DataFrame) -> pd.DataFrame:
    if calc.empty:
        return pd.DataFrame()

    lane_agg = (
        calc.groupby("LANE")["DURATION_HOURS"]
            .agg(["count", "min", "max", "mean", "median"])
            .sort_values("count", ascending=False)
    )
    carrier_agg = (
        calc.groupby(["LANE", "CARRIER_NAME", "CARRIER_SCAC"], dropna=False)["DURATION_HOURS"]
            .agg(["count", "min", "max", "mean", "median"])
    )

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
# Trend data
# ============================================================================

def _metric_prefix(metric: str) -> str:
    return "AVG" if metric == "Average" else "MEDIAN"


def _trend_col_names(metric: str) -> Tuple[str, str, str, str]:
    """Return (days_col, hours_col, duration_hours_col, decimal_days_col)."""
    p = _metric_prefix(metric)
    return f"{p}_DAYS", f"{p}_HOURS", f"{p}_DURATION_HOURS", f"{p}_DECIMAL_DAYS"


@st.cache_data(show_spinner=False, max_entries=64)
def build_trend_frame(
    calc: pd.DataFrame,
    lane_selection: str,
    carrier_selection: Tuple[str, ...],
    metric: str,
    aggregation: str,
) -> pd.DataFrame:
    """
    Aggregate per (time bucket, series). Columns are named based on metric
    so it's unambiguous what the number represents:
        AVG_DAYS / AVG_HOURS       — days+hours split of the average
        AVG_DURATION_HOURS         — the average expressed as a single hours number
        AVG_DECIMAL_DAYS           — same in decimal days (chart Y-axis)
        SHIPMENT_COUNT             — how many shipments fed into that bucket
    (Same for MEDIAN_* when metric == 'Median'.)
    """
    if calc.empty:
        return pd.DataFrame()

    df = calc.copy()
    df["SHIPMENT_CREATED_DATE"] = pd.to_datetime(df["SHIPMENT_CREATED_DATE"], errors="coerce")
    df = df.dropna(subset=["SHIPMENT_CREATED_DATE"])

    if lane_selection != "All Lanes":
        df = df[df["LANE"] == lane_selection]

    carriers = list(carrier_selection)
    if "All Carriers" in carriers:
        df = df.assign(_SERIES="All Carriers")
    else:
        df = df[df["CARRIER_NAME"].isin(carriers)]
        df = df.assign(_SERIES=df["CARRIER_NAME"])

    if df.empty:
        return pd.DataFrame()

    if aggregation == "Monthly":
        df["BUCKET"] = df["SHIPMENT_CREATED_DATE"].dt.to_period("M").dt.to_timestamp()
    else:
        df["BUCKET"] = df["SHIPMENT_CREATED_DATE"].dt.to_period("W").dt.start_time

    agg_func = "mean" if metric == "Average" else "median"
    days_col, hours_col, dur_col, dec_col = _trend_col_names(metric)

    grouped = (
        df.groupby(["BUCKET", "_SERIES"])["DURATION_HOURS"]
          .agg([agg_func, "count"])
          .rename(columns={agg_func: dur_col, "count": "SHIPMENT_COUNT"})
          .reset_index()
          .rename(columns={"_SERIES": "SERIES"})
    )
    d_s, h_s = hours_to_components_vec(grouped[dur_col])
    grouped[days_col] = d_s
    grouped[hours_col] = h_s
    grouped[dur_col] = grouped[dur_col].round(2)
    grouped[dec_col] = (grouped[dur_col] / 24).round(3)

    # Order columns nicely
    grouped = grouped[["BUCKET", "SERIES", "SHIPMENT_COUNT",
                       days_col, hours_col, dur_col, dec_col]]
    grouped.sort_values(["BUCKET", "SERIES"], inplace=True)
    return grouped


# ============================================================================
# Plotly figure builder
# ============================================================================

def make_trend_figure(
    trend: pd.DataFrame,
    start_code: str,
    end_code: str,
    aggregation_level: str,
    metric: str,
    agg_period: str,
):
    """Build the Plotly line chart from a trend frame."""
    days_col, hours_col, dur_col, dec_col = _trend_col_names(metric)

    fig = px.line(
        trend,
        x="BUCKET",
        y=dec_col,
        color="SERIES",
        markers=True,
        color_discrete_sequence=TREND_COLORS,
        labels={
            "BUCKET": f"{agg_period} bucket (anchor: SHIPMENT_CREATED_DATE)",
            dec_col: f"{metric} transit (days)",
            "SERIES": "Carrier",
        },
        title=(
            f"{metric} transit — {start_code} → {end_code} "
            f"({agg_period.lower()}, {aggregation_level.lower()}-level)"
        ),
    )
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Bucket: %{x|%Y-%m-%d}<br>"
            f"{metric} transit per shipment: "
            "%{customdata[0]}d %{customdata[1]}h "
            "(%{customdata[2]:.2f} hrs)<br>"
            "n = %{customdata[3]} shipments<extra></extra>"
        ),
        customdata=trend[[days_col, hours_col, dur_col, "SHIPMENT_COUNT"]].to_numpy(),
    )
    fig.update_layout(
        height=520,
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(title="Carrier", orientation="v"),
        margin=dict(l=60, r=40, t=70, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", rangemode="tozero")
    return fig


def fig_to_png_bytes(fig, width=1600, height=800, scale=2) -> bytes:
    """Render figure to PNG with vivid colors preserved."""
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
    ws = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns):
        sample = df[col].astype(str).head(200).tolist()
        max_len = max([len(str(col))] + [len(v) for v in sample]) if sample else len(str(col))
        ws.set_column(i, i, min(max_len + 2, 40))
    ws.freeze_panes(1, 0)


@st.cache_data(show_spinner=False, max_entries=8)
def write_excel(
    calc: pd.DataFrame, missed: pd.DataFrame, lane_sum: pd.DataFrame,
    start_code: str, end_code: str, aggregation_level: str, raw_row_count: int,
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
                "Unique lanes", "Generated at (UTC)",
            ],
            "Value": [
                MILESTONE_LABELS.get(start_code, start_code),
                MILESTONE_LABELS.get(end_code, end_code),
                aggregation_level,
                raw_row_count, len(calc), len(missed),
                calc["LANE"].nunique() if not calc.empty else 0,
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            ],
        })
        meta.to_excel(writer, sheet_name="Run Info", index=False)
        _autosize(writer, "Run Info", meta)
    buf.seek(0)
    return buf.getvalue()


def write_trend_single(trend: pd.DataFrame, metric: str, agg_period: str) -> bytes:
    """One-sheet Excel for the currently displayed trend."""
    buf = io.BytesIO()
    sheet = f"{metric}-{agg_period}"[:31]
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        trend.to_excel(w, sheet_name=sheet, index=False)
        _autosize(w, sheet, trend)
    return buf.getvalue()


def write_trend_all_four(
    calc: pd.DataFrame, lane_selection: str, carrier_selection: Tuple[str, ...],
) -> bytes:
    """4-sheet Excel: Average-Weekly, Average-Monthly, Median-Weekly, Median-Monthly."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for metric in ("Average", "Median"):
            for agg in ("Weekly", "Monthly"):
                tf = build_trend_frame(calc, lane_selection, carrier_selection, metric, agg)
                sheet = f"{metric}-{agg}"[:31]
                if tf.empty:
                    pd.DataFrame({"info": ["No data for this combination."]}) \
                        .to_excel(w, sheet_name=sheet, index=False)
                else:
                    tf.to_excel(w, sheet_name=sheet, index=False)
                    _autosize(w, sheet, tf)
    return buf.getvalue()


def write_all_four_png_zip(
    calc: pd.DataFrame, lane_selection: str, carrier_selection: Tuple[str, ...],
    start_code: str, end_code: str, aggregation_level: str,
) -> bytes:
    """Zip containing 4 PNG charts (all metric × period combos)."""
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for metric in ("Average", "Median"):
            for agg in ("Weekly", "Monthly"):
                tf = build_trend_frame(calc, lane_selection, carrier_selection, metric, agg)
                if tf.empty:
                    continue
                fig = make_trend_figure(tf, start_code, end_code, aggregation_level, metric, agg)
                png = fig_to_png_bytes(fig)
                fname = f"trend_{metric.lower()}_{agg.lower()}.png"
                zf.writestr(fname, png)
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
    "Upload a raw shipment file, pick start/end milestones and aggregation "
    "level — get per-shipment transit times, a lane × carrier breakdown, and "
    "interactive trends."
)

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader(
        "Raw shipment file (CSV or XLSX)",
        type=["csv", "xlsx", "xls"],
        help="Must contain SHIPMENT_ID, MASTER_SHIPMENT_ID, CONTAINER_NUMBER, "
             "CARRIER_NAME/SCAC, POL/POD fields, SHIPMENT_CREATED_DATE, and "
             "milestone columns (CEP, CGI, CLL, VDL, VAD, CDD, CGO, CER).",
    )

    st.header("2. Configure")
    agg_level = st.radio(
        "Aggregation level",
        options=["Container", "Master Shipment"],
        index=0,
        help=("**Container** = one row per CONTAINER_NUMBER.\n\n"
              "**Master Shipment** = collapse containers under the same "
              "MASTER_SHIPMENT_ID into a single row (first non-null milestone "
              "per group). A CONTAINER_COUNT column shows how many were rolled up."),
    )
    start_code = st.selectbox(
        "Transit START milestone",
        options=MILESTONE_ORDER, index=2,
        format_func=lambda c: MILESTONE_LABELS[c],
    )
    end_code = st.selectbox(
        "Transit END milestone",
        options=MILESTONE_ORDER, index=4,
        format_func=lambda c: MILESTONE_LABELS[c],
    )
    st.caption("ℹ️ VDL/VAD auto-fall back to their _P44 variants when null.")

    run = st.button("Run analysis", type="primary", use_container_width=True)

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

if not run and "results" not in st.session_state:
    st.success(f"Loaded {len(raw):,} rows. Configure in the sidebar, then click **Run analysis**.")
    with st.expander("Preview raw data (first 20 rows)"):
        st.dataframe(raw.head(20), use_container_width=True)
    st.stop()

needs_run = run or st.session_state.get("results", {}).get("params") != (
    _bytes_hash(file_bytes), start_code, end_code, agg_level
)
if needs_run:
    with st.spinner("Computing transits…"):
        try:
            calc, missed = compute_transit(raw, start_code, end_code, agg_level)
            lane_sum = build_lane_summary(calc)
        except Exception as e:
            st.error(f"Computation failed: {e}")
            st.stop()
        st.session_state["results"] = {
            "calc": calc, "missed": missed, "lane_sum": lane_sum,
            "start_code": start_code, "end_code": end_code,
            "agg_level": agg_level, "raw_count": len(raw),
            "params": (_bytes_hash(file_bytes), start_code, end_code, agg_level),
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
    "📈 Trends",
])

with tab1:
    st.subheader(f"Transit: {start_code} → {end_code}  ·  level: {agg_level}")
    st.caption("`DURATION_HOURS` is each shipment's transit expressed as a single "
               "number in hours (for sorting). `TRANSIT_DAYS` + `TRANSIT_HOURS` "
               "are the days + hours split (e.g., 36h → 1d 12h).")
    if len(calc) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing first {MAX_PREVIEW_ROWS:,} of {len(calc):,} rows. "
                   "Full data is in the Excel download below.")
    st.dataframe(calc.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Shipments excluded from transit calculation")
    st.caption("Missing milestone(s) or end timestamp not after start.")
    if not missed.empty:
        st.bar_chart(missed["MISSED_REASON"].value_counts())
    if len(missed) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing first {MAX_PREVIEW_ROWS:,} of {len(missed):,} rows.")
    st.dataframe(missed.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Per-lane breakdown")
    st.caption("First row per lane block = ALL CARRIERS for that lane. "
               "Following rows = each carrier (ordered by descending volume). "
               "`AVG_DURATION_HOURS` is the average transit (single number, in hours) "
               "across shipments in that group — same for MIN / MAX / MEDIAN.")
    st.dataframe(lane_sum, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Transit trend over time")
    st.caption("The carrier dropdown only lists carriers that have at least one "
               "**calculated** (eligible) shipment on the selected lane.")

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

        trend = build_trend_frame(
            calc, lane_choice, tuple(carrier_choice), metric_choice, agg_choice
        )

        if trend.empty:
            st.info("No data for this lane × carrier combination.")
        else:
            fig = make_trend_figure(
                trend, start_code, end_code, agg_level, metric_choice, agg_choice
            )
            st.plotly_chart(fig, use_container_width=True)

            # Inline reading guide for the columns the user will see in the data table
            days_col, hours_col, dur_col, dec_col = _trend_col_names(metric_choice)
            with st.expander("Underlying data — how to read the columns"):
                st.markdown(
                    f"- **BUCKET** — start date of the {agg_choice.lower()} period.\n"
                    f"- **SERIES** — carrier (or ‘All Carriers’ when aggregated).\n"
                    f"- **SHIPMENT_COUNT** — how many eligible shipments contributed to that point.\n"
                    f"- **`{days_col}` / `{hours_col}`** — the {metric_choice.lower()} expressed as "
                    f"days + hours (e.g., `42d 16h`). These are the d/h split, NOT a sum.\n"
                    f"- **`{dur_col}`** — the same {metric_choice.lower()} expressed as a single number in hours "
                    f"(e.g., 1024.46). This is the average/median **per shipment**, not a total across shipments.\n"
                    f"- **`{dec_col}`** — the same {metric_choice.lower()} as decimal days "
                    f"(this is what the chart plots on the Y-axis)."
                )
                st.dataframe(trend, use_container_width=True, hide_index=True)

            # -------- Download options --------
            st.markdown("**Export trend chart / data**")
            scope = st.radio(
                "Scope",
                ["Current view only", "All 4 combinations (Avg/Median × Weekly/Monthly)"],
                horizontal=True,
                key="trend_scope",
            )

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if scope == "Current view only":
                    try:
                        png_bytes = fig_to_png_bytes(fig)
                        st.download_button(
                            "⬇️ Download chart as PNG",
                            data=png_bytes,
                            file_name=_filename(
                                "trend", lane_choice, carrier_choice, "png",
                                suffix=f"{metric_choice.lower()}-{agg_choice.lower()}",
                            ),
                            mime="image/png",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.caption(f"PNG export failed: {e}")
                else:
                    with st.spinner("Rendering 4 charts…"):
                        zip_bytes = write_all_four_png_zip(
                            calc, lane_choice, tuple(carrier_choice),
                            start_code, end_code, agg_level,
                        )
                    st.download_button(
                        "⬇️ Download all 4 charts as ZIP (PNG)",
                        data=zip_bytes,
                        file_name=_filename("trend-all4", lane_choice, carrier_choice, "zip"),
                        mime="application/zip",
                        use_container_width=True,
                    )

            with col_dl2:
                if scope == "Current view only":
                    st.download_button(
                        "⬇️ Download trend data as Excel",
                        data=write_trend_single(trend, metric_choice, agg_choice),
                        file_name=_filename(
                            "trend", lane_choice, carrier_choice, "xlsx",
                            suffix=f"{metric_choice.lower()}-{agg_choice.lower()}",
                        ),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                else:
                    with st.spinner("Building 4-sheet Excel…"):
                        xlsx_bytes = write_trend_all_four(
                            calc, lane_choice, tuple(carrier_choice)
                        )
                    st.download_button(
                        "⬇️ Download all 4 as Excel (4 sheets)",
                        data=xlsx_bytes,
                        file_name=_filename("trend-all4", lane_choice, carrier_choice, "xlsx"),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
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
