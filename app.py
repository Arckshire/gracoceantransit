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
ALL_MILESTONE_COLS = sorted({c for _, _, srcs in MILESTONES for c in srcs})

# Minimum columns required for the tool to run. Milestone columns are checked
# at calculation time (a missing milestone column is treated as all-null,
# which just routes shipments to the missed sheet — the tool still runs).
REQUIRED_COLUMNS = {
    "SHIPMENT_ID", "MASTER_SHIPMENT_ID", "CONTAINER_NUMBER",
    "CARRIER_NAME", "CARRIER_SCAC",
    "POL_LOCODE", "POL", "POL_COUNTRY",
    "POD_LOCODE", "POD", "POD_COUNTRY",
    "SHIPMENT_CREATED_DATE",
}

# Cap how many rows we render in the on-screen preview tables. The full data
# always goes to the Excel download — this only protects the UI.
MAX_PREVIEW_ROWS = 10_000


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
    Collapse to one row per MASTER_SHIPMENT_ID.

    Strategy: groupby().first() (skipna=True) picks the first non-null value
    in each column per master shipment. Containers under the same master
    shipment normally share milestones; if one container has a milestone and
    another doesn't, we keep the populated value (more permissive).

    Adds CONTAINER_COUNT (how many containers were rolled up).
    """
    if "MASTER_SHIPMENT_ID" not in df.columns:
        return df

    container_counts = (
        df.groupby("MASTER_SHIPMENT_ID")["CONTAINER_NUMBER"]
          .nunique()
          .rename("CONTAINER_COUNT")
    )
    # .first() returns the first non-NA value per column per group — fast & vectorized
    grouped = (
        df.groupby("MASTER_SHIPMENT_ID", as_index=False, sort=False)
          .first()
    )
    grouped = grouped.merge(container_counts, on="MASTER_SHIPMENT_ID", how="left")
    grouped["CONTAINER_COUNT"] = grouped["CONTAINER_COUNT"].fillna(1).astype(int)
    return grouped


# ============================================================================
# Milestone resolution
# ============================================================================

def resolve_milestone(df: pd.DataFrame, code: str) -> pd.Series:
    """
    Return a tz-aware UTC datetime Series for the milestone. For VDL/VAD,
    falls back to _P44 whenever the primary column is null. Tolerant to
    missing source columns (treats them as all-null).
    """
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
    """Add POL_FULL, POD_FULL, LANE columns vectorized."""
    out = df.copy()

    def _fmt(locode_col, city_col, ctry_col):
        loc  = out[locode_col].fillna("").astype(str).str.strip()
        city = out[city_col].fillna("").astype(str).str.strip()
        ctry = out[ctry_col].fillna("").astype(str).str.strip()
        place = np.where(
            (city != "") & (ctry != ""), city + ", " + ctry,
            np.where(city != "", city, ctry)
        )
        place = pd.Series(place, index=out.index)
        # Join "LOCODE - place" only when both pieces exist
        result = np.where(
            (loc != "") & (place != ""), loc + " - " + place,
            np.where(loc != "", loc, place)
        )
        result = pd.Series(result, index=out.index)
        result = result.where(result != "", "Unknown")
        return result

    out["POL_FULL"] = _fmt("POL_LOCODE", "POL", "POL_COUNTRY")
    out["POD_FULL"] = _fmt("POD_LOCODE", "POD", "POD_COUNTRY")
    out["LANE"] = out["POL_FULL"] + " → " + out["POD_FULL"]
    return out


# ============================================================================
# Duration helpers — vectorized
# ============================================================================

def hours_to_components_vec(hours: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Vectorized conversion of fractional hours -> (days, remainder_hours) as
    Int64 (nullable). Carries 24h up to a day. Handles NaN.
    """
    # Round to minute precision to avoid float jitter
    total_minutes = (hours * 60).round()
    days = np.floor_divide(total_minutes, 24 * 60)
    rem_minutes = total_minutes - days * 24 * 60
    hours_part = (rem_minutes / 60).round()

    # 23h59.5m -> 24h after rounding -> bump to next day
    carry = hours_part == 24
    days = days.where(~carry, days + 1)
    hours_part = hours_part.where(~carry, 0)

    days_int = days.astype("Int64")
    hours_int = hours_part.astype("Int64")
    return days_int, hours_int


def hours_to_components(total_hours: float) -> Tuple[int, int]:
    """Scalar version for summary rows. Returns (np.nan, np.nan) if null."""
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
    aggregation_level: str,   # "Container" or "Master Shipment"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (calculated_df, missed_df).
      calculated_df: both milestones present AND end > start.
      missed_df:     anything else, with a MISSED_REASON column.
    """
    base = df
    if aggregation_level == "Master Shipment":
        base = aggregate_to_master_shipment(df)

    work = build_lane_keys(base).reset_index(drop=True)

    start_ts = resolve_milestone(work, start_code)
    end_ts   = resolve_milestone(work, end_code)
    work["START_TS"] = start_ts
    work["END_TS"]   = end_ts

    delta_hours = (end_ts - start_ts).dt.total_seconds() / 3600.0
    work["TRANSIT_HOURS"] = delta_hours

    # Reason classification (vectorized)
    reason = pd.Series("", index=work.index, dtype="object")
    reason = reason.mask(start_ts.isna() & end_ts.isna(), f"Missing both {start_code} and {end_code}")
    reason = reason.mask(start_ts.isna() & end_ts.notna(), f"Missing {start_code} (start)")
    reason = reason.mask(start_ts.notna() & end_ts.isna(), f"Missing {end_code} (end)")
    neg_mask = start_ts.notna() & end_ts.notna() & (delta_hours <= 0)
    reason = reason.mask(neg_mask, f"{end_code} not after {start_code} (non-positive transit)")
    work["MISSED_REASON"] = reason

    valid_mask = start_ts.notna() & end_ts.notna() & (delta_hours > 0)

    # Build display column list — include MSID + container count when aggregated
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
    calc = work.loc[valid_mask, base_cols + ["START_TS", "END_TS", "TRANSIT_HOURS"]].copy()
    calc.rename(columns={"START_TS": f"{start_code}_TS", "END_TS": f"{end_code}_TS"}, inplace=True)

    d_series, h_series = hours_to_components_vec(calc["TRANSIT_HOURS"])
    calc["TRANSIT_DAYS"] = d_series
    calc["TRANSIT_HOURS_REMAINDER"] = h_series
    calc["TOTAL_HOURS"] = calc["TRANSIT_HOURS"].round(2)
    calc.drop(columns=["TRANSIT_HOURS"], inplace=True)

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
    """Build a row dict from a pre-aggregated stats row."""
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
        "MIN_TOTAL_HOURS":    round(agg_row["min"], 2),
        "MAX_TOTAL_HOURS":    round(agg_row["max"], 2),
        "AVG_TOTAL_HOURS":    round(agg_row["mean"], 2),
        "MEDIAN_TOTAL_HOURS": round(agg_row["median"], 2),
    }


@st.cache_data(show_spinner=False, max_entries=8)
def build_lane_summary(calc: pd.DataFrame) -> pd.DataFrame:
    """
    Lane block: ALL CARRIERS row first (lane name shown), then one row per
    carrier (lane left blank), ordered by descending volume within each block.
    Blocks themselves ordered by descending lane volume.
    """
    if calc.empty:
        return pd.DataFrame()

    # Two groupbys, then iterate — O(N) instead of O(L*N)
    lane_agg = (
        calc.groupby("LANE")["TOTAL_HOURS"]
            .agg(["count", "min", "max", "mean", "median"])
            .sort_values("count", ascending=False)
    )
    carrier_agg = (
        calc.groupby(["LANE", "CARRIER_NAME", "CARRIER_SCAC"], dropna=False)["TOTAL_HOURS"]
            .agg(["count", "min", "max", "mean", "median"])
    )

    rows: List[dict] = []
    for lane, lstats in lane_agg.iterrows():
        rows.append(_stats_dict(lane, "ALL CARRIERS", "ALL", lstats, lstats["count"]))
        # Carriers within this lane, by descending volume
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

@st.cache_data(show_spinner=False, max_entries=32)
def build_trend_frame(
    calc: pd.DataFrame,
    lane_selection: str,
    carrier_selection: Tuple[str, ...],   # tuple so it's hashable for cache
    metric: str,
    aggregation: str,
) -> pd.DataFrame:
    """Aggregate per time bucket × series (carrier or 'All Carriers')."""
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
    grouped = (
        df.groupby(["BUCKET", "_SERIES"])["TOTAL_HOURS"]
          .agg([agg_func, "count"])
          .rename(columns={agg_func: "VALUE_TOTAL_HOURS", "count": "VOLUME"})
          .reset_index()
          .rename(columns={"_SERIES": "SERIES"})
    )
    d_s, h_s = hours_to_components_vec(grouped["VALUE_TOTAL_HOURS"])
    grouped["VALUE_DAYS"] = d_s
    grouped["VALUE_HOURS_REMAINDER"] = h_s
    grouped["VALUE_TOTAL_HOURS"] = grouped["VALUE_TOTAL_HOURS"].round(2)
    grouped["VALUE_DAYS_DECIMAL"] = (grouped["VALUE_TOTAL_HOURS"] / 24).round(3)
    grouped.sort_values(["BUCKET", "SERIES"], inplace=True)
    return grouped


# ============================================================================
# Excel writer
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
    calc: pd.DataFrame,
    missed: pd.DataFrame,
    lane_sum: pd.DataFrame,
    start_code: str,
    end_code: str,
    aggregation_level: str,
    raw_row_count: int,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        # Sheet 1: Calculated
        calc_out = _strip_tz(calc)
        calc_out.to_excel(writer, sheet_name="Calculated Transit", index=False)
        _autosize(writer, "Calculated Transit", calc_out)

        # Sheet 2: Missed
        missed_out = _strip_tz(missed)
        missed_out.to_excel(writer, sheet_name="Missed Milestones", index=False)
        _autosize(writer, "Missed Milestones", missed_out)

        # Sheet 3: Lane × Carrier summary
        lane_sum.to_excel(writer, sheet_name="Lane x Carrier Summary", index=False)
        _autosize(writer, "Lane x Carrier Summary", lane_sum)

        # Sheet 4: Run Info
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
                raw_row_count,
                len(calc),
                len(missed),
                calc["LANE"].nunique() if not calc.empty else 0,
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            ],
        })
        meta.to_excel(writer, sheet_name="Run Info", index=False)
        _autosize(writer, "Run Info", meta)

    buf.seek(0)
    return buf.getvalue()


def _chart_filename(lane, carriers, metric, agg, ext) -> str:
    lane_part = "all-lanes" if lane == "All Lanes" else lane.split(" - ")[0].lower()
    if "All Carriers" in carriers:
        carr_part = "all-carriers"
    elif len(carriers) == 1:
        carr_part = carriers[0].lower().replace(" ", "-")
    else:
        carr_part = f"{len(carriers)}-carriers"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"trend_{lane_part}_{carr_part}_{metric.lower()}_{agg.lower()}_{ts}.{ext}"


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
        help=(
            "**Container** = one row per CONTAINER_NUMBER.\n\n"
            "**Master Shipment** = collapse containers under the same "
            "MASTER_SHIPMENT_ID into a single row (first non-null milestone "
            "per group). A CONTAINER_COUNT column shows how many were rolled up."
        ),
    )
    start_code = st.selectbox(
        "Transit START milestone",
        options=MILESTONE_ORDER, index=2,  # CLL
        format_func=lambda c: MILESTONE_LABELS[c],
    )
    end_code = st.selectbox(
        "Transit END milestone",
        options=MILESTONE_ORDER, index=4,  # VAD
        format_func=lambda c: MILESTONE_LABELS[c],
    )
    st.caption("ℹ️ VDL/VAD auto-fall back to their _P44 variants when null.")

    run = st.button("Run analysis", type="primary", use_container_width=True)

# Block until a file is uploaded
if uploaded is None:
    st.info("⬅️ Upload a CSV or XLSX file to begin.")
    st.stop()

# Load & validate
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

# Sanity preview before any computation
if not run and "results" not in st.session_state:
    st.success(f"Loaded {len(raw):,} rows. Configure in the sidebar, then click **Run analysis**.")
    with st.expander("Preview raw data (first 20 rows)"):
        st.dataframe(raw.head(20), use_container_width=True)
    st.stop()

# Detect parameter changes — invalidate cached results if config changed
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
            "calc": calc,
            "missed": missed,
            "lane_sum": lane_sum,
            "start_code": start_code,
            "end_code": end_code,
            "agg_level": agg_level,
            "raw_count": len(raw),
            "params": (_bytes_hash(file_bytes), start_code, end_code, agg_level),
        }

results    = st.session_state["results"]
calc       = results["calc"]
missed     = results["missed"]
lane_sum   = results["lane_sum"]
start_code = results["start_code"]
end_code   = results["end_code"]
agg_level  = results["agg_level"]

# ---- Top-line metrics ----
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Input rows ({agg_level.lower()}-level after agg)",
          f"{(len(calc) + len(missed)):,}",
          delta=f"raw: {results['raw_count']:,}",
          delta_color="off")
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
               "Following rows = each carrier (ordered by descending volume).")
    st.dataframe(lane_sum, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Transit trend over time")

    if calc.empty:
        st.info("No calculated transits to plot.")
    else:
        lanes_sorted = (
            calc.groupby("LANE").size().sort_values(ascending=False).index.tolist()
        )
        col_a, col_b, col_c, col_d = st.columns([2, 2, 1, 1])

        with col_a:
            lane_choice = st.selectbox("Lane", options=["All Lanes"] + lanes_sorted, index=0)

        # Carrier pool depends on lane
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
            fig = px.line(
                trend,
                x="BUCKET",
                y="VALUE_DAYS_DECIMAL",
                color="SERIES",
                markers=True,
                labels={
                    "BUCKET": f"{agg_choice} bucket (anchor: SHIPMENT_CREATED_DATE)",
                    "VALUE_DAYS_DECIMAL": f"{metric_choice} transit (days)",
                    "SERIES": "Carrier",
                },
                title=f"{metric_choice} transit — {start_code} → {end_code} "
                      f"({agg_choice.lower()}, {agg_level.lower()}-level)",
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Bucket: %{x|%Y-%m-%d}<br>"
                    "Transit: %{customdata[0]}d %{customdata[1]}h "
                    "(%{customdata[2]} total hrs)<br>"
                    "Shipments: %{customdata[3]}<extra></extra>"
                ),
                customdata=trend[
                    ["VALUE_DAYS", "VALUE_HOURS_REMAINDER",
                     "VALUE_TOTAL_HOURS", "VOLUME"]
                ].to_numpy(),
            )
            fig.update_layout(height=520, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Underlying trend data"):
                st.dataframe(trend, use_container_width=True, hide_index=True)

            # Chart exports — shareable alongside the main Excel
            export_cols = st.columns(2)
            with export_cols[0]:
                try:
                    png_bytes = fig.to_image(format="png", width=1400, height=700, scale=2)
                    st.download_button(
                        "⬇️ Download chart as PNG",
                        data=png_bytes,
                        file_name=_chart_filename(lane_choice, carrier_choice, metric_choice, agg_choice, "png"),
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception:
                    st.caption("PNG export unavailable (install `kaleido` per requirements.txt).")
            with export_cols[1]:
                trend_buf = io.BytesIO()
                with pd.ExcelWriter(trend_buf, engine="xlsxwriter") as w:
                    trend.to_excel(w, sheet_name="Trend Data", index=False)
                    _autosize(w, "Trend Data", trend)
                st.download_button(
                    "⬇️ Download trend data as Excel",
                    data=trend_buf.getvalue(),
                    file_name=_chart_filename(lane_choice, carrier_choice, metric_choice, agg_choice, "xlsx"),
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
