# Ocean Shipment Transit Analyzer

A Streamlit app for computing per-shipment ocean transit times between any two
milestones, summarizing them by lane × carrier, and exploring trends over time.

Built for raw shipment files exported from the project44 platform, but works
with any file that follows the same column schema.

---

## What it does

1. **Upload** a raw shipment file (CSV or XLSX).
2. **Pick the aggregation level**:
   - **Container** — one row per `CONTAINER_NUMBER` (default).
   - **Master Shipment** — collapse containers under the same
     `MASTER_SHIPMENT_ID` into a single representative row. Useful when
     multiple containers move together and you want master-shipment-level
     volumes/averages instead of container-level. A `CONTAINER_COUNT` column
     is added so you can still see how many containers each row represents.
3. **Pick two milestones**: a transit *start* and a transit *end*.
   - Available milestones: `CEP`, `CGI`, `CLL`, `VDL`, `VAD`, `CDD`, `CGO`, `CER`.
   - For `VDL` and `VAD`, the carrier-reported value is used; if it's null,
     the tool falls back to the `_P44` computed variant automatically.
4. **Get an Excel report** with four sheets:
   - **Calculated Transit** — one row per eligible shipment with `START_TS`,
     `END_TS`, `TRANSIT_DAYS`, `TRANSIT_HOURS`, `TOTAL_HOURS`.
     36 hours displays as `TRANSIT_DAYS=1, TRANSIT_HOURS=12` — not `1.5 days`.
     `TOTAL_HOURS` is the same transit expressed as a single number in hours
     (e.g., `36.00`) — useful for sorting / filtering, and the value that
     feeds every aggregate stat downstream.
   - **Missed Milestones** — shipments excluded because a milestone was null
     or the end timestamp was not strictly after the start. `MISSED_REASON`
     explains each row.
   - **Lane × Carrier Summary** — per unique lane block: one `ALL CARRIERS`
     row first, then one row per carrier on that lane (ordered by descending
     volume). Statistics are min / max / average / median in days + hours.
     `AVG_TOTAL_HOURS` is the **average per shipment** for that group
     expressed in total hours (NOT a sum across shipments). Same idea for
     `MIN/MAX/MEDIAN_TOTAL_HOURS`.
   - **Run Info** — milestones, aggregation level, row counts, timestamp,
     code version.
5. **Trends & Consistency tab** — TWO interactive charts:
   - **Chart 1 — Transit time.** Per-carrier lines (one color per carrier)
     plus a bold black "Combined (all selected)" line, with gray bars in the
     background showing container volume per bucket. Use this to set SAP
     lead times and compare carrier speed.
   - **Chart 2 — Transit-time consistency (std dev).** Per-carrier dotted
     std-dev lines plus a bold dashed black combined std-dev line, same
     gray volume bars. Lower = more consistent. Use this to pick reliable
     carriers. Buckets where a carrier had only **1 shipment** are shown
     as a hollow marker — std dev is mathematically undefined with one
     data point.
   - Filters: single-select **Lane** (or "All Lanes"), multi-select
     **Carrier(s)** (scoped to the chosen lane — only carriers with at
     least one eligible shipment appear). "All Carriers" is mutually
     exclusive with specific carriers.
   - Toggles: **Average** vs **Median** (drives Chart 1's center line),
     **Weekly** vs **Monthly** bucket. Time bucket is anchored on
     `SHIPMENT_CREATED_DATE`. Overlay checkboxes let you hide volume bars,
     the combined line, or per-carrier lines.
   - **Export scope toggle**:
     - "Current view" → zip with the 2 PNGs you're looking at + Excel
       with a Key sheet and 2 data sheets.
     - "Full pack" → zip with **6 PNGs** (transit avg/median ×
       weekly/monthly, plus std-dev weekly + monthly) and an Excel with
       a **Key sheet + 6 data sheets**. This is what you'd send a customer
       as the trend deliverable for a single lane.
   - **📦 Bulk lane export** — at the bottom of the Trends tab. Multi-select
     any number of lanes (or one click "include all"), hit Generate, and
     get ONE master ZIP back containing **one folder per lane**. Each lane
     folder has the 6 PNGs + `data_with_key.xlsx` (Key + 6 sheets) — same
     content as if you'd run the Full pack manually on that lane. A
     `README.txt` at the root of the master ZIP lists every lane included
     and what each file is. Lanes stay independent — no combining or
     averaging across lanes. Use this when a customer asks for the
     standard pack across many lanes at once.
   - Data column naming (each row = one carrier OR `Combined (all selected)` in one bucket):
     - `BUCKET` — start date of the week/month period.
     - `SERIES` — carrier name, or `Combined (all selected)`.
     - `SHIPMENT_COUNT` — containers from THIS series in this bucket.
     - `TOTAL_BUCKET_VOLUME` — containers in the bucket across ALL selected
       carriers (same value across every row in that bucket).
     - `AVG_DAYS` / `AVG_HOURS` — average **per shipment** as d+h
       (e.g., `42d 16h`). NOT a sum. Same idea for `MEDIAN_*`.
     - `AVG_TOTAL_HOURS` — average per shipment as a single hours number
       (for sorting). Same idea for `MEDIAN_TOTAL_HOURS`.
     - `AVG_DECIMAL_DAYS` / `MEDIAN_DECIMAL_DAYS` — what the chart Y-axis plots.
     - `STDEV_HOURS` / `STDEV_DECIMAL_DAYS` — std dev of transit times in
       that bucket. Blank when `SHIPMENT_COUNT < 2`.
     - `NOTES` — flags edge cases like `"1 shipment only — std dev N/A"`.

---

## Troubleshooting

If you upgrade the app (e.g., pull a new version from GitHub onto Streamlit
Cloud) and see a `KeyError: Column not found` after the deploy, it means
your browser still has a stale session pointing at the previous run's
DataFrame. Two fixes, both safe:

- Click **🔄 Reset all caches & session** in the sidebar's Troubleshooting
  expander, then re-upload.
- Or just refresh the page hard (Ctrl/Cmd + Shift + R).

The app embeds a `CODE_VERSION` constant into every cache key, so any
schema-breaking change automatically forces a fresh compute the next time
you hit Run. The Reset button is there for the in-between moment if you've
been clicking around in a stale session.

---

## Lane definition

```
{POL_LOCODE} - {POL_CITY}, {POL_COUNTRY}  →  {POD_LOCODE} - {POD_CITY}, {POD_COUNTRY}
```

Example: `VNHPH - Hai Phong, VN → USSAV - Savannah, US`

---

## Required input columns

```
SHIPMENT_ID, MASTER_SHIPMENT_ID, CONTAINER_NUMBER,
CARRIER_NAME, CARRIER_SCAC,
POL_LOCODE, POL, POL_COUNTRY,
POD_LOCODE, POD, POD_COUNTRY,
SHIPMENT_CREATED_DATE,
CEP, CGI, CLL, VDL, VDL_P44, VAD, VAD_P44, CDD, CGO, CER
```

Extra columns are ignored. A missing milestone column is tolerated — its
shipments will simply land in the Missed sheet.

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically <http://localhost:8501>).

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo (`app.py`, `requirements.txt`, README).
2. Go to <https://share.streamlit.io>, click **New app**, point it at the repo
   and at `app.py`.
3. Streamlit Cloud installs from `requirements.txt` and starts the app. No
   secrets are required — the app is stateless and only processes the file
   uploaded at runtime.

---

## Performance notes

- File reads, milestone resolution, transit computation, lane summary, and
  trend frames are all cached with `@st.cache_data` keyed on the file
  contents and the user's selections. Re-running with the same settings is
  effectively instant; switching tabs or trend filters never recomputes the
  whole pipeline.
- Duration conversion (`hours → days + hours`) and lane key construction
  are fully vectorized.
- Lane summary uses two `groupby` calls instead of a per-lane loop.
- On-screen previews are capped at 10 000 rows; the full data always goes
  into the Excel download.
- Tested cleanly against a 1 885-row Generac export and a 200 000-row
  stress sample.

---

## Data handling

- All timestamps are converted to UTC internally before subtraction.
- Eligibility: `start_ts not null AND end_ts not null AND end_ts > start_ts`.
- A non-positive difference routes the shipment to the Missed sheet with
  reason `"<END> not after <START> (non-positive transit)"`.
- `_P44` fallback only fires for VDL and VAD.
- In Master Shipment mode, the representative row for each
  `MASTER_SHIPMENT_ID` uses the first non-null value per column across that
  group's containers (`groupby().first()`).
- Hour rounding: total transit is computed in fractional hours, then
  converted to days + integer-hour remainder. Carry-over at 24h is handled
  (`23h 59.5m → 1d 0h`, not `0d 24h`).
