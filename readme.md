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
     `END_TS`, `TRANSIT_DAYS`, `TRANSIT_HOURS_REMAINDER`, `TOTAL_HOURS`.
     36 hours displays as `Days=1, Hours=12` — not `1.5 days`.
   - **Missed Milestones** — shipments excluded because a milestone was null
     or the end timestamp was not strictly after the start. `MISSED_REASON`
     explains each row.
   - **Lane × Carrier Summary** — per unique lane block: one `ALL CARRIERS`
     row first, then one row per carrier on that lane (ordered by descending
     volume). Statistics are min / max / average / median in days + hours
     plus numeric `*_TOTAL_HOURS` columns for sorting.
   - **Run Info** — milestones, aggregation level, row counts, timestamp.
5. **Trends tab** — interactive chart with:
   - Single-select **Lane** (or "All Lanes").
   - Multi-select **Carrier(s)**, scoped to the chosen lane. "All Carriers"
     is mutually exclusive with specific carriers (enforced in code).
   - Toggle between **Average** and **Median** transit time.
   - Toggle between **Weekly** and **Monthly** aggregation.
   - Time bucket is anchored on `SHIPMENT_CREATED_DATE`.
   - Download the rendered chart as **PNG** and the underlying data as a
     standalone Excel — share alongside the main report.

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
