# Employment Equity data – analyst assignment

Cleaning, restructuring and exploratory analysis of the EEA2 extracts (`ee_meta.csv`, `ee_profile_all.csv`), 500 employers, 2013–2024.

## Run

```bash
pip install pandas matplotlib
python 01_clean.py      # data/raw -> data/clean (+ cleaning_log.txt)
python 02_analysis.py   # data/clean -> outputs/tables, outputs/charts
```

Place the two supplied CSVs in `data/raw/` as `ee_meta.csv` and `ee_profile_all.csv`.

## Files

| Path | Contents |
| --- | --- |
| `01_clean.py` | Reshapes the wide employer file to employer-year, normalises text, removes placeholders, flags the unreliable 2016–18 sector codes, derives one canonical row per employer; tidies and validates the workforce file, reshapes it long (race × gender), merges the two |
| `02_analysis.py` | All numbers and charts in the write-up |
| `data/clean/firm_year_meta.csv` | Employer attributes, one row per employer-year |
| `data/clean/firms.csv` | One row per employer: canonical sector, SETA, business type, province, size band, QA flags |
| `data/clean/workforce_long.csv` | Year × employer × level × race × gender → headcount (420,000 rows) |
| `data/clean/panel.csv` | `workforce_long` joined to employer attributes – analysis-ready |
| `data/clean/cleaning_log.txt` | What was found and changed, with counts |
| `outputs/tables/*.csv` | 13 summary tables behind the findings |
| `outputs/charts/*.png` | Static versions of the charts |

## Key decisions

- **KZN Department of Education excluded from pooled figures.** It is 31–47% of all headcount each year and is miscoded in the metadata; it is analysed separately.
- **Industry sector 2016–2018 not used.** 29% of employers have a different sector in exactly those three years (0–4% in others), consistent with a coding change in the source.
- **Canonical employer attributes = modal value across years**, with the number of distinct values kept as a QA flag.
- **"Workforce" = permanent employees; "Black" = African, Coloured, Indian** (EE Act definition).
- The panel is balanced, so change over time is within the same employers.
