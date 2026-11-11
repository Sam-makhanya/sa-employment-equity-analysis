"""
01_clean.py - Inspect, clean and restructure the Employment Equity (EEA2) extracts.

Inputs  : data/raw/ee_meta.csv, data/raw/ee_profile_all.csv
Outputs : data/clean/
    firm_year_meta.csv   one row per employer x submission year (cleaned, long)
    firms.csv            one row per employer, canonical (time-invariant) attributes + QA flags
    workforce_long.csv   tidy: year x employer x occupational level x race x gender -> headcount
    panel.csv            workforce_long merged with firm attributes (analysis-ready)
    cleaning_log.txt     what was found and what was done

Issues found during inspection (see cleaning_log.txt for counts):
  meta
    * Wide layout: 6 attributes x 12 years = 72 columns, years in shuffled order.
    * Free-text names: double spaces, merged words, placeholder values
      ("SAME AS ABOVE", "PUBLIC ENTITY", bare registration numbers), missing values.
    * City "-" used as a missing marker; double spaces ("CAPE  TOWN").
    * industry_sector for 2016-2018 disagrees with the firm's other years for ~28% of
      firms (vs 1-4% in other years), identically in all three years -> looks like a
      different coding scheme / bad extract, so those years are not trusted.
    * Attributes drift across years (sector, SETA, business type, city) -> a single
      canonical value per firm is taken as the mode of the trusted years.
  profile
    * Inconsistent column casing and a typo ("foriegn_male").
    * Occupational level label "Temporary" (2024 only) vs "Temporary employees".
    * Panel is balanced (500 firms x 12 years x 7 levels), no negatives, no duplicates.
    * One employer (KZN Provincial Department of Education) is ~35% of all headcount
      and is coded as "Arts, Entertainment and Recreation" / "Educational Institution";
      it is flagged so results can be shown with and without it.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/clean")
OUT.mkdir(parents=True, exist_ok=True)
log = []


def note(msg):
    print(msg)
    log.append(msg)


# ----------------------------------------------------------------------------
# 1. Employer metadata: wide -> long
# ----------------------------------------------------------------------------
meta_wide = pd.read_csv(RAW / "ee_meta.csv", dtype=str)
note(f"meta: {meta_wide.shape[0]} employers, {meta_wide.shape[1]} columns (wide by year)")

long = meta_wide.melt(id_vars="ee_ref", var_name="col", value_name="value")
long[["field", "year"]] = long["col"].str.extract(r"^(.*)_(\d{4})$")
long["year"] = long["year"].astype(int)
meta = (long.pivot_table(index=["ee_ref", "year"], columns="field",
                         values="value", aggfunc="first", dropna=False)
            .reset_index())
meta.columns.name = None
meta = meta.rename(columns={"dti_registration_name": "reg_name",
                            "province_physical": "province",
                            "city_town_physical": "city",
                            "seta_classification": "seta"})


def clean_text(s):
    if pd.isna(s):
        return np.nan
    s = re.sub(r"\s+", " ", str(s)).strip().upper()
    return s or np.nan


for c in ["reg_name", "industry_sector", "seta", "business_type", "province", "city"]:
    meta[c] = meta[c].map(clean_text)

# business type: keep readable title case
meta["business_type"] = meta["business_type"].str.title().str.replace("Non-Profit", "Non-profit")

# placeholders -> missing
PLACEHOLDER_NAMES = {"SAME AS ABOVE", "PUBLIC ENTITY", "N/A", "NA", "-", "NONE"}
is_placeholder = (meta["reg_name"].isin(PLACEHOLDER_NAMES)
                  | meta["reg_name"].str.fullmatch(r"[A-Z]?\d{3,}", na=False))
note(f"meta: {is_placeholder.sum()} firm-year registration names were placeholders "
     f"(e.g. 'SAME AS ABOVE', bare reg. numbers) -> set missing")
meta.loc[is_placeholder, "reg_name"] = np.nan
n_city = meta["city"].isin(["-", ""]).sum()
meta.loc[meta["city"].isin(["-", ""]), "city"] = np.nan
note(f"meta: {n_city} firm-year city values were '-' -> set missing")

# Normalise common legal-suffix spellings so "PTY LTD", "(PTY) LTD", "PTYLTD" agree
def norm_name(s):
    if pd.isna(s):
        return s
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"\bPTY\s*LTD\b|\bPTYLTD\b|\bPTY \(LTD\b", "PTY LTD", s)
    s = re.sub(r"\bEDMS\s*BPK\b", "EDMS BPK", s)
    return re.sub(r"\s+", " ", s).strip()

meta["reg_name"] = meta["reg_name"].map(norm_name)

# Diagnose the 2016-18 industry block -------------------------------------
def mode_or_nan(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else np.nan

trusted = meta[~meta["year"].isin([2016, 2017, 2018])]
mode_sector_trusted = trusted.groupby("ee_ref")["industry_sector"].agg(mode_or_nan)
dev = (meta["industry_sector"] != meta["ee_ref"].map(mode_sector_trusted)) & meta["industry_sector"].notna()
dev_by_year = (dev.groupby(meta["year"]).mean() * 100).round(1)
note("meta: % of firms whose industry_sector differs from their modal sector, by year:\n"
     + dev_by_year.to_string())
meta["industry_sector_raw"] = meta["industry_sector"]
meta.loc[meta["year"].isin([2016, 2017, 2018]), "industry_sector"] = np.nan
note("meta: industry_sector for 2016-2018 treated as unreliable (kept as industry_sector_raw)")

meta.to_csv(OUT / "firm_year_meta.csv", index=False)

# ----------------------------------------------------------------------------
# 2. One canonical row per firm
# ----------------------------------------------------------------------------
def last_valid(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan

meta = meta.sort_values(["ee_ref", "year"])
g = meta.groupby("ee_ref")
firms = pd.DataFrame({
    "reg_name_latest": g["reg_name"].agg(last_valid),
    "n_names": g["reg_name"].nunique(),
    "industry_sector": g["industry_sector"].agg(mode_or_nan),
    "seta": g["seta"].agg(mode_or_nan),
    "business_type": g["business_type"].agg(mode_or_nan),
    "province": g["province"].agg(mode_or_nan),
    "city": g["city"].agg(mode_or_nan),
    "n_sectors": g["industry_sector"].nunique(),
    "n_business_types": g["business_type"].nunique(),
    "n_provinces": g["province"].nunique(),
}).reset_index()

for c in ["n_names", "n_sectors", "n_business_types", "n_provinces"]:
    note(f"firms: distribution of {c}: {firms[c].value_counts().sort_index().to_dict()}")

# simple sector grouping for charts (18 -> 8 groups)
SECTOR_GROUP = {
    "MANUFACTURING": "Manufacturing",
    "AGRICULTURE, FORESTRY & FISHING": "Agriculture",
    "WHOLESALE AND RETAIL TRADE; REPAIR OF MOTOR VEHICLES AND MOTORCYCLES": "Wholesale & retail",
    "CONSTRUCTION": "Construction",
    "TRANSPORTATION AND STORAGE": "Transport & storage",
    "FINANCIAL AND INSURANCE ACTIVITIES": "Finance & business services",
    "PROFESSIONAL, SCIENTIFIC AND TECHNICAL ACTIVITIES": "Finance & business services",
    "ADMINISTRATIVE AND SUPPORT ACTIVITIES": "Finance & business services",
    "REAL ESTATE ACTIVITIES": "Finance & business services",
    "INFORMATION AND COMMUNICATION": "Finance & business services",
    "MINING AND QUARRYING": "Mining & utilities",
    "ELECTRICITY, GAS, STEAM AND AIR CONDITIONING SUPPLY": "Mining & utilities",
    "WATER SUPPLY; SEWERAGE, WASTE MANAGEMENT AND REMEDIATION ACTIVITIES": "Mining & utilities",
    "ACCOMMODATION AND FOOD SERVICE ACTIVITIES": "Hospitality & arts",
    "ARTS, ENTERTAINMENT AND RECREATION": "Hospitality & arts",
    "PUBLIC ADMINISTRATION AND DEFENCE, COMPULSORY SOCIAL SECURITY": "Public admin, health & education",
    "HUMAN HEALTH AND SOCIAL WORK ACTIVITIES": "Public admin, health & education",
    "EDUCATION": "Public admin, health & education",
}
firms["sector_group"] = firms["industry_sector"].map(SECTOR_GROUP).fillna("Unknown")

# ----------------------------------------------------------------------------
# 3. Workforce profile: tidy columns, harmonise levels, long format
# ----------------------------------------------------------------------------
prof = pd.read_csv(RAW / "ee_profile_all.csv")
note(f"profile: {prof.shape[0]} rows; columns as supplied: {list(prof.columns)}")
prof.columns = [c.lower().replace("foriegn", "foreign") for c in prof.columns]
prof = prof.rename(columns={"submission_year": "year"})

LEVEL_MAP = {
    "Top management": "Top management",
    "Senior management": "Senior management",
    "Professionally qualified and experienced specialists and mid-management": "Professional / mid-management",
    "Skilled technical and academically qualified workers, junior management, supervisors, foremen, and superintendents": "Skilled technical / junior management",
    "Semi-skilled and discretionary decision making": "Semi-skilled",
    "Unskilled and defined decision making": "Unskilled",
    "Temporary employees": "Temporary employees",
    "Temporary": "Temporary employees",
}
note(f"profile: 'Temporary' label used in {(prof.occupation_level=='Temporary').sum()} rows "
     f"(year(s) {sorted(int(y) for y in prof.loc[prof.occupation_level=="Temporary","year"].unique())}) -> 'Temporary employees'")
unmapped = set(prof["occupation_level"]) - set(LEVEL_MAP)
assert not unmapped, unmapped
prof["level"] = prof["occupation_level"].map(LEVEL_MAP)
LEVEL_ORDER = ["Top management", "Senior management", "Professional / mid-management",
               "Skilled technical / junior management", "Semi-skilled", "Unskilled",
               "Temporary employees"]

count_cols = [c for c in prof.columns if re.match(r"^(african|coloured|indian|white|foreign)_(male|female)$", c)]
assert len(count_cols) == 10
assert (prof[count_cols] >= 0).all().all(), "negative counts"
assert not prof.duplicated(["year", "ee_ref", "level"]).any(), "duplicate firm-year-level"
assert prof.groupby(["ee_ref", "year"]).size().eq(7).all(), "panel not balanced"
note("profile: checks passed - no negatives, no duplicate keys, balanced 500 x 12 x 7 panel")

wf = prof.melt(id_vars=["year", "ee_ref", "level"], value_vars=count_cols,
               var_name="group", value_name="headcount")
wf[["race", "gender"]] = wf["group"].str.split("_", expand=True)
wf["race"] = wf["race"].str.title()
wf["gender"] = wf["gender"].str.title()
wf["level"] = pd.Categorical(wf["level"], LEVEL_ORDER, ordered=True)
wf["permanent"] = wf["level"] != "Temporary employees"
wf = wf.drop(columns="group")[["year", "ee_ref", "level", "permanent", "race", "gender", "headcount"]]
wf.to_csv(OUT / "workforce_long.csv", index=False)

# ----------------------------------------------------------------------------
# 4. Firm size, outlier flag, merge
# ----------------------------------------------------------------------------
size = wf.groupby(["ee_ref", "year"])["headcount"].sum().unstack()
firms = firms.merge(size.median(axis=1).rename("median_headcount").reset_index(), on="ee_ref")
firms["size_band"] = pd.cut(firms["median_headcount"], [0, 49, 149, 499, np.inf],
                            labels=["<50", "50-149", "150-499", "500+"])
share_top = size.sum(axis=1).max() / size.values.sum()
big = size.sum(axis=1).idxmax()
firms["dominant_outlier"] = firms["ee_ref"] == big
note(f"profile: largest employer {big} ({firms.loc[firms.dominant_outlier,'reg_name_latest'].iat[0]}) "
     f"= {share_top:.1%} of all headcount; flagged dominant_outlier")

firms.to_csv(OUT / "firms.csv", index=False)
panel = wf.merge(firms[["ee_ref", "sector_group", "industry_sector", "business_type",
                        "province", "size_band", "dominant_outlier"]], on="ee_ref", how="left")
assert len(panel) == len(wf)
panel.to_csv(OUT / "panel.csv", index=False)
note(f"panel: {len(panel):,} rows written")

(OUT / "cleaning_log.txt").write_text("\n".join(log))
