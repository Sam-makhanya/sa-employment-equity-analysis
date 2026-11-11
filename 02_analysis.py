"""
02_analysis.py - Explore the cleaned EE panel and produce tables + charts.

Reads data/clean/panel.csv (from 01_clean.py). Writes outputs/tables/*.csv and outputs/charts/*.png.

Analytical choices
  * The KZN Department of Education is ~35% of all headcount, so pooled figures exclude it
    unless stated (it is reported separately).
  * "Workforce" = permanent employees (temporary employees reported separately).
  * The panel is balanced (same 500 employers every year), so changes over time are
    within-employer change, not employers entering or leaving the sample.
  * "Black" follows the EE Act definition: African, Coloured and Indian (foreign nationals excluded).
"""
from pathlib import Path
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
T = Path("outputs/tables"); C = Path("outputs/charts")
T.mkdir(parents=True, exist_ok=True); C.mkdir(parents=True, exist_ok=True)

p = pd.read_csv("data/clean/panel.csv")
LEVELS = ["Top management", "Senior management", "Professional / mid-management",
          "Skilled technical / junior management", "Semi-skilled", "Unskilled", "Temporary employees"]
p["level"] = pd.Categorical(p["level"], LEVELS, ordered=True)
p["black"] = p["race"].isin(["African", "Coloured", "Indian"])
x = p[~p["dominant_outlier"]]            # main analysis sample: 499 employers

# palette (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def share(df, by, col, val):
    t = df.groupby(by + [col], observed=True)["headcount"].sum().unstack(col)
    return (t[val].sum(axis=1) if isinstance(val, list) else t[val]) / t.sum(axis=1) * 100


def save(fig, name):
    fig.tight_layout(); fig.savefig(C / name, dpi=180); plt.close(fig)


# ---------------------------------------------------------------- 0. data overview
size = p.groupby(["ee_ref", "year"])["headcount"].sum().unstack()
overview = pd.DataFrame({
    "total_headcount": size.sum(),
    "kzn_doe_share_pct": (size.loc[p.loc[p.dominant_outlier, "ee_ref"].iat[0]] / size.sum() * 100).round(1),
    "median_firm_headcount": size.median(),
})
overview.to_csv(T / "00_overview_by_year.csv")

# ---------------------------------------------------------------- 1. race by level
race_level = (x[x.year.isin([2013, 2024])]
              .groupby(["year", "level", "race"], observed=True)["headcount"].sum()
              .unstack("race"))
race_level_pct = race_level.div(race_level.sum(axis=1), axis=0).mul(100).round(1)
race_level_pct["headcount"] = race_level.sum(axis=1)
race_level_pct.to_csv(T / "01_race_by_level_2013_2024.csv")

perm = x[x.permanent]
wf_race = share(perm, ["year"], "race", "White")
tm = x[x.level == "Top management"]
tm_race = tm.groupby(["year", "race"])["headcount"].sum().unstack()
tm_race_pct = tm_race.div(tm_race.sum(axis=1), axis=0).mul(100).round(1)
tm_race_pct["top_mgmt_headcount"] = tm_race.sum(axis=1)
wf_race_all = perm.groupby(["year", "race"])["headcount"].sum().unstack()
tm_race_pct.join(wf_race_all.div(wf_race_all.sum(axis=1), axis=0).mul(100).round(1)
                 .add_prefix("workforce_")).to_csv(T / "02_top_mgmt_vs_workforce_race_by_year.csv")

# Chart 1: white share by level, 2013 vs 2024 (dumbbell)
lv = [l for l in LEVELS if l != "Temporary employees"]
w13 = race_level_pct.loc[2013, "White"].reindex(lv); w24 = race_level_pct.loc[2024, "White"].reindex(lv)
fig, ax = plt.subplots(figsize=(8, 4.2))
yy = range(len(lv))[::-1]
for y, a, b in zip(yy, w13, w24):
    ax.plot([a, b], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
ax.scatter(w13, yy, s=70, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3, label="2013")
ax.scatter(w24, yy, s=70, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3, label="2024")
for y, a, b in zip(yy, w13, w24):
    if abs(a - b) < 3:
        ax.text(max(a, b) + 1.5, y, f"{b:.0f}% (both years)", color=INK, fontsize=8.5, va="center"); continue
    ax.text(a + 1.5, y + 0.22, f"{a:.0f}%", color=INK2, fontsize=8.5)
    ax.text(b - 1.5, y + 0.22, f"{b:.0f}%", color=INK, fontsize=8.5, ha="right")
ax.axvline(wf_race.loc[2024], color=INK2, lw=1, ls="--")
ax.text(wf_race.loc[2024] + 0.8, -0.85, f"White share of permanent\nworkforce, 2024: {wf_race.loc[2024]:.0f}%",
        fontsize=8, color=INK2, va="bottom")
ax.set_yticks(list(yy)); ax.set_yticklabels(lv); ax.set_xlim(0, 72); ax.set_ylim(-1, len(lv) - 0.4)
ax.set_xlabel("White share of employees at that level (%)"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right", ncol=2)
fig.suptitle("White share has fallen at every level, but top management is still 54% white", x=0.01, ha="left", fontsize=11, color=INK)
save(fig, "01_white_share_by_level.png")

# Chart 2: top management race over time
fig, ax = plt.subplots(figsize=(8, 4))
for race, col in [("White", ORANGE), ("African", BLUE), ("Indian", AQUA), ("Coloured", VIOLET)]:
    s = tm_race_pct[race]
    ax.plot(s.index, s.values, color=col, lw=2, marker="o", ms=4)
    ax.text(2024.25, s.iloc[-1], f"{race} {s.iloc[-1]:.0f}%", color=INK, va="center", fontsize=9)
ax.set_xlim(2012.6, 2026.2); ax.set_ylim(0, 72); ax.set_xticks(range(2013, 2025))
ax.set_ylabel("Share of top management (%)"); ax.grid(axis="x", visible=False)
fig.suptitle("Top management: white share down 11 points in 11 years", x=0.01, ha="left", fontsize=11, color=INK)
save(fig, "02_top_mgmt_race_trend.png")

# ---------------------------------------------------------------- 2. firm-level top management
f = (x[x.level == "Top management"]
     .assign(white=lambda d: d.headcount.where(d.race == "White", 0),
             blk=lambda d: d.headcount.where(d.black, 0),
             fem=lambda d: d.headcount.where(d.gender == "Female", 0))
     .groupby(["ee_ref", "year"])[["headcount", "white", "blk", "fem"]].sum().reset_index())
f = f[f.headcount > 0]
f["all_white"] = f.white == f.headcount
f["no_black"] = f.blk == 0
f["no_women"] = f.fem == 0
firm_tm = f.groupby("year").agg(firms_with_top_mgmt=("ee_ref", "size"),
                                pct_all_white=("all_white", "mean"),
                                pct_no_black=("no_black", "mean"),
                                pct_no_women=("no_women", "mean"),
                                median_team_size=("headcount", "median"))
for c in ["pct_all_white", "pct_no_black", "pct_no_women"]:
    firm_tm[c] = (firm_tm[c] * 100).round(1)
firm_tm.to_csv(T / "03_firm_level_top_mgmt.csv")
a = f[f.year == 2013].set_index("ee_ref").all_white; b = f[f.year == 2024].set_index("ee_ref").all_white
trans = pd.crosstab(a.rename("all_white_2013"), b.rename("all_white_2024"))
trans.to_csv(T / "04_all_white_transition_2013_2024.csv")

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(firm_tm.index, firm_tm.pct_all_white, color=ORANGE, lw=2, marker="o", ms=4)
ax.plot(firm_tm.index, firm_tm.pct_no_women, color=VIOLET, lw=2, marker="o", ms=4)
ax.text(2024.25, firm_tm.pct_all_white.iloc[-1], f"All-white\ntop management {firm_tm.pct_all_white.iloc[-1]:.0f}%", fontsize=9, va="center")
ax.text(2024.25, firm_tm.pct_no_women.iloc[-1], f"No women in\ntop management {firm_tm.pct_no_women.iloc[-1]:.0f}%", fontsize=9, va="center")
ax.axvspan(2019, 2024, color=GRID, alpha=0.35, lw=0)
ax.text(2021.5, 69, "plateau", ha="center", color=INK2, fontsize=8.5)
ax.set_xlim(2012.6, 2026.6); ax.set_ylim(40, 72); ax.set_xticks(range(2013, 2025))
ax.set_ylabel("% of employers (with a top-mgmt team)"); ax.grid(axis="x", visible=False)
fig.suptitle("Half of employers still have an all-white top team, and progress stopped around 2019", x=0.01, ha="left", fontsize=11, color=INK)
save(fig, "03_firms_all_white_top_mgmt.png")

# ---------------------------------------------------------------- 3. sector / size / type, 2024
x24 = x[x.year == 2024]

def gap_table(key):
    rows = []
    for k, d in x24.groupby(key, observed=True):
        pm = d[d.permanent]; t = d[d.level == "Top management"]
        rows.append({key: k, "employers": d.ee_ref.nunique(),
                     "permanent_headcount": pm.headcount.sum(),
                     "white_pct_workforce": pm[pm.race == "White"].headcount.sum() / pm.headcount.sum() * 100,
                     "top_mgmt_headcount": t.headcount.sum(),
                     "white_pct_top_mgmt": t[t.race == "White"].headcount.sum() / t.headcount.sum() * 100,
                     "female_pct_top_mgmt": t[t.gender == "Female"].headcount.sum() / t.headcount.sum() * 100})
    out = pd.DataFrame(rows).round(1)
    out["top_mgmt_to_workforce_ratio"] = (out.white_pct_top_mgmt / out.white_pct_workforce).round(1)
    return out.sort_values("white_pct_top_mgmt", ascending=False)

sect = gap_table("sector_group"); sect = sect[sect.sector_group != "Unknown"]
sect.to_csv(T / "05_sector_2024.csv", index=False)
size_t = gap_table("size_band"); size_t.to_csv(T / "06_size_band_2024.csv", index=False)
gap_table("business_type").to_csv(T / "07_business_type_2024.csv", index=False)
gap_table("province").to_csv(T / "08_province_2024.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.4))
s = sect.iloc[::-1]
yy = range(len(s))
for y, a, b in zip(yy, s.white_pct_workforce, s.white_pct_top_mgmt):
    ax.plot([a, b], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
ax.scatter(s.white_pct_workforce, yy, s=70, color=AQUA, edgecolor=SURF, linewidth=2, zorder=3, label="Permanent workforce")
ax.scatter(s.white_pct_top_mgmt, yy, s=70, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3, label="Top management")
for y, b, n in zip(yy, s.white_pct_top_mgmt, s.employers):
    ax.text(b + 2, y, f"{b:.0f}%", va="center", fontsize=8.5, color=INK)
ax.set_yticks(list(yy)); ax.set_yticklabels([f"{g} ({n})" for g, n in zip(s.sector_group, s.employers)])
ax.set_xlim(0, 100); ax.set_xlabel("White share, 2024 (%)  ·  (n employers)"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2)
fig.suptitle("Agriculture: 9% of workers are white, 84% of top managers are", x=0.01, ha="left", fontsize=11, color=INK)
save(fig, "04_sector_gap_2024.png")

# ---------------------------------------------------------------- 4. occupational shift & gender
lvl = x.groupby(["year", "level"], observed=True)["headcount"].sum().unstack()
lvl.to_csv(T / "09_headcount_by_level_year.csv")
chg = ((lvl.loc[2024] / lvl.loc[2013] - 1) * 100).round(1).reindex(LEVELS)
chg.rename("pct_change_2013_2024").to_csv(T / "10_headcount_change_by_level.csv")
fem = x.groupby(["year", "level", "gender"], observed=True)["headcount"].sum().unstack()
fem_pct = (fem.Female / fem.sum(axis=1) * 100).round(1).unstack("level")[LEVELS]
fem_pct.to_csv(T / "11_female_share_by_level_year.csv")

fig, ax = plt.subplots(figsize=(8, 4))
cols = [BLUE if v >= 0 else ORANGE for v in chg.values]
ax.barh(range(len(chg))[::-1], chg.values, color=cols, height=0.6)
for y, v in zip(range(len(chg))[::-1], chg.values):
    ax.text(v + (1 if v >= 0 else -1), y, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.set_yticks(range(len(chg))[::-1]); ax.set_yticklabels(chg.index); ax.axvline(0, color=INK2, lw=1)
ax.set_xlim(-40, 15); ax.set_xlabel("Change in headcount, 2013 to 2024 (%)"); ax.grid(axis="y", visible=False)
fig.suptitle("Jobs shrank at the bottom: unskilled and temporary headcount down ~30%", x=0.01, ha="left", fontsize=11, color=INK)
save(fig, "05_headcount_change_by_level.png")

# ---------------------------------------------------------------- 5. the dominant employer
k = p[p.dominant_outlier].groupby(["year", "level"], observed=True)["headcount"].sum().unstack()[LEVELS]
k.to_csv(T / "12_kzn_doe_by_level_year.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(k.index, k["Temporary employees"] / 1000, color=ORANGE, lw=2, marker="o", ms=4)
ax.plot(k.index, k["Professional / mid-management"] / 1000, color=BLUE, lw=2, marker="o", ms=4)
ax.plot(k.index, k["Skilled technical / junior management"] / 1000, color=AQUA, lw=2, marker="o", ms=4)
for col, lab, c in [("Temporary employees", "Temporary", ORANGE),
                    ("Professional / mid-management", "Professional / mid-mgmt", BLUE),
                    ("Skilled technical / junior management", "Skilled technical", AQUA)]:
    ax.text(2024.25, k[col].iloc[-1] / 1000, lab, fontsize=9, va="center")
ax.set_xlim(2012.6, 2026.8); ax.set_xticks(range(2013, 2025)); ax.set_ylabel("Headcount (thousands)")
ax.grid(axis="x", visible=False)
fig.suptitle("KZN Dept of Education: a 2019 reclassification and a 2022 temporary-staff spike", x=0.01, ha="left", fontsize=11, color=INK)
save(fig, "06_kzn_doe.png")

print(overview); print(race_level_pct); print(tm_race_pct); print(firm_tm); print(trans)
print(sect); print(size_t); print(chg); print(fem_pct[["Top management", "Senior management"]])
