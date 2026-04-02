"""
data_pipeline.py
Loads, cleans, and merges the three raw data sources into a single
model-ready DataFrame.  Import this from any other module.
"""

import re
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
COMBINE_FILE = "Copy_of_TAMIDS_Combine_Data.xlsx"
CFB_FILE     = "Copy_of_TE_CFB_Stats_2004-2025.xlsx"
NFL_FILE     = "Copy_of_TE_NFL_STATS_2000_2024.xlsx"


# ── helpers ────────────────────────────────────────────────────────────────

def _parse_draft(s):
    """Return (round, pick, year) from 'Team / Xst / Ypick / ZZZZ' string."""
    if pd.isna(s):
        return None, None, None
    m = re.search(r"(\d+)(?:st|nd|rd|th) / (\d+)(?:st|nd|rd|th) pick / (\d+)", str(s))
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def _parse_height(h):
    """Height is stored as a datetime-like string '2026-06-04' → 6'4\" = 76 in."""
    if pd.isna(h):
        return None
    try:
        parts = str(h).split("-")
        if len(parts) >= 3:
            return int(parts[1]) * 12 + int(parts[2].split(" ")[0])
    except Exception:
        pass
    return None


# ── loaders ────────────────────────────────────────────────────────────────

def load_combine(sheet="2000-2025 TE Data"):
    df = pd.read_excel(COMBINE_FILE, sheet_name=sheet)
    df[["round", "pick", "draft_year"]] = df["Drafted (tm/rnd/yr)"].apply(
        lambda x: pd.Series(_parse_draft(x))
    )
    df["height_in"] = df["Ht"].apply(_parse_height)
    return df


def load_combine_2026():
    df = pd.read_excel(COMBINE_FILE, sheet_name="2026 TE Data")
    df["height_in"] = df["Ht"].apply(_parse_height)
    # 2026 players have no draft slot yet
    df["round"] = None
    df["pick"]  = None
    df["draft_year"] = 2026
    return df


def load_cfb():
    df = pd.read_excel(CFB_FILE, sheet_name="2004-2025")
    reg = df[df["Season Type"] == "regular"].copy()

    # Career college totals per player
    agg = reg.groupby("Player Name").agg(
        cfb_seasons       = ("Season",         "nunique"),
        cfb_rec_yds       = ("receiving_YDS",   "sum"),
        cfb_rec_td        = ("receiving_TD",    "sum"),
        cfb_rec           = ("receiving_REC",   "sum"),
        cfb_ypr           = ("receiving_YPR",   "mean"),
        cfb_rush_yds      = ("rushing_YDS",     "sum"),
        cfb_rush_td       = ("rushing_TD",      "sum"),
        cfb_ppa_total     = ("ppa_overall_total","sum"),
        cfb_ppa_pass      = ("ppa_pass_total",  "sum"),
        cfb_usage_pass    = ("usage_pass",      "mean"),
        cfb_usage_overall = ("usage_overall",   "mean"),
    ).reset_index().rename(columns={"Player Name": "name"})

    # Derived: receiving yards per season
    agg["cfb_rec_yds_per_season"] = agg["cfb_rec_yds"] / agg["cfb_seasons"].clip(lower=1)
    agg["cfb_rec_td_per_season"]  = agg["cfb_rec_td"]  / agg["cfb_seasons"].clip(lower=1)

    return agg


def load_nfl():
    df = pd.read_excel(NFL_FILE)
    reg = df[df["season_type"] == "REG"].copy()

    career = reg.groupby("player_name").agg(
        career_rec_yds   = ("receiving_yards", "sum"),
        career_rec_td    = ("receiving_tds",   "sum"),
        career_rec       = ("receptions",      "sum"),
        career_epa       = ("receiving_epa",   "sum"),
        career_games     = ("games",           "sum"),
        career_tgt       = ("targets",         "sum"),
        career_fantasy   = ("fantasy_points_ppr","sum"),
        career_seasons   = ("season",          "nunique"),
        career_yac       = ("receiving_yards_after_catch","sum"),
        peak_rec_yds     = ("receiving_yards", "max"),     # best single season
        peak_fantasy     = ("fantasy_points_ppr","max"),
    ).reset_index().rename(columns={"player_name": "name"})

    # Efficiency derived
    career["career_ypr"]  = career["career_rec_yds"] / career["career_rec"].clip(lower=1)
    career["career_catch_rate"] = career["career_rec"] / career["career_tgt"].clip(lower=1)

    return career


# ── master merge ───────────────────────────────────────────────────────────

def build_dataset():
    """Returns (df_train, df_2026) with all features merged."""
    combine  = load_combine()
    cfb      = load_cfb()
    nfl      = load_nfl()
    p2026    = load_combine_2026()

    # Merge combine → CFB → NFL
    def merge_all(base):
        df = base.merge(cfb, left_on="Player", right_on="name", how="left")
        df = df.merge(nfl, left_on="Player", right_on="name", how="left")
        return df

    train = merge_all(combine)
    test  = merge_all(p2026)

    # ── Success label ──────────────────────────────────────────────────────
    # Threshold: career ≥ 1 500 rec yards OR ≥ 10 TDs  (meaningful starter)
    train["success"] = (
        (train["career_rec_yds"] > 1500) | (train["career_rec_td"] >= 10)
    ).astype(float)
    # Players with no NFL data at all → unknown, keep NaN
    train.loc[train["career_rec_yds"].isna(), "success"] = np.nan

    # ── Career EPA normalised to games played ──────────────────────────────
    for df in [train, test]:
        df["career_epa_per_game"] = df["career_epa"] / df["career_games"].clip(lower=1)

    return train, test


# ── feature spec ───────────────────────────────────────────────────────────

FEATURES = [
    # combine / athleticism
    "40yd", "Wt", "height_in", "Vertical", "Broad Jump", "3Cone", "Shuttle",
    # draft capital
    "pick",
    # college production
    "cfb_rec_yds", "cfb_rec_td", "cfb_rec",
    "cfb_ypr", "cfb_ppa_total", "cfb_ppa_pass",
    "cfb_usage_pass", "cfb_rec_yds_per_season", "cfb_rec_td_per_season",
    "cfb_rush_yds",
]

FEATURE_LABELS = {
    "40yd":                  "40-Yard Dash",
    "Wt":                    "Weight (lbs)",
    "height_in":             "Height (in)",
    "Vertical":              "Vertical Jump",
    "Broad Jump":            "Broad Jump",
    "3Cone":                 "3-Cone Drill",
    "Shuttle":               "Shuttle Time",
    "pick":                  "Draft Pick #",
    "cfb_rec_yds":           "Career College Rec Yds",
    "cfb_rec_td":            "Career College Rec TDs",
    "cfb_rec":               "Career College Receptions",
    "cfb_ypr":               "College Yards / Reception",
    "cfb_ppa_total":         "College Overall PPA",
    "cfb_ppa_pass":          "College Pass PPA",
    "cfb_usage_pass":        "College Pass Usage Rate",
    "cfb_rec_yds_per_season":"College Rec Yds / Season",
    "cfb_rec_td_per_season": "College Rec TDs / Season",
    "cfb_rush_yds":          "Career College Rush Yds",
}

if __name__ == "__main__":
    train, test = build_dataset()
    print(f"Train rows : {len(train)}")
    print(f"Labelled   : {train['success'].notna().sum()}")
    print(f"Success rate: {train['success'].mean():.1%}")
    print(f"\n2026 prospects: {len(test)}")
    print(test[["Player", "40yd", "Wt", "cfb_rec_yds"]].to_string())
