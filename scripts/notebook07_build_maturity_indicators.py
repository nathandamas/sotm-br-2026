from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd

FINAL_REFERENCE_MONTHS = 12
INITIAL_GROWTH_MONTHS = 24
RECENT_MONTHS = 24
PLATEAU_WINDOW_MONTHS = 12
PLATEAU_REL_TOL = 0.02
PLATEAU_ABS_TOL_M = 50.0
PLATEAU_PERSISTENCE_FRACTION = 0.80
MILESTONES = (0.50, 0.80, 0.90, 0.95)


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, start.parent, *start.parents]:
        if (candidate / "notebooks").exists() and (candidate / "Petropolis").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def month_distance(a, b):
    if pd.isna(a) or pd.isna(b):
        return pd.NA
    return int(b.ordinal - a.ordinal)


def longest_streak(values, target=True):
    best = current = 0
    for value in values:
        if bool(value) == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def normalized_hhi(values):
    x = np.asarray(values, dtype=float)
    total = x.sum()
    if total <= 0:
        return np.nan
    shares = x[x > 0] / total
    n = len(shares)
    if n == 1:
        return 1.0
    hhi = float(np.square(shares).sum())
    return float((hhi - 1 / n) / (1 - 1 / n))


def safe_ratio(num, den):
    return np.nan if den is None or pd.isna(den) or den == 0 else float(num / den)


def robust_window_net_slope(values, endpoint_span=3):
    """Robust cumulative slope using median levels at both window endpoints."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return np.nan
    span = max(1, min(int(endpoint_span), len(y) // 2 if len(y) >= 4 else 1))
    start_level = float(np.median(y[:span]))
    end_level = float(np.median(y[-span:]))
    separation = len(y) - 1
    return float((end_level - start_level) / separation) if separation > 0 else np.nan


def detect_plateau(g, reference_m, first_90_idx):
    if reference_m <= 0 or first_90_idx is None:
        return None, np.nan
    tol = max(PLATEAU_ABS_TOL_M, PLATEAU_REL_TOL * reference_m)
    road = g["road_length_m"].to_numpy(dtype=float)
    for start in range(first_90_idx, len(g) - PLATEAU_WINDOW_MONTHS + 1):
        window = road[start : start + PLATEAU_WINDOW_MONTHS]
        if np.nanmax(window) - np.nanmin(window) > tol:
            continue
        tail = road[start:]
        persistence = np.mean(np.abs(tail - reference_m) <= tol)
        if persistence >= PLATEAU_PERSISTENCE_FRACTION:
            return start, float(persistence)
    return None, np.nan


PROJECT_ROOT = find_project_root(Path.cwd())
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "Petropolis"
TIMESERIES_FILE = RESULTS_DIR / "petropolis_ohsome_monthly_timeseries_895_cells.parquet"
GRID_FILE = RESULTS_DIR / "petropolis_grid_1km_with_temporal_summary.parquet"
OUTPUT_INDICATORS = RESULTS_DIR / "petropolis_osm_maturity_indicators_895_cells.parquet"
OUTPUT_GEO = RESULTS_DIR / "petropolis_grid_1km_with_maturity_indicators.parquet"
OUTPUT_DICTIONARY = RESULTS_DIR / "petropolis_osm_maturity_indicator_dictionary.csv"

for p in (TIMESERIES_FILE, GRID_FILE):
    if not p.exists():
        raise FileNotFoundError(f"Required Notebook 06 output not found: {p}")

panel = pd.read_parquet(TIMESERIES_FILE)
grid = gpd.read_parquet(GRID_FILE)
panel["month_period"] = pd.PeriodIndex(panel["month"], freq="M")
panel = panel.sort_values(["grid_1km_id", "month_period"]).reset_index(drop=True)

PRIMARY = [
    "road_length_m",
    "road_element_count",
    "contributors_count",
    "contributions_count",
    "creation_contributions_count",
    "deletion_contributions_count",
    "tag_change_contributions_count",
    "geometry_change_contributions_count",
]
expected_rows = 895 * panel["month"].nunique()
input_checks = {
    "panel_rows_complete": len(panel) == expected_rows,
    "cell_month_unique": not panel.duplicated(["grid_1km_id", "month"]).any(),
    "panel_895_cells": panel["grid_1km_id"].nunique() == 895,
    "grid_895_cells": len(grid) == 895,
    "ids_match_grid": set(panel["grid_1km_id"].astype(str)) == set(grid["grid_1km_id"].astype(str)),
    "primary_non_null": panel[PRIMARY].notna().all().all(),
    "primary_nonnegative": (panel[PRIMARY] >= 0).all().all(),
}
print(pd.DataFrame(input_checks.items(), columns=["check", "pass"]).to_string(index=False))
if not all(input_checks.values()):
    raise ValueError(f"Notebook 06 input validation failed: {input_checks}")

rows = []
for grid_id, g in panel.groupby("grid_1km_id", sort=True):
    g = g.sort_values("month_period").reset_index(drop=True).copy()
    months = g["month_period"]
    last_month = months.iloc[-1]
    road = g["road_length_m"].astype(float)
    delta = road.diff()
    active = g["contributions_count"].gt(0).to_numpy()

    road_idx = np.flatnonzero(road.to_numpy() > 0)
    first_idx = int(road_idx[0]) if len(road_idx) else None
    first_month = months.iloc[first_idx] if first_idx is not None else pd.NaT
    mapping_age = month_distance(first_month, last_month) if first_idx is not None else pd.NA

    reference_m = float(road.tail(FINAL_REFERENCE_MONTHS).median())
    current_m = float(road.iloc[-1])
    historical_max_m = float(road.max())
    saturation_ratio = safe_ratio(current_m, reference_m)

    milestone_idx = {}
    milestone_time = {}
    for threshold in MILESTONES:
        if reference_m > 0:
            idxs = np.flatnonzero(road.to_numpy() >= threshold * reference_m)
            idx = int(idxs[0]) if len(idxs) else None
        else:
            idx = None
        milestone_idx[threshold] = idx
        milestone_time[threshold] = (
            month_distance(first_month, months.iloc[idx])
            if idx is not None and first_idx is not None
            else pd.NA
        )

    plateau_idx, plateau_tail_fraction = detect_plateau(g, reference_m, milestone_idx[0.90])
    plateau_month = months.iloc[plateau_idx] if plateau_idx is not None else pd.NaT
    plateau_duration = month_distance(plateau_month, last_month) + 1 if plateau_idx is not None else pd.NA

    positive_delta = delta.where(delta > 0)
    if positive_delta.notna().any():
        peak_growth_i = int(positive_delta.idxmax())
        peak_growth_month = months.iloc[peak_growth_i]
        peak_growth_m = float(delta.iloc[peak_growth_i])
    else:
        peak_growth_month = pd.NaT
        peak_growth_m = 0.0

    if first_idx is not None:
        initial_road = road.iloc[first_idx : min(first_idx + INITIAL_GROWTH_MONTHS, len(g))]
        growth_initial = robust_window_net_slope(initial_road.to_numpy(), endpoint_span=3)
    else:
        growth_initial = np.nan
    recent_road = road.tail(min(RECENT_MONTHS, len(road)))
    growth_recent = robust_window_net_slope(recent_road.to_numpy(), endpoint_span=3)
    deceleration = (
        growth_initial - growth_recent
        if not (pd.isna(growth_initial) or pd.isna(growth_recent))
        else np.nan
    )
    deceleration_ratio = (
        safe_ratio(deceleration, abs(growth_initial))
        if not pd.isna(deceleration) and abs(growth_initial) > 1e-9
        else np.nan
    )

    active_months = int(active.sum())
    active_ratio = float(active.mean())
    recent_active_ratio = float(active[-min(RECENT_MONTHS, len(active)) :].mean())
    active_indices = np.flatnonzero(active)
    months_since_last_activity = int(len(g) - 1 - active_indices[-1]) if len(active_indices) else pd.NA
    post_plateau_active_ratio = float(active[plateau_idx:].mean()) if plateau_idx is not None else np.nan
    persistence_index = float(math.sqrt(active_ratio * recent_active_ratio))

    contrib = g["contributions_count"].astype(float)
    users = g["contributors_count"].astype(float)
    contrib_total = float(contrib.sum())
    contributor_month_sum = float(users.sum())
    hist = contrib.iloc[:-RECENT_MONTHS] if len(g) > RECENT_MONTHS else contrib.iloc[0:0]
    recent = contrib.tail(RECENT_MONTHS)
    historical_mean = float(hist.mean()) if len(hist) else np.nan
    recent_mean = float(recent.mean())

    active_users = users[users > 0]
    if len(active_users) >= 2 and active_users.mean() > 0:
        contributor_volatility = float(active_users.std(ddof=0) / active_users.mean())
    elif len(active_users) == 1:
        contributor_volatility = 0.0
    else:
        contributor_volatility = np.nan

    type_totals = {
        "creation": float(g["creation_contributions_count"].sum()),
        "deletion": float(g["deletion_contributions_count"].sum()),
        "tag_change": float(g["tag_change_contributions_count"].sum()),
        "geometry_change": float(g["geometry_change_contributions_count"].sum()),
    }
    type_event_sum = sum(type_totals.values())

    rows.append(
        {
            "grid_1km_id": str(grid_id),
            "first_mapping_month": str(first_month) if first_idx is not None else None,
            "mapping_age_months": mapping_age,
            "current_road_length_km": current_m / 1000.0,
            "recent_reference_road_length_km": reference_m / 1000.0,
            "historical_max_road_length_km": historical_max_m / 1000.0,
            "saturation_ratio": saturation_ratio,
            "peak_growth_month": str(peak_growth_month) if not pd.isna(peak_growth_month) else None,
            "peak_growth_m": peak_growth_m,
            "positive_growth_months": int((delta > 0).sum()),
            "negative_growth_months": int((delta < 0).sum()),
            "growth_rate_initial_m_per_month": growth_initial,
            "growth_rate_recent_m_per_month": growth_recent,
            "growth_deceleration_m_per_month": deceleration,
            "growth_deceleration_ratio": deceleration_ratio,
            "time_to_50pct_months": milestone_time[0.50],
            "time_to_80pct_months": milestone_time[0.80],
            "time_to_90pct_months": milestone_time[0.90],
            "time_to_95pct_months": milestone_time[0.95],
            "months_since_90pct": month_distance(months.iloc[milestone_idx[0.90]], last_month) if milestone_idx[0.90] is not None else pd.NA,
            "plateau_start_month": str(plateau_month) if plateau_idx is not None else None,
            "plateau_duration_months": plateau_duration,
            "plateau_tail_stability_fraction": plateau_tail_fraction,
            "active_months": active_months,
            "active_month_ratio": active_ratio,
            "longest_active_streak_months": longest_streak(active, True),
            "longest_inactive_streak_months": longest_streak(active, False),
            "months_since_last_activity": months_since_last_activity,
            "post_plateau_active_month_ratio": post_plateau_active_ratio,
            "recent_activity_ratio": recent_active_ratio,
            "persistence_index": persistence_index,
            "total_contributions": contrib_total,
            "mean_monthly_contributions": float(contrib.mean()),
            "median_monthly_contributions": float(contrib.median()),
            "peak_monthly_contributions": float(contrib.max()),
            "total_contributor_months": contributor_month_sum,
            "mean_monthly_contributors": float(users.mean()),
            "median_monthly_contributors": float(users.median()),
            "peak_monthly_contributors": float(users.max()),
            "contributions_per_contributor_month": safe_ratio(contrib_total, contributor_month_sum),
            "creation_contributions_total": type_totals["creation"],
            "deletion_contributions_total": type_totals["deletion"],
            "tag_change_contributions_total": type_totals["tag_change"],
            "geometry_change_contributions_total": type_totals["geometry_change"],
            "creation_to_total_contributions_ratio": safe_ratio(type_totals["creation"], contrib_total),
            "deletion_to_total_contributions_ratio": safe_ratio(type_totals["deletion"], contrib_total),
            "tag_change_to_total_contributions_ratio": safe_ratio(type_totals["tag_change"], contrib_total),
            "geometry_change_to_total_contributions_ratio": safe_ratio(type_totals["geometry_change"], contrib_total),
            "contribution_temporal_concentration_hhi": normalized_hhi(contrib),
            "recent_mean_monthly_contributions": recent_mean,
            "historical_mean_monthly_contributions": historical_mean,
            "recent_to_historical_activity_ratio": safe_ratio(recent_mean, historical_mean),
            "contributor_volatility_proxy": contributor_volatility,
            "expansion_share_proxy": safe_ratio(type_totals["creation"], type_event_sum),
            "maintenance_share_proxy": safe_ratio(type_totals["tag_change"] + type_totals["geometry_change"], type_event_sum),
            "deletion_share_proxy": safe_ratio(type_totals["deletion"], type_event_sum),
        }
    )

indicators = pd.DataFrame(rows)

# Explicit QC includes non-degeneracy checks discovered by Notebook 08.
def qc_status(condition, warning=False):
    if condition:
        return "PASS"
    return "WARNING" if warning else "FAIL"

ratio_cols = ["active_month_ratio", "recent_activity_ratio", "persistence_index"]
milestone_cols = ["time_to_50pct_months", "time_to_80pct_months", "time_to_90pct_months", "time_to_95pct_months", "months_since_90pct"]
unexpected_core_na = indicators[["grid_1km_id", "current_road_length_km", "active_months", "active_month_ratio", "recent_activity_ratio", "persistence_index", "total_contributions"]].isna().sum().sum()
checks = [
    ("rows_895", qc_status(len(indicators) == 895), len(indicators)),
    ("ids_unique", qc_status(indicators["grid_1km_id"].is_unique), indicators["grid_1km_id"].nunique()),
    ("ids_match_grid", qc_status(set(indicators["grid_1km_id"]) == set(grid["grid_1km_id"].astype(str))), ""),
    ("core_no_unexpected_na", qc_status(unexpected_core_na == 0), int(unexpected_core_na)),
    ("road_lengths_nonnegative", qc_status((indicators["current_road_length_km"] >= 0).all()), float(indicators["current_road_length_km"].min())),
    ("activity_ratios_0_1", qc_status(((indicators[ratio_cols] >= 0) & (indicators[ratio_cols] <= 1)).all().all()), ""),
    ("milestones_nonnegative_when_defined", qc_status((indicators[milestone_cols].dropna() >= 0).all().all()), ""),
    ("plateau_duration_nonnegative_when_defined", qc_status((indicators["plateau_duration_months"].dropna() >= 0).all()), ""),
    ("plateau_stability_at_least_80pct_when_defined", qc_status((indicators["plateau_tail_stability_fraction"].dropna() >= PLATEAU_PERSISTENCE_FRACTION).all()), ""),
    ("growth_initial_non_degenerate", qc_status(indicators["growth_rate_initial_m_per_month"].dropna().nunique() > 1), int(indicators["growth_rate_initial_m_per_month"].dropna().nunique())),
    ("growth_deceleration_defined_for_some_cells", qc_status(indicators["growth_deceleration_ratio"].notna().sum() > 0), int(indicators["growth_deceleration_ratio"].notna().sum())),
    ("contributor_volatility_non_degenerate", qc_status(indicators["contributor_volatility_proxy"].dropna().nunique() > 1), int(indicators["contributor_volatility_proxy"].dropna().nunique())),
]
qc = pd.DataFrame(checks, columns=["check", "status", "value"])
print(qc.to_string(index=False))
if (qc["status"] == "FAIL").any():
    raise ValueError("Notebook 07 quality control contains FAIL results.")

# Variable dictionary.
dictionary_rows = [
    ("grid_1km_id", "Official 1 km grid-cell identifier", "IBGE identifier", "id", "Spatial key"),
    ("first_mapping_month", "First month with mapped road length > 0", "first t where L_t > 0", "YYYY-MM", "Intrinsic onset of mapped-road presence"),
    ("mapping_age_months", "Months from first mapped-road presence to final complete month", "t_end - t_first", "months", "Intrinsic mapping age"),
    ("current_road_length_km", "Mapped road length in final complete month", "L_T / 1000", "km", "Current mapped-road state"),
    ("recent_reference_road_length_km", "Median mapped road length over final 12 months", "median(L_{T-11:T}) / 1000", "km", "Robust recent intrinsic reference; not external completeness"),
    ("saturation_ratio", "Current road length relative to recent reference", "L_T / median(L_{T-11:T})", "ratio", "Convergence to recent intrinsic level; may exceed 1"),
    ("growth_rate_initial_m_per_month", "Robust net road-length slope in first 24 months after mapping onset", "(median(last 3 L)-median(first 3 L))/(window months-1)", "m/month", "Early cumulative growth rate resistant to isolated monthly anomalies"),
    ("growth_rate_recent_m_per_month", "Robust net road-length slope in last 24 months", "(median(last 3 L)-median(first 3 L))/(window months-1)", "m/month", "Recent cumulative growth rate resistant to isolated monthly anomalies"),
    ("growth_deceleration_m_per_month", "Difference between robust early and recent net growth slopes", "g_initial-g_recent", "m/month", "Positive values indicate deceleration of mapped-road expansion"),
    ("growth_deceleration_ratio", "Growth deceleration relative to absolute early growth", "(g_initial-g_recent)/abs(g_initial)", "ratio", "Relative deceleration; undefined when early growth is effectively zero"),
    ("time_to_90pct_months", "Months from mapping onset to 90% of recent reference", "t_90-t_first", "months", "Trajectory milestone"),
    ("plateau_duration_months", "Months from plateau start through final complete month", "T-t_plateau+1", "months", "Duration of sustained plateau"),
    ("active_month_ratio", "Fraction of complete months with at least one contribution", "active months/all months", "ratio 0-1", "Long-run activity persistence"),
    ("recent_activity_ratio", "Fraction of last 24 months with contributions", "active recent months/24", "ratio 0-1", "Recent activity persistence"),
    ("persistence_index", "Geometric mean of long-run and recent activity ratios", "sqrt(active_month_ratio*recent_activity_ratio)", "ratio 0-1", "Transparent joint persistence indicator"),
    ("total_contributions", "Sum of monthly OSM contributions", "sum(C_t)", "contributions", "Total activity intensity"),
    ("contribution_temporal_concentration_hhi", "Normalized HHI of contributions across active months", "normalized sum((C_t/sum C)^2)", "ratio 0-1", "0 dispersed over time; 1 concentrated in one month"),
    ("contributor_volatility_proxy", "Coefficient of variation of contributor counts across active months", "sd(U|U>0)/mean(U|U>0)", "ratio", "Contributor-intensity volatility; not identity-based churn"),
    ("expansion_share_proxy", "Creation-type events relative to all counted type events", "creation/(creation+deletion+tag+geometry)", "ratio", "Expansion-oriented event proxy; event types can overlap"),
    ("maintenance_share_proxy", "Tag and geometry change events relative to all counted type events", "(tag+geometry)/(creation+deletion+tag+geometry)", "ratio", "Maintenance-oriented event proxy; event types can overlap"),
]
dictionary = pd.DataFrame(dictionary_rows, columns=["variable", "definition", "formula", "unit", "interpretation"])
for col in indicators.columns:
    if col not in set(dictionary["variable"]):
        dictionary.loc[len(dictionary)] = [col, "Derived component indicator; see Notebook 07 code and column name.", "See implementation.", "see variable", "Retained as a separate decomposable component."]

geo = grid.copy()
overlap = [c for c in indicators.columns if c != "grid_1km_id" and c in geo.columns]
if overlap:
    geo = geo.drop(columns=overlap)
geo = geo.merge(indicators, on="grid_1km_id", how="left", validate="one_to_one")
if len(geo) != 895 or not geo["grid_1km_id"].is_unique:
    raise ValueError("GeoParquet merge failed integrity checks.")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
indicators.to_parquet(OUTPUT_INDICATORS, index=False)
geo.to_parquet(OUTPUT_GEO, index=False)
dictionary.sort_values("variable").to_csv(OUTPUT_DICTIONARY, index=False)

print("Cells with roads:", int((indicators["current_road_length_km"] > 0).sum()))
print("Cells with detected plateau:", int(indicators["plateau_start_month"].notna().sum()))
print("Cells with any contributions:", int((indicators["total_contributions"] > 0).sum()))
print("Initial growth unique non-null values:", indicators["growth_rate_initial_m_per_month"].dropna().nunique())
print("Growth deceleration ratio defined cells:", indicators["growth_deceleration_ratio"].notna().sum())
print("Contributor volatility unique non-null values:", indicators["contributor_volatility_proxy"].dropna().nunique())
print("Saved:", OUTPUT_INDICATORS)
print("Saved:", OUTPUT_GEO)
print("Saved:", OUTPUT_DICTIONARY)
