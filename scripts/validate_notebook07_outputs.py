from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'data' / 'processed' / 'Petropolis'
IND_FILE = RESULTS / 'petropolis_osm_maturity_indicators_895_cells.parquet'
DICT_FILE = RESULTS / 'petropolis_osm_maturity_indicator_dictionary.csv'

ind = pd.read_parquet(IND_FILE)

specs = {
'grid_1km_id': ('Official 1 km grid-cell identifier.', 'IBGE 1 km grid identifier.', 'id', 'Spatial key; unique by cell.'),
'first_mapping_month': ('First complete month in which mapped road length is greater than zero.', 'min(t | L_t > 0)', 'YYYY-MM', 'Intrinsic onset of mapped-road presence; undefined for cells without roads.'),
'mapping_age_months': ('Elapsed months from first mapped-road presence through the final complete month.', 'T - t_first', 'months', 'Intrinsic mapping age; undefined for cells without roads.'),
'current_road_length_km': ('Mapped road length in the final complete month.', 'L_T / 1000', 'km', 'Current mapped-road state.'),
'recent_reference_road_length_km': ('Robust recent mapped-road reference based on the final 12 complete months.', 'median(L_{T-11:T}) / 1000', 'km', 'Intrinsic reference only; not external completeness.'),
'historical_max_road_length_km': ('Maximum mapped road length observed over the complete monthly series.', 'max_t(L_t) / 1000', 'km', 'Historical maximum mapped-road state.'),
'saturation_ratio': ('Current road length relative to the recent 12-month median reference.', 'L_T / median(L_{T-11:T})', 'ratio', 'Intrinsic convergence indicator; may exceed 1 after recent expansion and is undefined when the reference is zero.'),
'peak_growth_month': ('Month of the largest positive month-to-month road-length change.', 'argmax_t(Delta L_t | Delta L_t > 0)', 'YYYY-MM', 'Timing of the largest observed expansion pulse; undefined if no positive change is observed.'),
'peak_growth_m': ('Largest positive month-to-month road-length change.', 'max(Delta L_t, Delta L_t > 0)', 'm/month', 'Magnitude of the largest observed expansion pulse; zero when no positive change occurs.'),
'positive_growth_months': ('Number of months with positive net road-length change.', 'sum_t[Delta L_t > 0]', 'months', 'Frequency of net mapped-road expansion.'),
'negative_growth_months': ('Number of months with negative net road-length change.', 'sum_t[Delta L_t < 0]', 'months', 'Frequency of net mapped-road contraction or deletion.'),
'growth_rate_initial_m_per_month': ('Median net monthly road-length change during the first 24 months after mapped-road onset, excluding the onset jump itself.', 'median(Delta L_t), t in first 24 post-onset months', 'm/month', 'Robust post-onset early growth rate; does not duplicate the initial mapping pulse captured by peak_growth_m.'),
'growth_rate_recent_m_per_month': ('Median net monthly road-length change during the final 24 complete months.', 'median(Delta L_t), t in final 24 months', 'm/month', 'Robust recent net growth rate.'),
'growth_deceleration_m_per_month': ('Difference between early post-onset and recent median growth rates.', 'g_initial - g_recent', 'm/month', 'Positive values indicate slower recent growth than early post-onset growth.'),
'growth_deceleration_ratio': ('Growth deceleration normalized by the absolute early post-onset growth rate.', '(g_initial - g_recent) / |g_initial|', 'ratio', 'Relative deceleration; undefined when early median growth is zero or unavailable.'),
'time_to_50pct_months': ('Months from mapped-road onset to first reaching 50% of the recent reference.', 't_50 - t_first', 'months', 'Intrinsic trajectory milestone; undefined for cells without a positive recent reference.'),
'time_to_80pct_months': ('Months from mapped-road onset to first reaching 80% of the recent reference.', 't_80 - t_first', 'months', 'Intrinsic trajectory milestone; undefined for cells without a positive recent reference.'),
'time_to_90pct_months': ('Months from mapped-road onset to first reaching 90% of the recent reference.', 't_90 - t_first', 'months', 'Intrinsic trajectory milestone used as the earliest possible plateau start.'),
'time_to_95pct_months': ('Months from mapped-road onset to first reaching 95% of the recent reference.', 't_95 - t_first', 'months', 'Intrinsic trajectory milestone.'),
'months_since_90pct': ('Months elapsed from first reaching 90% of the recent reference to the final complete month.', 'T - t_90', 'months', 'Duration spent at or beyond the intrinsic 90% milestone, regardless of later fluctuations.'),
'plateau_start_month': ('First month at or after the 90% milestone starting a 12-month low-range window and satisfying the tail persistence rule.', 'first t >= t_90 with range(L_{t:t+11}) <= max(50 m, 0.02*L_ref) and tail persistence >= 0.80', 'YYYY-MM', 'Descriptive plateau onset; not proof of external completeness.'),
'plateau_duration_months': ('Months from detected plateau start through the final complete month.', 'T - t_plateau + 1', 'months', 'Duration of the descriptive plateau regime.'),
'plateau_tail_stability_fraction': ('Fraction of months from plateau start onward lying within the plateau tolerance around the recent reference.', 'mean_t[|L_t - L_ref| <= tol], t >= t_plateau', 'ratio 0-1', 'Tail stability conditional on a detected plateau; required to be at least 0.80.'),
'active_months': ('Number of complete months with at least one road-related OSM contribution.', 'sum_t[C_t > 0]', 'months', 'Total duration of observed contribution activity.'),
'active_month_ratio': ('Fraction of all 224 complete months with at least one contribution.', 'active_months / 224', 'ratio 0-1', 'Long-run contribution persistence.'),
'longest_active_streak_months': ('Longest run of consecutive months with at least one contribution.', 'max consecutive run of [C_t > 0]', 'months', 'Maximum sustained activity spell.'),
'longest_inactive_streak_months': ('Longest run of consecutive months with zero contributions.', 'max consecutive run of [C_t = 0]', 'months', 'Maximum inactivity spell.'),
'months_since_last_activity': ('Months from the last active month to the final complete month.', 'T - max(t | C_t > 0)', 'months', 'Recency of contribution activity; undefined for cells with no contributions.'),
'post_plateau_active_month_ratio': ('Fraction of months from detected plateau start onward with at least one contribution.', 'mean_t[C_t > 0], t >= t_plateau', 'ratio 0-1', 'Persistence of editing after descriptive plateau; undefined when no plateau is detected.'),
'recent_activity_ratio': ('Fraction of the final 24 complete months with at least one contribution.', 'sum_{T-23:T}[C_t > 0] / 24', 'ratio 0-1', 'Recent contribution persistence.'),
'persistence_index': ('Geometric mean of long-run and recent activity ratios.', 'sqrt(active_month_ratio * recent_activity_ratio)', 'ratio 0-1', 'Transparent joint persistence indicator; equals zero if either component is zero.'),
'total_contributions': ('Total road-related OSM contributions across all complete months.', 'sum_t(C_t)', 'contributions', 'Whole-period activity intensity.'),
'mean_monthly_contributions': ('Mean road-related contributions per complete month.', 'mean_t(C_t)', 'contributions/month', 'Average monthly activity including inactive months.'),
'median_monthly_contributions': ('Median road-related contributions per complete month.', 'median_t(C_t)', 'contributions/month', 'Robust typical monthly activity including inactive months.'),
'peak_monthly_contributions': ('Maximum road-related contributions in a single month.', 'max_t(C_t)', 'contributions/month', 'Largest monthly editing pulse.'),
'total_contributor_months': ('Sum of monthly unique-contributor counts.', 'sum_t(U_t)', 'contributor-months', 'Contributor-month exposure; not the number of unique contributors across the whole period.'),
'mean_monthly_contributors': ('Mean monthly number of unique contributors.', 'mean_t(U_t)', 'contributors/month', 'Average contributor intensity including inactive months.'),
'median_monthly_contributors': ('Median monthly number of unique contributors.', 'median_t(U_t)', 'contributors/month', 'Robust typical contributor intensity including inactive months.'),
'peak_monthly_contributors': ('Maximum monthly number of unique contributors.', 'max_t(U_t)', 'contributors/month', 'Largest observed contributor pulse.'),
'contributions_per_contributor_month': ('Total contributions divided by summed monthly contributor counts.', 'sum_t(C_t) / sum_t(U_t)', 'contributions/contributor-month', 'Activity intensity per contributor-month; not per unique person over the whole period.'),
'creation_contributions_total': ('Sum of monthly contributions classified as creation.', 'sum_t(C_creation,t)', 'contributions', 'Expansion-oriented contribution count; contribution-type filters may overlap.'),
'deletion_contributions_total': ('Sum of monthly contributions classified as deletion.', 'sum_t(C_deletion,t)', 'contributions', 'Deletion-oriented contribution count; contribution-type filters may overlap.'),
'tag_change_contributions_total': ('Sum of monthly contributions classified as tag change.', 'sum_t(C_tag,t)', 'contributions', 'Attribute-maintenance contribution count; contribution-type filters may overlap.'),
'geometry_change_contributions_total': ('Sum of monthly contributions classified as geometry change.', 'sum_t(C_geometry,t)', 'contributions', 'Geometry-maintenance contribution count; contribution-type filters may overlap.'),
'creation_to_total_contributions_ratio': ('Creation-classified contributions relative to all contributions.', 'creation_total / total_contributions', 'ratio', 'Expansion-oriented frequency relative to all contributions; may coexist with other type classifications.'),
'deletion_to_total_contributions_ratio': ('Deletion-classified contributions relative to all contributions.', 'deletion_total / total_contributions', 'ratio', 'Deletion-oriented frequency relative to all contributions; may coexist with other type classifications.'),
'tag_change_to_total_contributions_ratio': ('Tag-change-classified contributions relative to all contributions.', 'tag_change_total / total_contributions', 'ratio', 'Attribute-maintenance frequency relative to all contributions; may coexist with other type classifications.'),
'geometry_change_to_total_contributions_ratio': ('Geometry-change-classified contributions relative to all contributions.', 'geometry_change_total / total_contributions', 'ratio', 'Geometry-maintenance frequency relative to all contributions; may coexist with other type classifications.'),
'contribution_temporal_concentration_hhi': ('Normalized Herfindahl-Hirschman concentration of contributions across active months.', '(sum_i s_i^2 - 1/n)/(1 - 1/n), s_i=C_i/sum(C), active months only', 'ratio 0-1', 'Zero approaches temporal dispersion; one means all contributions occur in a single active month; undefined when total contributions are zero.'),
'recent_mean_monthly_contributions': ('Mean contributions in the final 24 complete months.', 'mean(C_{T-23:T})', 'contributions/month', 'Recent activity intensity.'),
'historical_mean_monthly_contributions': ('Mean contributions before the final 24 complete months.', 'mean(C_{1:T-24})', 'contributions/month', 'Earlier-period activity intensity.'),
'recent_to_historical_activity_ratio': ('Recent mean monthly contributions divided by the earlier-period mean.', 'recent_mean / historical_mean', 'ratio', 'Values above 1 indicate higher recent activity; undefined when historical mean is zero.'),
'contributor_volatility_proxy': ('Median absolute month-to-month change in contributor count normalized by mean contributor count in active months.', 'median(|Delta U_t|) / mean(U_t | U_t>0)', 'ratio', 'Contributor-intensity volatility proxy; not identity-based contributor churn.'),
'expansion_share_proxy': ('Creation-type event count relative to the sum of the four counted type-event totals.', 'creation / (creation + deletion + tag_change + geometry_change)', 'ratio 0-1', 'Share of counted type-events oriented to expansion. Denominator may double-count contributions with multiple type labels.'),
'maintenance_share_proxy': ('Tag-change plus geometry-change event counts relative to the sum of the four counted type-event totals.', '(tag_change + geometry_change) / (creation + deletion + tag_change + geometry_change)', 'ratio 0-1', 'Share of counted type-events oriented to maintenance. Denominator may double-count multi-labelled contributions.'),
'deletion_share_proxy': ('Deletion-type event count relative to the sum of the four counted type-event totals.', 'deletion / (creation + deletion + tag_change + geometry_change)', 'ratio 0-1', 'Share of counted type-events oriented to deletion. Together with expansion and maintenance proxies it sums to one when defined, but this is a type-event composition, not a unique-contribution composition.'),
}

missing_specs = sorted(set(ind.columns) - set(specs))
extra_specs = sorted(set(specs) - set(ind.columns))
if missing_specs or extra_specs:
    raise ValueError(f'Dictionary specification mismatch: missing={missing_specs}, extra={extra_specs}')

dictionary = pd.DataFrame([
    {'variable': col, 'definition': specs[col][0], 'formula': specs[col][1], 'unit': specs[col][2], 'interpretation': specs[col][3]}
    for col in ind.columns
])
if dictionary[['definition','formula','unit','interpretation']].isna().any().any():
    raise ValueError('Dictionary contains null documentation fields.')
if dictionary.astype(str).apply(lambda s: s.str.contains('See notebook implementation|Derived component indicator', case=False, regex=True)).any().any():
    raise ValueError('Dictionary still contains generic placeholders.')
dictionary.sort_values('variable').to_csv(DICT_FILE, index=False)

road_mask = ind['current_road_length_km'] > 0
activity_mask = ind['total_contributions'] > 0
plateau_mask = ind['plateau_start_month'].notna()

checks = {
    'rows_895': len(ind) == 895,
    'ids_unique': ind['grid_1km_id'].is_unique,
    'hhi_na_exactly_no_activity': ind['contribution_temporal_concentration_hhi'].isna().equals(~activity_mask),
    'plateau_only_for_cells_with_roads': bool((~plateau_mask | road_mask).all()),
    'plateau_stability_ge_0_80': bool((ind.loc[plateau_mask, 'plateau_tail_stability_fraction'] >= 0.80).all()),
    'type_event_proxy_composition': bool(np.allclose(
        ind.loc[ind['expansion_share_proxy'].notna(), ['expansion_share_proxy','maintenance_share_proxy','deletion_share_proxy']].sum(axis=1),
        1.0, atol=1e-9
    )),
    'dictionary_complete': len(dictionary) == len(ind.columns) and dictionary['variable'].is_unique,
}
for name, ok in checks.items():
    print(f'{name}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise ValueError(f'Post-execution Notebook 07 validation failed: {checks}')

road_cells = int(road_mask.sum())
plateau_cells = int(plateau_mask.sum())
activity_cells = int(activity_mask.sum())
print(f'Cells with roads: {road_cells}')
print(f'Cells with detected plateau: {plateau_cells} ({plateau_cells/road_cells:.1%} of road cells)')
print(f'Cells with any contributions: {activity_cells}')
print('Expected NA counts:')
for col in [
    'first_mapping_month','mapping_age_months','saturation_ratio','time_to_50pct_months','time_to_80pct_months',
    'time_to_90pct_months','time_to_95pct_months','months_since_90pct','plateau_start_month','plateau_duration_months',
    'post_plateau_active_month_ratio','contribution_temporal_concentration_hhi','recent_to_historical_activity_ratio',
    'contributor_volatility_proxy','expansion_share_proxy','maintenance_share_proxy','deletion_share_proxy'
]:
    print(f'  {col}: {int(ind[col].isna().sum())}')

print('Selected indicator distributions:')
selected = [
    'time_to_50pct_months','time_to_80pct_months','time_to_90pct_months','time_to_95pct_months',
    'months_since_90pct','plateau_duration_months','active_month_ratio','recent_activity_ratio','persistence_index',
    'contribution_temporal_concentration_hhi','recent_to_historical_activity_ratio',
    'expansion_share_proxy','maintenance_share_proxy','deletion_share_proxy'
]
print(ind[selected].describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).T.to_string())
