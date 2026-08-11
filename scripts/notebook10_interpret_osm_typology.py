from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, fisher_exact
from libpysal.weights import Queen

RANDOM_STATE = 42
PERMUTATIONS = 999
ALPHA = 0.05


def find_root(start: Path) -> Path:
    p = start.resolve()
    for c in [p, *p.parents]:
        if (c / 'notebooks').exists() and (c / 'scripts').exists():
            return c
    raise FileNotFoundError('Repository root not found.')


ROOT = find_root(Path.cwd())
OUT = ROOT / 'data' / 'processed' / 'Petropolis'
FIG = OUT / 'figures' / 'notebook10'
FIG.mkdir(parents=True, exist_ok=True)

TYPOLOGY_FILE = OUT / 'petropolis_osm_cell_typology_895_cells.parquet'
SPATIAL_FILE = OUT / 'petropolis_grid_1km_with_osm_typology.parquet'
PROFILE_FILE = OUT / 'petropolis_osm_typology_profiles.csv'
SELECTED_MODEL_FILE = OUT / 'petropolis_osm_typology_selected_model.csv'

for p in [TYPOLOGY_FILE, SPATIAL_FILE, PROFILE_FILE, SELECTED_MODEL_FILE]:
    if not p.exists():
        raise FileNotFoundError(p)

typology = pd.read_parquet(TYPOLOGY_FILE)
gdf = gpd.read_parquet(SPATIAL_FILE)
profiles_nb09 = pd.read_csv(PROFILE_FILE)
selected_model = pd.read_csv(SELECTED_MODEL_FILE)

typology['grid_1km_id'] = typology['grid_1km_id'].astype(str)
gdf['grid_1km_id'] = gdf['grid_1km_id'].astype(str)

assert len(typology) == 895 and typology['grid_1km_id'].is_unique
assert len(gdf) == 895 and gdf['grid_1km_id'].is_unique
assert set(typology['grid_1km_id']) == set(gdf['grid_1km_id'])
assert gdf['typology_class_id'].notna().all()

CLASS_ORDER = [0, 1, 2]
CLASS_LABELS = {
    0: 'Structural class: no current mapped roads',
    1: 'Provisional cluster 1',
    2: 'Provisional cluster 2',
}

print('Input rows:', len(gdf))
print('CRS:', gdf.crs)
print('Class sizes:')
print(gdf.groupby(['typology_class_id', 'typology_label']).size().to_string())

# -----------------------------------------------------------------------------
# 1. Substantive indicator sets
# -----------------------------------------------------------------------------
# Variables available for all or most classes, useful for broad characterization.
BROAD_INDICATORS = [
    'current_road_length_km',
    'active_month_ratio',
    'recent_activity_ratio',
    'persistence_index',
    'total_contributions',
    'total_contributor_months',
    'mean_monthly_contributions',
    'peak_monthly_contributions',
    'contribution_temporal_concentration_hhi',
    'recent_to_historical_activity_ratio',
    'expansion_share_proxy',
    'maintenance_share_proxy',
    'deletion_share_proxy',
    'contributor_volatility_proxy',
]

# Variables whose substantive interpretation requires a mapped-road history.
ROAD_HISTORY_INDICATORS = [
    'mapping_age_months',
    'peak_growth_m',
    'positive_growth_months',
    'negative_growth_months',
    'growth_rate_initial_m_per_month',
    'growth_rate_recent_m_per_month',
    'growth_deceleration_m_per_month',
    'time_to_50pct_months',
    'time_to_80pct_months',
    'time_to_90pct_months',
    'time_to_95pct_months',
    'months_since_90pct',
    'plateau_duration_months',
    'plateau_tail_stability_fraction',
    'post_plateau_active_month_ratio',
]

BROAD_INDICATORS = [c for c in BROAD_INDICATORS if c in gdf.columns]
ROAD_HISTORY_INDICATORS = [c for c in ROAD_HISTORY_INDICATORS if c in gdf.columns]

# -----------------------------------------------------------------------------
# 2. Article-ready class summary
# -----------------------------------------------------------------------------
class_rows = []
for cid in CLASS_ORDER:
    group = gdf[gdf['typology_class_id'].eq(cid)].copy()
    class_rows.append({
        'typology_class_id': cid,
        'typology_label': CLASS_LABELS[cid],
        'n_cells': len(group),
        'share_of_895_pct': len(group) / len(gdf) * 100.0,
        'median_current_road_length_km': pd.to_numeric(group.get('current_road_length_km'), errors='coerce').median(),
        'median_active_month_ratio': pd.to_numeric(group.get('active_month_ratio'), errors='coerce').median(),
        'median_recent_activity_ratio': pd.to_numeric(group.get('recent_activity_ratio'), errors='coerce').median(),
        'median_total_contributions': pd.to_numeric(group.get('total_contributions'), errors='coerce').median(),
        'median_total_contributor_months': pd.to_numeric(group.get('total_contributor_months'), errors='coerce').median(),
        'median_temporal_concentration_hhi': pd.to_numeric(group.get('contribution_temporal_concentration_hhi'), errors='coerce').median(),
        'median_mapping_age_months': pd.to_numeric(group.get('mapping_age_months'), errors='coerce').median(),
        'median_time_to_90pct_months': pd.to_numeric(group.get('time_to_90pct_months'), errors='coerce').median(),
        'median_plateau_duration_months': pd.to_numeric(group.get('plateau_duration_months'), errors='coerce').median(),
    })
class_summary = pd.DataFrame(class_rows)

# Long-form descriptive profile with median and IQR.
profile_rows = []
for cid in CLASS_ORDER:
    group = gdf[gdf['typology_class_id'].eq(cid)]
    for var in sorted(set(BROAD_INDICATORS + ROAD_HISTORY_INDICATORS)):
        s = pd.to_numeric(group[var], errors='coerce').dropna()
        profile_rows.append({
            'typology_class_id': cid,
            'typology_label': CLASS_LABELS[cid],
            'variable': var,
            'n_nonmissing': len(s),
            'mean': s.mean() if len(s) else np.nan,
            'std': s.std(ddof=1) if len(s) > 1 else np.nan,
            'median': s.median() if len(s) else np.nan,
            'q25': s.quantile(0.25) if len(s) else np.nan,
            'q75': s.quantile(0.75) if len(s) else np.nan,
            'iqr': (s.quantile(0.75) - s.quantile(0.25)) if len(s) else np.nan,
            'min': s.min() if len(s) else np.nan,
            'max': s.max() if len(s) else np.nan,
        })
indicator_profiles = pd.DataFrame(profile_rows)

# -----------------------------------------------------------------------------
# 3. Omnibus tests across all three classes
# -----------------------------------------------------------------------------
def epsilon_squared_kruskal(H, n, k):
    if n <= k:
        return np.nan
    return max(0.0, float((H - k + 1) / (n - k)))


omnibus_rows = []
for var in BROAD_INDICATORS:
    groups = []
    ns = []
    for cid in CLASS_ORDER:
        s = pd.to_numeric(gdf.loc[gdf.typology_class_id.eq(cid), var], errors='coerce').dropna()
        groups.append(s.to_numpy())
        ns.append(len(s))
    valid_groups = [x for x in groups if len(x) > 0]
    if len(valid_groups) < 2 or sum(len(x) for x in valid_groups) < 10:
        continue
    H, p = kruskal(*valid_groups)
    omnibus_rows.append({
        'variable': var,
        'test': 'Kruskal-Wallis',
        'H': float(H),
        'p_value': float(p),
        'epsilon_squared': epsilon_squared_kruskal(H, sum(ns), len(valid_groups)),
        'n_class_0': ns[0],
        'n_class_1': ns[1],
        'n_class_2': ns[2],
    })
omnibus_tests = pd.DataFrame(omnibus_rows)

# Benjamini-Hochberg correction for the family of omnibus tests.
def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out

if len(omnibus_tests):
    omnibus_tests['p_adj_bh'] = bh_adjust(omnibus_tests['p_value'].values)
    omnibus_tests['significant_fdr_0_05'] = omnibus_tests['p_adj_bh'].lt(ALPHA)

# -----------------------------------------------------------------------------
# 4. Pairwise tests and interpretable effect sizes
# -----------------------------------------------------------------------------
def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # Efficient rank-based equivalent to pairwise dominance.
    U, _ = mannwhitneyu(x, y, alternative='two-sided')
    return float((2 * U) / (len(x) * len(y)) - 1)


pairwise_rows = []
for var in BROAD_INDICATORS:
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        x = pd.to_numeric(gdf.loc[gdf.typology_class_id.eq(a), var], errors='coerce').dropna().to_numpy()
        y = pd.to_numeric(gdf.loc[gdf.typology_class_id.eq(b), var], errors='coerce').dropna().to_numpy()
        if len(x) < 3 or len(y) < 3:
            continue
        U, p = mannwhitneyu(x, y, alternative='two-sided')
        pairwise_rows.append({
            'variable': var,
            'class_a': a,
            'class_b': b,
            'label_a': CLASS_LABELS[a],
            'label_b': CLASS_LABELS[b],
            'n_a': len(x),
            'n_b': len(y),
            'median_a': float(np.median(x)),
            'median_b': float(np.median(y)),
            'mannwhitney_u': float(U),
            'p_value': float(p),
            'cliffs_delta_a_vs_b': cliffs_delta(x, y),
        })
pairwise_tests = pd.DataFrame(pairwise_rows)
if len(pairwise_tests):
    pairwise_tests['p_adj_bh'] = bh_adjust(pairwise_tests['p_value'].values)
    pairwise_tests['significant_fdr_0_05'] = pairwise_tests['p_adj_bh'].lt(ALPHA)

# Focused road-history comparisons: cluster 1 versus cluster 2 only.
road_cluster_rows = []
for var in ROAD_HISTORY_INDICATORS:
    x = pd.to_numeric(gdf.loc[gdf.typology_class_id.eq(1), var], errors='coerce').dropna().to_numpy()
    y = pd.to_numeric(gdf.loc[gdf.typology_class_id.eq(2), var], errors='coerce').dropna().to_numpy()
    if len(x) < 3 or len(y) < 3:
        continue
    U, p = mannwhitneyu(x, y, alternative='two-sided')
    road_cluster_rows.append({
        'variable': var,
        'n_cluster_1': len(x),
        'n_cluster_2': len(y),
        'median_cluster_1': float(np.median(x)),
        'median_cluster_2': float(np.median(y)),
        'mannwhitney_u': float(U),
        'p_value': float(p),
        'cliffs_delta_cluster1_vs_2': cliffs_delta(x, y),
    })
road_cluster_tests = pd.DataFrame(road_cluster_rows)
if len(road_cluster_tests):
    road_cluster_tests['p_adj_bh'] = bh_adjust(road_cluster_tests['p_value'].values)
    road_cluster_tests['significant_fdr_0_05'] = road_cluster_tests['p_adj_bh'].lt(ALPHA)

# -----------------------------------------------------------------------------
# 5. Binary prevalence tests for interpretable activity states
# -----------------------------------------------------------------------------
binary_defs = {
    'recent_activity_present': gdf['recent_activity_ratio'].fillna(0).gt(0),
    'plateau_detected': gdf['plateau_duration_months'].notna(),
    'high_temporal_concentration_hhi_ge_0_25': gdf['contribution_temporal_concentration_hhi'].ge(0.25),
}

binary_rows = []
for name, mask in binary_defs.items():
    # Compare the two modeled road-present clusters; structural no-road is not
    # substantively comparable for plateau-related states.
    sub = pd.DataFrame({
        'class_id': gdf['typology_class_id'],
        'state': mask.astype(int),
    })
    tab = pd.crosstab(sub.loc[sub.class_id.isin([1, 2]), 'class_id'], sub.loc[sub.class_id.isin([1, 2]), 'state'])
    if tab.shape == (2, 2):
        odds, p = fisher_exact(tab.values)
        binary_rows.append({
            'state': name,
            'test': 'Fisher exact',
            'odds_ratio_cluster1_vs_2': float(odds),
            'p_value': float(p),
            'cluster1_positive_n': int(tab.loc[1, 1]) if 1 in tab.columns else 0,
            'cluster1_total_n': int(tab.loc[1].sum()),
            'cluster2_positive_n': int(tab.loc[2, 1]) if 1 in tab.columns else 0,
            'cluster2_total_n': int(tab.loc[2].sum()),
        })
binary_tests = pd.DataFrame(binary_rows)
if len(binary_tests):
    binary_tests['p_adj_bh'] = bh_adjust(binary_tests['p_value'].values)
    binary_tests['significant_fdr_0_05'] = binary_tests['p_adj_bh'].lt(ALPHA)

# -----------------------------------------------------------------------------
# 6. Spatial structure of the typology
# -----------------------------------------------------------------------------
w = Queen.from_dataframe(gdf.reset_index(drop=True), ids=list(range(len(gdf))), silence_warnings=True)
labels = gdf.reset_index(drop=True)['typology_class_id'].astype(int).to_numpy()
edges = [(i, j) for i, nei in w.neighbors.items() for j in nei if i < j]

same_obs = float(np.mean([labels[i] == labels[j] for i, j in edges])) if edges else np.nan
rng = np.random.default_rng(RANDOM_STATE)
perm_same = []
for _ in range(PERMUTATIONS):
    perm = rng.permutation(labels)
    perm_same.append(np.mean([perm[i] == perm[j] for i, j in edges]))
perm_same = np.asarray(perm_same, dtype=float)
p_spatial = float((1 + np.sum(perm_same >= same_obs)) / (PERMUTATIONS + 1))

transition = pd.DataFrame(0, index=CLASS_ORDER, columns=CLASS_ORDER, dtype=int)
for i, j in edges:
    a, b = int(labels[i]), int(labels[j])
    transition.loc[a, b] += 1
    if a != b:
        transition.loc[b, a] += 1

transition_long = (
    transition.rename_axis('class_a').reset_index()
    .melt(id_vars='class_a', var_name='class_b', value_name='edge_count')
)

# Connected components by class using Queen adjacency.
def component_sizes_for_class(cid):
    nodes = {i for i, x in enumerate(labels) if x == cid}
    visited = set()
    sizes = []
    for start in nodes:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in w.neighbors.get(u, []):
                if v in nodes and v not in visited:
                    visited.add(v)
                    stack.append(v)
        sizes.append(size)
    return sorted(sizes, reverse=True)

spatial_rows = []
for cid in CLASS_ORDER:
    sizes = component_sizes_for_class(cid)
    class_n = int((labels == cid).sum())
    spatial_rows.append({
        'typology_class_id': cid,
        'typology_label': CLASS_LABELS[cid],
        'n_cells': class_n,
        'n_connected_components': len(sizes),
        'largest_component_n': sizes[0] if sizes else 0,
        'largest_component_share': (sizes[0] / class_n) if sizes and class_n else np.nan,
    })
spatial_class_summary = pd.DataFrame(spatial_rows)
spatial_global_summary = pd.DataFrame([{
    'n_queen_edges': len(edges),
    'observed_same_class_neighbor_share': same_obs,
    'permutation_mean_same_class_share': float(perm_same.mean()),
    'permutation_sd_same_class_share': float(perm_same.std(ddof=1)),
    'permutation_p_upper': p_spatial,
    'permutations': PERMUTATIONS,
    'queen_islands': len(w.islands),
}])

# -----------------------------------------------------------------------------
# 7. Provisional substantive interpretation from original-scale medians
# -----------------------------------------------------------------------------
# Interpretation remains descriptive. It does not rename the stored classes.
interpretation_rows = []
for cid in CLASS_ORDER:
    row = class_summary.loc[class_summary.typology_class_id.eq(cid)].iloc[0]
    if cid == 0:
        interpretation = (
            'Structural absence of currently mapped highway-line geometry in the 1 km cell; '
            'this class is not evidence of real-world road absence or OSM incompleteness.'
        )
    else:
        other = 2 if cid == 1 else 1
        cur = class_summary.loc[class_summary.typology_class_id.eq(cid)].iloc[0]
        oth = class_summary.loc[class_summary.typology_class_id.eq(other)].iloc[0]
        length_rel = 'higher' if cur.median_current_road_length_km > oth.median_current_road_length_km else 'lower'
        activity_rel = 'higher' if cur.median_active_month_ratio > oth.median_active_month_ratio else 'lower'
        recent_rel = 'higher' if cur.median_recent_activity_ratio > oth.median_recent_activity_ratio else 'lower'
        interpretation = (
            f'Road-present intrinsic mapping regime with {length_rel} median mapped-road length, '
            f'{activity_rel} historical activity persistence, and {recent_rel} recent activity than the other modeled cluster. '
            'Interpretation is relative to OSM history only.'
        )
    interpretation_rows.append({
        'typology_class_id': cid,
        'typology_label': CLASS_LABELS[cid],
        'provisional_substantive_interpretation': interpretation,
    })
interpretation_table = pd.DataFrame(interpretation_rows)

# -----------------------------------------------------------------------------
# 8. Article figures
# -----------------------------------------------------------------------------
# Figure A — Typology map.
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(column='typology_class_id', categorical=True, legend=True, ax=ax, linewidth=0.15, edgecolor='0.55')
ax.set_title('Intrinsic OSM mapping-history typology — Petrópolis, 1 km cells')
ax.set_axis_off()
fig.tight_layout()
fig.savefig(FIG / 'figure_typology_map.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure B — Robust standardized median profile of substantive indicators.
PROFILE_FIG_VARS = [
    c for c in [
        'current_road_length_km', 'mapping_age_months', 'time_to_90pct_months',
        'plateau_duration_months', 'active_month_ratio', 'recent_activity_ratio',
        'total_contributions', 'contribution_temporal_concentration_hhi',
        'expansion_share_proxy', 'contributor_volatility_proxy'
    ] if c in gdf.columns
]
med = gdf.groupby('typology_class_id')[PROFILE_FIG_VARS].median(numeric_only=True).reindex(CLASS_ORDER)
standardized = med.copy().astype(float)
for c in PROFILE_FIG_VARS:
    col = pd.to_numeric(gdf[c], errors='coerce')
    center = col.median()
    iqr = col.quantile(0.75) - col.quantile(0.25)
    if pd.isna(iqr) or iqr == 0:
        standardized[c] = 0.0
    else:
        standardized[c] = (med[c] - center) / iqr
standardized = standardized.clip(-3, 3)

fig, ax = plt.subplots(figsize=(12, 4.8))
im = ax.imshow(standardized.values, aspect='auto', vmin=-3, vmax=3)
ax.set_yticks(range(len(CLASS_ORDER)))
ax.set_yticklabels([CLASS_LABELS[i] for i in CLASS_ORDER])
ax.set_xticks(range(len(PROFILE_FIG_VARS)))
ax.set_xticklabels(PROFILE_FIG_VARS, rotation=45, ha='right')
ax.set_title('Robust standardized median indicator profiles by typology class')
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='Median relative to overall median (IQR units)')
fig.tight_layout()
fig.savefig(FIG / 'figure_standardized_class_profiles.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure C — Key original-scale distributions.
KEY_BOXPLOT_VARS = [
    c for c in ['current_road_length_km', 'active_month_ratio', 'total_contributions', 'time_to_90pct_months']
    if c in gdf.columns
]
for var in KEY_BOXPLOT_VARS:
    values = [pd.to_numeric(gdf.loc[gdf.typology_class_id.eq(cid), var], errors='coerce').dropna().to_numpy() for cid in CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.boxplot(values, labels=[str(i) for i in CLASS_ORDER], showfliers=False)
    ax.set_xlabel('Typology class ID')
    ax.set_ylabel(var)
    ax.set_title(f'Distribution of {var} by typology class')
    fig.tight_layout()
    fig.savefig(FIG / f'figure_boxplot_{var}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# Figure D — Neighbor transition matrix.
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(transition.values, aspect='equal')
ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_yticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_ORDER)
ax.set_yticklabels(CLASS_ORDER)
ax.set_xlabel('Neighbor class')
ax.set_ylabel('Focal class')
ax.set_title('Queen-neighbor class transition counts')
for i in range(len(CLASS_ORDER)):
    for j in range(len(CLASS_ORDER)):
        ax.text(j, i, str(int(transition.iloc[i, j])), ha='center', va='center')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIG / 'figure_neighbor_transition_matrix.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# -----------------------------------------------------------------------------
# 9. Save article-ready tables
# -----------------------------------------------------------------------------
files = {
    'class_summary': OUT / 'petropolis_osm_typology_article_class_summary.csv',
    'indicator_profiles': OUT / 'petropolis_osm_typology_article_indicator_profiles.csv',
    'omnibus_tests': OUT / 'petropolis_osm_typology_article_kruskal_tests.csv',
    'pairwise_tests': OUT / 'petropolis_osm_typology_article_pairwise_tests.csv',
    'road_cluster_tests': OUT / 'petropolis_osm_typology_article_road_cluster_tests.csv',
    'binary_tests': OUT / 'petropolis_osm_typology_article_binary_tests.csv',
    'spatial_global': OUT / 'petropolis_osm_typology_article_spatial_global.csv',
    'spatial_classes': OUT / 'petropolis_osm_typology_article_spatial_classes.csv',
    'neighbor_transitions': OUT / 'petropolis_osm_typology_article_neighbor_transitions.csv',
    'interpretation': OUT / 'petropolis_osm_typology_article_interpretation.csv',
}

class_summary.to_csv(files['class_summary'], index=False)
indicator_profiles.to_csv(files['indicator_profiles'], index=False)
omnibus_tests.to_csv(files['omnibus_tests'], index=False)
pairwise_tests.to_csv(files['pairwise_tests'], index=False)
road_cluster_tests.to_csv(files['road_cluster_tests'], index=False)
binary_tests.to_csv(files['binary_tests'], index=False)
spatial_global_summary.to_csv(files['spatial_global'], index=False)
spatial_class_summary.to_csv(files['spatial_classes'], index=False)
transition_long.to_csv(files['neighbor_transitions'], index=False)
interpretation_table.to_csv(files['interpretation'], index=False)

# Compact LaTeX tables for manuscript drafting.
class_summary.to_latex(OUT / 'petropolis_osm_typology_article_class_summary.tex', index=False, float_format='%.3f')
if len(road_cluster_tests):
    road_cluster_tests.to_latex(OUT / 'petropolis_osm_typology_article_road_cluster_tests.tex', index=False, float_format='%.4f')

# -----------------------------------------------------------------------------
# 10. Quality control
# -----------------------------------------------------------------------------
checks = {
    'rows_895': len(gdf) == 895,
    'ids_unique': gdf['grid_1km_id'].is_unique,
    'exactly_three_operational_classes': set(gdf['typology_class_id'].astype(int)) == {0, 1, 2},
    'class_summary_complete': len(class_summary) == 3 and class_summary['n_cells'].sum() == 895,
    'spatial_edges_exist': len(edges) > 0,
    'spatial_permutation_valid': np.isfinite(same_obs) and np.isfinite(p_spatial),
    'article_tables_written': all(p.exists() for p in files.values()),
    'article_figures_written': all((FIG / name).exists() for name in [
        'figure_typology_map.png',
        'figure_standardized_class_profiles.png',
        'figure_neighbor_transition_matrix.png',
    ]),
    'broad_test_pvalues_valid': omnibus_tests['p_value'].between(0, 1).all() if len(omnibus_tests) else False,
    'road_cluster_test_pvalues_valid': road_cluster_tests['p_value'].between(0, 1).all() if len(road_cluster_tests) else False,
}
qc = pd.DataFrame([{'check': k, 'status': 'PASS' if bool(v) else 'FAIL'} for k, v in checks.items()])
qc.to_csv(OUT / 'petropolis_osm_typology_article_quality_checks.csv', index=False)

print('\nClass summary:')
print(class_summary.to_string(index=False))
print('\nOmnibus tests:')
print(omnibus_tests.sort_values('p_adj_bh').to_string(index=False) if len(omnibus_tests) else 'None')
print('\nRoad-present cluster comparisons:')
print(road_cluster_tests.sort_values('p_adj_bh').to_string(index=False) if len(road_cluster_tests) else 'None')
print('\nSpatial global summary:')
print(spatial_global_summary.to_string(index=False))
print('\nSpatial class summary:')
print(spatial_class_summary.to_string(index=False))
print('\nProvisional substantive interpretation:')
print(interpretation_table.to_string(index=False))
print('\nQuality control:')
print(qc.to_string(index=False))

if not all(checks.values()):
    raise RuntimeError(f'Notebook 10 quality-control failure: {checks}')

print('\nInterpretation boundary: the typology characterizes intrinsic OSM mapping-history regimes. It does not demonstrate external completeness, positional accuracy, real-world road absence, or socioeconomic status.')
