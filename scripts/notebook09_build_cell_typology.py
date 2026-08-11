from pathlib import Path
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from libpysal.weights import Queen

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 180)
RANDOM_SEED = 42
K_VALUES = range(2, 9)
BOOTSTRAP_REPEATS = 50
SUBSAMPLE_FRACTION = 0.80
MIN_CLUSTER_N = 20
CLIP_LO = 0.01
CLIP_HI = 0.99


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, start.parent, *start.parents]:
        if (candidate / 'notebooks').exists():
            return candidate
    raise FileNotFoundError('Could not locate project root.')


def spatial_agreement(labels, ids, weights):
    label_map = dict(zip(ids, labels))
    same = total = 0
    for i in ids:
        for j in weights.neighbors.get(i, []):
            if j in label_map:
                total += 1
                same += int(label_map[i] == label_map[j])
    return np.nan if total == 0 else same / total


def fit_labels(method, k, X):
    if method == 'kmeans':
        return KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=50).fit_predict(X)
    if method == 'hierarchical':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
    raise ValueError(method)


def stability_against_reference(method, k, X, reference_labels, repeats=BOOTSTRAP_REPEATS):
    rng = np.random.default_rng(RANDOM_SEED + 1000 * k + (0 if method == 'kmeans' else 100))
    n = len(X)
    m = int(round(SUBSAMPLE_FRACTION * n))
    aris = []
    for _ in range(repeats):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        labels_sub = fit_labels(method, k, X[idx])
        aris.append(adjusted_rand_score(reference_labels[idx], labels_sub))
    return float(np.mean(aris)), float(np.std(aris)), float(np.quantile(aris, 0.10))


PROJECT_ROOT = find_project_root(Path.cwd())
PROCESSED = PROJECT_ROOT / 'data' / 'processed' / 'Petropolis'
FIGURES = PROCESSED / 'notebook09_figures'
FIGURES.mkdir(parents=True, exist_ok=True)
FEATURE_FILE = PROCESSED / 'petropolis_osm_indicator_feature_matrix_895_cells.parquet'
GRID_FILE = PROCESSED / 'petropolis_grid_1km_with_maturity_indicators.parquet'
CORR_FILE = PROCESSED / 'petropolis_osm_spearman_correlation.csv'

for p in [FEATURE_FILE, GRID_FILE, CORR_FILE]:
    if not p.exists():
        raise FileNotFoundError(p)

features_all = pd.read_parquet(FEATURE_FILE)
grid = gpd.read_parquet(GRID_FILE)
corr = pd.read_csv(CORR_FILE, index_col=0)
assert len(features_all) == 895 and features_all['grid_1km_id'].is_unique
assert len(grid) == 895 and grid['grid_1km_id'].is_unique

SELECTED_FEATURES = [
    'current_road_length_km',
    'mapping_age_months',
    'growth_deceleration_m_per_month',
    'time_to_90pct_months',
    'plateau_duration_months',
    'active_month_ratio',
    'recent_activity_ratio',
    'contributions_per_contributor_month',
    'contribution_temporal_concentration_hhi',
    'contributor_volatility_proxy',
    'expansion_share_proxy',
    'deletion_share_proxy',
]
missing_features = [c for c in SELECTED_FEATURES if c not in features_all.columns]
if missing_features:
    raise KeyError(f'Missing selected features: {missing_features}')

selected_corr = corr.loc[SELECTED_FEATURES, SELECTED_FEATURES]
max_pair_abs_rho = max(abs(selected_corr.iloc[i, j]) for i in range(len(SELECTED_FEATURES)) for j in range(i+1, len(SELECTED_FEATURES)))
print('Selected features:', len(SELECTED_FEATURES))
print('Maximum selected pair |Spearman rho|:', round(max_pair_abs_rho, 4))
if max_pair_abs_rho >= 0.85:
    raise RuntimeError('Selected feature set still contains a strongly redundant pair.')

X_raw = features_all.set_index('grid_1km_id')[SELECTED_FEATURES].copy()
prep_rows = []
for col in SELECTED_FEATURES:
    median = X_raw[col].median()
    filled = X_raw[col].fillna(median)
    lo = filled.quantile(CLIP_LO)
    hi = filled.quantile(CLIP_HI)
    X_raw[col] = filled.clip(lo, hi)
    prep_rows.append({
        'variable': col,
        'missing_n': int(features_all[col].isna().sum()),
        'median_imputation': float(median),
        'clip_1pct': float(lo),
        'clip_99pct': float(hi),
        'unique_after_preprocessing': int(X_raw[col].nunique()),
    })
preprocessing = pd.DataFrame(prep_rows)
if (preprocessing['unique_after_preprocessing'] < 2).any():
    raise RuntimeError('A selected feature is constant after preprocessing.')

scaler = RobustScaler(quantile_range=(25, 75))
X = scaler.fit_transform(X_raw)
ids = X_raw.index.astype(str).tolist()

# Queen contiguity is used only as a diagnostic, never as a clustering constraint.
grid_aligned = grid.set_index('grid_1km_id').loc[ids].reset_index()
w = Queen.from_dataframe(grid_aligned, ids=ids, silence_warnings=True)

rows = []
solutions = {}
for method in ['kmeans', 'hierarchical']:
    for k in K_VALUES:
        labels = fit_labels(method, k, X)
        solutions[(method, k)] = labels
        counts = pd.Series(labels).value_counts()
        stab_mean, stab_sd, stab_p10 = stability_against_reference(method, k, X, labels)
        rows.append({
            'method': method,
            'k': k,
            'silhouette': float(silhouette_score(X, labels)),
            'calinski_harabasz': float(calinski_harabasz_score(X, labels)),
            'davies_bouldin': float(davies_bouldin_score(X, labels)),
            'min_cluster_n': int(counts.min()),
            'max_cluster_n': int(counts.max()),
            'min_cluster_fraction': float(counts.min() / len(labels)),
            'stability_ari_mean': stab_mean,
            'stability_ari_sd': stab_sd,
            'stability_ari_p10': stab_p10,
            'queen_same_cluster_edge_fraction': float(spatial_agreement(labels, ids, w)),
        })
metrics = pd.DataFrame(rows)
metrics['admissible_size'] = metrics['min_cluster_n'] >= MIN_CLUSTER_N
metrics['admissible_stability'] = metrics['stability_ari_mean'] >= 0.60
metrics['admissible'] = metrics['admissible_size'] & metrics['admissible_stability']

# Transparent selection rule: among solutions with >=20 cells in every cluster and mean
# subsample ARI >=0.60, maximize silhouette. If none satisfy both constraints, retain
# the size constraint and maximize stability first, then silhouette.
admissible = metrics[metrics['admissible']].copy()
if len(admissible):
    best_row = admissible.sort_values(['silhouette', 'stability_ari_mean', 'k'], ascending=[False, False, True]).iloc[0]
    selection_rule = 'max silhouette among solutions with min_cluster_n>=20 and mean subsample ARI>=0.60'
else:
    size_ok = metrics[metrics['admissible_size']].copy()
    if size_ok.empty:
        raise RuntimeError('No clustering solution satisfies the minimum cluster size constraint.')
    best_row = size_ok.sort_values(['stability_ari_mean', 'silhouette', 'k'], ascending=[False, False, True]).iloc[0]
    selection_rule = 'fallback: max stability among solutions with min_cluster_n>=20, then silhouette'

best_method = str(best_row['method'])
best_k = int(best_row['k'])
best_labels = solutions[(best_method, best_k)]
print('\nClustering comparison:')
print(metrics.to_string(index=False))
print('\nSelected solution:', best_method, 'k=', best_k)
print('Selection rule:', selection_rule)

# Stable provisional numbering: order clusters by median active_month_ratio, then current road length.
label_frame = pd.DataFrame({'grid_1km_id': ids, 'raw_cluster': best_labels}).merge(
    features_all[['grid_1km_id'] + SELECTED_FEATURES], on='grid_1km_id', how='left'
)
ordering = (label_frame.groupby('raw_cluster')
            .agg(active=('active_month_ratio', 'median'), road=('current_road_length_km', 'median'))
            .sort_values(['active', 'road'])
            .index.tolist())
remap = {raw: i + 1 for i, raw in enumerate(ordering)}
label_frame['cluster_id'] = label_frame['raw_cluster'].map(remap).astype(int)
label_frame['cluster_label'] = label_frame['cluster_id'].map(lambda x: f'Provisional type {x}')

profile_median = label_frame.groupby(['cluster_id', 'cluster_label'])[SELECTED_FEATURES].median().reset_index()
profile_mean = label_frame.groupby(['cluster_id', 'cluster_label'])[SELECTED_FEATURES].mean().reset_index()
cluster_sizes = label_frame.groupby(['cluster_id', 'cluster_label']).size().reset_index(name='n_cells')

# Standardized median profiles for interpretation only.
profile_scaled = []
for cluster_id, g in label_frame.groupby('cluster_id'):
    rec = {'cluster_id': cluster_id, 'cluster_label': f'Provisional type {cluster_id}'}
    for j, col in enumerate(SELECTED_FEATURES):
        vals = X_raw.loc[g['grid_1km_id'], col]
        center = X_raw[col].median()
        scale = X_raw[col].quantile(.75) - X_raw[col].quantile(.25)
        rec[col] = 0.0 if scale == 0 else float((vals.median() - center) / scale)
    profile_scaled.append(rec)
profile_scaled = pd.DataFrame(profile_scaled)

# Pairwise algorithm agreement for the selected k.
km_labels = solutions[('kmeans', best_k)]
hc_labels = solutions[('hierarchical', best_k)]
algorithm_agreement_ari = float(adjusted_rand_score(km_labels, hc_labels))

# Spatial diagnostic of final solution.
final_spatial_agreement = float(spatial_agreement(best_labels, ids, w))

# Save classification and geometry.
typology = features_all.merge(label_frame[['grid_1km_id', 'cluster_id', 'cluster_label']], on='grid_1km_id', how='left')
typology['selected_method'] = best_method
typology['selected_k'] = best_k
typology['selection_rule'] = selection_rule
typology['algorithm_agreement_ari_at_selected_k'] = algorithm_agreement_ari
typology['queen_same_cluster_edge_fraction'] = final_spatial_agreement
geo = grid.merge(label_frame[['grid_1km_id', 'cluster_id', 'cluster_label']], on='grid_1km_id', how='left')

OUT_TYPOLOGY = PROCESSED / 'petropolis_osm_cell_typology_895_cells.parquet'
OUT_GEO = PROCESSED / 'petropolis_grid_1km_with_osm_typology.parquet'
OUT_METRICS = PROCESSED / 'petropolis_osm_typology_model_selection.csv'
OUT_STABILITY = PROCESSED / 'petropolis_osm_typology_stability.csv'
OUT_PROFILE_MEDIAN = PROCESSED / 'petropolis_osm_typology_profiles_median.csv'
OUT_PROFILE_MEAN = PROCESSED / 'petropolis_osm_typology_profiles_mean.csv'
OUT_PROFILE_SCALED = PROCESSED / 'petropolis_osm_typology_profiles_robust_scaled_median.csv'
OUT_SIZES = PROCESSED / 'petropolis_osm_typology_cluster_sizes.csv'
OUT_PREP = PROCESSED / 'petropolis_osm_typology_preprocessing.csv'
OUT_FEATURES = PROCESSED / 'petropolis_osm_typology_selected_features.csv'

pd.DataFrame({'variable': SELECTED_FEATURES}).to_csv(OUT_FEATURES, index=False)
preprocessing.to_csv(OUT_PREP, index=False)
metrics.to_csv(OUT_METRICS, index=False)
metrics[['method','k','stability_ari_mean','stability_ari_sd','stability_ari_p10']].to_csv(OUT_STABILITY, index=False)
profile_median.to_csv(OUT_PROFILE_MEDIAN, index=False)
profile_mean.to_csv(OUT_PROFILE_MEAN, index=False)
profile_scaled.to_csv(OUT_PROFILE_SCALED, index=False)
cluster_sizes.to_csv(OUT_SIZES, index=False)
typology.to_parquet(OUT_TYPOLOGY, index=False)
geo.to_parquet(OUT_GEO, index=False)

# Diagnostic figures.
fig, ax = plt.subplots(figsize=(9, 5))
for method, g in metrics.groupby('method'):
    ax.plot(g['k'], g['silhouette'], marker='o', label=method)
ax.axvline(best_k, linestyle='--', linewidth=1)
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Silhouette score')
ax.set_title('Clustering solution comparison')
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / 'typology_silhouette_by_k.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
for method, g in metrics.groupby('method'):
    ax.plot(g['k'], g['stability_ari_mean'], marker='o', label=method)
ax.axhline(0.60, linestyle='--', linewidth=1)
ax.axvline(best_k, linestyle='--', linewidth=1)
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Mean subsample ARI')
ax.set_title('Clustering stability under 80% subsampling')
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / 'typology_stability_by_k.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 8))
geo.plot(column='cluster_id', categorical=True, legend=True, ax=ax, linewidth=0.15, edgecolor='black')
ax.set_title(f'Provisional intrinsic OSM cell typology — {best_method}, k={best_k}')
ax.set_axis_off()
fig.tight_layout()
fig.savefig(FIGURES / 'petropolis_osm_cell_typology_map.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, max(4, best_k * 0.8)))
mat = profile_scaled.set_index('cluster_label')[SELECTED_FEATURES]
im = ax.imshow(mat.values, aspect='auto', vmin=-2, vmax=2)
ax.set_yticks(range(len(mat.index)))
ax.set_yticklabels(mat.index)
ax.set_xticks(range(len(SELECTED_FEATURES)))
ax.set_xticklabels(SELECTED_FEATURES, rotation=90, fontsize=8)
ax.set_title('Robust-scaled median profiles of provisional types')
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
fig.tight_layout()
fig.savefig(FIGURES / 'petropolis_osm_typology_profile_heatmap.png', dpi=200)
plt.close(fig)

checks = {
    'rows_895': len(typology) == 895,
    'ids_unique': typology['grid_1km_id'].is_unique,
    'selected_features_nonredundant': max_pair_abs_rho < 0.85,
    'selected_features_nonconstant': (preprocessing['unique_after_preprocessing'] >= 2).all(),
    'all_cells_classified': typology['cluster_id'].notna().all(),
    'cluster_count_matches_k': typology['cluster_id'].nunique() == best_k,
    'minimum_cluster_size_ok': cluster_sizes['n_cells'].min() >= MIN_CLUSTER_N,
    'selected_solution_stability_reasonable': float(best_row['stability_ari_mean']) >= 0.60 or selection_rule.startswith('fallback'),
    'output_files_exist': all(p.exists() for p in [OUT_TYPOLOGY, OUT_GEO, OUT_METRICS, OUT_STABILITY, OUT_PROFILE_MEDIAN, OUT_PROFILE_MEAN, OUT_PROFILE_SCALED, OUT_SIZES, OUT_PREP, OUT_FEATURES]),
}
qc = pd.DataFrame([{'check': k, 'status': 'PASS' if bool(v) else 'FAIL'} for k,v in checks.items()])
print('\nNotebook 09 quality control:')
print(qc.to_string(index=False))
if not all(checks.values()):
    raise RuntimeError(f'Notebook 09 QC failure: {checks}')

print('\nCluster sizes:')
print(cluster_sizes.to_string(index=False))
print('\nAlgorithm agreement ARI at selected k:', round(algorithm_agreement_ari, 4))
print('Queen same-cluster edge fraction:', round(final_spatial_agreement, 4))
print('\nInterpretation boundary: cluster labels are provisional descriptors of intrinsic OSM history and contribution dynamics; they are not external completeness or quality classes.')
