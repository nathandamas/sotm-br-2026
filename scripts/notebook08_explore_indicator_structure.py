from pathlib import Path
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from libpysal.weights import Queen
from esda.moran import Moran

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 180)


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, start.parent, *start.parents]:
        if (candidate / 'notebooks').exists():
            return candidate
    raise FileNotFoundError('Could not locate project root.')


PROJECT_ROOT = find_project_root(Path.cwd())
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'Petropolis'
INDICATOR_FILE = PROCESSED_DIR / 'petropolis_osm_maturity_indicators_895_cells.parquet'
GRID_FILE = PROCESSED_DIR / 'petropolis_grid_1km_with_maturity_indicators.parquet'
DICTIONARY_FILE = PROCESSED_DIR / 'petropolis_osm_maturity_indicator_dictionary.csv'

for path in [INDICATOR_FILE, GRID_FILE, DICTIONARY_FILE]:
    if not path.exists():
        raise FileNotFoundError(path)

indicators = pd.read_parquet(INDICATOR_FILE)
grid = gpd.read_parquet(GRID_FILE)
dictionary = pd.read_csv(DICTIONARY_FILE)

print('Python:', sys.version.split()[0])
print('Indicator rows:', len(indicators))
print('Grid rows:', len(grid))
print('Grid CRS:', grid.crs)

assert len(indicators) == 895
assert indicators['grid_1km_id'].is_unique
assert len(grid) == 895
assert grid['grid_1km_id'].is_unique
assert set(indicators['grid_1km_id']) == set(grid['grid_1km_id'])
print('PASS — Notebook 07 outputs cover the same 895 unique cells.')

FAMILIES = {
    'network_state': [
        'current_road_length_km', 'recent_reference_road_length_km',
        'historical_max_road_length_km', 'mapping_age_months'
    ],
    'growth': [
        'peak_growth_m', 'positive_growth_months', 'negative_growth_months',
        'growth_rate_initial_m_per_month', 'growth_rate_recent_m_per_month',
        'growth_deceleration_m_per_month'
    ],
    'intrinsic_saturation': [
        'saturation_ratio', 'time_to_50pct_months', 'time_to_80pct_months',
        'time_to_90pct_months', 'time_to_95pct_months', 'months_since_90pct',
        'plateau_duration_months', 'plateau_tail_stability_fraction'
    ],
    'persistence': [
        'active_months', 'active_month_ratio', 'longest_active_streak_months',
        'longest_inactive_streak_months', 'months_since_last_activity',
        'post_plateau_active_month_ratio', 'recent_activity_ratio', 'persistence_index'
    ],
    'contribution_dynamics': [
        'total_contributions', 'mean_monthly_contributions', 'median_monthly_contributions',
        'peak_monthly_contributions', 'total_contributor_months', 'mean_monthly_contributors',
        'peak_monthly_contributors', 'contributions_per_contributor_month',
        'contribution_temporal_concentration_hhi', 'recent_to_historical_activity_ratio',
        'contributor_volatility_proxy', 'expansion_share_proxy', 'maintenance_share_proxy',
        'deletion_share_proxy'
    ],
}
available = {family: [v for v in values if v in indicators.columns] for family, values in FAMILIES.items()}
for family, values in available.items():
    print(f'{family}: {len(values)} variables')

numeric_cols = indicators.select_dtypes(include=['number']).columns.tolist()
rows = []
for col in numeric_cols:
    s = indicators[col]
    nonnull = s.dropna()
    rows.append({
        'variable': col,
        'n': len(s),
        'missing_n': int(s.isna().sum()),
        'missing_pct': float(s.isna().mean() * 100),
        'zero_n': int((nonnull == 0).sum()),
        'zero_pct_nonmissing': float((nonnull == 0).mean() * 100) if len(nonnull) else np.nan,
        'min': nonnull.min() if len(nonnull) else np.nan,
        'q25': nonnull.quantile(0.25) if len(nonnull) else np.nan,
        'median': nonnull.median() if len(nonnull) else np.nan,
        'mean': nonnull.mean() if len(nonnull) else np.nan,
        'q75': nonnull.quantile(0.75) if len(nonnull) else np.nan,
        'max': nonnull.max() if len(nonnull) else np.nan,
        'unique_n': int(nonnull.nunique()),
    })
distribution_diagnostics = pd.DataFrame(rows).sort_values(['missing_pct', 'variable'], ascending=[False, True])
print('\nDistribution and missingness diagnostics:')
print(distribution_diagnostics.to_string(index=False))

analysis_candidates = sorted(set(sum(available.values(), [])))
corr = indicators[analysis_candidates].corr(method='spearman', min_periods=100)
pairs = []
for i, a in enumerate(corr.columns):
    for j, b in enumerate(corr.columns):
        if j <= i:
            continue
        r = corr.loc[a, b]
        if pd.notna(r):
            pairs.append({'variable_a': a, 'variable_b': b, 'spearman_rho': float(r), 'abs_rho': abs(float(r))})
correlation_pairs = pd.DataFrame(pairs).sort_values('abs_rho', ascending=False)
strong_redundancy = correlation_pairs[correlation_pairs['abs_rho'] >= 0.85].copy()
print('\nStrong redundancy pairs (|rho| >= 0.85):', len(strong_redundancy))
print(strong_redundancy.head(40).to_string(index=False))

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr.columns, fontsize=7)
ax.set_title('Spearman correlation matrix of intrinsic OSM indicators')
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
plt.tight_layout()
plt.show()

PCA_FEATURES = [
    'current_road_length_km', 'mapping_age_months', 'peak_growth_m',
    'growth_deceleration_m_per_month', 'time_to_90pct_months', 'months_since_90pct',
    'plateau_duration_months', 'active_month_ratio', 'recent_activity_ratio',
    'longest_inactive_streak_months', 'total_contributions', 'total_contributor_months',
    'contributions_per_contributor_month', 'contribution_temporal_concentration_hhi',
    'recent_to_historical_activity_ratio', 'contributor_volatility_proxy',
    'expansion_share_proxy', 'maintenance_share_proxy'
]
PCA_FEATURES = [c for c in PCA_FEATURES if c in indicators.columns]
X = indicators.set_index('grid_1km_id')[PCA_FEATURES].copy()
imputation_table = pd.DataFrame({
    'variable': PCA_FEATURES,
    'missing_n': [int(X[c].isna().sum()) for c in PCA_FEATURES],
    'median_used_for_pca': [float(X[c].median()) if X[c].notna().any() else np.nan for c in PCA_FEATURES],
})
for col in PCA_FEATURES:
    X[col] = X[col].fillna(X[col].median())

scaler = RobustScaler(quantile_range=(25, 75))
X_scaled = scaler.fit_transform(X)
pca = PCA()
scores = pca.fit_transform(X_scaled)
explained = pd.DataFrame({
    'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'explained_variance_ratio': pca.explained_variance_ratio_,
    'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_),
})
loadings = pd.DataFrame(
    pca.components_.T,
    index=PCA_FEATURES,
    columns=[f'PC{i+1}' for i in range(len(PCA_FEATURES))],
)
print('\nPCA imputation table:')
print(imputation_table.to_string(index=False))
print('\nPCA explained variance:')
print(explained.head(10).to_string(index=False))
print('\nLargest absolute PC1 loadings:')
print(loadings[['PC1', 'PC2', 'PC3']].sort_values('PC1', key=np.abs, ascending=False).head(20).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(1, min(11, len(explained) + 1)), explained['cumulative_explained_variance'].head(10), marker='o')
ax.set_xlabel('Number of principal components')
ax.set_ylabel('Cumulative explained variance')
ax.set_ylim(0, 1.02)
ax.set_title('PCA cumulative explained variance')
ax.grid(alpha=0.2)
plt.show()

SPATIAL_INDICATORS = [
    'current_road_length_km', 'mapping_age_months', 'active_month_ratio',
    'recent_activity_ratio', 'persistence_index', 'total_contributions',
    'contribution_temporal_concentration_hhi', 'time_to_90pct_months',
    'plateau_duration_months'
]
SPATIAL_INDICATORS = [c for c in SPATIAL_INDICATORS if c in grid.columns]
moran_rows = []
for col in SPATIAL_INDICATORS:
    subset = grid[['grid_1km_id', col, 'geometry']].dropna(subset=[col]).copy()
    if len(subset) < 30 or subset[col].nunique() < 2:
        moran_rows.append({'variable': col, 'n': len(subset), 'moran_i': np.nan, 'p_sim': np.nan, 'z_sim': np.nan, 'islands': np.nan})
        continue
    w = Queen.from_dataframe(subset, ids=subset['grid_1km_id'].tolist(), silence_warnings=True)
    w.transform = 'r'
    y = subset.set_index('grid_1km_id').loc[w.id_order, col].astype(float).to_numpy()
    mi = Moran(y, w, permutations=999, two_tailed=True)
    moran_rows.append({
        'variable': col,
        'n': len(subset),
        'moran_i': float(mi.I),
        'p_sim': float(mi.p_sim),
        'z_sim': float(mi.z_sim),
        'islands': len(w.islands),
    })
moran_results = pd.DataFrame(moran_rows).sort_values('moran_i', ascending=False)
print('\nGlobal Moran diagnostics:')
print(moran_results.to_string(index=False))

pca_scores = pd.DataFrame(scores[:, :3], index=X.index, columns=['PC1', 'PC2', 'PC3']).reset_index()
pca_map = grid[['grid_1km_id', 'geometry']].merge(pca_scores, on='grid_1km_id', how='left')
for pc in ['PC1', 'PC2', 'PC3']:
    fig, ax = plt.subplots(figsize=(9, 8))
    pca_map.plot(column=pc, legend=True, ax=ax, linewidth=0.1, edgecolor='black')
    ax.set_title(f'Spatial pattern of {pc}')
    ax.set_axis_off()
    plt.show()

FEATURE_MATRIX_FILE = PROCESSED_DIR / 'petropolis_osm_indicator_feature_matrix_895_cells.parquet'
PCA_LOADINGS_FILE = PROCESSED_DIR / 'petropolis_osm_pca_loadings.csv'
PCA_EXPLAINED_FILE = PROCESSED_DIR / 'petropolis_osm_pca_explained_variance.csv'
CORRELATION_FILE = PROCESSED_DIR / 'petropolis_osm_spearman_correlation.csv'
REDUNDANCY_FILE = PROCESSED_DIR / 'petropolis_osm_strong_redundancy_pairs.csv'
MORAN_FILE = PROCESSED_DIR / 'petropolis_osm_global_moran.csv'
DISTRIBUTION_FILE = PROCESSED_DIR / 'petropolis_osm_indicator_distribution_diagnostics.csv'
IMPUTATION_FILE = PROCESSED_DIR / 'petropolis_osm_pca_imputation_diagnostics.csv'

feature_matrix = indicators.merge(pca_scores, on='grid_1km_id', how='left')
feature_matrix.to_parquet(FEATURE_MATRIX_FILE, index=False)
loadings.reset_index(names='variable').to_csv(PCA_LOADINGS_FILE, index=False)
explained.to_csv(PCA_EXPLAINED_FILE, index=False)
corr.to_csv(CORRELATION_FILE)
strong_redundancy.to_csv(REDUNDANCY_FILE, index=False)
moran_results.to_csv(MORAN_FILE, index=False)
distribution_diagnostics.to_csv(DISTRIBUTION_FILE, index=False)
imputation_table.to_csv(IMPUTATION_FILE, index=False)

checks = {
    'rows_895': len(feature_matrix) == 895,
    'ids_unique': feature_matrix['grid_1km_id'].is_unique,
    'pca_scores_complete': feature_matrix[['PC1', 'PC2', 'PC3']].notna().all().all(),
    'pca_variance_sums_to_one': np.isclose(explained['explained_variance_ratio'].sum(), 1.0),
    'correlation_matrix_symmetric': np.allclose(corr.values, corr.values.T, equal_nan=True),
    'moran_results_created': len(moran_results) == len(SPATIAL_INDICATORS),
    'moran_i_finite_when_tested': moran_results.loc[moran_results['moran_i'].notna(), 'moran_i'].between(-1, 1).all(),
    'output_files_exist': all(p.exists() for p in [
        FEATURE_MATRIX_FILE, PCA_LOADINGS_FILE, PCA_EXPLAINED_FILE, CORRELATION_FILE,
        REDUNDANCY_FILE, MORAN_FILE, DISTRIBUTION_FILE, IMPUTATION_FILE
    ]),
}
qc = pd.DataFrame([
    {'check': name, 'status': 'PASS' if bool(ok) else 'FAIL'}
    for name, ok in checks.items()
])
print('\nNotebook 08 quality control:')
print(qc.to_string(index=False))
print(qc['status'].value_counts().reindex(['PASS', 'FAIL'], fill_value=0).to_string())
if not all(checks.values()):
    raise RuntimeError(f'Notebook 08 quality-control failure: {checks}')

print('\nSaved outputs:')
for path in [
    FEATURE_MATRIX_FILE, PCA_LOADINGS_FILE, PCA_EXPLAINED_FILE, CORRELATION_FILE,
    REDUNDANCY_FILE, MORAN_FILE, DISTRIBUTION_FILE, IMPUTATION_FILE
]:
    print(path)

print('\nInterpretation boundary: these are intrinsic OSM diagnostics, not external completeness or quality measures.')
