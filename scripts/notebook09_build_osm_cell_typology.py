from pathlib import Path
import json
import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from libpysal.weights import Queen

RANDOM_STATE = 42
K_RANGE = range(2, 8)
BOOTSTRAP_REPS = 40
SUBSAMPLE_FRACTION = 0.80
MIN_CLUSTER_FRACTION = 0.05
STABILITY_TARGET = 0.75
SILHOUETTE_TOLERANCE = 0.02


def find_root(start: Path) -> Path:
    p = start.resolve()
    for c in [p, *p.parents]:
        if (c / 'notebooks').exists() and (c / 'scripts').exists():
            return c
    raise FileNotFoundError('Repository root not found.')

ROOT = find_root(Path.cwd())
OUT = ROOT / 'data' / 'processed' / 'Petropolis'
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_FILE = OUT / 'petropolis_osm_indicator_feature_matrix_895_cells.parquet'
GRID_FILE = OUT / 'petropolis_grid_1km_with_maturity_indicators.parquet'
REDUNDANCY_FILE = OUT / 'petropolis_osm_strong_redundancy_pairs.csv'
DIST_FILE = OUT / 'petropolis_osm_indicator_distribution_diagnostics.csv'

for p in [FEATURE_FILE, GRID_FILE, REDUNDANCY_FILE, DIST_FILE]:
    if not p.exists():
        raise FileNotFoundError(f'Required Notebook 08/07 input missing: {p}')

features = pd.read_parquet(FEATURE_FILE)
grid = gpd.read_parquet(GRID_FILE)
redundancy = pd.read_csv(REDUNDANCY_FILE)
distribution = pd.read_csv(DIST_FILE)

print('Feature rows:', len(features), 'Grid rows:', len(grid))
if len(features) != 895 or len(grid) != 895:
    raise ValueError('Expected 895 cells from upstream validated outputs.')
if not features['grid_1km_id'].is_unique or not grid['grid_1km_id'].is_unique:
    raise ValueError('grid_1km_id must be unique.')
if set(features['grid_1km_id'].astype(str)) != set(grid['grid_1km_id'].astype(str)):
    raise ValueError('Feature and geometry IDs differ.')

# Reduced feature set: one interpretable representative per conceptual dimension.
# Variables shown by Notebook 08 to be all-missing, constant, or strongly redundant are excluded.
SELECTED_FEATURES = [
    'current_road_length_km',
    'mapping_age_months',
    'peak_growth_m',
    'time_to_90pct_months',
    'plateau_duration_months',
    'active_month_ratio',
    'longest_inactive_streak_months',
    'total_contributions',
    'contribution_temporal_concentration_hhi',
    'recent_activity_ratio',
    'expansion_share_proxy',
]
EXCLUDED_DEGENERATE = [
    'growth_deceleration_ratio',
    'growth_rate_initial_m_per_month',
    'growth_deceleration_m_per_month',
    'contributor_volatility_proxy',
]
for col in SELECTED_FEATURES:
    if col not in features.columns:
        raise KeyError(f'Missing selected feature: {col}')

selection_rows = []
for col in SELECTED_FEATURES:
    d = distribution.loc[distribution['variable'].eq(col)]
    miss = float(d['missing_pct'].iloc[0]) if len(d) else float(features[col].isna().mean()*100)
    r = redundancy[(redundancy.variable_a.eq(col)) | (redundancy.variable_b.eq(col))]
    strongest = float(r.abs_rho.max()) if len(r) else np.nan
    selection_rows.append({'variable':col,'selected':True,'missing_pct':miss,'strongest_abs_rho_in_full_set':strongest})
for col in EXCLUDED_DEGENERATE:
    d = distribution.loc[distribution['variable'].eq(col)]
    selection_rows.append({'variable':col,'selected':False,'missing_pct':float(d['missing_pct'].iloc[0]) if len(d) else np.nan,
                           'strongest_abs_rho_in_full_set':np.nan,
                           'exclusion_reason':'all-missing or zero-variance in Notebook 08 diagnostics'})
selection_df = pd.DataFrame(selection_rows)
selection_df.to_csv(OUT / 'petropolis_osm_typology_feature_selection.csv', index=False)

# Structural separation: cells with no current mapped road length are retained as an explicit class.
features = features.copy()
features['grid_1km_id'] = features['grid_1km_id'].astype(str)
road_mask = features['current_road_length_km'].fillna(0).gt(0)
model_df = features.loc[road_mask, ['grid_1km_id', *SELECTED_FEATURES]].copy()
no_road_df = features.loc[~road_mask, ['grid_1km_id']].copy()
print('Road-present cells modeled:', len(model_df), 'No-current-road structural cells:', len(no_road_df))
if len(model_df) != 781 or len(no_road_df) != 114:
    warnings.warn(f'Expected 781 road-present and 114 no-road cells; observed {len(model_df)} and {len(no_road_df)}.')

# Plateau non-detection is structural; zero duration means no detected plateau among road-present cells.
model_df['plateau_duration_months'] = model_df['plateau_duration_months'].fillna(0)
impute_cols = [c for c in SELECTED_FEATURES if c != 'plateau_duration_months']
imputer = SimpleImputer(strategy='median')
X_raw = model_df[SELECTED_FEATURES].copy()
X_raw[impute_cols] = imputer.fit_transform(X_raw[impute_cols])
imputation = pd.DataFrame({
    'variable': SELECTED_FEATURES,
    'missing_before_modeling': [int(model_df[c].isna().sum()) for c in SELECTED_FEATURES],
    'treatment': ['zero_if_plateau_not_detected' if c == 'plateau_duration_months' else 'median_within_road_present' for c in SELECTED_FEATURES]
})
imputation.to_csv(OUT / 'petropolis_osm_typology_imputation.csv', index=False)

scaler = RobustScaler()
X = scaler.fit_transform(X_raw)


def fit_labels(method, k, Xdata):
    if method == 'kmeans':
        return KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit_predict(Xdata)
    if method == 'ward':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xdata)
    raise ValueError(method)

full_labels = {}
metric_rows = []
for method in ['kmeans', 'ward']:
    for k in K_RANGE:
        lab = fit_labels(method, k, X)
        full_labels[(method, k)] = lab
        sizes = pd.Series(lab).value_counts()
        metric_rows.append({
            'method':method,'k':k,
            'silhouette':silhouette_score(X, lab),
            'calinski_harabasz':calinski_harabasz_score(X, lab),
            'davies_bouldin':davies_bouldin_score(X, lab),
            'min_cluster_n':int(sizes.min()),
            'min_cluster_fraction':float(sizes.min()/len(lab)),
            'max_cluster_fraction':float(sizes.max()/len(lab)),
        })
metrics = pd.DataFrame(metric_rows)

# Deterministic 80% subsample stability using ARI against the full-sample partition.
rng = np.random.default_rng(RANDOM_STATE)
stability_rows = []
n_sub = int(round(len(X)*SUBSAMPLE_FRACTION))
for method in ['kmeans','ward']:
    for k in K_RANGE:
        reference = full_labels[(method,k)]
        aris = []
        for rep in range(BOOTSTRAP_REPS):
            idx = np.sort(rng.choice(len(X), size=n_sub, replace=False))
            lab_sub = fit_labels(method,k,X[idx])
            aris.append(adjusted_rand_score(reference[idx],lab_sub))
        stability_rows.append({'method':method,'k':k,'stability_ari_mean':float(np.mean(aris)),
                               'stability_ari_sd':float(np.std(aris,ddof=1)),
                               'stability_ari_min':float(np.min(aris)),
                               'repetitions':BOOTSTRAP_REPS,'subsample_fraction':SUBSAMPLE_FRACTION})
stability = pd.DataFrame(stability_rows)
metrics = metrics.merge(stability,on=['method','k'])

# Queen-neighbor agreement is a diagnostic, not a spatial clustering constraint.
grid2 = grid.copy(); grid2['grid_1km_id'] = grid2['grid_1km_id'].astype(str)
road_grid = grid2[grid2['grid_1km_id'].isin(model_df['grid_1km_id'])].copy().reset_index(drop=True)
order_map = {gid:i for i,gid in enumerate(model_df['grid_1km_id'])}
road_grid['_model_order'] = road_grid['grid_1km_id'].map(order_map)
road_grid = road_grid.sort_values('_model_order').reset_index(drop=True)
try:
    w = Queen.from_dataframe(road_grid, ids=road_grid.index.tolist(), silence_warnings=True)
except TypeError:
    w = Queen.from_dataframe(road_grid, ids=road_grid.index.tolist())
edge_pairs = []
for i, neis in w.neighbors.items():
    for j in neis:
        if i < j:
            edge_pairs.append((i,j))
for ix,row in metrics.iterrows():
    lab = full_labels[(row.method,int(row.k))]
    metrics.loc[ix,'queen_neighbor_agreement'] = np.mean([lab[i]==lab[j] for i,j in edge_pairs]) if edge_pairs else np.nan
metrics.to_csv(OUT/'petropolis_osm_typology_model_selection.csv',index=False)
stability.to_csv(OUT/'petropolis_osm_typology_stability.csv',index=False)

# Selection rule: require >=5% minimum cluster size and mean subsample ARI >=.75.
# Choose the smallest k within 0.02 silhouette of the best admissible solution.
admissible = metrics[(metrics.min_cluster_fraction>=MIN_CLUSTER_FRACTION)&(metrics.stability_ari_mean>=STABILITY_TARGET)].copy()
if admissible.empty:
    admissible = metrics[metrics.min_cluster_fraction>=MIN_CLUSTER_FRACTION].copy()
    selection_note = 'No model met ARI >= 0.75; selected from models satisfying the 5% minimum cluster fraction.'
else:
    selection_note = 'Selected among models with minimum cluster fraction >=5% and mean subsample ARI >=0.75.'
best_sil = admissible.silhouette.max()
near = admissible[admissible.silhouette>=best_sil-SILHOUETTE_TOLERANCE].copy()
near = near.sort_values(['k','silhouette','stability_ari_mean'],ascending=[True,False,False])
selected = near.iloc[0]
method = str(selected.method); k = int(selected.k)
labels = full_labels[(method,k)]
print('Selected model:', method, 'k=', k)
print(selection_note)
print(selected.to_string())

# Sensitivity comparison with PCA-only clustering; PCA is not the primary model.
from sklearn.decomposition import PCA
pca = PCA(n_components=min(5,X.shape[1]), random_state=RANDOM_STATE)
scores = pca.fit_transform(X)
pca_lab = KMeans(n_clusters=k,n_init=50,random_state=RANDOM_STATE).fit_predict(scores[:,:min(3,scores.shape[1])])
pca_comparison_ari = adjusted_rand_score(labels,pca_lab)

road_assignment = pd.DataFrame({'grid_1km_id':model_df['grid_1km_id'].values,'typology_cluster_internal':labels.astype(int)})
prof_order = (road_assignment.merge(features[['grid_1km_id','current_road_length_km','total_contributions']],on='grid_1km_id')
              .groupby('typology_cluster_internal')[['current_road_length_km','total_contributions']].median()
              .sort_values(['current_road_length_km','total_contributions']).index.tolist())
id_map = {old:i+1 for i,old in enumerate(prof_order)}
road_assignment['typology_class_id'] = road_assignment['typology_cluster_internal'].map(id_map)
road_assignment['typology_label'] = road_assignment['typology_class_id'].map(lambda x:f'Provisional cluster {x}')
no_assignment = no_road_df.copy(); no_assignment['typology_cluster_internal']=-1; no_assignment['typology_class_id']=0; no_assignment['typology_label']='Structural class: no current mapped roads'
assignment = pd.concat([no_assignment,road_assignment],ignore_index=True).sort_values('grid_1km_id')
assignment['clustering_method']=method
assignment['modeled_k']=k
assignment['is_structural_no_road_class']=assignment.typology_class_id.eq(0)

joined = assignment.merge(features,on='grid_1km_id',how='left')
profile_rows=[]
for cid,g in joined.groupby('typology_class_id'):
    for var in SELECTED_FEATURES:
        vals = pd.to_numeric(g[var],errors='coerce')
        profile_rows.append({'typology_class_id':cid,'typology_label':g.typology_label.iloc[0],
                             'n_cells':len(g),'variable':var,'mean':vals.mean(),'median':vals.median(),
                             'q25':vals.quantile(.25),'q75':vals.quantile(.75),'missing_n':int(vals.isna().sum())})
profiles = pd.DataFrame(profile_rows)
profiles.to_csv(OUT/'petropolis_osm_typology_profiles.csv',index=False)

summary = pd.DataFrame([{
    'selected_method':method,'selected_k_modeled_road_cells':k,
    'n_road_present_modeled':len(model_df),'n_structural_no_road':len(no_road_df),'n_total':len(assignment),
    'silhouette':selected.silhouette,'calinski_harabasz':selected.calinski_harabasz,
    'davies_bouldin':selected.davies_bouldin,'stability_ari_mean':selected.stability_ari_mean,
    'stability_ari_sd':selected.stability_ari_sd,'min_cluster_fraction':selected.min_cluster_fraction,
    'queen_neighbor_agreement':selected.queen_neighbor_agreement,'pca3_sensitivity_ari':pca_comparison_ari,
    'selection_rule':selection_note
}])
summary.to_csv(OUT/'petropolis_osm_typology_selected_model.csv',index=False)

checks = {
    'rows_895':len(assignment)==895,
    'ids_unique':assignment.grid_1km_id.is_unique,
    'ids_match_grid':set(assignment.grid_1km_id)==set(grid2.grid_1km_id),
    'no_unassigned':assignment.typology_class_id.notna().all(),
    'structural_no_road_exact':int((assignment.typology_class_id==0).sum())==len(no_road_df),
    'modeled_cluster_min_fraction_ok':float(selected.min_cluster_fraction)>=MIN_CLUSTER_FRACTION,
    'selected_features_finite_after_imputation':np.isfinite(X).all(),
    'profile_covers_all_classes':profiles.typology_class_id.nunique()==assignment.typology_class_id.nunique(),
}
qc = pd.DataFrame([{'check':name,'status':'PASS' if value else 'FAIL'} for name,value in checks.items()])
qc.to_csv(OUT/'petropolis_osm_typology_quality_checks.csv',index=False)
print('\nQuality checks:')
print(qc.to_string(index=False))
if not all(checks.values()):
    raise ValueError('Notebook 09 quality checks failed.')

final_cols=['grid_1km_id','typology_class_id','typology_label','typology_cluster_internal','clustering_method','modeled_k','is_structural_no_road_class']
assignment[final_cols].to_parquet(OUT/'petropolis_osm_cell_typology_895_cells.parquet',index=False)
spatial = grid2.merge(assignment[final_cols],on='grid_1km_id',how='left',validate='one_to_one')
spatial.to_parquet(OUT/'petropolis_grid_1km_with_osm_typology.parquet',index=False)

fig,ax=plt.subplots(figsize=(10,10))
spatial.plot(column='typology_class_id',categorical=True,legend=True,ax=ax,linewidth=0.15,edgecolor='0.6')
ax.set_title(f'Provisional intrinsic OSM cell typology — {method}, k={k} + structural no-road class')
ax.set_axis_off(); fig.tight_layout()
fig.savefig(OUT/'petropolis_osm_cell_typology_map.png',dpi=180,bbox_inches='tight'); plt.close(fig)

print('\nClass sizes:')
print(assignment.groupby(['typology_class_id','typology_label']).size().rename('n').to_string())
print('\nPCA-only 3-component sensitivity ARI versus original-indicator solution:',round(pca_comparison_ari,3))
print('Outputs written to:',OUT)
