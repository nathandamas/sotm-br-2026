from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from libpysal.weights import Queen

RANDOM_STATE = 42
K_RANGE = range(2, 9)
RESAMPLE_REPS = 50
SUBSAMPLE_FRACTION = 0.80
MIN_CLUSTER_FRACTION = 0.05
STABILITY_TARGET = 0.75
SILHOUETTE_TOLERANCE = 0.02
CLIP_LOW, CLIP_HIGH = 0.01, 0.99


def find_root(start: Path) -> Path:
    p = start.resolve()
    for c in [p, *p.parents]:
        if (c / 'notebooks').exists() and (c / 'scripts').exists():
            return c
    raise FileNotFoundError('Repository root not found.')

ROOT = find_root(Path.cwd())
OUT = ROOT / 'data' / 'processed' / 'Petropolis'
FEATURE_FILE = OUT / 'petropolis_osm_indicator_feature_matrix_895_cells.parquet'
GRID_FILE = OUT / 'petropolis_grid_1km_with_maturity_indicators.parquet'
REDUNDANCY_FILE = OUT / 'petropolis_osm_strong_redundancy_pairs.csv'
DIST_FILE = OUT / 'petropolis_osm_indicator_distribution_diagnostics.csv'
for p in [FEATURE_FILE, GRID_FILE, REDUNDANCY_FILE, DIST_FILE]:
    if not p.exists():
        raise FileNotFoundError(p)

features = pd.read_parquet(FEATURE_FILE)
grid = gpd.read_parquet(GRID_FILE)
redundancy = pd.read_csv(REDUNDANCY_FILE)
distribution = pd.read_csv(DIST_FILE)
features['grid_1km_id'] = features['grid_1km_id'].astype(str)
grid['grid_1km_id'] = grid['grid_1km_id'].astype(str)
assert len(features) == 895 and features['grid_1km_id'].is_unique
assert len(grid) == 895 and grid['grid_1km_id'].is_unique
assert set(features.grid_1km_id) == set(grid.grid_1km_id)

# Final Notebook 08 confirms that every selected pair has |Spearman rho| < 0.85.
# PCA scores are diagnostic only and are not used as the primary clustering space.
SELECTED = [
    'current_road_length_km',
    'mapping_age_months',
    'growth_deceleration_m_per_month',
    'time_to_90pct_months',
    'plateau_duration_months',
    'active_month_ratio',
    'recent_activity_ratio',
    'contribution_temporal_concentration_hhi',
    'expansion_share_proxy',
    'contributor_volatility_proxy',
]
for c in SELECTED:
    if c not in features.columns:
        raise KeyError(c)
    d = distribution.loc[distribution.variable.eq(c)]
    if len(d) and (int(d.unique_n.iloc[0]) <= 1 or float(d.missing_pct.iloc[0]) > 40):
        raise ValueError(f'Invalid selected feature: {c}')
strong_selected = redundancy[
    redundancy.variable_a.isin(SELECTED) & redundancy.variable_b.isin(SELECTED) & redundancy.abs_rho.ge(0.85)
]
if len(strong_selected):
    raise ValueError(f'Selected features contain strong redundancy:\n{strong_selected}')

selection = []
for c in SELECTED:
    d = distribution.loc[distribution.variable.eq(c)]
    selection.append({
        'variable': c,
        'selected': True,
        'missing_pct': float(d.missing_pct.iloc[0]) if len(d) else float(features[c].isna().mean()*100),
        'unique_n': int(d.unique_n.iloc[0]) if len(d) else int(features[c].dropna().nunique()),
        'criterion': 'interpretable representative; no selected pair has |Spearman rho| >= 0.85'
    })
pd.DataFrame(selection).to_csv(OUT/'petropolis_osm_typology_feature_selection.csv', index=False)

# Cells with no current mapped roads are a structural class, not median-imputed road histories.
road_mask = features.current_road_length_km.fillna(0).gt(0)
model = features.loc[road_mask, ['grid_1km_id', *SELECTED]].copy().reset_index(drop=True)
no_road = features.loc[~road_mask, ['grid_1km_id']].copy()
print('Road-present cells modeled:', len(model))
print('Structural no-road cells:', len(no_road))

# Median imputation within road-present cells and 1–99% clipping before RobustScaler.
X = model[SELECTED].astype(float).copy()
pre_rows = []
for c in SELECTED:
    missing_n = int(X[c].isna().sum())
    med = float(X[c].median())
    q01, q99 = float(X[c].quantile(CLIP_LOW)), float(X[c].quantile(CLIP_HIGH))
    X[c] = X[c].fillna(med).clip(q01, q99)
    pre_rows.append({'variable':c,'missing_n':missing_n,'median_imputation':med,'clip_q01':q01,'clip_q99':q99})
pd.DataFrame(pre_rows).to_csv(OUT/'petropolis_osm_typology_preprocessing.csv', index=False)
Z = RobustScaler(quantile_range=(25,75)).fit_transform(X)
if not np.isfinite(Z).all():
    raise ValueError('Non-finite standardized feature matrix.')


def fit_labels(method, k, z):
    if method == 'kmeans':
        return KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit_predict(z)
    if method == 'ward':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(z)
    raise ValueError(method)

solutions, metric_rows = {}, []
for method in ['kmeans','ward']:
    for k in K_RANGE:
        lab = fit_labels(method,k,Z)
        solutions[(method,k)] = lab
        sz = pd.Series(lab).value_counts()
        metric_rows.append({
            'method':method,'k':k,
            'silhouette':float(silhouette_score(Z,lab)),
            'calinski_harabasz':float(calinski_harabasz_score(Z,lab)),
            'davies_bouldin':float(davies_bouldin_score(Z,lab)),
            'min_cluster_n':int(sz.min()),
            'min_cluster_fraction':float(sz.min()/len(lab)),
            'max_cluster_fraction':float(sz.max()/len(lab)),
        })
metrics = pd.DataFrame(metric_rows)

# Deterministic 80% subsample stability using ARI.
rng = np.random.default_rng(RANDOM_STATE)
stab_rows=[]
nsub=int(round(len(Z)*SUBSAMPLE_FRACTION))
for method in ['kmeans','ward']:
    for k in K_RANGE:
        ref=solutions[(method,k)]; aris=[]
        for rep in range(RESAMPLE_REPS):
            idx=np.sort(rng.choice(len(Z),size=nsub,replace=False))
            aris.append(adjusted_rand_score(ref[idx],fit_labels(method,k,Z[idx])))
        stab_rows.append({'method':method,'k':k,'stability_ari_mean':float(np.mean(aris)),
                          'stability_ari_sd':float(np.std(aris,ddof=1)),'stability_ari_p05':float(np.quantile(aris,.05)),
                          'repetitions':RESAMPLE_REPS,'subsample_fraction':SUBSAMPLE_FRACTION})
stability=pd.DataFrame(stab_rows)
metrics=metrics.merge(stability,on=['method','k'])

# Queen-neighbor agreement is descriptive only; it is not optimized.
road_grid=grid[grid.grid_1km_id.isin(model.grid_1km_id)].copy()
order={gid:i for i,gid in enumerate(model.grid_1km_id)}
road_grid['_order']=road_grid.grid_1km_id.map(order)
road_grid=road_grid.sort_values('_order').reset_index(drop=True)
w=Queen.from_dataframe(road_grid,ids=road_grid.index.tolist(),silence_warnings=True)
edges=[(i,j) for i,nei in w.neighbors.items() for j in nei if i<j]
for ix,r in metrics.iterrows():
    lab=solutions[(r.method,int(r.k))]
    metrics.loc[ix,'queen_neighbor_agreement']=float(np.mean([lab[i]==lab[j] for i,j in edges])) if edges else np.nan

# Selection: >=5% minimum cluster share and mean ARI >=0.75; then smallest k within
# 0.02 silhouette of the admissible best, favoring parsimony.
admissible=metrics[(metrics.min_cluster_fraction>=MIN_CLUSTER_FRACTION)&(metrics.stability_ari_mean>=STABILITY_TARGET)].copy()
if admissible.empty:
    admissible=metrics[metrics.min_cluster_fraction>=MIN_CLUSTER_FRACTION].copy()
    selection_note='No model met mean ARI >= 0.75; restricted to minimum cluster share >=5%.'
else:
    selection_note='Admissible: minimum cluster share >=5% and mean subsample ARI >=0.75.'
best_sil=float(admissible.silhouette.max())
near=admissible[admissible.silhouette>=best_sil-SILHOUETTE_TOLERANCE].sort_values(['k','silhouette','stability_ari_mean'],ascending=[True,False,False])
chosen=near.iloc[0]
method,k=str(chosen.method),int(chosen.k)
labels=solutions[(method,k)]
metrics.to_csv(OUT/'petropolis_osm_typology_model_selection.csv',index=False)
stability.to_csv(OUT/'petropolis_osm_typology_stability.csv',index=False)

# PCA-only sensitivity analysis, explicitly secondary.
pc=PCA(n_components=min(5,Z.shape[1]),random_state=RANDOM_STATE).fit_transform(Z)
pca_labels=KMeans(n_clusters=k,n_init=50,random_state=RANDOM_STATE).fit_predict(pc[:,:min(3,pc.shape[1])])
pca_sensitivity_ari=float(adjusted_rand_score(labels,pca_labels))

road_assignment=pd.DataFrame({'grid_1km_id':model.grid_1km_id,'cluster_internal':labels.astype(int)})
order_df=(road_assignment.merge(features[['grid_1km_id','current_road_length_km','active_month_ratio']],on='grid_1km_id')
          .groupby('cluster_internal')[['current_road_length_km','active_month_ratio']].median()
          .sort_values(['current_road_length_km','active_month_ratio']))
id_map={old:i+1 for i,old in enumerate(order_df.index.tolist())}
road_assignment['typology_class_id']=road_assignment.cluster_internal.map(id_map)
road_assignment['typology_label']=road_assignment.typology_class_id.map(lambda x:f'Provisional cluster {x}')
no_assignment=no_road.copy(); no_assignment['cluster_internal']=-1; no_assignment['typology_class_id']=0
no_assignment['typology_label']='Structural class: no current mapped roads'
assignment=pd.concat([no_assignment,road_assignment],ignore_index=True)
assignment['clustering_method']=method; assignment['modeled_k']=k
assignment['is_structural_no_road_class']=assignment.typology_class_id.eq(0)

joined=assignment.merge(features,on='grid_1km_id',how='left')
profiles=[]
for cid,g in joined.groupby('typology_class_id'):
    for var in SELECTED:
        s=pd.to_numeric(g[var],errors='coerce')
        profiles.append({'typology_class_id':int(cid),'typology_label':g.typology_label.iloc[0],'n_cells':len(g),
                         'variable':var,'n_nonmissing':int(s.notna().sum()),'mean':s.mean(),'median':s.median(),
                         'q25':s.quantile(.25),'q75':s.quantile(.75)})
profiles=pd.DataFrame(profiles)
profiles.to_csv(OUT/'petropolis_osm_typology_profiles.csv',index=False)

summary=pd.DataFrame([{'selected_method':method,'selected_k_road_present':k,'road_present_n':len(model),
                       'structural_no_road_n':len(no_road),'total_n':len(assignment),'silhouette':chosen.silhouette,
                       'calinski_harabasz':chosen.calinski_harabasz,'davies_bouldin':chosen.davies_bouldin,
                       'stability_ari_mean':chosen.stability_ari_mean,'stability_ari_sd':chosen.stability_ari_sd,
                       'min_cluster_fraction':chosen.min_cluster_fraction,'queen_neighbor_agreement':chosen.queen_neighbor_agreement,
                       'pca3_sensitivity_ari':pca_sensitivity_ari,'selection_rule':selection_note}])
summary.to_csv(OUT/'petropolis_osm_typology_selected_model.csv',index=False)

final_cols=['grid_1km_id','typology_class_id','typology_label','cluster_internal','clustering_method','modeled_k','is_structural_no_road_class']
assignment=assignment.sort_values('grid_1km_id')
assignment[final_cols].to_parquet(OUT/'petropolis_osm_cell_typology_895_cells.parquet',index=False)
spatial=grid.merge(assignment[final_cols],on='grid_1km_id',how='left',validate='one_to_one')
spatial.to_parquet(OUT/'petropolis_grid_1km_with_osm_typology.parquet',index=False)

fig,ax=plt.subplots(figsize=(10,10))
spatial.plot(column='typology_class_id',categorical=True,legend=True,ax=ax,linewidth=.15,edgecolor='0.6')
ax.set_title(f'Provisional intrinsic OSM cell typology — {method}, k={k} + structural no-road class')
ax.set_axis_off(); fig.tight_layout(); fig.savefig(OUT/'petropolis_osm_cell_typology_map.png',dpi=180,bbox_inches='tight'); plt.close(fig)

checks={
 'rows_895':len(assignment)==895,
 'ids_unique':assignment.grid_1km_id.is_unique,
 'ids_match_grid':set(assignment.grid_1km_id)==set(grid.grid_1km_id),
 'all_cells_classified':assignment.typology_class_id.notna().all(),
 'selected_features_nonredundant':len(strong_selected)==0,
 'selected_features_finite_after_preprocessing':np.isfinite(Z).all(),
 'minimum_cluster_fraction_ok':float(chosen.min_cluster_fraction)>=MIN_CLUSTER_FRACTION,
 'stability_finite':np.isfinite(chosen.stability_ari_mean),
 'profiles_cover_all_classes':profiles.typology_class_id.nunique()==assignment.typology_class_id.nunique(),
}
qc=pd.DataFrame([{'check':n,'status':'PASS' if v else 'FAIL'} for n,v in checks.items()])
qc.to_csv(OUT/'petropolis_osm_typology_quality_checks.csv',index=False)
print('Selected indicators:',SELECTED)
print('\nModel-selection metrics:\n',metrics.to_string(index=False))
print('\nChosen solution:\n',summary.to_string(index=False))
print('\nClass sizes:\n',assignment.groupby(['typology_class_id','typology_label']).size().to_string())
print('\nQuality control:\n',qc.to_string(index=False))
if not all(checks.values()): raise RuntimeError(checks)
print('\nInterpretation boundary: classes summarize intrinsic OSM mapping-history patterns, not external completeness or data quality.')
