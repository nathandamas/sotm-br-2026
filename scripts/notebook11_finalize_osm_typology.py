from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np


def find_root(start: Path) -> Path:
    p = start.resolve()
    for c in [p, *p.parents]:
        if (c / 'notebooks').exists() and (c / 'scripts').exists():
            return c
    raise FileNotFoundError('Repository root not found.')

ROOT = find_root(Path.cwd())
OUT = ROOT / 'data' / 'processed' / 'Petropolis'
OUT.mkdir(parents=True, exist_ok=True)

TYPOLOGY = OUT / 'petropolis_osm_cell_typology_895_cells.parquet'
SPATIAL = OUT / 'petropolis_grid_1km_with_osm_typology.parquet'
CLASS_SUMMARY = OUT / 'petropolis_osm_typology_article_class_summary.csv'
ROAD_TESTS = OUT / 'petropolis_osm_typology_article_road_cluster_tests.csv'
PAIRWISE = OUT / 'petropolis_osm_typology_article_pairwise_tests.csv'
SPATIAL_GLOBAL = OUT / 'petropolis_osm_typology_article_spatial_global.csv'
SPATIAL_CLASSES = OUT / 'petropolis_osm_typology_article_spatial_classes.csv'
QC10 = OUT / 'petropolis_osm_typology_article_quality_checks.csv'

for p in [TYPOLOGY, SPATIAL, CLASS_SUMMARY, ROAD_TESTS, PAIRWISE, SPATIAL_GLOBAL, SPATIAL_CLASSES, QC10]:
    if not p.exists():
        raise FileNotFoundError(p)

typ = pd.read_parquet(TYPOLOGY)
gdf = gpd.read_parquet(SPATIAL)
summary = pd.read_csv(CLASS_SUMMARY)
road = pd.read_csv(ROAD_TESTS)
pairwise = pd.read_csv(PAIRWISE)
sp_global = pd.read_csv(SPATIAL_GLOBAL)
sp_class = pd.read_csv(SPATIAL_CLASSES)
qc10 = pd.read_csv(QC10)

assert len(typ) == 895 and typ['grid_1km_id'].is_unique
assert len(gdf) == 895 and gdf['grid_1km_id'].is_unique
assert (qc10['status'] == 'PASS').all()

FINAL_LABELS = {
    0: 'No currently mapped OSM road lines',
    1: 'Lower-intensity, longer-plateau mapping regime',
    2: 'Higher-intensity, expansion-pulse mapping regime',
}
SHORT_LABELS = {
    0: 'No current OSM road lines',
    1: 'Lower-intensity / longer-plateau',
    2: 'Higher-intensity / expansion-pulse',
}

# Preserve provenance explicitly.
typ = typ.copy()
typ['provisional_typology_label'] = typ['typology_label']
typ['substantive_typology_label'] = typ['typology_class_id'].map(FINAL_LABELS)
typ['substantive_typology_short_label'] = typ['typology_class_id'].map(SHORT_LABELS)

gdf = gdf.copy()
gdf['provisional_typology_label'] = gdf['typology_label']
gdf['substantive_typology_label'] = gdf['typology_class_id'].map(FINAL_LABELS)
gdf['substantive_typology_short_label'] = gdf['typology_class_id'].map(SHORT_LABELS)

# Evidence table used to justify names.
def get_pair(var):
    d = pairwise[(pairwise['variable'] == var) & (pairwise['class_a'] == 1) & (pairwise['class_b'] == 2)]
    return d.iloc[0] if len(d) else None

def get_road(var):
    d = road[road['variable'] == var]
    return d.iloc[0] if len(d) else None

rows = []
for cid in [0,1,2]:
    s = summary[summary.typology_class_id.eq(cid)].iloc[0]
    rows.append({
        'typology_class_id': cid,
        'provisional_label': s['typology_label'],
        'substantive_label': FINAL_LABELS[cid],
        'n_cells': int(s['n_cells']),
        'share_pct': float(s['share_of_895_pct']),
        'median_current_road_length_km': s['median_current_road_length_km'],
        'median_active_month_ratio': s['median_active_month_ratio'],
        'median_total_contributions': s['median_total_contributions'],
        'median_mapping_age_months': s['median_mapping_age_months'],
        'median_time_to_90pct_months': s['median_time_to_90pct_months'],
        'median_plateau_duration_months': s['median_plateau_duration_months'],
    })
class_table = pd.DataFrame(rows)

# Explicit empirical naming evidence for the two road-present clusters.
evidence_vars = [
    'current_road_length_km', 'active_month_ratio', 'total_contributions',
    'total_contributor_months', 'contributor_volatility_proxy'
]
evidence = []
for var in evidence_vars:
    r = get_pair(var)
    if r is not None:
        evidence.append({
            'variable': var,
            'median_cluster_1': r['median_a'],
            'median_cluster_2': r['median_b'],
            'cliffs_delta_cluster1_vs_2': r['cliffs_delta_a_vs_b'],
            'p_adj_bh': r['p_adj_bh'],
            'significant_fdr_0_05': r['significant_fdr_0_05'],
            'evidence_family': 'activity_or_network_state',
        })
for var in ['peak_growth_m','positive_growth_months','growth_rate_initial_m_per_month','growth_deceleration_m_per_month','months_since_90pct','plateau_duration_months']:
    r = get_road(var)
    if r is not None:
        evidence.append({
            'variable': var,
            'median_cluster_1': r['median_cluster_1'],
            'median_cluster_2': r['median_cluster_2'],
            'cliffs_delta_cluster1_vs_2': r['cliffs_delta_cluster1_vs_2'],
            'p_adj_bh': r['p_adj_bh'],
            'significant_fdr_0_05': r['significant_fdr_0_05'],
            'evidence_family': 'road_history',
        })
name_evidence = pd.DataFrame(evidence)

# Article-oriented textual synthesis, deliberately conservative.
results_text = (
    'The intrinsic OSM typology comprised three operational classes across 895 1-km cells. '
    'A structural class contained 114 cells (12.7%) with no currently mapped OSM highway-line geometry. '
    'Among the 781 road-present cells, 718 were assigned to a lower-intensity, longer-plateau mapping regime and 63 to a higher-intensity, expansion-pulse mapping regime. '
    'The higher-intensity regime had a median mapped-road length of 4.75 km versus 2.27 km, a median of 96 versus 28 contributions, and a higher active-month ratio (0.0625 versus 0.0402). '
    'Its road-history profile was distinguished most strongly by an initial growth rate near 95.2 m/month versus 0.0, a growth-deceleration metric near 93.8 m/month versus 0.0, more positive-growth months (7 versus 4), and a larger peak growth event (about 2.14 km versus 1.40 km). '
    'By contrast, the lower-intensity regime showed a longer median plateau duration (70 versus 59 months) and more months since first reaching 90% of its recent intrinsic road-length reference (75 versus 68 months). '
    'These labels therefore summarize relative mapping-history regimes only and must not be interpreted as external completeness, positional accuracy, or real-world road availability.'
)

sp = sp_global.iloc[0]
spatial_text = (
    f"The typology was spatially structured: {sp['observed_same_class_neighbor_share']:.3f} of Queen-neighbor edges connected cells of the same class, "
    f"compared with a permutation expectation of {sp['permutation_mean_same_class_share']:.3f} (p={sp['permutation_p_upper']:.3f}; 999 permutations). "
    'Spatial coherence was evaluated descriptively and was not imposed as a clustering constraint.'
)

limitations_text = (
    'The class labels are descriptive summaries of intrinsic OpenStreetMap history. The structural no-road class denotes absence of currently mapped highway-line geometry in the analyzed snapshot, not absence of physical roads. '
    'The two road-present regimes are relative clusters derived from selected intrinsic indicators and do not constitute external quality or completeness classes. '
    'Recent activity prevalence did not differ significantly between the two road-present clusters after multiple-testing correction, so the final names avoid implying persistently higher current activity.'
)

# Final outputs.
OUT_TYP = OUT / 'petropolis_osm_cell_typology_final_895_cells.parquet'
OUT_GEO = OUT / 'petropolis_grid_1km_with_osm_typology_final.parquet'
OUT_CLASS = OUT / 'petropolis_osm_typology_final_class_summary.csv'
OUT_EVID = OUT / 'petropolis_osm_typology_final_naming_evidence.csv'
OUT_SPATIAL = OUT / 'petropolis_osm_typology_final_spatial_summary.csv'
OUT_TEXT = OUT / 'petropolis_osm_typology_final_results_text.md'
OUT_METHOD = OUT / 'petropolis_osm_typology_final_method_notes.md'
OUT_TEX = OUT / 'petropolis_osm_typology_final_class_summary.tex'
OUT_QC = OUT / 'petropolis_osm_typology_final_quality_checks.csv'

typ.to_parquet(OUT_TYP, index=False)
gdf.to_parquet(OUT_GEO, index=False)
class_table.to_csv(OUT_CLASS, index=False)
name_evidence.to_csv(OUT_EVID, index=False)
sp_global.to_csv(OUT_SPATIAL, index=False)
class_table.to_latex(OUT_TEX, index=False, float_format='%.3f')
OUT_TEXT.write_text('# Results-ready synthesis\n\n' + results_text + '\n\n' + spatial_text + '\n', encoding='utf-8')
OUT_METHOD.write_text('# Methodological interpretation boundary\n\n' + limitations_text + '\n', encoding='utf-8')

checks = {
    'rows_895_typology': len(typ) == 895,
    'rows_895_spatial': len(gdf) == 895,
    'ids_unique_typology': typ['grid_1km_id'].is_unique,
    'ids_unique_spatial': gdf['grid_1km_id'].is_unique,
    'ids_match': set(typ['grid_1km_id']) == set(gdf['grid_1km_id']),
    'all_final_labels_present': typ['substantive_typology_label'].notna().all(),
    'three_final_labels': typ['substantive_typology_label'].nunique() == 3,
    'provisional_labels_preserved': 'provisional_typology_label' in typ.columns,
    'class_counts_preserved': typ.groupby('typology_class_id').size().to_dict() == {0:114,1:718,2:63},
    'naming_evidence_nonempty': len(name_evidence) >= 8,
    'notebook10_qc_all_pass': (qc10['status'] == 'PASS').all(),
}
qc = pd.DataFrame([{'check': k, 'status': 'PASS' if v else 'FAIL'} for k,v in checks.items()])
qc.to_csv(OUT_QC, index=False)

print('Final labels:')
for k,v in FINAL_LABELS.items():
    print(f'  {k}: {v}')
print('\nFinal class table:')
print(class_table.to_string(index=False))
print('\nNaming evidence:')
print(name_evidence.to_string(index=False))
print('\nSpatial synthesis:')
print(spatial_text)
print('\nQuality control:')
print(qc.to_string(index=False))
if not all(checks.values()):
    raise RuntimeError(checks)

print('\nInterpretation boundary: final labels summarize intrinsic OSM mapping-history regimes, not external completeness or quality.')
