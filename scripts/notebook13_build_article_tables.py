from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / 'data' / 'processed' / 'Petropolis'
OUT = REPO_ROOT / 'tables' / 'article'
OUT.mkdir(parents=True, exist_ok=True)

FINAL_LABELS = {
    0: 'No currently mapped OSM road lines',
    1: 'Lower-intensity, longer-plateau mapping regime',
    2: 'Higher-intensity, expansion-pulse mapping regime',
}

DISPLAY = {
    'current_road_length_km': ('Current mapped road length', 'km'),
    'mapping_age_months': ('Age of mapped-road presence', 'months'),
    'active_month_ratio': ('Active-month ratio', 'proportion'),
    'recent_activity_ratio': ('Recent-activity ratio', 'proportion'),
    'persistence_index': ('Persistence index', 'index'),
    'total_contributions': ('Cumulative OSM contributions', 'count'),
    'total_contributor_months': ('Contributor-months', 'count'),
    'contribution_temporal_concentration_hhi': ('Temporal concentration of contributions', 'HHI'),
    'contributor_volatility_proxy': ('Contributor volatility proxy', 'CV'),
    'peak_growth_m': ('Peak monthly road growth', 'm'),
    'positive_growth_months': ('Positive-growth months', 'months'),
    'growth_rate_initial_m_per_month': ('Initial road-growth rate', 'm/month'),
    'growth_rate_recent_m_per_month': ('Recent road-growth rate', 'm/month'),
    'growth_deceleration_m_per_month': ('Growth deceleration', 'm/month'),
    'time_to_90pct_months': ('Time to 90% of intrinsic recent reference', 'months'),
    'months_since_90pct': ('Months since reaching 90% reference', 'months'),
    'plateau_duration_months': ('Intrinsic plateau duration', 'months'),
    'plateau_tail_stability_fraction': ('Plateau-tail stability fraction', 'proportion'),
    'post_plateau_active_month_ratio': ('Post-plateau active-month ratio', 'proportion'),
}


def fmt_num(x, digits=2):
    if pd.isna(x):
        return 'NA'
    if abs(float(x)) >= 1000:
        return f'{float(x):,.0f}'
    return f'{float(x):.{digits}f}'


def fmt_p(x):
    if pd.isna(x):
        return 'NA'
    x = float(x)
    if x < 0.001:
        return '<0.001'
    return f'{x:.3f}'


def save_table(df, stem, caption, note):
    csv_path = OUT / f'{stem}.csv'
    md_path = OUT / f'{stem}.md'
    tex_path = OUT / f'{stem}.tex'
    df.to_csv(csv_path, index=False)
    md = f'### {caption}\n\n' + df.to_markdown(index=False) + f'\n\n*Note.* {note}\n'
    md_path.write_text(md, encoding='utf-8')
    tex = df.to_latex(index=False, escape=True, caption=caption, label=f'tab:{stem}', longtable=False)
    tex += f'\n% Note: {note}\n'
    tex_path.write_text(tex, encoding='utf-8')


class_summary = pd.read_csv(ROOT / 'petropolis_osm_typology_article_class_summary.csv')
profiles = pd.read_csv(ROOT / 'petropolis_osm_typology_article_indicator_profiles.csv')
road_tests = pd.read_csv(ROOT / 'petropolis_osm_typology_article_road_cluster_tests.csv')
pairwise = pd.read_csv(ROOT / 'petropolis_osm_typology_article_pairwise_tests.csv')
spatial_global = pd.read_csv(ROOT / 'petropolis_osm_typology_article_spatial_global.csv')
spatial_classes = pd.read_csv(ROOT / 'petropolis_osm_typology_article_spatial_classes.csv')
final_summary = pd.read_csv(ROOT / 'petropolis_osm_typology_final_class_summary.csv')

t1 = final_summary.copy()
if 'final_typology_label' not in t1.columns:
    label_col = [c for c in t1.columns if 'label' in c.lower()][0]
    t1 = t1.rename(columns={label_col: 'final_typology_label'})
count_col = [c for c in t1.columns if c in ('n_cells','cell_count','count')]
if count_col:
    t1 = t1.rename(columns={count_col[0]: 'Cells'})
else:
    t1['Cells'] = class_summary['n_cells']
share_col = [c for c in t1.columns if 'share' in c.lower() or 'percent' in c.lower() or 'pct' in c.lower()]
if share_col:
    t1 = t1.rename(columns={share_col[0]: 'Share (%)'})
else:
    t1['Share (%)'] = t1['Cells'] / 895 * 100
if 'typology_class_id' not in t1.columns:
    t1['typology_class_id'] = [0,1,2]
t1['Class'] = t1['typology_class_id'].map(FINAL_LABELS)
t1 = t1[['Class','Cells','Share (%)']]
t1['Share (%)'] = t1['Share (%)'].map(lambda x: round(float(x), 2))
t1.loc[len(t1)] = ['Total study grid', int(t1['Cells'].sum()), round(float(t1['Share (%)'].sum()),2)]
save_table(t1, 'table_01_study_grid_typology_composition',
           'Study grid and final OSM mapping-regime typology composition',
           'The study grid contains 895 one-kilometre cells. Class 0 is a structural state, while Classes 1 and 2 are road-present intrinsic OSM mapping regimes.')

profile_vars = [
    'current_road_length_km','mapping_age_months','active_month_ratio','recent_activity_ratio',
    'persistence_index','total_contributions','total_contributor_months',
    'contribution_temporal_concentration_hhi','contributor_volatility_proxy',
    'time_to_90pct_months','plateau_duration_months'
]
rows=[]
for var in profile_vars:
    if var not in set(profiles['variable']):
        continue
    d = profiles[profiles['variable']==var].set_index('typology_class_id')
    label, unit = DISPLAY.get(var,(var,''))
    row={'Indicator':label,'Unit':unit}
    for cid in [0,1,2]:
        if cid not in d.index:
            val='NA'
        else:
            r=d.loc[cid]
            if pd.isna(r['median']):
                val='NA'
            else:
                val=f"{fmt_num(r['median'])} [{fmt_num(r['q25'])}–{fmt_num(r['q75'])}]"
        row[FINAL_LABELS[cid]]=val
    rows.append(row)
t2=pd.DataFrame(rows)
save_table(t2, 'table_02_substantive_class_profiles',
           'Substantive profiles of the three final classes',
           'Values are median [Q1–Q3] on original indicator scales. NA denotes metrics that are structurally undefined or unavailable. The no-road class is retained as a structural class rather than treated as a third modeled road-history regime.')

road_keep = [
    'peak_growth_m','positive_growth_months','growth_rate_initial_m_per_month',
    'growth_deceleration_m_per_month','time_to_90pct_months','months_since_90pct',
    'plateau_duration_months','plateau_tail_stability_fraction','post_plateau_active_month_ratio'
]
contrib_keep = [
    'active_month_ratio','recent_activity_ratio','persistence_index','total_contributions',
    'total_contributor_months','contribution_temporal_concentration_hhi','contributor_volatility_proxy',
    'current_road_length_km'
]
comp=[]
for _,r in road_tests[road_tests['variable'].isin(road_keep)].iterrows():
    comp.append({'variable':r['variable'],'median_cluster_1':r['median_cluster_1'],'median_cluster_2':r['median_cluster_2'],'cliffs_delta_cluster1_vs_2':r['cliffs_delta_cluster1_vs_2'],'mannwhitney_u':r['mannwhitney_u'],'p_value':r['p_value'],'p_adj_bh':r['p_adj_bh']})
p12 = pairwise[(pairwise['class_a']==1)&(pairwise['class_b']==2)&(pairwise['variable'].isin(contrib_keep))]
for _,r in p12.iterrows():
    comp.append({'variable':r['variable'],'median_cluster_1':r['median_a'],'median_cluster_2':r['median_b'],'cliffs_delta_cluster1_vs_2':r['cliffs_delta_a_vs_b'],'mannwhitney_u':r['mannwhitney_u'],'p_value':r['p_value'],'p_adj_bh':r['p_adj_bh']})
t3raw=pd.DataFrame(comp).drop_duplicates('variable')
rows=[]
for _,r in t3raw.iterrows():
    lab,unit=DISPLAY.get(r['variable'],(r['variable'],''))
    rows.append({'Indicator':lab,'Unit':unit,'Lower-intensity regime median':fmt_num(r['median_cluster_1']),'Higher-intensity regime median':fmt_num(r['median_cluster_2']),"Cliff's delta (lower vs higher)":fmt_num(r['cliffs_delta_cluster1_vs_2'],3),'Mann–Whitney U':fmt_num(r['mannwhitney_u'],1),'p':fmt_p(r['p_value']),'BH-adjusted p':fmt_p(r['p_adj_bh'])})
t3=pd.DataFrame(rows)
save_table(t3, 'table_03_road_present_regime_comparison',
           'Focused comparison of the two road-present OSM mapping regimes',
           "Cliff's delta is oriented as lower-intensity minus higher-intensity; negative values indicate larger values in the higher-intensity expansion-pulse regime. P-values are two-sided Mann–Whitney tests, with Benjamini–Hochberg correction.")

g = spatial_global.iloc[0]
rows=[
    {'Spatial metric':'Queen-neighbor edges','Value':int(g['n_queen_edges'])},
    {'Spatial metric':'Observed same-class Queen adjacency','Value':f"{100*g['observed_same_class_neighbor_share']:.2f}%"},
    {'Spatial metric':'Permutation expectation','Value':f"{100*g['permutation_mean_same_class_share']:.2f}%"},
    {'Spatial metric':'Permutation SD','Value':f"{100*g['permutation_sd_same_class_share']:.2f} percentage points"},
    {'Spatial metric':'Empirical upper-tail p-value','Value':fmt_p(g['permutation_p_upper'])},
    {'Spatial metric':'Permutations','Value':int(g['permutations'])},
    {'Spatial metric':'Queen islands','Value':int(g['queen_islands'])},
]
for _,r in spatial_classes.iterrows():
    cid=int(r['typology_class_id'])
    rows.append({'Spatial metric':f"Connected components — {FINAL_LABELS[cid]}",'Value':int(r['n_connected_components'])})
    rows.append({'Spatial metric':f"Largest-component share — {FINAL_LABELS[cid]}",'Value':f"{100*r['largest_component_share']:.2f}%"})
t4=pd.DataFrame(rows)
save_table(t4, 'table_04_spatial_coherence_summary',
           'Spatial coherence of the final typology',
           'Queen contiguity was evaluated after clustering and did not constrain the cluster solution. The empirical p-value is based on 999 random label permutations.')

t5=pd.DataFrame([
    {'Class':FINAL_LABELS[0],'Operational definition':'1 km cells with no currently mapped OSM highway-line geometry in the analyzed snapshot.','Substantive interpretation':'Structural map state; road-history indicators tied to current mapped-road presence can be undefined.','Key limitation':'Does not imply physical absence of roads and is not an OSM incompleteness class.'},
    {'Class':FINAL_LABELS[1],'Operational definition':'Road-present cells assigned to the larger k-means cluster after robust preprocessing of non-redundant intrinsic OSM-history indicators.','Substantive interpretation':'Lower median mapped-road length and historical contribution intensity, with a comparatively longer intrinsic plateau.','Key limitation':'A relative intrinsic history regime; not a class of completeness, positional accuracy, or external data quality.'},
    {'Class':FINAL_LABELS[2],'Operational definition':'Road-present cells assigned to the smaller k-means cluster after the same validated preprocessing and model selection.','Substantive interpretation':'Higher mapped-road and contribution intensity, with stronger expansion pulses followed by pronounced deceleration.','Key limitation':'A relative intrinsic history regime; it does not establish superior mapping quality or real-world infrastructure density.'}
])
save_table(t5, 'table_05_class_definitions_and_limitations',
           'Definitions, substantive interpretation, and limitations of the final classes',
           'All labels describe intrinsic OpenStreetMap mapping histories within this study. They are not external validation classes and should not be described as completeness or accuracy categories.')

checks = pd.DataFrame([
    ('table_1_total_895', int(t1.iloc[:-1]['Cells'].sum()) == 895),
    ('table_1_three_classes', len(t1)-1 == 3),
    ('table_2_has_profiles', len(t2) >= 8),
    ('table_3_has_effect_sizes', len(t3) >= 10 and t3["Cliff's delta (lower vs higher)"].notna().all()),
    ('table_4_spatial_metrics', len(t4) >= 10),
    ('table_5_three_definitions', len(t5) == 3),
])
checks.columns=['check','pass']
checks['status']=np.where(checks['pass'],'PASS','FAIL')
checks.to_csv(OUT/'table_quality_checks.csv', index=False)
if not checks['pass'].all():
    raise RuntimeError('One or more article-table quality checks failed:\n' + checks.to_string(index=False))

manifest=[]
for p in sorted(OUT.glob('table_*')):
    if p.is_file():
        manifest.append({'file':p.name,'bytes':p.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT/'table_manifest.csv', index=False)
print('Generated article tables:')
print(pd.DataFrame(manifest).to_string(index=False))
print('\nQuality checks:')
print(checks.to_string(index=False))
