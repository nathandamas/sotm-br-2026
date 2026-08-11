from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely.geometry import LineString
import contextily as cx

ROOT = Path.cwd()
while not (ROOT / 'notebooks').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
OUT = ROOT / 'data' / 'output' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
INPUT = ROOT / 'data' / 'processed' / 'Petropolis' / 'petropolis_grid_1km_with_osm_typology_final.parquet'
if not INPUT.exists():
    raise FileNotFoundError(INPUT)

gdf = gpd.read_parquet(INPUT)
assert len(gdf) == 895
assert gdf['grid_1km_id'].astype(str).is_unique

# Metric cartographic CRS for Petrópolis; contextily will warp Positron to it.
MAP_CRS = 'EPSG:31983'  # SIRGAS 2000 / UTM zone 23S
WGS84 = 'EPSG:4326'
g = gdf.to_crs(MAP_CRS)

DATA_SOURCE = 'IBGE Census Grid; OpenStreetMap history via ohsome API'
BASEMAP_SOURCE = 'CARTO Positron © OpenStreetMap contributors © CARTO'
CRS_LABEL = 'CRS/SRC: SIRGAS 2000 / UTM zone 23S — EPSG:31983'


def nice_step(span, target=5):
    raw = span / target
    p = 10 ** math.floor(math.log10(raw))
    for m in [1, 2, 2.5, 5, 10]:
        if m * p >= raw:
            return m * p
    return 10 * p


def add_scalebar(ax, length_km=None):
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    width = xmax - xmin; height = ymax - ymin
    if length_km is None:
        desired = width / 5 / 1000
        candidates = np.array([1,2,5,10,20,25,50])
        length_km = float(candidates[np.argmin(np.abs(candidates-desired))])
    x0 = xmin + 0.06 * width; y0 = ymin + 0.055 * height
    L = length_km * 1000
    ax.plot([x0, x0+L], [y0, y0], color='black', lw=3, zorder=20)
    ax.plot([x0, x0], [y0-0.006*height, y0+0.006*height], color='black', lw=1.5, zorder=20)
    ax.plot([x0+L, x0+L], [y0-0.006*height, y0+0.006*height], color='black', lw=1.5, zorder=20)
    ax.text(x0+L/2, y0+0.012*height, f'{length_km:g} km', ha='center', va='bottom', fontsize=8, zorder=20)


def add_north(ax):
    ax.annotate('N', xy=(0.94,0.92), xytext=(0.94,0.82), xycoords='axes fraction', textcoords='axes fraction',
                ha='center', va='center', fontsize=11, fontweight='bold',
                arrowprops=dict(facecolor='black', edgecolor='black', width=2.5, headwidth=9, headlength=11), zorder=30)


def add_geographic_graticule(ax):
    to_ll = Transformer.from_crs(MAP_CRS, WGS84, always_xy=True)
    to_map = Transformer.from_crs(WGS84, MAP_CRS, always_xy=True)
    xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
    corners=[to_ll.transform(x,y) for x,y in [(xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)]]
    lons=[p[0] for p in corners]; lats=[p[1] for p in corners]
    lon_step=nice_step(max(lons)-min(lons),4); lat_step=nice_step(max(lats)-min(lats),4)
    lon_vals=np.arange(math.floor(min(lons)/lon_step)*lon_step, max(lons)+lon_step, lon_step)
    lat_vals=np.arange(math.floor(min(lats)/lat_step)*lat_step, max(lats)+lat_step, lat_step)
    for lon in lon_vals:
        lat_seq=np.linspace(min(lats)-0.05, max(lats)+0.05, 80)
        xy=[to_map.transform(lon,lat) for lat in lat_seq]
        ax.plot([p[0] for p in xy],[p[1] for p in xy], ls='--', lw=0.45, color='0.35', alpha=0.35, zorder=2)
        x,y=to_map.transform(lon,min(lats))
        if xmin < x < xmax:
            ax.text(x, ymin+0.012*(ymax-ymin), f'{abs(lon):.2f}°W', ha='center', va='bottom', fontsize=7, color='0.25', zorder=25)
    for lat in lat_vals:
        lon_seq=np.linspace(min(lons)-0.05, max(lons)+0.05, 80)
        xy=[to_map.transform(lon,lat) for lon in lon_seq]
        ax.plot([p[0] for p in xy],[p[1] for p in xy], ls='--', lw=0.45, color='0.35', alpha=0.35, zorder=2)
        x,y=to_map.transform(min(lons),lat)
        if ymin < y < ymax:
            ax.text(xmin+0.008*(xmax-xmin), y, f'{abs(lat):.2f}°S', ha='left', va='center', fontsize=7, color='0.25', zorder=25)


def footer(fig, note=''):
    txt = f'{CRS_LABEL}\nData: {DATA_SOURCE}\nBasemap: {BASEMAP_SOURCE}'
    if note:
        txt += f'\n{note}'
    fig.text(0.02, 0.018, txt, ha='left', va='bottom', fontsize=7.5)


def base_figure(title):
    fig, ax = plt.subplots(figsize=(8.2,8.6))
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    xmin,ymin,xmax,ymax = g.total_bounds
    dx=(xmax-xmin)*0.04; dy=(ymax-ymin)*0.04
    ax.set_xlim(xmin-dx,xmax+dx); ax.set_ylim(ymin-dy,ymax+dy)
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=MAP_CRS, attribution=False, zoom='auto')
    except Exception as e:
        print('WARNING: Positron unavailable; continuing without basemap:', repr(e))
    add_geographic_graticule(ax); add_scalebar(ax); add_north(ax)
    ax.set_axis_off()
    return fig, ax


def save(fig, stem):
    fig.savefig(OUT/f'{stem}.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT/f'{stem}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_continuous(column, title, legend_label, stem, log=False, note=''):
    if column not in g.columns:
        print('SKIP missing:', column); return
    fig,ax=base_figure(title)
    s=pd.to_numeric(g[column], errors='coerce')
    valid=s.replace([np.inf,-np.inf],np.nan).dropna()
    if valid.empty:
        print('SKIP empty:', column); plt.close(fig); return
    if log:
        positive=valid[valid>0]
        vmin=max(float(positive.quantile(.02)),1e-6); vmax=float(positive.quantile(.98))
        norm=LogNorm(vmin=vmin,vmax=max(vmax,vmin*1.01))
        zero_mask=s.fillna(0).le(0)
        g.loc[zero_mask].plot(ax=ax, facecolor='0.88', edgecolor='white', linewidth=.18, alpha=.90, zorder=5)
        g.loc[~zero_mask & s.notna()].plot(ax=ax, column=column, cmap='viridis', norm=norm, edgecolor='white', linewidth=.18, alpha=.86, zorder=6)
        sm=mpl.cm.ScalarMappable(norm=norm,cmap='viridis'); sm.set_array([])
        cb=fig.colorbar(sm,ax=ax,fraction=.035,pad=.015); cb.set_label(legend_label,fontsize=9)
        ax.legend(handles=[Patch(facecolor='0.88',edgecolor='0.6',label='0 / no recorded activity')],loc='lower right',fontsize=7,frameon=True)
    else:
        vmin=float(valid.quantile(.02)); vmax=float(valid.quantile(.98))
        if math.isclose(vmin,vmax): vmin=float(valid.min()); vmax=float(valid.max())
        norm=Normalize(vmin=vmin,vmax=vmax)
        g.plot(ax=ax,column=column,cmap='viridis',norm=norm,edgecolor='white',linewidth=.18,alpha=.86,zorder=6,missing_kwds={'color':'0.88','label':'Not defined'})
        sm=mpl.cm.ScalarMappable(norm=norm,cmap='viridis'); sm.set_array([])
        cb=fig.colorbar(sm,ax=ax,fraction=.035,pad=.015); cb.set_label(legend_label,fontsize=9)
    footer(fig, note + (' Values outside the 2–98% display range are clipped to endpoint colors.' if not log else ' Logarithmic color normalization; positive display range uses the 2–98% quantiles.'))
    save(fig,stem)


def plot_typology():
    fig,ax=base_figure('Intrinsic OpenStreetMap mapping regimes')
    labels={0:'No currently mapped OSM road lines',1:'Lower-intensity, longer-plateau mapping regime',2:'Higher-intensity, expansion-pulse mapping regime'}
    cmap=mpl.colormaps['viridis']; colors=[cmap(.08),cmap(.52),cmap(.92)]
    for cid in [0,1,2]:
        g[g.typology_class_id.eq(cid)].plot(ax=ax,facecolor=colors[cid],edgecolor='white',linewidth=.22,alpha=.88,zorder=6)
    handles=[Patch(facecolor=colors[c],edgecolor='white',label=f'{c}: {labels[c]}') for c in [0,1,2]]
    ax.legend(handles=handles,loc='lower right',fontsize=7.2,frameon=True,title='Final typology',title_fontsize=8)
    footer(fig,'Classes describe intrinsic OSM mapping histories; they are not external completeness or accuracy classes.')
    save(fig,'figure_01_final_typology')

plot_typology()
plot_continuous('current_road_length_km','Current mapped road length','Road length (km)','figure_02_current_road_length')
plot_continuous('mapping_age_months','Age of mapped-road presence','Mapping age (months)','figure_03_mapping_age')
plot_continuous('active_month_ratio','Persistence of mapping activity','Active-month ratio','figure_04_active_month_ratio')
plot_continuous('total_contributions','Cumulative OSM contributions','Contributions (log scale)','figure_05_total_contributions',log=True)
plot_continuous('contribution_temporal_concentration_hhi','Temporal concentration of contributions','Temporal concentration (HHI)','figure_06_temporal_concentration')
plot_continuous('time_to_90pct_months','Time to 90% of intrinsic recent reference','Months to 90%','figure_07_time_to_90pct',note='The 90% reference is intrinsic to each OSM trajectory, not an external completeness benchmark.')
plot_continuous('plateau_duration_months','Duration of intrinsic plateau','Plateau duration (months)','figure_08_plateau_duration',note='Plateau is an intrinsic trajectory descriptor, not a completeness statement.')

manifest=[]
for p in sorted(OUT.glob('figure_*.*')):
    manifest.append({'file':p.name,'bytes':p.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT/'figure_manifest.csv',index=False)
print(pd.DataFrame(manifest).to_string(index=False))
print('Article map production complete:', OUT)
