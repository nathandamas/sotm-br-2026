from pathlib import Path
import math
import textwrap

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Patch, Rectangle, Polygon
from pyproj import Transformer
import contextily as cx

# -----------------------------------------------------------------------------
# Paths and validated inputs
# -----------------------------------------------------------------------------
ROOT = Path.cwd()
while not (ROOT / "notebooks").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

OUT = ROOT / "data" / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GRID_FILE = ROOT / "data" / "processed" / "Petropolis" / "petropolis_grid_1km_with_osm_typology_final.parquet"
MUNICIPALITY_FILE = ROOT / "Petropolis" / "Petropolis_2025_4674_IBGE.parquet"

for path in [GRID_FILE, MUNICIPALITY_FILE]:
    if not path.exists():
        raise FileNotFoundError(path)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
gdf = gpd.read_parquet(GRID_FILE)
municipality = gpd.read_parquet(MUNICIPALITY_FILE)

assert len(gdf) == 895
assert gdf["grid_1km_id"].astype(str).is_unique
assert gdf.crs is not None and municipality.crs is not None

MAP_CRS = "EPSG:31983"  # SIRGAS 2000 / UTM zone 23S
WGS84 = "EPSG:4326"

g = gdf.to_crs(MAP_CRS).copy()
muni = municipality.to_crs(MAP_CRS).dissolve().reset_index(drop=True)
muni_ll = municipality.to_crs(WGS84).dissolve().reset_index(drop=True)

city_point = muni_ll.geometry.iloc[0].representative_point()
CITY_LON, CITY_LAT = city_point.x, city_point.y

DATA_SOURCE_LINES = [
    "OpenStreetMap road history via ohsome API",
    "1 km reference grid derived from IBGE source grid",
    "Municipal boundary: IBGE (2025)",
]
BASEMAP_LINES = [
    "CARTO Positron",
    "© OpenStreetMap contributors, © CARTO",
]
CRS_LINES = [
    "Map CRS: SIRGAS 2000 / UTM zone 23S (EPSG:31983)",
    "Graticule: geographic longitude/latitude",
    "Datum: SIRGAS 2000",
]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def dms_label(value, axis):
    hemi = ("W" if value < 0 else "E") if axis == "lon" else ("S" if value < 0 else "N")
    av = abs(value)
    deg = int(av)
    minutes = int(round((av - deg) * 60))
    if minutes == 60:
        deg += 1
        minutes = 0
    return f"{deg}°{minutes:02d}′{hemi}"


def nice_geo_step(span):
    # Prefer 3, 6, 10, 15, 20 or 30 arc-minute spacing.
    candidates = np.array([3, 6, 10, 15, 20, 30, 60, 120]) / 60.0
    target = span / 4.0
    return float(candidates[np.argmin(np.abs(candidates - target))])


def add_geographic_graticule(ax):
    to_ll = Transformer.from_crs(MAP_CRS, WGS84, always_xy=True)
    to_map = Transformer.from_crs(WGS84, MAP_CRS, always_xy=True)
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()

    corners = [to_ll.transform(x, y) for x, y in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]]
    lons = [p[0] for p in corners]; lats = [p[1] for p in corners]
    lon_step = nice_geo_step(max(lons) - min(lons))
    lat_step = nice_geo_step(max(lats) - min(lats))

    lon_vals = np.arange(math.floor(min(lons) / lon_step) * lon_step, max(lons) + lon_step, lon_step)
    lat_vals = np.arange(math.floor(min(lats) / lat_step) * lat_step, max(lats) + lat_step, lat_step)

    for lon in lon_vals:
        lat_seq = np.linspace(min(lats) - 0.05, max(lats) + 0.05, 120)
        xy = [to_map.transform(lon, lat) for lat in lat_seq]
        ax.plot([p[0] for p in xy], [p[1] for p in xy], ls="--", lw=0.6, color="0.38", alpha=0.35, zorder=2)
        x, _ = to_map.transform(lon, min(lats))
        if xmin < x < xmax:
            ax.text(x, ymin + 0.010 * (ymax-ymin), dms_label(lon, "lon"), ha="center", va="bottom", fontsize=9.5, color="0.20", zorder=30)

    for lat in lat_vals:
        lon_seq = np.linspace(min(lons) - 0.05, max(lons) + 0.05, 120)
        xy = [to_map.transform(lon, lat) for lon in lon_seq]
        ax.plot([p[0] for p in xy], [p[1] for p in xy], ls="--", lw=0.6, color="0.38", alpha=0.35, zorder=2)
        _, y = to_map.transform(min(lons), lat)
        if ymin < y < ymax:
            ax.text(xmin + 0.008 * (xmax-xmin), y, dms_label(lat, "lat"), ha="left", va="center", rotation=90, fontsize=9.5, color="0.20", zorder=30)


def add_scalebar(ax, length_km=10):
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    width = xmax - xmin; height = ymax - ymin
    L = length_km * 1000.0
    x0 = xmax - 0.07 * width - L
    y0 = ymin + 0.065 * height
    seg = L / 2.0
    h = 0.013 * height

    ax.add_patch(Rectangle((x0, y0), seg, h, facecolor="black", edgecolor="black", lw=0.8, zorder=40))
    ax.add_patch(Rectangle((x0 + seg, y0), seg, h, facecolor="white", edgecolor="black", lw=0.8, zorder=40))
    for x in [x0, x0 + seg, x0 + L]:
        ax.plot([x, x], [y0, y0 + h], color="black", lw=0.8, zorder=41)
    ax.text(x0, y0 + 1.5*h, "0", ha="center", va="bottom", fontsize=9.5, zorder=41)
    ax.text(x0 + seg, y0 + 1.5*h, f"{length_km/2:g}", ha="center", va="bottom", fontsize=9.5, zorder=41)
    ax.text(x0 + L, y0 + 1.5*h, f"{length_km:g} km", ha="center", va="bottom", fontsize=9.5, zorder=41)


def add_north_arrow(ax):
    # Classic upward-pointing north arrow matching the visual language of the approved mock-up.
    x = 0.945
    apex_y = 0.935
    base_y = 0.855
    half_w = 0.022
    triangle = Polygon(
        [(x, apex_y), (x-half_w, base_y), (x+half_w, base_y)],
        closed=True,
        facecolor="black",
        edgecolor="black",
        transform=ax.transAxes,
        zorder=50,
        clip_on=False,
    )
    ax.add_patch(triangle)
    ax.text(x, 0.955, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=14, fontweight="bold", color="black", zorder=51)


def load_south_america():
    # Natural Earth 1:110m countries. Vector geometry avoids the distorted web-tile inset.
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(url)
    continent_col = "CONTINENT" if "CONTINENT" in world.columns else "continent"
    south = world[world[continent_col].astype(str).str.lower().eq("south america")].copy()
    if south.empty:
        raise RuntimeError("Natural Earth South America subset is empty.")
    return south.to_crs(WGS84)


SOUTH_AMERICA = load_south_america()


def draw_location_map(ax):
    ax.set_title("LOCATION", fontsize=12.5, fontweight="bold", pad=7)
    SOUTH_AMERICA.plot(ax=ax, facecolor="#eeeeee", edgecolor="#777777", linewidth=0.7, zorder=1)
    ax.set_xlim(-84, -32)
    ax.set_ylim(-57, 14)
    ax.set_aspect("equal", adjustable="box")
    ax.scatter([CITY_LON], [CITY_LAT], marker="o", s=46, facecolors="none", edgecolors="#d43f2f", linewidths=1.8, zorder=5)
    ax.scatter([CITY_LON], [CITY_LAT], marker="+", s=110, color="#d43f2f", linewidths=1.6, zorder=6)
    ax.text(CITY_LON + 2.0, CITY_LAT - 1.3, "Petrópolis", color="#d43f2f", fontsize=11.5, fontweight="bold", ha="left", va="center", zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color("0.35")


def draw_side_panel(fig, legend_drawer):
    # Left information panel.
    panel = fig.add_axes([0.018, 0.035, 0.245, 0.885])
    panel.set_xticks([]); panel.set_yticks([])
    panel.set_facecolor("white")
    for sp in panel.spines.values():
        sp.set_linewidth(0.8); sp.set_color("0.25")

    # Location inset.
    loc = fig.add_axes([0.033, 0.615, 0.215, 0.275])
    draw_location_map(loc)

    # CRS block.
    y = 0.595
    for line in CRS_LINES:
        fig.text(0.035, y, line, ha="left", va="top", fontsize=10.2)
        y -= 0.022

    # Separator.
    fig.lines.append(mpl.lines.Line2D([0.018, 0.263], [0.515, 0.515], transform=fig.transFigure, color="0.35", lw=0.8))
    fig.text(0.032, 0.495, "LEGEND", ha="left", va="top", fontsize=12.5, fontweight="bold")

    # Municipal boundary legend item.
    fig.patches.append(Rectangle((0.035, 0.455), 0.027, 0.022, transform=fig.transFigure, facecolor="white", edgecolor="black", lw=1.9, clip_on=False))
    fig.text(0.070, 0.466, "Petrópolis municipal boundary", ha="left", va="center", fontsize=10.5)

    # Thematic legend starts below the boundary item.
    legend_drawer(fig, 0.425)

    # Data-source / basemap block at bottom, leaving generous whitespace.
    fig.lines.append(mpl.lines.Line2D([0.018, 0.263], [0.235, 0.235], transform=fig.transFigure, color="0.35", lw=0.8))
    fig.text(0.032, 0.220, "DATA SOURCES", ha="left", va="top", fontsize=11.5, fontweight="bold")
    yy = 0.192
    for line in DATA_SOURCE_LINES:
        fig.text(0.034, yy, f"• {line}", ha="left", va="top", fontsize=9.8)
        yy -= 0.021
    fig.text(0.032, 0.112, "BASEMAP", ha="left", va="top", fontsize=11.5, fontweight="bold")
    yy = 0.084
    for line in BASEMAP_LINES:
        fig.text(0.034, yy, line, ha="left", va="top", fontsize=9.8)
        yy -= 0.020


def base_figure(title, note, legend_drawer):
    # Wide article layout. Main map and lower note have exactly the same horizontal extent.
    fig = plt.figure(figsize=(15.0, 10.0), facecolor="white")
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=18, fontweight="bold")

    draw_side_panel(fig, legend_drawer)

    map_left, map_width = 0.285, 0.695
    ax = fig.add_axes([map_left, 0.145, map_width, 0.775])

    # Extent uses both the real municipal boundary and the original intersecting 1 km cells.
    bounds = np.vstack([g.total_bounds, muni.total_bounds])
    xmin, ymin = bounds[:, [0,1]].min(axis=0)
    xmax, ymax = bounds[:, [2,3]].max(axis=0)
    dx = (xmax - xmin) * 0.035
    dy = (ymax - ymin) * 0.035
    ax.set_xlim(xmin-dx, xmax+dx)
    ax.set_ylim(ymin-dy, ymax+dy)
    ax.set_aspect("equal", adjustable="box")

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=MAP_CRS, attribution=False, zoom="auto", alpha=0.90)
    except Exception as e:
        print("WARNING: Positron unavailable; continuing without basemap:", repr(e))

    add_geographic_graticule(ax)
    add_scalebar(ax, length_km=10)
    add_north_arrow(ax)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.9); sp.set_color("0.25")

    note_ax = fig.add_axes([map_left, 0.035, map_width, 0.075])
    note_ax.set_xticks([]); note_ax.set_yticks([])
    note_ax.set_facecolor("white")
    for sp in note_ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("0.25")
    wrapped = textwrap.fill(note, width=145)
    note_ax.text(0.5, 0.5, wrapped, ha="center", va="center", fontsize=10.5, transform=note_ax.transAxes)

    return fig, ax


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_discrete_typology_legend(fig, y0):
    labels = {
        0: "No currently mapped OSM road lines\n(114 cells — 12.74%)",
        1: "Lower-intensity, longer-plateau mapping regime\n(718 cells — 80.22%)",
        2: "Higher-intensity, expansion-pulse mapping regime\n(63 cells — 7.04%)",
    }
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(.08), cmap(.52), cmap(.90)]
    fig.text(0.034, y0, "Final typology (OSM mapping regime)", ha="left", va="top", fontsize=10.8, fontweight="bold")
    yy = y0 - 0.040
    for cid in [0,1,2]:
        fig.patches.append(Rectangle((0.038, yy-0.010), 0.025, 0.025, transform=fig.transFigure, facecolor=colors[cid], edgecolor="none", clip_on=False))
        fig.text(0.070, yy+0.002, labels[cid], ha="left", va="center", fontsize=10.0)
        yy -= 0.078


def make_continuous_legend(column, label, log=False):
    s = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if log:
        positive = valid[valid > 0]
        vmin = max(float(positive.quantile(.02)), 1e-6)
        vmax = float(positive.quantile(.98))
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin*1.01))
    else:
        vmin = float(valid.quantile(.02)); vmax = float(valid.quantile(.98))
        if math.isclose(vmin, vmax):
            vmin, vmax = float(valid.min()), float(valid.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

    def drawer(fig, y0):
        fig.text(0.034, y0, label, ha="left", va="top", fontsize=10.8, fontweight="bold")
        cax = fig.add_axes([0.040, y0-0.175, 0.030, 0.145])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.ax.tick_params(labelsize=9.5, width=0.7, length=3)
        cb.outline.set_linewidth(0.7)
        fig.text(0.082, y0-0.103, "High", ha="left", va="center", fontsize=9.5)
        fig.text(0.082, y0-0.168, "Low", ha="left", va="center", fontsize=9.5)
        if log:
            fig.text(0.034, y0-0.205, "Color scale: logarithmic", ha="left", va="top", fontsize=9.2, color="0.30")
        fig.patches.append(Rectangle((0.038, y0-0.245), 0.025, 0.022, transform=fig.transFigure, facecolor="0.88", edgecolor="0.65", lw=0.6, clip_on=False))
        fig.text(0.070, y0-0.234, "Not defined / no recorded activity", ha="left", va="center", fontsize=9.5)

    return norm, drawer


def plot_typology():
    title = "FINAL OSM MAPPING MATURITY TYPOLOGY — PETRÓPOLIS (RJ), BRAZIL"
    note = (
        "Final classification derived from intrinsic historical dynamics of OpenStreetMap road mapping. "
        "The classes describe mapping regimes and do not represent external assessments of completeness, positional accuracy, or real-world road availability."
    )
    fig, ax = base_figure(title, note, draw_discrete_typology_legend)
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(.08), cmap(.52), cmap(.90)]
    for cid in [0,1,2]:
        g[g["typology_class_id"].eq(cid)].plot(ax=ax, facecolor=colors[cid], edgecolor="white", linewidth=0.35, alpha=0.90, zorder=8)
    muni.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=25)
    save(fig, "figure_01_final_typology")


def plot_continuous(column, title, legend_label, stem, note, log=False):
    if column not in g.columns:
        raise KeyError(column)
    s = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        raise ValueError(f"No valid values for {column}")

    norm, legend_drawer = make_continuous_legend(column, legend_label, log=log)
    fig, ax = base_figure(title, note, legend_drawer)

    if log:
        positive_mask = s > 0
        g.loc[~positive_mask].plot(ax=ax, facecolor="0.88", edgecolor="white", linewidth=0.25, alpha=0.92, zorder=7)
        g.loc[positive_mask].plot(ax=ax, column=column, cmap="viridis", norm=norm, edgecolor="white", linewidth=0.25, alpha=0.88, zorder=8)
    else:
        g.plot(ax=ax, column=column, cmap="viridis", norm=norm, edgecolor="white", linewidth=0.25, alpha=0.88, zorder=8, missing_kwds={"color":"0.88"})

    muni.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=25)
    save(fig, stem)


# -----------------------------------------------------------------------------
# Produce the eight article maps
# -----------------------------------------------------------------------------
plot_typology()

plot_continuous(
    "current_road_length_km",
    "CURRENT MAPPED ROAD LENGTH — PETRÓPOLIS (RJ), BRAZIL",
    "Current mapped road length (km)",
    "figure_02_current_road_length",
    "Current OpenStreetMap road length aggregated by the original 1 km IBGE analytical cells. The black line is the actual Petrópolis municipal boundary used in the study; cells are preserved in their original geometry.",
)

plot_continuous(
    "mapping_age_months",
    "MAPPING AGE — PETRÓPOLIS (RJ), BRAZIL",
    "Mapping age (months)",
    "figure_03_mapping_age",
    "Mapping age measures how long OSM road mapping has been present in each 1 km cell, expressed in months over the validated monthly history used by the study.",
)

plot_continuous(
    "active_month_ratio",
    "ACTIVE-MONTH RATIO — PETRÓPOLIS (RJ), BRAZIL",
    "Active-month ratio",
    "figure_04_active_month_ratio",
    "Share of months with recorded OpenStreetMap road-mapping activity in each 1 km cell over the complete monthly observation period.",
)

plot_continuous(
    "total_contributions",
    "TOTAL OSM CONTRIBUTIONS — PETRÓPOLIS (RJ), BRAZIL",
    "Cumulative OSM contributions",
    "figure_05_total_contributions",
    "Cumulative OpenStreetMap contribution counts aggregated by 1 km analytical cell. A logarithmic color scale is used to preserve spatial contrast across the strongly right-skewed distribution.",
    log=True,
)

plot_continuous(
    "contribution_temporal_concentration_hhi",
    "CONTRIBUTION TEMPORAL CONCENTRATION (HHI) — PETRÓPOLIS (RJ), BRAZIL",
    "Temporal concentration (HHI)",
    "figure_06_temporal_concentration",
    "Herfindahl–Hirschman Index (HHI) of monthly contribution concentration. Higher values indicate that contributions are concentrated in fewer months; lower values indicate a more temporally dispersed history.",
)

plot_continuous(
    "time_to_90pct_months",
    "TIME TO 90% OF INTRINSIC RECENT REFERENCE — PETRÓPOLIS (RJ), BRAZIL",
    "Time to 90% (months)",
    "figure_07_time_to_90pct",
    "Months required for each cell to reach 90% of its intrinsic recent road-length reference. The reference is internal to each OSM trajectory and is not an external completeness benchmark.",
)

plot_continuous(
    "plateau_duration_months",
    "INTRINSIC PLATEAU DURATION — PETRÓPOLIS (RJ), BRAZIL",
    "Intrinsic plateau duration (months)",
    "figure_08_plateau_duration",
    "Duration of the intrinsic plateau phase identified from each cell's OSM road-mapping trajectory. Plateau is a trajectory descriptor and must not be interpreted as external completeness.",
)

manifest = []
for p in sorted(OUT.glob("figure_*.*")):
    manifest.append({"file": p.name, "bytes": p.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT / "figure_manifest.csv", index=False)

print("Municipality source:", MUNICIPALITY_FILE)
print("Municipality CRS:", municipality.crs)
print("Map CRS:", MAP_CRS)
print("Municipality representative point (lon, lat):", (CITY_LON, CITY_LAT))
print("South America countries loaded:", len(SOUTH_AMERICA))
print(pd.DataFrame(manifest).to_string(index=False))
print("Article map production complete:", OUT)
