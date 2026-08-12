from pathlib import Path
import math
import textwrap

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Rectangle, Polygon
from matplotlib.ticker import MaxNLocator
from pyproj import Transformer
import contextily as cx

ROOT = Path.cwd()
while not (ROOT / "notebooks").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

OUT = ROOT / "data" / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
GRID_FILE = ROOT / "data" / "processed" / "Petropolis" / "petropolis_grid_1km_with_osm_typology_final.parquet"
MUNICIPALITY_FILE = ROOT / "Petropolis" / "Petropolis_2025_4674_IBGE.parquet"
for path in (GRID_FILE, MUNICIPALITY_FILE):
    if not path.exists():
        raise FileNotFoundError(path)

gdf = gpd.read_parquet(GRID_FILE)
municipality = gpd.read_parquet(MUNICIPALITY_FILE)
assert len(gdf) == 895
assert gdf["grid_1km_id"].astype(str).is_unique
assert gdf.crs is not None and municipality.crs is not None

MAP_CRS = "EPSG:31983"
WGS84 = "EPSG:4326"
g = gdf.to_crs(MAP_CRS).copy()
muni = municipality.to_crs(MAP_CRS).dissolve().reset_index(drop=True)
muni_geom = muni.geometry.iloc[0]
centroid_map = muni_geom.centroid
centroid_ll = gpd.GeoSeries([centroid_map], crs=MAP_CRS).to_crs(WGS84).iloc[0]
CITY_LON, CITY_LAT = float(centroid_ll.x), float(centroid_ll.y)

g_display = g.copy()
g_display["geometry"] = g_display.geometry.intersection(muni_geom)
g_display = g_display.loc[~g_display.geometry.is_empty].copy()

DATA_SOURCE_LINES = [
    "OpenStreetMap road history via ohsome API",
    "1 km reference grid derived from IBGE
  source grid",
    "Municipal boundary: IBGE (2025)",
]
BASEMAP_LINES = [
    "CARTO Positron",
    "(© OpenStreetMap contributors, © CARTO)",
]
CRS_LINES = [
    "Coordinate System: SIRGAS 2000",
    "Projection: UTM Zone 23S (EPSG:31983)",
    "Datum: SIRGAS 2000",
]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def dms_label(value, axis):
    hemi = ("W" if value < 0 else "E") if axis == "lon" else ("S" if value < 0 else "N")
    av = abs(float(value))
    deg = int(av)
    minutes = int(round((av - deg) * 60))
    if minutes == 60:
        deg += 1
        minutes = 0
    return f"{deg}°{minutes:02d}′{hemi}"


def nice_geo_step(span):
    candidates = np.array([3, 6, 10, 15, 20, 30, 60]) / 60.0
    return float(candidates[np.argmin(np.abs(candidates - span / 4.0))])


def style_box(ax, linewidth=0.9):
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color("0.25")


def hide_panel_frame(ax):
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_equal_scale_extent(fig, ax, bounds, padding=0.05):
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    cx0, cy0 = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    width = (xmax - xmin) * (1 + 2 * padding)
    height = (ymax - ymin) * (1 + 2 * padding)
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    target = bbox.width / bbox.height
    if width / height < target:
        width = height * target
    else:
        height = width / target
    ax.set_xlim(cx0 - width / 2, cx0 + width / 2)
    ax.set_ylim(cy0 - height / 2, cy0 + height / 2)
    ax.set_aspect("auto")


def add_geographic_graticule(ax):
    """Draw graticule and labels around the inner map frame.

    The map itself is inset inside a larger map panel, so labels can safely sit
    just outside the map frame without crossing into neighbouring panels or
    being clipped by the figure boundary.
    """
    to_ll = Transformer.from_crs(MAP_CRS, WGS84, always_xy=True)
    to_map = Transformer.from_crs(WGS84, MAP_CRS, always_xy=True)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    corners = [
        to_ll.transform(x, y)
        for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    ]
    lons, lats = [p[0] for p in corners], [p[1] for p in corners]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_step = nice_geo_step(max_lon - min_lon)
    lat_step = nice_geo_step(max_lat - min_lat)
    lon_vals = np.arange(math.ceil(min_lon / lon_step) * lon_step, max_lon + 1e-9, lon_step)
    lat_vals = np.arange(math.ceil(min_lat / lat_step) * lat_step, max_lat + 1e-9, lat_step)

    for lon in lon_vals:
        seq = np.linspace(min_lat - 0.03, max_lat + 0.03, 180)
        xy = np.array([to_map.transform(lon, lat) for lat in seq])
        ax.plot(xy[:, 0], xy[:, 1], "--", lw=0.55, color="0.45", alpha=0.28, zorder=3)
        xb, _ = to_map.transform(lon, min_lat)
        xt, _ = to_map.transform(lon, max_lat)
        label = dms_label(lon, "lon")
        if xmin < xb < xmax:
            ax.plot([xb, xb], [ymin, ymin + 0.010 * height], color="0.15", lw=0.8, zorder=35)
            ax.text(
                xb, ymin - 0.014 * height, label,
                ha="center", va="top", fontsize=10.2,
                color="0.15", clip_on=False, zorder=35,
            )
        if xmin < xt < xmax:
            ax.plot([xt, xt], [ymax, ymax - 0.010 * height], color="0.15", lw=0.8, zorder=35)
            ax.text(
                xt, ymax + 0.014 * height, label,
                ha="center", va="bottom", fontsize=10.2,
                color="0.15", clip_on=False, zorder=35,
            )

    for lat in lat_vals:
        seq = np.linspace(min_lon - 0.03, max_lon + 0.03, 180)
        xy = np.array([to_map.transform(lon, lat) for lon in seq])
        ax.plot(xy[:, 0], xy[:, 1], "--", lw=0.55, color="0.45", alpha=0.28, zorder=3)
        _, yl = to_map.transform(min_lon, lat)
        _, yr = to_map.transform(max_lon, lat)
        label = dms_label(lat, "lat")
        if ymin < yl < ymax:
            ax.plot([xmin, xmin + 0.010 * width], [yl, yl], color="0.15", lw=0.8, zorder=35)
            ax.text(
                xmin - 0.012 * width, yl, label,
                ha="right", va="center", rotation=90, fontsize=10.2,
                color="0.15", clip_on=False, zorder=35,
            )
        if ymin < yr < ymax:
            ax.plot([xmax, xmax - 0.010 * width], [yr, yr], color="0.15", lw=0.8, zorder=35)
            ax.text(
                xmax + 0.012 * width, yr, label,
                ha="left", va="center", rotation=-90, fontsize=10.2,
                color="0.15", clip_on=False, zorder=35,
            )


def add_scalebar(ax, length_km=15, segments=3):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    total = length_km * 1000.0
    seg = total / segments
    x0 = xmax - 0.065 * width - total
    y0 = ymin + 0.060 * height
    h = 0.012 * height
    for i in range(segments):
        ax.add_patch(Rectangle(
            (x0 + i * seg, y0), seg, h,
            facecolor="black" if i % 2 == 0 else "white",
            edgecolor="black", lw=0.8, zorder=45,
        ))
    for i in range(segments + 1):
        x = x0 + i * seg
        ax.plot([x, x], [y0, y0 + h], color="black", lw=0.8, zorder=46)
        label = f"{int(i * length_km / segments)}" + (" km" if i == segments else "")
        ax.text(x, y0 + 1.6 * h, label, ha="center", va="bottom", fontsize=10.2, zorder=46)


def add_north_arrow(ax):
    x, y_top, y_wing, y_notch, half_w = 0.945, 0.925, 0.842, 0.862, 0.027
    ax.add_patch(Polygon(
        [(x, y_top), (x - half_w, y_wing), (x, y_notch), (x + half_w, y_wing)],
        closed=True, facecolor="black", edgecolor="black", lw=0.8,
        transform=ax.transAxes, zorder=60, clip_on=False,
    ))
    ax.text(
        x, y_top + 0.018, "N", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=14.5, fontweight="bold", zorder=61,
    )


def load_locator_layers():
    countries_url = (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/ne_110m_admin_0_countries.geojson"
    )
    states_url = (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/ne_50m_admin_1_states_provinces.geojson"
    )
    world = gpd.read_file(countries_url)
    continent_col = "CONTINENT" if "CONTINENT" in world.columns else "continent"
    south = world[world[continent_col].astype(str).str.lower().eq("south america")].copy().to_crs(WGS84)
    if south.empty:
        raise RuntimeError("Natural Earth South America subset is empty.")

    name_col = next((c for c in ("ADMIN", "admin", "NAME", "name") if c in south.columns), None)
    if name_col is None:
        raise RuntimeError("Natural Earth country-name field was not found.")
    brazil = south[south[name_col].astype(str).str.lower().eq("brazil")].copy()
    if brazil.empty:
        raise RuntimeError("Brazil was not found in Natural Earth countries.")

    states = gpd.read_file(states_url).to_crs(WGS84)
    state_mask = pd.Series(False, index=states.index)
    for col in ("admin", "ADMIN", "adm0_name", "geonunit", "adm0_a3", "sov_a3", "iso_a2"):
        if col in states.columns:
            values = states[col].astype(str).str.strip().str.lower()
            state_mask |= values.eq("brazil") | values.eq("bra") | values.eq("br")
    brazil_states = states[state_mask].copy()
    if brazil_states.empty:
        raise RuntimeError(
            "Brazilian states were not found in Natural Earth 50m admin-1 data. "
            f"Available columns: {list(states.columns)}"
        )
    return south, brazil, brazil_states


SOUTH_AMERICA, BRAZIL, BRAZIL_STATES = load_locator_layers()


def draw_location_panel(ax):
    style_box(ax)
    ax.text(
        0.5, 0.965, "LOCATION", transform=ax.transAxes,
        ha="center", va="top", fontsize=10.8, fontweight="bold",
    )

    # Crop the locator to the part of South America surrounding Brazil so the
    # country is visually dominant, as in the approved reference. Brazilian
    # state boundaries are shown to reproduce the requested contextual detail.
    m = ax.inset_axes([0.055, 0.245, 0.89, 0.665])
    SOUTH_AMERICA.plot(
        ax=m, facecolor="#f0f0f0", edgecolor="#a3a3a3", linewidth=0.55, zorder=1,
    )
    BRAZIL.plot(
        ax=m, facecolor="#e6e6e6", edgecolor="#858585", linewidth=0.75, zorder=2,
    )
    BRAZIL_STATES.boundary.plot(ax=m, color="#9a9a9a", linewidth=0.42, zorder=3)
    m.set_xlim(-82.0, -34.0)
    m.set_ylim(-36.0, 7.0)
    m.set_aspect("equal", adjustable="datalim")

    m.scatter(
        [CITY_LON], [CITY_LAT], marker="o", s=56,
        facecolors="none", edgecolors="#d44735", linewidths=1.6, zorder=5,
    )
    m.scatter(
        [CITY_LON], [CITY_LAT], marker="+", s=130,
        color="#d44735", linewidths=1.6, zorder=6,
    )
    m.text(
        CITY_LON + 1.4, CITY_LAT - 0.6, "Petrópolis",
        color="#d44735", fontsize=10.4, fontweight="bold",
        ha="left", va="center", zorder=7,
    )
    m.set_xticks([])
    m.set_yticks([])
    for spine in m.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.35")

    for i, line in enumerate(CRS_LINES):
        ax.text(
            0.055, 0.205 - i * 0.058, line,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.2, color="0.10",
        )


def draw_sources_panel(ax):
    style_box(ax)
    ax.text(
        0.055, 0.92, "DATA SOURCES", transform=ax.transAxes,
        ha="left", va="top", fontsize=10.8, fontweight="bold",
    )
    ax.text(
        0.065, 0.78, f"• {DATA_SOURCE_LINES[0]}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.7,
    )
    ax.text(
        0.065, 0.60, f"• {DATA_SOURCE_LINES[1]}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.7, linespacing=1.08,
    )
    ax.text(
        0.065, 0.39, f"• {DATA_SOURCE_LINES[2]}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.7,
    )

    ax.text(
        0.055, 0.25, "BASEMAP", transform=ax.transAxes,
        ha="left", va="top", fontsize=10.8, fontweight="bold",
    )
    ax.text(
        0.065, 0.13, BASEMAP_LINES[0], transform=ax.transAxes,
        ha="left", va="top", fontsize=8.7,
    )
    ax.text(
        0.065, 0.055, BASEMAP_LINES[1], transform=ax.transAxes,
        ha="left", va="top", fontsize=8.7,
    )


def add_boundary_legend(ax):
    # Deliberately generous separation from the LEGEND heading prevents the
    # symbol border from crossing the title at publication resolution.
    ax.add_patch(Rectangle(
        (0.060, 0.735), 0.105, 0.075,
        transform=ax.transAxes, facecolor="white",
        edgecolor="black", linewidth=1.8,
    ))
    ax.text(
        0.195, 0.773, "Petrópolis municipal boundary",
        transform=ax.transAxes, ha="left", va="center", fontsize=10.2,
    )


def draw_typology_legend(ax):
    style_box(ax)
    ax.text(
        0.055, 0.95, "LEGEND", transform=ax.transAxes,
        ha="left", va="top", fontsize=10.8, fontweight="bold",
    )
    add_boundary_legend(ax)
    ax.text(
        0.055, 0.625, "Final typology (OSM mapping regime)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11.1, fontweight="bold",
    )
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(0.08), cmap(0.52), cmap(0.90)]
    labels = [
        "No currently mapped OSM road lines\n(114 cells — 12.74%)",
        "Lower-intensity, longer-plateau\nmapping regime\n(718 cells — 80.22%)",
        "Higher-intensity, expansion-pulse\nmapping regime\n(63 cells — 7.04%)",
    ]
    for color, label, y in zip(colors, labels, [0.485, 0.295, 0.105]):
        ax.add_patch(Rectangle(
            (0.070, y - 0.038), 0.105, 0.090,
            transform=ax.transAxes, facecolor=color, edgecolor="none",
        ))
        ax.text(
            0.205, y + 0.006, label,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=9.8, linespacing=1.12,
        )


def make_continuous_legend(column, label, log=False):
    s = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        raise ValueError(f"No valid values for {column}")
    if log:
        pos = valid[valid > 0]
        vmin = max(float(pos.quantile(0.02)), 1e-6)
        vmax = float(pos.quantile(0.98))
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))
    else:
        vmin = float(valid.quantile(0.02))
        vmax = float(valid.quantile(0.98))
        if math.isclose(vmin, vmax):
            vmin, vmax = float(valid.min()), float(valid.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

    def drawer(ax):
        style_box(ax)
        ax.text(
            0.055, 0.95, "LEGEND", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.8, fontweight="bold",
        )
        add_boundary_legend(ax)
        ax.text(
            0.055, 0.625, textwrap.fill(label, width=35),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.9, fontweight="bold", linespacing=1.12,
        )

        cax = ax.inset_axes([0.095, 0.145, 0.10, 0.355])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax, orientation="vertical")
        cb.ax.tick_params(labelsize=9.3, length=3)
        cb.outline.set_linewidth(0.7)
        if not log:
            cb.locator = MaxNLocator(nbins=5)
            cb.update_ticks()

        if s.isna().any() or (log and (s <= 0).any()):
            ax.add_patch(Rectangle(
                (0.33, 0.22), 0.095, 0.075,
                transform=ax.transAxes, facecolor="0.88",
                edgecolor="0.55", linewidth=0.7,
            ))
            ax.text(
                0.46, 0.258, "No data / not defined",
                transform=ax.transAxes, ha="left", va="center", fontsize=9.5,
            )
        if log:
            ax.text(
                0.33, 0.425, "Logarithmic\ncolour scale",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=9.5, linespacing=1.15,
            )
    return norm, drawer


def base_figure(title, note, legend_drawer):
    fig = plt.figure(figsize=(16.0, 10.0), facecolor="white")
    fig.suptitle(title, x=0.5, y=0.972, fontsize=18, fontweight="bold", ha="center")

    outer = fig.add_gridspec(
        1, 2,
        left=0.022, right=0.985, bottom=0.040, top=0.910,
        width_ratios=[0.255, 0.745], wspace=0.012,
    )
    left = outer[0].subgridspec(
        3, 1, height_ratios=[0.39, 0.35, 0.26], hspace=0.020,
    )
    right = outer[1].subgridspec(
        2, 1, height_ratios=[0.875, 0.125], hspace=0.020,
    )

    location_ax = fig.add_subplot(left[0])
    legend_ax = fig.add_subplot(left[1])
    sources_ax = fig.add_subplot(left[2])
    map_panel = fig.add_subplot(right[0])
    note_panel = fig.add_subplot(right[1])

    draw_location_panel(location_ax)
    legend_drawer(legend_ax)
    draw_sources_panel(sources_ax)

    # The outer map panel is only a safe layout envelope. The actual map frame
    # is inset, leaving dedicated room for geographic labels on all four sides.
    # This keeps labels inside the figure and away from the left information
    # column while reproducing the reference-map composition.
    hide_panel_frame(map_panel)
    map_ax = map_panel.inset_axes([0.055, 0.060, 0.890, 0.880])
    style_box(map_ax, 0.9)
    set_equal_scale_extent(fig, map_ax, muni.total_bounds, 0.050)

    try:
        cx.add_basemap(
            map_ax,
            source=cx.providers.CartoDB.Positron,
            crs=MAP_CRS,
            attribution=False,
            zoom="auto",
            alpha=0.86,
        )
    except Exception as exc:
        print("WARNING: CARTO Positron unavailable; continuing without basemap:", repr(exc))

    add_geographic_graticule(map_ax)
    add_scalebar(map_ax, 15, 3)
    add_north_arrow(map_ax)

    # Match the interpretation box width to the actual map frame, not to the
    # invisible map-panel envelope used for coordinate-label clearance.
    hide_panel_frame(note_panel)
    note_ax = note_panel.inset_axes([0.055, 0.02, 0.890, 0.96])
    style_box(note_ax, 0.9)
    note_ax.text(
        0.5, 0.50, textwrap.fill(note, width=110),
        transform=note_ax.transAxes, ha="center", va="center",
        fontsize=10.5, linespacing=1.25, color="0.08",
    )
    return fig, map_ax


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def plot_typology():
    title = "FINAL OSM MAPPING MATURITY TYPOLOGY — PETRÓPOLIS (RJ), BRAZIL"
    note = (
        "This typology represents the final classification derived from the intrinsic historical dynamics of "
        "OpenStreetMap road-mapping activity, based on logistic growth, intrinsic saturation, and contribution "
        "patterns. It does not represent external assessments of completeness or accuracy."
    )
    fig, ax = base_figure(title, note, draw_typology_legend)
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(0.08), cmap(0.52), cmap(0.90)]
    for cid, color in enumerate(colors):
        g_display[g_display["typology_class_id"].eq(cid)].plot(
            ax=ax, facecolor=color, edgecolor="#565656",
            linewidth=0.28, alpha=0.92, zorder=10,
        )
    muni.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=30)
    save(fig, "figure_01_final_typology")


def plot_continuous(column, title, legend_label, stem, note, log=False):
    if column not in g_display.columns:
        raise KeyError(column)
    s = pd.to_numeric(g_display[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        raise ValueError(f"No valid values for {column}")
    norm, drawer = make_continuous_legend(column, legend_label, log)
    fig, ax = base_figure(title, note, drawer)
    if log:
        mask = s > 0
        g_display.loc[~mask].plot(
            ax=ax, facecolor="0.88", edgecolor="#666666",
            linewidth=0.22, alpha=0.92, zorder=9,
        )
        g_display.loc[mask].plot(
            ax=ax, column=column, cmap="viridis", norm=norm,
            edgecolor="#666666", linewidth=0.22, alpha=0.90, zorder=10,
        )
    else:
        g_display.plot(
            ax=ax, column=column, cmap="viridis", norm=norm,
            edgecolor="#666666", linewidth=0.22, alpha=0.90, zorder=10,
            missing_kwds={"color": "0.88", "edgecolor": "#777777", "linewidth": 0.22},
        )
    muni.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=30)
    save(fig, stem)


plot_typology()
plot_continuous(
    "current_road_length_km",
    "CURRENT MAPPED ROAD LENGTH — PETRÓPOLIS (RJ), BRAZIL",
    "Current mapped road length (km)",
    "figure_02_current_road_length",
    "Current OpenStreetMap road length aggregated by the original 1 km IBGE analytical cells. Cells are clipped to the municipal boundary for display only; the underlying analytical values are unchanged.",
)
plot_continuous(
    "mapping_age_months",
    "MAPPING AGE — PETRÓPOLIS (RJ), BRAZIL",
    "Mapping age (months)",
    "figure_03_mapping_age",
    "Mapping age measures how long OSM road mapping has been present in each 1 km cell, expressed in months over the validated monthly history used in the study.",
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
    "Cumulative OpenStreetMap contribution counts aggregated by 1 km analytical cell. A logarithmic colour scale preserves spatial contrast across the strongly right-skewed distribution.",
    log=True,
)
plot_continuous(
    "contribution_temporal_concentration_hhi",
    "CONTRIBUTION TEMPORAL CONCENTRATION (HHI) — PETRÓPOLIS (RJ), BRAZIL",
    "Temporal concentration (HHI)",
    "figure_06_temporal_concentration",
    "Herfindahl–Hirschman Index (HHI) of monthly contribution concentration. Higher values indicate activity concentrated in fewer months; lower values indicate a more temporally dispersed contribution history.",
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
print("Municipality source CRS:", municipality.crs)
print("Map CRS:", MAP_CRS)
print("Municipality centroid in EPSG:4326 (lon, lat):", (CITY_LON, CITY_LAT))
print("South America countries loaded:", len(SOUTH_AMERICA))
print("Brazilian states loaded:", len(BRAZIL_STATES))
print("Display cells after clipping:", len(g_display))
print(pd.DataFrame(manifest).to_string(index=False))
print("Article map production complete:", OUT)
