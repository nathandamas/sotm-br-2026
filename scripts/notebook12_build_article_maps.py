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

# -----------------------------------------------------------------------------
# Paths and validated inputs
# -----------------------------------------------------------------------------
ROOT = Path.cwd()
while not (ROOT / "notebooks").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

OUT = ROOT / "data" / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GRID_FILE = (
    ROOT
    / "data"
    / "processed"
    / "Petropolis"
    / "petropolis_grid_1km_with_osm_typology_final.parquet"
)
MUNICIPALITY_FILE = ROOT / "Petropolis" / "Petropolis_2025_4674_IBGE.parquet"

for path in (GRID_FILE, MUNICIPALITY_FILE):
    if not path.exists():
        raise FileNotFoundError(path)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
gdf = gpd.read_parquet(GRID_FILE)
municipality = gpd.read_parquet(MUNICIPALITY_FILE)

assert len(gdf) == 895
assert gdf["grid_1km_id"].astype(str).is_unique
assert gdf.crs is not None
assert municipality.crs is not None

MAP_CRS = "EPSG:31983"  # SIRGAS 2000 / UTM zone 23S
WGS84 = "EPSG:4326"

g = gdf.to_crs(MAP_CRS).copy()
muni = municipality.to_crs(MAP_CRS).dissolve().reset_index(drop=True)
muni_geom = muni.geometry.iloc[0]

# The location marker is the geometric centroid calculated in the projected
# metric CRS and only then transformed to geographic coordinates.  This avoids
# the bias that can arise when a centroid is calculated directly in EPSG:4326.
centroid_map = muni_geom.centroid
centroid_ll = gpd.GeoSeries([centroid_map], crs=MAP_CRS).to_crs(WGS84).iloc[0]
CITY_LON, CITY_LAT = float(centroid_ll.x), float(centroid_ll.y)

# The analytical grid is never modified.  For cartographic display only, cells
# are clipped to the real municipal boundary so the map follows the actual
# Petrópolis perimeter while preserving the original cell-level values.
g_display = g.copy()
g_display["geometry"] = g_display.geometry.intersection(muni_geom)
g_display = g_display.loc[~g_display.geometry.is_empty].copy()

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
    "Coordinate System: SIRGAS 2000",
    "Projection: UTM Zone 23S (EPSG:31983)",
    "Datum: SIRGAS 2000",
]

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
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
    # Prefer familiar cartographic intervals.  Six arc-minutes (0.1°) is
    # generally appropriate for the Petrópolis map extent.
    candidates = np.array([3, 6, 10, 15, 20, 30, 60]) / 60.0
    target = span / 4.0
    return float(candidates[np.argmin(np.abs(candidates - target))])


def set_equal_scale_extent(fig, ax, bounds, padding=0.045):
    """Fill the whole map box without distorting the projected geometry."""
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    cx0 = (xmin + xmax) / 2.0
    cy0 = (ymin + ymax) / 2.0
    width = (xmax - xmin) * (1.0 + 2.0 * padding)
    height = (ymax - ymin) * (1.0 + 2.0 * padding)

    fig.canvas.draw()
    bbox = ax.get_window_extent()
    target_ratio = bbox.width / bbox.height
    data_ratio = width / height

    if data_ratio < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio

    ax.set_xlim(cx0 - width / 2.0, cx0 + width / 2.0)
    ax.set_ylim(cy0 - height / 2.0, cy0 + height / 2.0)
    ax.set_aspect("auto")


def add_geographic_graticule(ax):
    """Draw geographic graticule and coordinate labels on all four borders."""
    to_ll = Transformer.from_crs(MAP_CRS, WGS84, always_xy=True)
    to_map = Transformer.from_crs(WGS84, MAP_CRS, always_xy=True)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width = xmax - xmin
    height = ymax - ymin

    corners = [
        to_ll.transform(x, y)
        for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    ]
    lons = [p[0] for p in corners]
    lats = [p[1] for p in corners]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    lon_step = nice_geo_step(max_lon - min_lon)
    lat_step = nice_geo_step(max_lat - min_lat)
    lon_vals = np.arange(
        math.ceil(min_lon / lon_step) * lon_step,
        max_lon + 1e-9,
        lon_step,
    )
    lat_vals = np.arange(
        math.ceil(min_lat / lat_step) * lat_step,
        max_lat + 1e-9,
        lat_step,
    )

    # Meridians + top/bottom labels.
    for lon in lon_vals:
        lat_seq = np.linspace(min_lat - 0.03, max_lat + 0.03, 180)
        xy = np.array([to_map.transform(lon, lat) for lat in lat_seq])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            ls="--",
            lw=0.55,
            color="0.45",
            alpha=0.28,
            zorder=3,
        )

        xb, _ = to_map.transform(lon, min_lat)
        xt, _ = to_map.transform(lon, max_lat)
        label = dms_label(lon, "lon")
        if xmin < xb < xmax:
            ax.plot([xb, xb], [ymin, ymin + 0.010 * height], color="0.15", lw=0.8, zorder=35)
            ax.text(
                xb,
                ymin - 0.014 * height,
                label,
                ha="center",
                va="top",
                fontsize=10.5,
                color="0.15",
                clip_on=False,
                zorder=35,
            )
        if xmin < xt < xmax:
            ax.plot([xt, xt], [ymax, ymax - 0.010 * height], color="0.15", lw=0.8, zorder=35)
            ax.text(
                xt,
                ymax + 0.014 * height,
                label,
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="0.15",
                clip_on=False,
                zorder=35,
            )

    # Parallels + left/right labels.
    for lat in lat_vals:
        lon_seq = np.linspace(min_lon - 0.03, max_lon + 0.03, 180)
        xy = np.array([to_map.transform(lon, lat) for lon in lon_seq])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            ls="--",
            lw=0.55,
            color="0.45",
            alpha=0.28,
            zorder=3,
        )

        _, yl = to_map.transform(min_lon, lat)
        _, yr = to_map.transform(max_lon, lat)
        label = dms_label(lat, "lat")
        if ymin < yl < ymax:
            ax.plot([xmin, xmin + 0.010 * width], [yl, yl], color="0.15", lw=0.8, zorder=35)
            ax.text(
                xmin - 0.012 * width,
                yl,
                label,
                ha="right",
                va="center",
                rotation=90,
                fontsize=10.5,
                color="0.15",
                clip_on=False,
                zorder=35,
            )
        if ymin < yr < ymax:
            ax.plot([xmax, xmax - 0.010 * width], [yr, yr], color="0.15", lw=0.8, zorder=35)
            ax.text(
                xmax + 0.012 * width,
                yr,
                label,
                ha="left",
                va="center",
                rotation=-90,
                fontsize=10.5,
                color="0.15",
                clip_on=False,
                zorder=35,
            )


def add_scalebar(ax, length_km=15, segments=3):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width = xmax - xmin
    height = ymax - ymin

    total = length_km * 1000.0
    seg = total / segments
    x0 = xmax - 0.065 * width - total
    y0 = ymin + 0.060 * height
    bar_h = 0.012 * height

    for i in range(segments):
        ax.add_patch(
            Rectangle(
                (x0 + i * seg, y0),
                seg,
                bar_h,
                facecolor="black" if i % 2 == 0 else "white",
                edgecolor="black",
                linewidth=0.8,
                zorder=45,
            )
        )
    for i in range(segments + 1):
        x = x0 + i * seg
        ax.plot([x, x], [y0, y0 + bar_h], color="black", lw=0.8, zorder=46)
        label = f"{int(i * length_km / segments)}"
        if i == segments:
            label += " km"
        ax.text(
            x,
            y0 + 1.6 * bar_h,
            label,
            ha="center",
            va="bottom",
            fontsize=10.2,
            zorder=46,
        )


def add_north_arrow(ax):
    """North arrow matching the approved black spearhead visual style."""
    x = 0.945
    y_top = 0.925
    y_wing = 0.842
    y_notch = 0.862
    half_w = 0.027

    arrow = Polygon(
        [
            (x, y_top),
            (x - half_w, y_wing),
            (x, y_notch),
            (x + half_w, y_wing),
        ],
        closed=True,
        facecolor="black",
        edgecolor="black",
        linewidth=0.8,
        transform=ax.transAxes,
        zorder=60,
        clip_on=False,
    )
    ax.add_patch(arrow)
    ax.text(
        x,
        y_top + 0.018,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14.5,
        fontweight="bold",
        color="black",
        zorder=61,
    )


def load_south_america():
    url = (
        "https://raw.githubusercontent.com/nvkelso/"
        "natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    )
    world = gpd.read_file(url)
    continent_col = "CONTINENT" if "CONTINENT" in world.columns else "continent"
    south = world[
        world[continent_col].astype(str).str.lower().eq("south america")
    ].copy()
    if south.empty:
        raise RuntimeError("Natural Earth South America subset is empty.")
    return south.to_crs(WGS84)


SOUTH_AMERICA = load_south_america()


def style_box(ax, linewidth=0.9):
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color("0.25")


def draw_location_panel(ax):
    style_box(ax)
    ax.text(
        0.5,
        0.965,
        "LOCATION",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    map_ax = ax.inset_axes([0.055, 0.245, 0.89, 0.655])
    SOUTH_AMERICA.plot(
        ax=map_ax,
        facecolor="#ededed",
        edgecolor="#777777",
        linewidth=0.65,
        zorder=1,
    )
    map_ax.set_xlim(-82.5, -33.0)
    map_ax.set_ylim(-56.0, 13.0)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.scatter(
        [CITY_LON],
        [CITY_LAT],
        marker="o",
        s=52,
        facecolors="none",
        edgecolors="#d44735",
        linewidths=1.8,
        zorder=5,
    )
    map_ax.scatter(
        [CITY_LON],
        [CITY_LAT],
        marker="+",
        s=125,
        color="#d44735",
        linewidths=1.7,
        zorder=6,
    )
    map_ax.text(
        CITY_LON + 1.6,
        CITY_LAT - 0.7,
        "Petrópolis",
        color="#d44735",
        fontsize=11.2,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=7,
    )
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    for spine in map_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.35")

    y0 = 0.205
    for i, line in enumerate(CRS_LINES):
        ax.text(
            0.055,
            y0 - i * 0.058,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="0.10",
        )


def draw_sources_panel(ax):
    style_box(ax)
    ax.text(
        0.055,
        0.92,
        "DATA SOURCES",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="bold",
    )
    y = 0.75
    for line in DATA_SOURCE_LINES:
        ax.text(
            0.065,
            y,
            f"• {line}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
        )
        y -= 0.13

    ax.text(
        0.055,
        0.34,
        "BASEMAP",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.0,
        fontweight="bold",
    )
    ax.text(
        0.065,
        0.20,
        BASEMAP_LINES[0],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
    )
    ax.text(
        0.065,
        0.08,
        BASEMAP_LINES[1],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
    )


def draw_typology_legend(ax):
    style_box(ax)
    ax.text(
        0.055,
        0.94,
        "LEGEND",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="bold",
    )

    # Municipal boundary symbol.
    ax.add_patch(
        Rectangle(
            (0.060, 0.815),
            0.105,
            0.085,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=1.8,
        )
    )
    ax.text(
        0.20,
        0.858,
        "Petrópolis municipal boundary",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
    )

    ax.text(
        0.055,
        0.70,
        "Final typology (OSM mapping regime)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.3,
        fontweight="bold",
    )

    cmap = mpl.colormaps["viridis"]
    colors = [cmap(0.08), cmap(0.52), cmap(0.90)]
    labels = [
        "No currently mapped OSM road lines\n(114 cells — 12.74%)",
        "Lower-intensity, longer-plateau\nmapping regime\n(718 cells — 80.22%)",
        "Higher-intensity, expansion-pulse\nmapping regime\n(63 cells — 7.04%)",
    ]
    ys = [0.555, 0.345, 0.135]
    for color, label, y in zip(colors, labels, ys):
        ax.add_patch(
            Rectangle(
                (0.070, y - 0.040),
                0.105,
                0.095,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            0.205,
            y + 0.006,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10.0,
            linespacing=1.15,
        )


def make_continuous_legend(column, label, log=False):
    s = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        raise ValueError(f"No valid values for {column}")

    if log:
        positive = valid[valid > 0]
        vmin = max(float(positive.quantile(0.02)), 1e-6)
        vmax = float(positive.quantile(0.98))
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
            0.055,
            0.94,
            "LEGEND",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13.5,
            fontweight="bold",
        )
        ax.add_patch(
            Rectangle(
                (0.060, 0.815),
                0.105,
                0.085,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="black",
                linewidth=1.8,
            )
        )
        ax.text(
            0.20,
            0.858,
            "Petrópolis municipal boundary",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10.5,
        )
        ax.text(
            0.055,
            0.70,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11.2,
            fontweight="bold",
        )

        cax = ax.inset_axes([0.095, 0.18, 0.10, 0.43])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax, orientation="vertical")
        cb.ax.tick_params(labelsize=9.5, length=3)
        if not log:
            cb.locator = MaxNLocator(nbins=5)
            cb.update_ticks()
        cb.outline.set_linewidth(0.7)

        if s.isna().any() or (log and (s <= 0).any()):
            ax.add_patch(
                Rectangle(
                    (0.33, 0.25),
                    0.095,
                    0.08,
                    transform=ax.transAxes,
                    facecolor="0.88",
                    edgecolor="0.55",
                    linewidth=0.7,
                )
            )
            ax.text(
                0.46,
                0.29,
                "No data / not defined",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=9.8,
            )

        if log:
            ax.text(
                0.33,
                0.47,
                "Logarithmic\ncolour scale",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=9.8,
                linespacing=1.2,
            )

    return norm, drawer


def base_figure(title, note, legend_drawer):
    """Publication layout modelled on the approved reference composition."""
    fig = plt.figure(figsize=(16.0, 10.0), facecolor="white")
    fig.suptitle(title, x=0.5, y=0.975, fontsize=18, fontweight="bold", ha="center")

    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        left=0.022,
        right=0.985,
        bottom=0.040,
        top=0.930,
        width_ratios=[0.255, 0.745],
        wspace=0.012,
    )
    left = outer[0].subgridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.40, 0.34, 0.26],
        hspace=0.020,
    )
    right = outer[1].subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=[0.875, 0.125],
        hspace=0.020,
    )

    location_ax = fig.add_subplot(left[0])
    legend_ax = fig.add_subplot(left[1])
    sources_ax = fig.add_subplot(left[2])
    map_ax = fig.add_subplot(right[0])
    note_ax = fig.add_subplot(right[1])

    draw_location_panel(location_ax)
    legend_drawer(legend_ax)
    draw_sources_panel(sources_ax)

    # Main map frame: its width is exactly the same as the note box below.
    style_box(map_ax, linewidth=0.9)
    set_equal_scale_extent(fig, map_ax, muni.total_bounds, padding=0.050)

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
    add_scalebar(map_ax, length_km=15, segments=3)
    add_north_arrow(map_ax)

    # Interpretation box.
    style_box(note_ax, linewidth=0.9)
    wrapped = textwrap.fill(note, width=118)
    note_ax.text(
        0.5,
        0.50,
        wrapped,
        transform=note_ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.8,
        linespacing=1.30,
        color="0.08",
    )

    return fig, map_ax


def save(fig, stem):
    # 600 dpi requested for publication-ready raster outputs.  Do not use
    # bbox_inches='tight': retaining the designed canvas prevents clipping and
    # keeps every map at an identical layout/size.
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Maps
# -----------------------------------------------------------------------------
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
        subset = g_display[g_display["typology_class_id"].eq(cid)]
        subset.plot(
            ax=ax,
            facecolor=color,
            edgecolor="#565656",
            linewidth=0.28,
            alpha=0.92,
            zorder=10,
        )
    muni.boundary.plot(ax=ax, color="black", linewidth=2.0, zorder=30)
    save(fig, "figure_01_final_typology")


def plot_continuous(column, title, legend_label, stem, note, log=False):
    if column not in g_display.columns:
        raise KeyError(column)

    s = pd.to_numeric(g_display[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        raise ValueError(f"No valid values for {column}")

    norm, legend_drawer = make_continuous_legend(column, legend_label, log=log)
    fig, ax = base_figure(title, note, legend_drawer)

    if log:
        positive_mask = s > 0
        g_display.loc[~positive_mask].plot(
            ax=ax,
            facecolor="0.88",
            edgecolor="#666666",
            linewidth=0.22,
            alpha=0.92,
            zorder=9,
        )
        g_display.loc[positive_mask].plot(
            ax=ax,
            column=column,
            cmap="viridis",
            norm=norm,
            edgecolor="#666666",
            linewidth=0.22,
            alpha=0.90,
            zorder=10,
        )
    else:
        g_display.plot(
            ax=ax,
            column=column,
            cmap="viridis",
            norm=norm,
            edgecolor="#666666",
            linewidth=0.22,
            alpha=0.90,
            zorder=10,
            missing_kwds={
                "color": "0.88",
                "edgecolor": "#777777",
                "linewidth": 0.22,
            },
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

# -----------------------------------------------------------------------------
# Manifest and diagnostics
# -----------------------------------------------------------------------------
manifest = []
for path in sorted(OUT.glob("figure_*.*")):
    manifest.append({"file": path.name, "bytes": path.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT / "figure_manifest.csv", index=False)

print("Municipality source:", MUNICIPALITY_FILE)
print("Municipality source CRS:", municipality.crs)
print("Map CRS:", MAP_CRS)
print("Municipality centroid in EPSG:4326 (lon, lat):", (CITY_LON, CITY_LAT))
print("South America countries loaded:", len(SOUTH_AMERICA))
print("Display cells after clipping:", len(g_display))
print(pd.DataFrame(manifest).to_string(index=False))
print("Article map production complete:", OUT)
