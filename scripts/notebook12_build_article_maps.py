from pathlib import Path
import math
import textwrap

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Patch, Rectangle
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

for path in [GRID_FILE, MUNICIPALITY_FILE]:
    if not path.exists():
        raise FileNotFoundError(path)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
gdf = gpd.read_parquet(GRID_FILE)
municipality_src = gpd.read_parquet(MUNICIPALITY_FILE)

assert len(gdf) == 895, f"Expected 895 grid cells, found {len(gdf)}"
assert gdf["grid_1km_id"].astype(str).is_unique
assert municipality_src.crs is not None

# The municipal source can contain one or more features; dissolve guarantees a
# single authoritative outline without approximating it from the grid.
municipality_src = municipality_src[["geometry"]].copy()
municipality_src["_municipality"] = 1
municipality_src = municipality_src.dissolve(by="_municipality", as_index=False)
assert len(municipality_src) == 1
assert municipality_src.geometry.notna().all()
assert municipality_src.geometry.is_valid.all()

MAP_CRS = "EPSG:31983"  # SIRGAS 2000 / UTM zone 23S
WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"

g = gdf.to_crs(MAP_CRS)
municipality = municipality_src.to_crs(MAP_CRS)
municipality_wgs84 = municipality_src.to_crs(WGS84)

# Real municipality centroid used only for the location inset.
location_point = municipality.to_crs(WEB_MERCATOR).geometry.iloc[0].centroid

TITLE_SUFFIX = "PETRÓPOLIS (RJ), BRAZIL"
BOUNDARY_LABEL = "Petrópolis municipal boundary"
MAP_CRS_LABEL = "SIRGAS 2000 / UTM zone 23S (EPSG:31983)"
GRATICULE_LABEL = "Geographic longitude/latitude — SIRGAS 2000"
BASEMAP_LABEL = "CARTO Positron (© OpenStreetMap contributors, © CARTO)"

# -----------------------------------------------------------------------------
# General cartographic helpers
# -----------------------------------------------------------------------------
def nice_step(span, target=5):
    if span <= 0:
        return 1.0
    raw = span / target
    p = 10 ** math.floor(math.log10(raw))
    for m in [1, 2, 2.5, 5, 10]:
        if m * p >= raw:
            return m * p
    return 10 * p


def degree_minute(value, axis):
    hemisphere = (
        "W" if axis == "lon" and value < 0
        else "E" if axis == "lon"
        else "S" if value < 0
        else "N"
    )
    av = abs(value)
    deg = int(math.floor(av))
    minute = int(round((av - deg) * 60))
    if minute == 60:
        deg += 1
        minute = 0
    return f"{deg}°{minute:02d}′{hemisphere}"


def add_scalebar(ax, length_km=None):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width = xmax - xmin
    height = ymax - ymin

    if length_km is None:
        desired = width / 4.5 / 1000
        candidates = np.array([1, 2, 5, 10, 15, 20, 25, 50], dtype=float)
        length_km = float(candidates[np.argmin(np.abs(candidates - desired))])

    # Alternating two-segment graphic scale, placed at lower right.
    total_m = length_km * 1000
    segment = total_m / 2
    x0 = xmax - 0.07 * width - total_m
    y0 = ymin + 0.055 * height
    bar_h = 0.012 * height

    for i, fc in enumerate(["black", "white"]):
        rect = Rectangle(
            (x0 + i * segment, y0), segment, bar_h,
            facecolor=fc, edgecolor="black", linewidth=0.8, zorder=40,
        )
        ax.add_patch(rect)

    for frac, label in [(0, "0"), (0.5, f"{length_km/2:g}"), (1, f"{length_km:g} km")]:
        ax.text(
            x0 + frac * total_m,
            y0 + 1.8 * bar_h,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="black",
            zorder=41,
        )


def add_north(ax):
    ax.annotate(
        "N",
        xy=(0.945, 0.89),
        xytext=(0.945, 0.965),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(
            facecolor="black",
            edgecolor="black",
            width=2.5,
            headwidth=10,
            headlength=12,
        ),
        zorder=50,
    )


def add_geographic_graticule(ax):
    to_ll = Transformer.from_crs(MAP_CRS, WGS84, always_xy=True)
    to_map = Transformer.from_crs(WGS84, MAP_CRS, always_xy=True)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    corners = [
        to_ll.transform(x, y)
        for x, y in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    ]
    lons = [p[0] for p in corners]
    lats = [p[1] for p in corners]

    lon_step = nice_step(max(lons) - min(lons), 5)
    lat_step = nice_step(max(lats) - min(lats), 5)
    lon_vals = np.arange(
        math.floor(min(lons) / lon_step) * lon_step,
        max(lons) + lon_step,
        lon_step,
    )
    lat_vals = np.arange(
        math.floor(min(lats) / lat_step) * lat_step,
        max(lats) + lat_step,
        lat_step,
    )

    for lon in lon_vals:
        lat_seq = np.linspace(min(lats) - 0.05, max(lats) + 0.05, 120)
        xy = [to_map.transform(lon, lat) for lat in lat_seq]
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        ax.plot(xs, ys, ls="--", lw=0.45, color="0.35", alpha=0.28, zorder=2)
        x_bottom, _ = to_map.transform(lon, min(lats))
        x_top, _ = to_map.transform(lon, max(lats))
        if xmin < x_bottom < xmax:
            label = degree_minute(lon, "lon")
            ax.text(
                x_bottom,
                ymin + 0.006 * (ymax - ymin),
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.25",
                zorder=45,
            )
            ax.text(
                x_top,
                ymax - 0.006 * (ymax - ymin),
                label,
                ha="center",
                va="top",
                fontsize=7,
                color="0.25",
                zorder=45,
            )

    for lat in lat_vals:
        lon_seq = np.linspace(min(lons) - 0.05, max(lons) + 0.05, 120)
        xy = [to_map.transform(lon, lat) for lon in lon_seq]
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        ax.plot(xs, ys, ls="--", lw=0.45, color="0.35", alpha=0.28, zorder=2)
        _, y_left = to_map.transform(min(lons), lat)
        _, y_right = to_map.transform(max(lons), lat)
        if ymin < y_left < ymax:
            label = degree_minute(lat, "lat")
            ax.text(
                xmin + 0.006 * (xmax - xmin),
                y_left,
                label,
                ha="left",
                va="center",
                rotation=90,
                fontsize=7,
                color="0.25",
                zorder=45,
            )
            ax.text(
                xmax - 0.006 * (xmax - xmin),
                y_right,
                label,
                ha="right",
                va="center",
                rotation=-90,
                fontsize=7,
                color="0.25",
                zorder=45,
            )


def style_panel(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("0.25")


def add_location_inset(left_ax):
    loc_ax = left_ax.inset_axes([0.055, 0.675, 0.89, 0.265])
    loc_ax.set_title("LOCATION", fontsize=8.5, fontweight="bold", pad=4)

    # South-America view in Web Mercator. CARTO Positron supplies the contextual
    # country/state outlines; the point itself comes from the real municipality.
    to_web = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True)
    x0, y0 = to_web.transform(-83.5, -57.5)
    x1, y1 = to_web.transform(-33.0, 14.5)
    loc_ax.set_xlim(x0, x1)
    loc_ax.set_ylim(y0, y1)
    try:
        cx.add_basemap(
            loc_ax,
            source=cx.providers.CartoDB.Positron,
            crs=WEB_MERCATOR,
            attribution=False,
            zoom=3,
        )
    except Exception as exc:
        print("WARNING: location basemap unavailable:", repr(exc))

    loc_ax.scatter(
        [location_point.x],
        [location_point.y],
        marker="o",
        s=32,
        facecolors="none",
        edgecolors="#d73027",
        linewidths=1.2,
        zorder=30,
    )
    loc_ax.scatter(
        [location_point.x],
        [location_point.y],
        marker="+",
        s=60,
        color="#d73027",
        linewidths=1.2,
        zorder=31,
    )
    loc_ax.annotate(
        "Petrópolis",
        xy=(location_point.x, location_point.y),
        xytext=(7, -5),
        textcoords="offset points",
        color="#d73027",
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=31,
    )
    loc_ax.set_axis_off()
    for spine in loc_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_edgecolor("0.4")


def section_title(ax, y, text):
    ax.text(
        0.05,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8.3,
        fontweight="bold",
        ha="left",
        va="top",
        color="0.1",
    )


def add_common_left_metadata(left_ax):
    left_ax.text(
        0.055,
        0.66,
        "Map CRS: SIRGAS 2000 / UTM zone 23S (EPSG:31983)\n"
        "Graticule: geographic longitude/latitude\n"
        "Datum: SIRGAS 2000",
        transform=left_ax.transAxes,
        fontsize=7.2,
        ha="left",
        va="top",
        linespacing=1.35,
    )
    left_ax.plot([0, 1], [0.61, 0.61], transform=left_ax.transAxes, color="0.35", lw=0.7)


def add_sources(left_ax):
    left_ax.plot([0, 1], [0.205, 0.205], transform=left_ax.transAxes, color="0.35", lw=0.7)
    section_title(left_ax, 0.195, "DATA SOURCES")
    sources = (
        "• OpenStreetMap road history via ohsome API\n"
        "• 1 km reference grid derived from IBGE source grid\n"
        "• Municipal boundary: IBGE (2025)"
    )
    left_ax.text(
        0.055,
        0.17,
        sources,
        transform=left_ax.transAxes,
        fontsize=6.8,
        ha="left",
        va="top",
        linespacing=1.35,
    )
    section_title(left_ax, 0.085, "BASEMAP")
    left_ax.text(
        0.055,
        0.06,
        "CARTO Positron\n© OpenStreetMap contributors, © CARTO",
        transform=left_ax.transAxes,
        fontsize=6.8,
        ha="left",
        va="top",
        linespacing=1.3,
    )


def make_layout(title, note):
    fig = plt.figure(figsize=(12.8, 9.2), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[0.255, 0.745],
        height_ratios=[0.885, 0.115],
        left=0.018,
        right=0.988,
        top=0.94,
        bottom=0.025,
        wspace=0.025,
        hspace=0.025,
    )

    left_ax = fig.add_subplot(gs[:, 0])
    map_ax = fig.add_subplot(gs[0, 1])
    note_ax = fig.add_subplot(gs[1, 1])

    style_panel(left_ax)
    style_panel(map_ax)
    style_panel(note_ax)

    fig.suptitle(
        title,
        x=0.5,
        y=0.977,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )

    add_location_inset(left_ax)
    add_common_left_metadata(left_ax)

    # Use the complete 1 km cell extent so original IBGE cell geometries are
    # preserved, including edge cells that intersect the municipality.
    xmin, ymin, xmax, ymax = g.total_bounds
    dx = (xmax - xmin) * 0.045
    dy = (ymax - ymin) * 0.045
    map_ax.set_xlim(xmin - dx, xmax + dx)
    map_ax.set_ylim(ymin - dy, ymax + dy)
    map_ax.set_aspect("equal")

    try:
        cx.add_basemap(
            map_ax,
            source=cx.providers.CartoDB.Positron,
            crs=MAP_CRS,
            attribution=False,
            zoom="auto",
        )
    except Exception as exc:
        print("WARNING: Positron unavailable; continuing without basemap:", repr(exc))

    add_geographic_graticule(map_ax)
    add_scalebar(map_ax)
    add_north(map_ax)
    map_ax.set_xticks([])
    map_ax.set_yticks([])

    note_ax.text(
        0.5,
        0.5,
        "\n".join(textwrap.wrap(note, width=135)),
        transform=note_ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        linespacing=1.35,
    )

    return fig, left_ax, map_ax, note_ax


def draw_boundary(map_ax):
    municipality.boundary.plot(
        ax=map_ax,
        color="black",
        linewidth=1.65,
        zorder=35,
    )


def add_boundary_legend(left_ax, y=0.575):
    section_title(left_ax, y, "LEGEND")
    patch = Rectangle(
        (0.06, y - 0.054),
        0.09,
        0.024,
        transform=left_ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        clip_on=False,
    )
    left_ax.add_patch(patch)
    left_ax.text(
        0.175,
        y - 0.042,
        BOUNDARY_LABEL,
        transform=left_ax.transAxes,
        fontsize=7.2,
        ha="left",
        va="center",
    )


def add_continuous_legend(left_ax, label, cmap, norm, missing_label="Not defined"):
    add_boundary_legend(left_ax)
    left_ax.text(
        0.055,
        0.49,
        label,
        transform=left_ax.transAxes,
        fontsize=7.7,
        fontweight="bold",
        ha="left",
        va="top",
    )
    cax = left_ax.inset_axes([0.075, 0.425, 0.83, 0.025])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.ax.tick_params(labelsize=6.4, length=2.5, pad=1.5)
    cb.outline.set_linewidth(0.6)

    missing_patch = Rectangle(
        (0.075, 0.365),
        0.055,
        0.021,
        transform=left_ax.transAxes,
        facecolor="0.88",
        edgecolor="0.65",
        linewidth=0.5,
    )
    left_ax.add_patch(missing_patch)
    left_ax.text(
        0.15,
        0.375,
        missing_label,
        transform=left_ax.transAxes,
        fontsize=6.8,
        ha="left",
        va="center",
    )
    add_sources(left_ax)


def add_typology_legend(left_ax, colors, labels, counts):
    add_boundary_legend(left_ax)
    left_ax.text(
        0.055,
        0.49,
        "Final typology (OSM mapping regime)",
        transform=left_ax.transAxes,
        fontsize=7.7,
        fontweight="bold",
        ha="left",
        va="top",
    )
    y = 0.455
    total = len(g)
    for cid in [0, 1, 2]:
        patch = Rectangle(
            (0.065, y - 0.016),
            0.065,
            0.026,
            transform=left_ax.transAxes,
            facecolor=colors[cid],
            edgecolor="white",
            linewidth=0.4,
        )
        left_ax.add_patch(patch)
        pct = 100 * counts[cid] / total
        txt = f"{labels[cid]}\n({counts[cid]} cells — {pct:.2f}%)"
        left_ax.text(
            0.15,
            y,
            txt,
            transform=left_ax.transAxes,
            fontsize=6.65,
            ha="left",
            va="center",
            linespacing=1.2,
        )
        y -= 0.075
    add_sources(left_ax)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Map producers
# -----------------------------------------------------------------------------
def plot_typology():
    title = f"FINAL OSM MAPPING MATURITY TYPOLOGY — {TITLE_SUFFIX}"
    note = (
        "This map presents the final classification derived from intrinsic OpenStreetMap road-mapping histories, "
        "combining road-network growth, plateau/maturity, persistence and contribution dynamics. The classes do not "
        "represent external completeness, positional accuracy or overall data-quality categories."
    )
    fig, left_ax, map_ax, _ = make_layout(title, note)

    labels = {
        0: "No currently mapped OSM road lines",
        1: "Lower-intensity, longer-plateau mapping regime",
        2: "Higher-intensity, expansion-pulse mapping regime",
    }
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(0.08), cmap(0.52), cmap(0.92)]
    counts = g["typology_class_id"].value_counts().reindex([0, 1, 2], fill_value=0).to_dict()

    for cid in [0, 1, 2]:
        g[g["typology_class_id"].eq(cid)].plot(
            ax=map_ax,
            facecolor=colors[cid],
            edgecolor="0.45",
            linewidth=0.20,
            alpha=0.90,
            zorder=10,
        )

    draw_boundary(map_ax)
    add_typology_legend(left_ax, colors, labels, counts)
    save(fig, "figure_01_final_typology")


def plot_continuous(column, title, legend_label, stem, log=False, note=""):
    if column not in g.columns:
        raise KeyError(f"Missing required map variable: {column}")

    s = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        raise ValueError(f"No valid values for {column}")

    cmap = mpl.colormaps["viridis"]

    if log:
        positive = valid[valid > 0]
        if positive.empty:
            raise ValueError(f"No positive values for logarithmic map {column}")
        vmin = max(float(positive.quantile(0.02)), 1e-6)
        vmax = float(positive.quantile(0.98))
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01), clip=True)
    else:
        vmin = float(valid.quantile(0.02))
        vmax = float(valid.quantile(0.98))
        if math.isclose(vmin, vmax):
            vmin = float(valid.min())
            vmax = float(valid.max())
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    fig, left_ax, map_ax, _ = make_layout(title, note)

    missing = s.isna()
    if missing.any():
        g.loc[missing].plot(
            ax=map_ax,
            facecolor="0.88",
            edgecolor="0.62",
            linewidth=0.18,
            alpha=0.95,
            zorder=9,
        )

    if log:
        zero_or_negative = s.notna() & s.le(0)
        if zero_or_negative.any():
            g.loc[zero_or_negative].plot(
                ax=map_ax,
                facecolor="0.88",
                edgecolor="0.62",
                linewidth=0.18,
                alpha=0.95,
                zorder=9,
            )
        mapped = s.notna() & s.gt(0)
    else:
        mapped = s.notna()

    g.loc[mapped].plot(
        ax=map_ax,
        column=column,
        cmap=cmap,
        norm=norm,
        edgecolor="0.45",
        linewidth=0.20,
        alpha=0.90,
        zorder=10,
    )

    # The authoritative IBGE municipal perimeter is always the uppermost
    # thematic feature, ensuring it remains visible on every map.
    draw_boundary(map_ax)

    missing_label = "Not defined"
    if log:
        missing_label = "0 / not defined"
    add_continuous_legend(left_ax, legend_label, cmap, norm, missing_label)
    save(fig, stem)


plot_typology()

plot_continuous(
    "current_road_length_km",
    f"CURRENT MAPPED ROAD LENGTH — {TITLE_SUFFIX}",
    "Current mapped road length (km)",
    "figure_02_current_road_length",
    note=(
        "Current OpenStreetMap road length aggregated by the original 1 km IBGE analytical cells. "
        "The black line is the actual Petrópolis municipal boundary used in the study. Display colors use the "
        "2nd–98th percentile range; underlying values are unchanged."
    ),
)

plot_continuous(
    "mapping_age_months",
    f"MAPPING AGE — {TITLE_SUFFIX}",
    "Mapping age (months)",
    "figure_03_mapping_age",
    note=(
        "Mapping age describes the duration of mapped-road presence in each analytical cell over the monthly OSM "
        "history analysed. Display colors use the 2nd–98th percentile range; underlying values are unchanged."
    ),
)

plot_continuous(
    "active_month_ratio",
    f"ACTIVE-MONTH RATIO — {TITLE_SUFFIX}",
    "Active-month ratio",
    "figure_04_active_month_ratio",
    note=(
        "The active-month ratio is the share of monthly observations with recorded OpenStreetMap road-mapping "
        "activity in each 1 km cell. Display colors use the 2nd–98th percentile range."
    ),
)

plot_continuous(
    "total_contributions",
    f"TOTAL OSM CONTRIBUTIONS — {TITLE_SUFFIX}",
    "Total OSM contributions (log scale)",
    "figure_05_total_contributions",
    log=True,
    note=(
        "Cumulative OpenStreetMap contribution counts are aggregated by 1 km analytical cell. A logarithmic color "
        "normalization is used because the distribution is strongly right-skewed; positive display limits use the "
        "2nd–98th percentiles."
    ),
)

plot_continuous(
    "contribution_temporal_concentration_hhi",
    f"CONTRIBUTION TEMPORAL CONCENTRATION (HHI) — {TITLE_SUFFIX}",
    "Temporal concentration (HHI)",
    "figure_06_temporal_concentration",
    note=(
        "The Herfindahl–Hirschman Index (HHI) summarizes temporal concentration of monthly contributions. Higher "
        "values indicate that activity is concentrated in fewer months. Display colors use the 2nd–98th percentile range."
    ),
)

plot_continuous(
    "time_to_90pct_months",
    f"TIME TO 90% OF INTRINSIC RECENT REFERENCE — {TITLE_SUFFIX}",
    "Time to 90% (months)",
    "figure_07_time_to_90pct",
    note=(
        "Time required for the mapped-road trajectory to reach 90% of its intrinsic recent reference level. The "
        "reference is internal to each OSM trajectory and must not be interpreted as external map completeness."
    ),
)

plot_continuous(
    "plateau_duration_months",
    f"INTRINSIC PLATEAU DURATION — {TITLE_SUFFIX}",
    "Intrinsic plateau duration (months)",
    "figure_08_plateau_duration",
    note=(
        "Duration of the intrinsic plateau detected in the mapped-road trajectory of each cell. Plateau is a "
        "descriptive property of OSM history and does not imply external completeness or positional accuracy."
    ),
)

# -----------------------------------------------------------------------------
# Output inventory
# -----------------------------------------------------------------------------
manifest = []
for p in sorted(OUT.glob("figure_*.*")):
    manifest.append({"file": p.name, "bytes": p.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT / "figure_manifest.csv", index=False)

print(pd.DataFrame(manifest).to_string(index=False))
print("Article map production complete:", OUT)
print("Municipality boundary source:", MUNICIPALITY_FILE)
print("Municipality source CRS:", municipality_src.crs)
print("Map CRS:", MAP_CRS)
