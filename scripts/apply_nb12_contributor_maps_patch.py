from pathlib import Path

path = Path("scripts/notebook12_build_article_maps.py")
text = path.read_text(encoding="utf-8")

# 1. Optional contributor-metrics input.
needle = 'MUNICIPALITY_FILE = ROOT / "Petropolis" / "Petropolis_2025_4674_IBGE.parquet"\n'
insert = (
    needle
    + 'CONTRIBUTOR_FILE = ROOT / "data" / "output" / "notebooks" / "14" / "petropolis_unique_contributors_895_cells.parquet"\n'
)
if 'CONTRIBUTOR_FILE = ROOT / "data" / "output" / "notebooks" / "14"' not in text:
    if needle not in text:
        raise RuntimeError("Could not locate MUNICIPALITY_FILE definition")
    text = text.replace(needle, insert, 1)

# 2. Merge contributor metrics when Notebook 14 output exists.
needle = 'gdf = gpd.read_parquet(GRID_FILE)\nmunicipality = gpd.read_parquet(MUNICIPALITY_FILE)\n'
replacement = '''gdf = gpd.read_parquet(GRID_FILE)
municipality = gpd.read_parquet(MUNICIPALITY_FILE)
if CONTRIBUTOR_FILE.exists():
    contributor_metrics = pd.read_parquet(CONTRIBUTOR_FILE)
    contributor_metrics["grid_1km_id"] = contributor_metrics["grid_1km_id"].astype(str)
    required_contributor_cols = {
        "grid_1km_id", "all_unique_contributors_count", "road_unique_contributors_count"
    }
    missing_contributor_cols = required_contributor_cols - set(contributor_metrics.columns)
    if missing_contributor_cols:
        raise KeyError(f"Contributor metrics missing columns: {sorted(missing_contributor_cols)}")
    if len(contributor_metrics) != 895 or not contributor_metrics["grid_1km_id"].is_unique:
        raise ValueError("Notebook 14 contributor metrics must contain 895 unique grid IDs.")
    gdf["grid_1km_id"] = gdf["grid_1km_id"].astype(str)
    gdf = gdf.merge(
        contributor_metrics[[
            "grid_1km_id", "all_unique_contributors_count", "road_unique_contributors_count"
        ]],
        on="grid_1km_id", how="left", validate="one_to_one",
    )
    if gdf[["all_unique_contributors_count", "road_unique_contributors_count"]].isna().any().any():
        raise ValueError("Contributor metrics merge produced missing values.")
    print("Loaded contributor metrics:", CONTRIBUTOR_FILE)
else:
    print("Contributor metrics not present; Figures 09–10 will be skipped.")
'''
if 'Loaded contributor metrics:' not in text:
    if needle not in text:
        raise RuntimeError("Could not locate grid/municipality read block")
    text = text.replace(needle, replacement, 1)

# 3. Allow only the all-OSM contributor map to use a more general source label.
text = text.replace('def draw_sources_panel(ax):\n', 'def draw_sources_panel(ax, first_line=None):\n', 1)
needle = '    style_box(ax)\n    ax.text(\n        0.055, 0.92, "DATA SOURCES", transform=ax.transAxes,\n'
replacement = '    style_box(ax)\n    source_line = DATA_SOURCE_LINES[0] if first_line is None else first_line\n    ax.text(\n        0.055, 0.92, "DATA SOURCES", transform=ax.transAxes,\n'
if 'source_line = DATA_SOURCE_LINES[0] if first_line is None else first_line' not in text:
    if needle not in text:
        raise RuntimeError("Could not locate draw_sources_panel heading block")
    text = text.replace(needle, replacement, 1)
text = text.replace('0.065, 0.78, f"• {DATA_SOURCE_LINES[0]}",', '0.065, 0.78, f"• {source_line}",', 1)

# 4. Pass optional source text through base_figure and plot_continuous.
text = text.replace(
    'def base_figure(title, note, legend_drawer):\n',
    'def base_figure(title, note, legend_drawer, source_first_line=None):\n',
    1,
)
text = text.replace(
    '    draw_sources_panel(sources_ax)\n',
    '    draw_sources_panel(sources_ax, source_first_line)\n',
    1,
)
text = text.replace(
    'def plot_continuous(column, title, legend_label, stem, note, log=False):\n',
    'def plot_continuous(column, title, legend_label, stem, note, log=False, source_first_line=None):\n',
    1,
)
text = text.replace(
    '    fig, ax = base_figure(title, note, drawer)\n',
    '    fig, ax = base_figure(title, note, drawer, source_first_line)\n',
    1,
)

# 5. Add Figures 09 and 10, only after Notebook 14 data exists.
marker = '''plot_continuous(
    "plateau_duration_months",
    "INTRINSIC PLATEAU DURATION — PETRÓPOLIS (RJ), BRAZIL",
    "Intrinsic plateau duration (months)",
    "figure_08_plateau_duration",
    "Duration of the intrinsic plateau phase identified from each cell's OSM road-mapping trajectory. Plateau is a trajectory descriptor and must not be interpreted as external completeness.",
)
'''
addition = marker + '''
if {"all_unique_contributors_count", "road_unique_contributors_count"}.issubset(g_display.columns):
    plot_continuous(
        "all_unique_contributors_count",
        "UNIQUE OSM CONTRIBUTORS — PETRÓPOLIS (RJ), BRAZIL",
        "Unique OSM contributors (users)",
        "figure_09_unique_osm_contributors",
        "Number of unique OpenStreetMap users who contributed to any OSM feature within each 1 km analytical cell over the same frozen historical interval used for the road-contributor analysis. This is an identity-based unique-user count, not a sum of monthly contributor counts.",
        source_first_line="OpenStreetMap contribution history via ohsome API",
    )
    plot_continuous(
        "road_unique_contributors_count",
        "UNIQUE ROAD-MAPPING CONTRIBUTORS — PETRÓPOLIS (RJ), BRAZIL",
        "Unique contributors to highway=* lines",
        "figure_10_unique_road_contributors",
        "Number of unique OpenStreetMap users who contributed to line elements tagged highway=* within each 1 km analytical cell over the frozen Notebook 05 historical interval. The count is the validated road-filtered /users/count metric and is not a contributor-month sum.",
    )
'''
if 'figure_09_unique_osm_contributors' not in text:
    if marker not in text:
        raise RuntimeError("Could not locate Figure 08 plot block")
    text = text.replace(marker, addition, 1)

path.write_text(text, encoding="utf-8")
print("Integrated optional Figures 09–10 into Notebook 12 map generator.")
