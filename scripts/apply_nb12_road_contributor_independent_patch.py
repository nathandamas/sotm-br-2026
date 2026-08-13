from pathlib import Path

path = Path("scripts/notebook12_build_article_maps.py")
text = path.read_text(encoding="utf-8")
original = text

# Add direct Notebook 05 source for the already validated road-user count.
needle = 'CONTRIBUTOR_FILE = ROOT / "data" / "output" / "notebooks" / "14" / "petropolis_unique_contributors_895_cells.parquet"\n'
addition = needle + '''ROAD_CONTRIBUTOR_FILE = ROOT / "data" / "output" / "notebooks" / "05" / "notebook-05-ohsome-production-results" / "production_outputs" / "petropolis_ohsome_895_cells.parquet"
'''
if 'ROAD_CONTRIBUTOR_FILE = ROOT / "data" / "output" / "notebooks" / "05"' not in text:
    if needle not in text:
        raise RuntimeError("CONTRIBUTOR_FILE definition not found")
    text = text.replace(needle, addition, 1)

# Merge road contributors directly from Notebook 05 before optional Notebook 14 merge.
needle = 'gdf = gpd.read_parquet(GRID_FILE)\nmunicipality = gpd.read_parquet(MUNICIPALITY_FILE)\n'
replacement = '''gdf = gpd.read_parquet(GRID_FILE)
municipality = gpd.read_parquet(MUNICIPALITY_FILE)
gdf["grid_1km_id"] = gdf["grid_1km_id"].astype(str)

if ROAD_CONTRIBUTOR_FILE.exists():
    road_contributors = pd.read_parquet(ROAD_CONTRIBUTOR_FILE)
    road_contributors["grid_1km_id"] = road_contributors["grid_1km_id"].astype(str)
    if "contributors_count" not in road_contributors.columns:
        raise KeyError("Notebook 05 road contributor output lacks contributors_count")
    if len(road_contributors) != 895 or not road_contributors["grid_1km_id"].is_unique:
        raise ValueError("Notebook 05 road contributor table must contain 895 unique grid IDs.")
    road_contributors["road_unique_contributors_count"] = pd.to_numeric(
        road_contributors["contributors_count"], errors="raise"
    )
    gdf = gdf.merge(
        road_contributors[["grid_1km_id", "road_unique_contributors_count"]],
        on="grid_1km_id", how="left", validate="one_to_one",
    )
    if gdf["road_unique_contributors_count"].isna().any():
        raise ValueError("Notebook 05 road contributor merge produced missing values.")
    print("Loaded validated road contributor metric:", ROAD_CONTRIBUTOR_FILE)
else:
    print("Notebook 05 road contributor materialization not found; Figure 10 will be skipped.")
'''
if 'Loaded validated road contributor metric:' not in text:
    if needle not in text:
        raise RuntimeError("Grid read block not found")
    text = text.replace(needle, replacement, 1)

# Avoid merging the same road column from Notebook 14; only merge the all-OSM metric there.
old = '''    required_contributor_cols = {
        "grid_1km_id", "all_unique_contributors_count", "road_unique_contributors_count"
    }
'''
new = '''    required_contributor_cols = {
        "grid_1km_id", "all_unique_contributors_count"
    }
'''
if old in text:
    text = text.replace(old, new, 1)

old = '''        contributor_metrics[[
            "grid_1km_id", "all_unique_contributors_count", "road_unique_contributors_count"
        ]],
'''
new = '''        contributor_metrics[[
            "grid_1km_id", "all_unique_contributors_count"
        ]],
'''
if old in text:
    text = text.replace(old, new, 1)

old = '''    if gdf[["all_unique_contributors_count", "road_unique_contributors_count"]].isna().any().any():
        raise ValueError("Contributor metrics merge produced missing values.")
'''
new = '''    if gdf[["all_unique_contributors_count"]].isna().any().any():
        raise ValueError("All-OSM contributor metrics merge produced missing values.")
'''
if old in text:
    text = text.replace(old, new, 1)

# Make Figure 09 and Figure 10 independent.
old = '''if {"all_unique_contributors_count", "road_unique_contributors_count"}.issubset(g_display.columns):
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
new = '''if "all_unique_contributors_count" in g_display.columns:
    plot_continuous(
        "all_unique_contributors_count",
        "UNIQUE OSM CONTRIBUTORS — PETRÓPOLIS (RJ), BRAZIL",
        "Unique OSM contributors (users)",
        "figure_09_unique_osm_contributors",
        "Number of unique OpenStreetMap users who contributed to any OSM feature within each 1 km analytical cell over the same frozen historical interval used for the road-contributor analysis. This is an identity-based unique-user count, not a sum of monthly contributor counts.",
        source_first_line="OpenStreetMap contribution history via ohsome API",
    )

if "road_unique_contributors_count" in g_display.columns:
    plot_continuous(
        "road_unique_contributors_count",
        "UNIQUE ROAD-MAPPING CONTRIBUTORS — PETRÓPOLIS (RJ), BRAZIL",
        "Unique contributors to highway=* lines",
        "figure_10_unique_road_contributors",
        "Number of unique OpenStreetMap users who contributed to line elements tagged highway=* within each 1 km analytical cell over the frozen Notebook 05 historical interval. The count is the validated road-filtered /users/count metric and is not a contributor-month sum.",
    )
'''
if old not in text:
    raise RuntimeError("Combined contributor map block not found")
text = text.replace(old, new, 1)

if text == original:
    raise RuntimeError("No changes applied")
path.write_text(text, encoding="utf-8")
print("Figure 10 is now sourced independently from validated Notebook 05 road contributors.")
