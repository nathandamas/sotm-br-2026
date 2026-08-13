from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path.cwd()
while not (ROOT / "notebooks").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

INPUT = ROOT / "data" / "processed" / "Petropolis" / "petropolis_grid_1km_with_osm_typology_final.parquet"
OUTPUT = ROOT / "data" / "processed" / "Petropolis" / "petropolis_figure12_metric.parquet"

gdf = gpd.read_parquet(INPUT)
contributions = pd.to_numeric(gdf["total_contributions"], errors="raise")
road_km = pd.to_numeric(gdf["current_road_length_km"], errors="raise")
metric = np.where(road_km > 0, contributions / road_km, np.nan)
gdf["contributions_per_mapped_road_km"] = metric
gdf.to_parquet(OUTPUT, index=False)
print(OUTPUT)
