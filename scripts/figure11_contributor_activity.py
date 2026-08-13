from pathlib import Path
import pandas as pd


def find_root(start: Path) -> Path:
    p = start.resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "notebooks").exists() and (candidate / "scripts").exists():
            return candidate
    raise FileNotFoundError("Repository root not found.")


ROOT = find_root(Path.cwd())
BASE_SCRIPT = ROOT / "scripts" / "notebook12_build_article_maps.py"
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(BASE_SCRIPT)

# Reuse the complete, already-approved cartographic setup and functions from
# Figure 01–10 without executing the existing figure-production calls. This
# keeps Figure 11 visually identical while avoiding duplication of the layout.
source = BASE_SCRIPT.read_text(encoding="utf-8")
marker = "\nplot_typology()\n"
if marker not in source:
    raise RuntimeError("Could not locate the article-map execution marker.")
prefix = source.split(marker, 1)[0]
namespace = {"__name__": "figure11_contributor_activity_base"}
exec(compile(prefix, str(BASE_SCRIPT), "exec"), namespace)

g_display = namespace["g_display"]
g = namespace["g"]
plot_continuous = namespace["plot_continuous"]
OUT = namespace["OUT"]

metric = "total_contributor_months"
if metric not in g_display.columns:
    raise KeyError(
        "The final typology GeoParquet does not contain total_contributor_months. "
        "Figure 11 must use the already-materialized monthly contributor series."
    )

values = pd.to_numeric(g[metric], errors="coerce")
if values.isna().any():
    raise ValueError("total_contributor_months contains unexpected missing values.")
if (values < 0).any():
    raise ValueError("total_contributor_months contains negative values.")

plot_continuous(
    metric,
    "CUMULATIVE CONTRIBUTOR ACTIVITY — PETRÓPOLIS (RJ), BRAZIL",
    "Contributor activity (contributor-months)",
    "figure_11_contributor_activity",
    "Contributor activity is measured as contributor-month exposure: the sum, across the complete monthly road-mapping history, of the number of unique OpenStreetMap users active in each month. A user active in multiple months is counted once in each such month; therefore this measures cumulative participation intensity, not the number of distinct people over the whole period.",
)

summary = pd.DataFrame([
    {
        "indicator": metric,
        "unit": "contributor-months",
        "n_cells": int(values.notna().sum()),
        "zero_cells": int(values.eq(0).sum()),
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p98": float(values.quantile(0.98)),
        "max": float(values.max()),
        "methodological_boundary": "Sum of monthly unique-contributor counts; not whole-period unique users.",
    }
])
summary.to_csv(OUT / "figure_11_contributor_activity_summary.csv", index=False)

manifest = []
for p in sorted(OUT.glob("figure_*.*")):
    if p.name != "figure_manifest.csv":
        manifest.append({"file": p.name, "bytes": p.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT / "figure_manifest.csv", index=False)

print(summary.to_string(index=False))
print("Saved:", OUT / "figure_11_contributor_activity.png")
print("Saved:", OUT / "figure_11_contributor_activity.pdf")
