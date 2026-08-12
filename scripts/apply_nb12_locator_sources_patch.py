from pathlib import Path
import re

path = Path("scripts/notebook12_build_article_maps.py")
text = path.read_text(encoding="utf-8")
original = text

old_constants = '''DATA_SOURCE_LINES = [
    "OpenStreetMap (osm.org) – vector data\n  (snapshot dated 2025-05-24)",
    "ohsome API – contribution history",
    "1 km reference grid",
    "Municipal boundary: IBGE (2022)",
]
BASEMAP_LINES = ["CARTO Positron (© OpenStreetMap contributors,\n© CARTO)"]
'''
new_constants = '''DATA_SOURCE_LINES = [
    "OpenStreetMap road history via ohsome API",
    "1 km reference grid derived from IBGE\n  source grid",
    "Municipal boundary: IBGE (2025)",
]
BASEMAP_LINES = [
    "CARTO Positron",
    "(© OpenStreetMap contributors, © CARTO)",
]
'''
if old_constants not in text:
    raise RuntimeError("Expected current DATA_SOURCE_LINES/BASEMAP_LINES block not found")
text = text.replace(old_constants, new_constants, 1)

old_aspect = '    m.set_aspect("equal", adjustable="box")\n'
new_aspect = '    m.set_aspect("equal", adjustable="datalim")\n'
if old_aspect not in text:
    raise RuntimeError("Expected locator aspect statement not found")
text = text.replace(old_aspect, new_aspect, 1)

pattern_sources = re.compile(r'def draw_sources_panel\(ax\):\n.*?\n\ndef add_boundary_legend\(ax\):', re.S)
new_sources = '''def draw_sources_panel(ax):
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


def add_boundary_legend(ax):'''
text, n = pattern_sources.subn(new_sources, text, count=1)
if n != 1:
    raise RuntimeError(f"Expected one draw_sources_panel function, replaced {n}")

if text == original:
    raise RuntimeError("No changes were applied")

path.write_text(text, encoding="utf-8")
print("Applied only the requested locator-frame and source/basemap text changes.")
