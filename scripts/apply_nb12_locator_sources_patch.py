from pathlib import Path
import re

path = Path("scripts/notebook12_build_article_maps.py")
text = path.read_text(encoding="utf-8")
original = text

pattern_constants = re.compile(
    r'DATA_SOURCE_LINES = \[.*?\]\nBASEMAP_LINES = \[.*?\]\nCRS_LINES = \[',
    re.S,
)
new_constants = r'''DATA_SOURCE_LINES = [
    "OpenStreetMap road history via ohsome API",
    "1 km reference grid derived from IBGE\n  source grid",
    "Municipal boundary: IBGE (2025)",
]
BASEMAP_LINES = [
    "CARTO Positron",
    "(© OpenStreetMap contributors, © CARTO)",
]
CRS_LINES = ['''
text, n = pattern_constants.subn(lambda m: new_constants, text, count=1)
if n != 1:
    raise RuntimeError(f"Expected one DATA_SOURCE_LINES/BASEMAP_LINES block, replaced {n}")

if 'm.set_aspect("equal", adjustable="box")' in text:
    text = text.replace(
        'm.set_aspect("equal", adjustable="box")',
        'm.set_aspect("equal", adjustable="datalim")',
        1,
    )
elif 'm.set_aspect("equal", adjustable="datalim")' not in text:
    raise RuntimeError("Expected locator aspect statement not found")

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
text, n = pattern_sources.subn(lambda m: new_sources, text, count=1)
if n != 1:
    raise RuntimeError(f"Expected one draw_sources_panel function, replaced {n}")

path.write_text(text, encoding="utf-8")
print("Applied only the requested locator-frame and source/basemap text changes.")
