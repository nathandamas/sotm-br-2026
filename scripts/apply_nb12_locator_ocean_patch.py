from pathlib import Path

path = Path("scripts/notebook12_build_article_maps.py")
text = path.read_text(encoding="utf-8")

needle = '    m = ax.inset_axes([0.055, 0.245, 0.89, 0.665])\n'
replacement = (
    '    m = ax.inset_axes([0.055, 0.245, 0.89, 0.665])\n'
    '    # Ocean background for the locator only; land polygons are drawn above it.\n'
    '    m.set_facecolor("#9ecae1")\n'
)

if 'm.set_facecolor("#9ecae1")' in text:
    print("Locator ocean background already applied.")
elif needle not in text:
    raise RuntimeError("Expected locator inset statement not found")
else:
    text = text.replace(needle, replacement, 1)
    path.write_text(text, encoding="utf-8")
    print("Applied #9ecae1 ocean background to locator map only.")
