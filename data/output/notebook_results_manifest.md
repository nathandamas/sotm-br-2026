# Notebook results audit

All notebooks 01–13 have a validated successful execution in the project pipeline. Notebook 03 is executed inside the Notebook 04 workflow and its executed notebook plus boundary outputs are preserved in the Notebook 04 artifact.

| Notebook | Status | Canonical run | Canonical artifact | Target folder in results repository |
|---|---|---:|---:|---|
| 01 — validate IBGE input data | SUCCESS | 31438989833 | 9082122977 | `notebook_01/` |
| 02 — build IBGE 1 km grid | SUCCESS | 31440862694 | 9082793177 | `notebook_02/` |
| 03 — prepare ohsome boundaries | SUCCESS (within NB04 workflow) | 31445861031 | 9084494455 | `notebook_03/` |
| 04 — query ohsome test cells | SUCCESS | 31445861031 | 9084494455 | `notebook_04/` |
| 05 — query all 895 cells | SUCCESS | 31446172260 | 9084655869 | `notebook_05/` |
| 06 — monthly OSM time series | SUCCESS | 31447247778 | 9085073919 | `notebook_06/` |
| 07 — maturity indicators | SUCCESS | 31451193839 | 9086389498 | `notebook_07/` |
| 08 — indicator structure | SUCCESS | 31452397134 | 9086782405 | `notebook_08/` |
| 09 — cell typology | SUCCESS | 31455714847 | 9087965919 | `notebook_09/` |
| 10 — typology interpretation | SUCCESS | 31484979441 | 9098746154 | `notebook_10/` |
| 11 — final typology | SUCCESS | 31486390648 | 9099287431 | `notebook_11/` |
| 12 — article maps | SUCCESS | 31487353842 | 9099676503 | `notebook_12/` |
| 13 — article tables | SUCCESS | 31487892038 | 9099854352 | `notebook_13/` |

## Current article-output locations

- Figures: `data/output/figures/`
- Tables: `data/output/tables/`

## Proposed results-repository structure

```text
README.md
manifest.csv
notebook_01/
notebook_02/
notebook_03/
notebook_04/
notebook_05/
notebook_06/
notebook_07/
notebook_08/
notebook_09/
notebook_10/
notebook_11/
notebook_12/
notebook_13/
```

Each folder should contain the canonical executed notebook and the outputs/artifacts attributable to that notebook. For Notebook 03, the relevant subset must be extracted from the canonical Notebook 04 artifact rather than duplicated wholesale.
