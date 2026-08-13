from pathlib import Path
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

API_BASE = "https://api.ohsome.org/v1"
BATCH_SIZE = 5
REQUEST_TIMEOUT = 240
MAX_ATTEMPTS = 5
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, start.parent, *start.parents]:
        if (candidate / "notebooks").exists() and (candidate / "Petropolis").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


ROOT = find_project_root(Path.cwd())
NB05 = ROOT / "data" / "output" / "notebooks" / "05" / "notebook-05-ohsome-production-results" / "production_outputs"
ROAD_METRICS_FILE = NB05 / "petropolis_ohsome_895_cells.parquet"
BOUNDARIES_FILE = NB05 / "petropolis_grid_1km_ohsome_boundaries.geojson"
OUT_DIR = ROOT / "data" / "output" / "notebooks" / "14"
RAW_DIR = ROOT / "data" / "raw" / "ohsome" / "Petropolis" / "all_unique_contributors_895_cells"
OUT_FILE = OUT_DIR / "petropolis_unique_contributors_895_cells.parquet"
QC_FILE = OUT_DIR / "unique_contributors_quality_checks.csv"
SUMMARY_FILE = OUT_DIR / "unique_contributors_summary.csv"

for path in (ROAD_METRICS_FILE, BOUNDARIES_FILE):
    if not path.exists():
        raise FileNotFoundError(path)
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

road = pd.read_parquet(ROAD_METRICS_FILE)
required = {
    "grid_1km_id",
    "contributors_count",
    "contributors_count_from",
    "contributors_count_to",
}
missing = required - set(road.columns)
if missing:
    raise KeyError(f"Notebook 05 road-contributor output is missing columns: {sorted(missing)}")

road = road.copy()
road["grid_1km_id"] = road["grid_1km_id"].astype(str)
if len(road) != 895 or not road["grid_1km_id"].is_unique:
    raise ValueError("Notebook 05 road-contributor table must contain 895 unique grid IDs.")

from_values = road["contributors_count_from"].dropna().astype(str).unique().tolist()
to_values = road["contributors_count_to"].dropna().astype(str).unique().tolist()
if len(from_values) != 1 or len(to_values) != 1:
    raise ValueError(f"Expected one frozen Notebook 05 contributor interval, got from={from_values}, to={to_values}")
FROM_TIMESTAMP = from_values[0]
TO_TIMESTAMP = to_values[0]
USERS_TIME = f"{FROM_TIMESTAMP}/{TO_TIMESTAMP}"

feature_collection = json.loads(BOUNDARIES_FILE.read_text(encoding="utf-8"))
features = feature_collection.get("features", [])
feature_by_id = {str(f.get("properties", {}).get("id")): f for f in features}
ids = sorted(road["grid_1km_id"].tolist())
if len(features) != 895 or set(ids) != set(feature_by_id):
    raise ValueError("Boundary collection does not match the 895 Notebook 05 grid IDs.")

batches = [ids[i : i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]
session = requests.Session()
session.headers.update({"User-Agent": "sotm-br-2026-research/0.5 (academic OSM analysis)"})


def batch_fc(batch_ids):
    return {"type": "FeatureCollection", "features": [feature_by_id[i] for i in batch_ids]}


def validate_body(body, expected_ids):
    groups = body.get("groupByResult") if isinstance(body, dict) else None
    if not isinstance(groups, list):
        raise ValueError("Missing groupByResult list in ohsome response.")
    returned = [str(g.get("groupByObject")) for g in groups]
    if len(returned) != len(set(returned)) or set(returned) != set(expected_ids):
        raise ValueError("Boundary IDs returned by ohsome do not match the requested batch.")
    for group in groups:
        result = group.get("result")
        if not isinstance(result, list) or len(result) != 1:
            raise ValueError(f"Expected exactly one interval for {group.get('groupByObject')}")
        item = result[0]
        if "value" not in item or not {"fromTimestamp", "toTimestamp"}.issubset(item):
            raise ValueError("Unexpected /users/count/groupBy/boundary response schema.")
    return True


def checkpoint_path(batch_no):
    return RAW_DIR / f"batch_{batch_no:03d}.json"


def request_or_cache(batch_no, batch_ids):
    path = checkpoint_path(batch_no)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        body = cached.get("response", {}).get("body")
        if cached.get("response", {}).get("status_code") == 200:
            validate_body(body, batch_ids)
            return body, "cache"

    data = {
        "bpolys": json.dumps(batch_fc(batch_ids), ensure_ascii=False, separators=(",", ":")),
        "time": USERS_TIME,
        "format": "json",
    }
    # Deliberately no `filter`: this request counts unique users contributing to
    # any OSM feature in each boundary, unlike Notebook 05's road-filtered count.
    url = f"{API_BASE}/users/count/groupBy/boundary"
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.post(url, data=data, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_error = exc
            transport_error = {
                "request": {
                    "endpoint": "/users/count/groupBy/boundary",
                    "batch_no": batch_no,
                    "ids": batch_ids,
                    "feature_count": len(batch_ids),
                    "filter": None,
                    "time": USERS_TIME,
                    "attempt": attempt,
                },
                "transport_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            path.with_suffix(f".attempt_{attempt:02d}.transport_error.json").write_text(
                json.dumps(transport_error, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(min(2 ** attempt, 30))
            continue
        try:
            body = response.json()
        except Exception:
            body = {"raw_text": response.text}
        envelope = {
            "request": {
                "endpoint": "/users/count/groupBy/boundary",
                "batch_no": batch_no,
                "ids": batch_ids,
                "feature_count": len(batch_ids),
                "filter": None,
                "time": USERS_TIME,
            },
            "response": {
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "body": body,
            },
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if response.ok:
            validate_body(body, batch_ids)
            path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
            return body, "request"
        error_path = path.with_suffix(".error.json")
        error_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
        last_error = RuntimeError(
            f"All-OSM contributor batch {batch_no} HTTP {response.status_code}: "
            f"{json.dumps(body, ensure_ascii=False)[:1000]}"
        )
        if response.status_code not in TRANSIENT_STATUS or attempt == MAX_ATTEMPTS:
            raise last_error
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 30)
        time.sleep(delay)
    raise last_error


def parse_body(body):
    rows = []
    for group in body["groupByResult"]:
        item = group["result"][0]
        rows.append(
            {
                "grid_1km_id": str(group["groupByObject"]),
                "all_unique_contributors_count": float(item["value"]),
                "all_unique_contributors_from": item["fromTimestamp"],
                "all_unique_contributors_to": item["toTimestamp"],
            }
        )
    return pd.DataFrame(rows)


parts = []
sources = {"request": 0, "cache": 0}
for batch_no, batch_ids in enumerate(batches, start=1):
    body, source = request_or_cache(batch_no, batch_ids)
    sources[source] += 1
    parts.append(parse_body(body))
    print(f"Batch {batch_no:02d}/{len(batches)}: {source}; cells={len(batch_ids)}")

all_users = pd.concat(parts, ignore_index=True)
if len(all_users) != 895 or not all_users["grid_1km_id"].is_unique:
    raise ValueError("All-OSM contributor query did not return 895 unique cells.")

result = road[[
    "grid_1km_id",
    "contributors_count",
    "contributors_count_from",
    "contributors_count_to",
]].rename(
    columns={
        "contributors_count": "road_unique_contributors_count",
        "contributors_count_from": "road_unique_contributors_from",
        "contributors_count_to": "road_unique_contributors_to",
    }
).merge(all_users, on="grid_1km_id", how="left", validate="one_to_one")

for col in ("all_unique_contributors_count", "road_unique_contributors_count"):
    result[col] = pd.to_numeric(result[col], errors="raise")

checks = [
    ("rows_895", len(result) == 895, len(result)),
    ("grid_ids_unique", result["grid_1km_id"].is_unique, result["grid_1km_id"].nunique()),
    ("all_counts_nonnegative", (result["all_unique_contributors_count"] >= 0).all(), result["all_unique_contributors_count"].min()),
    ("road_counts_nonnegative", (result["road_unique_contributors_count"] >= 0).all(), result["road_unique_contributors_count"].min()),
    (
        "road_subset_not_above_all",
        (result["road_unique_contributors_count"] <= result["all_unique_contributors_count"]).all(),
        int((result["road_unique_contributors_count"] > result["all_unique_contributors_count"]).sum()),
    ),
    (
        "same_frozen_interval",
        (result["road_unique_contributors_from"].astype(str) == result["all_unique_contributors_from"].astype(str)).all()
        and (result["road_unique_contributors_to"].astype(str) == result["all_unique_contributors_to"].astype(str)).all(),
        USERS_TIME,
    ),
]
qc = pd.DataFrame(checks, columns=["check", "pass", "value"])
qc["status"] = qc["pass"].map({True: "PASS", False: "FAIL"})
if not qc["pass"].all():
    print(qc.to_string(index=False))
    raise ValueError("Contributor-count QC failed.")

result.to_parquet(OUT_FILE, index=False)
qc[["check", "status", "value"]].to_csv(QC_FILE, index=False)
summary = pd.DataFrame([
    {
        "metric": "all_unique_contributors_count",
        "min": result["all_unique_contributors_count"].min(),
        "median": result["all_unique_contributors_count"].median(),
        "mean": result["all_unique_contributors_count"].mean(),
        "max": result["all_unique_contributors_count"].max(),
        "zero_cells": int(result["all_unique_contributors_count"].eq(0).sum()),
    },
    {
        "metric": "road_unique_contributors_count",
        "min": result["road_unique_contributors_count"].min(),
        "median": result["road_unique_contributors_count"].median(),
        "mean": result["road_unique_contributors_count"].mean(),
        "max": result["road_unique_contributors_count"].max(),
        "zero_cells": int(result["road_unique_contributors_count"].eq(0).sum()),
    },
])
summary.to_csv(SUMMARY_FILE, index=False)

print("Frozen contributor interval:", USERS_TIME)
print("Batch sources:", sources)
print(qc[["check", "status", "value"]].to_string(index=False))
print(summary.to_string(index=False))
print("Saved:", OUT_FILE)
