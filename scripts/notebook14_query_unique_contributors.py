from pathlib import Path
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests

API_BASE = "https://api.ohsome.org/v1"
REQUEST_TIMEOUT = 180
MAX_ATTEMPTS = 2
MAX_WORKERS = 12
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
PROGRESS_FILE = OUT_DIR / "unique_contributors_query_progress.csv"

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

HEADERS = {"User-Agent": "sotm-br-2026-research/0.6 (academic OSM analysis)"}


def single_fc(grid_id):
    return {"type": "FeatureCollection", "features": [feature_by_id[grid_id]]}


def validate_body(body, expected_id):
    groups = body.get("groupByResult") if isinstance(body, dict) else None
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError(f"Expected one groupByResult object for grid {expected_id}.")
    group = groups[0]
    returned_id = str(group.get("groupByObject"))
    if returned_id != str(expected_id):
        raise ValueError(f"ohsome returned grid {returned_id}, expected {expected_id}.")
    result = group.get("result")
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError(f"Expected exactly one time interval for grid {expected_id}.")
    item = result[0]
    if "value" not in item or not {"fromTimestamp", "toTimestamp"}.issubset(item):
        raise ValueError("Unexpected /users/count/groupBy/boundary response schema.")
    return True


def safe_id(grid_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(grid_id))


def checkpoint_path(grid_id):
    return RAW_DIR / f"cell_{safe_id(grid_id)}.json"


def parse_body(body):
    group = body["groupByResult"][0]
    item = group["result"][0]
    return {
        "grid_1km_id": str(group["groupByObject"]),
        "all_unique_contributors_count": float(item["value"]),
        "all_unique_contributors_from": item["fromTimestamp"],
        "all_unique_contributors_to": item["toTimestamp"],
    }


def load_cached(grid_id):
    path = checkpoint_path(grid_id)
    if not path.exists():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    response = cached.get("response", {})
    body = response.get("body")
    if response.get("status_code") != 200:
        return None
    try:
        validate_body(body, grid_id)
    except Exception:
        return None
    return parse_body(body)


def query_cell(grid_id):
    cached = load_cached(grid_id)
    if cached is not None:
        return grid_id, cached, "cache", None

    path = checkpoint_path(grid_id)
    url = f"{API_BASE}/users/count/groupBy/boundary"
    data = {
        "bpolys": json.dumps(single_fc(grid_id), ensure_ascii=False, separators=(",", ":")),
        "time": USERS_TIME,
        "format": "json",
    }
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(url, data=data, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_error = exc
            envelope = {
                "request": {
                    "endpoint": "/users/count/groupBy/boundary",
                    "grid_1km_id": grid_id,
                    "feature_count": 1,
                    "filter": None,
                    "time": USERS_TIME,
                    "attempt": attempt,
                },
                "transport_error": {"type": type(exc).__name__, "message": str(exc)},
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            path.with_suffix(f".attempt_{attempt:02d}.transport_error.json").write_text(
                json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** attempt, 15))
                continue
            return grid_id, None, "error", f"{type(exc).__name__}: {exc}"

        try:
            body = response.json()
        except Exception:
            body = {"raw_text": response.text}
        envelope = {
            "request": {
                "endpoint": "/users/count/groupBy/boundary",
                "grid_1km_id": grid_id,
                "feature_count": 1,
                "filter": None,
                "time": USERS_TIME,
                "attempt": attempt,
            },
            "response": {
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "body": body,
            },
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if response.ok:
            try:
                validate_body(body, grid_id)
            except Exception as exc:
                return grid_id, None, "error", f"SchemaError: {exc}"
            path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
            return grid_id, parse_body(body), "request", None

        path.with_suffix(f".attempt_{attempt:02d}.http_error.json").write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        last_error = RuntimeError(f"HTTP {response.status_code}: {json.dumps(body, ensure_ascii=False)[:500]}")
        if response.status_code not in TRANSIENT_STATUS or attempt == MAX_ATTEMPTS:
            return grid_id, None, "error", str(last_error)
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 15)
        time.sleep(delay)

    return grid_id, None, "error", str(last_error)


rows = []
progress = []
pre_cached = 0
pending_ids = []
for grid_id in ids:
    cached = load_cached(grid_id)
    if cached is None:
        pending_ids.append(grid_id)
    else:
        rows.append(cached)
        progress.append({"grid_1km_id": grid_id, "status": "cache", "error": ""})
        pre_cached += 1

print(
    f"Notebook 14 all-OSM unique users: {pre_cached} cached, {len(pending_ids)} pending, "
    f"workers={MAX_WORKERS}, timeout={REQUEST_TIMEOUT}s, attempts={MAX_ATTEMPTS}",
    flush=True,
)

if pending_ids:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(query_cell, grid_id): grid_id for grid_id in pending_ids}
        completed = 0
        for future in as_completed(futures):
            grid_id = futures[future]
            try:
                returned_id, row, source, error = future.result()
            except Exception as exc:
                returned_id, row, source, error = grid_id, None, "error", f"{type(exc).__name__}: {exc}"
            if str(returned_id) != str(grid_id):
                source = "error"
                error = f"Worker ID mismatch: returned {returned_id}, expected {grid_id}"
                row = None
            if row is not None:
                rows.append(row)
            progress.append({"grid_1km_id": grid_id, "status": source, "error": error or ""})
            completed += 1
            if completed % 10 == 0 or source == "error" or completed == len(pending_ids):
                success_now = sum(p["status"] in {"cache", "request"} for p in progress)
                print(
                    f"Progress: {success_now}/895 valid checkpoints; completed this run {completed}/{len(pending_ids)}; "
                    f"latest={grid_id} status={source}",
                    flush=True,
                )

progress_df = pd.DataFrame(progress).sort_values("grid_1km_id")
progress_df.to_csv(PROGRESS_FILE, index=False)

all_users = pd.DataFrame(rows)
if not all_users.empty:
    all_users["grid_1km_id"] = all_users["grid_1km_id"].astype(str)
    all_users = all_users.drop_duplicates("grid_1km_id", keep="last")

failed_ids = sorted(set(ids) - set(all_users.get("grid_1km_id", pd.Series(dtype=str)).astype(str)))
if failed_ids:
    print(f"Incomplete all-OSM unique-user query: {len(failed_ids)} cells remain. Checkpoints retained in {RAW_DIR}", flush=True)
    print("First remaining IDs:", failed_ids[:20], flush=True)
    raise RuntimeError(
        f"All-OSM unique-user query incomplete: {len(failed_ids)} of 895 cells remain. "
        "Successful per-cell checkpoints are intentionally retained for the next run."
    )

if len(all_users) != 895 or not all_users["grid_1km_id"].is_unique:
    raise ValueError("All-OSM contributor query did not produce 895 unique cells.")

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
print(qc[["check", "status", "value"]].to_string(index=False))
print(summary.to_string(index=False))
print("Saved:", OUT_FILE)
