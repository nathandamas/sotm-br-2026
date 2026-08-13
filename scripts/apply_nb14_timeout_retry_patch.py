from pathlib import Path

path = Path("scripts/notebook14_query_unique_contributors.py")
text = path.read_text(encoding="utf-8")
original = text

text = text.replace('BATCH_SIZE = 25\n', 'BATCH_SIZE = 5\n', 1)

old = '''    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = session.post(url, data=data, timeout=REQUEST_TIMEOUT)
        try:
            body = response.json()
        except Exception:
            body = {"raw_text": response.text}
'''
new = '''    for attempt in range(1, MAX_ATTEMPTS + 1):
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
'''
if old not in text:
    raise RuntimeError("Expected request loop was not found")
text = text.replace(old, new, 1)

if text == original:
    raise RuntimeError("No changes applied")
path.write_text(text, encoding="utf-8")
print("Notebook 14 patched: batch size 5 and transport-level retries enabled.")
