import json, os
from datetime import datetime, timezone

FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "data/feedback.jsonl")

def append_feedback(record: dict) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    record = dict(record)
    record["ts_utc"] = datetime.now(timezone.utc).isoformat()
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
