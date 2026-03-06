import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import gradio as gr
import requests

# -------------------------
# Config
# -------------------------
EVAL_CASES_PATH = Path("eval_cases.jsonl")
FEEDBACK_LOG_PATH = Path("feedback_log.jsonl")

API_BASE_URL_DEFAULT = "http://127.0.0.1:8000"
RECOMMEND_ENDPOINT = "/recommend"
HEALTH_ENDPOINT = "/health"

APP_VERSION = "v0.1"
MODEL_ID = "OzzeY72/biobert-medical-specialities"

# Must match your API's output specialist strings (+ optional extra choices)
SPECIALIST_CHOICES = [
    "Cardiologist",
    "Neurologist",
    "Pulmonologist",
    "Gastroenterologist",
    "Dermatologist",
    "Otolaryngologist (Ear, Nose and Throat Specialist)",
    "Orthopedic Surgeon",
    "Urologist",
    "Gynecologist / Obstetrician",
    "Psychiatrist",
    "Ophthalmologist (Eye Specialist)",
    "Endocrinologist",
    "Nephrologist (Kidney Specialist)",
    "Oncologist",
    "Rheumatologist",
    "Hematologist",
    "Allergist / Immunologist",
    "Pediatrician",
    "General Physician / Family Medicine",
    "Not sure / ambiguous",
]

WRITE_LOCK = Lock()


# -------------------------
# Helpers: JSONL
# -------------------------
def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip bad lines rather than crashing the whole UI
                continue
    return rows


def load_eval_cases() -> List[dict]:
    if not EVAL_CASES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EVAL_CASES_PATH}. Create it first (eval cases JSONL)."
        )
    cases = _read_jsonl(EVAL_CASES_PATH)

    # Basic schema sanity (non-fatal)
    required = {"case_id", "age", "gender", "symptoms", "severity", "duration"}
    cleaned = []
    for c in cases:
        if not required.issubset(set(c.keys())):
            continue
        cleaned.append(c)
    return cleaned


def load_reviewed_case_ids() -> set:
    rows = _read_jsonl(FEEDBACK_LOG_PATH)
    reviewed = set()
    for r in rows:
        cid = r.get("case_id")
        if cid:
            reviewed.add(cid)
    return reviewed


def append_feedback(entry: dict) -> None:
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with WRITE_LOCK:
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# -------------------------
# Helpers: API
# -------------------------
def api_health(base_url: str) -> Tuple[bool, str]:
    try:
        r = requests.get(base_url.rstrip("/") + HEALTH_ENDPOINT, timeout=5)
        if r.status_code == 200:
            return True, "✅ API reachable"
        return False, f"❌ API health returned {r.status_code}"
    except Exception as e:
        return False, f"❌ API not reachable: {e}"


def api_recommend(base_url: str, payload: dict) -> Tuple[bool, dict, str]:
    try:
        r = requests.post(
            base_url.rstrip("/") + RECOMMEND_ENDPOINT,
            json=payload,
            timeout=30
        )
        if r.status_code != 200:
            return False, {}, f"API error {r.status_code}: {r.text}"
        return True, r.json(), ""
    except Exception as e:
        return False, {}, f"Request failed: {e}"


# -------------------------
# Selection logic
# -------------------------
def pick_next_unreviewed(cases: List[dict], reviewed: set, start_idx: int = 0) -> Tuple[Optional[int], Optional[dict]]:
    n = len(cases)
    if n == 0:
        return None, None
    for i in range(start_idx, n):
        if cases[i]["case_id"] not in reviewed:
            return i, cases[i]
    # wrap
    for i in range(0, start_idx):
        if cases[i]["case_id"] not in reviewed:
            return i, cases[i]
    return None, None


def now_ist_iso() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).isoformat(timespec="seconds")


# -------------------------
# Gradio callbacks
# -------------------------
def init_app(api_base_url: str) -> Tuple[str, int, dict, dict, str, str, str, str]:
    """
    Returns:
    status_md, total_cases, current_case, current_output, case_json, out_json, progress_md, api_status_md
    """
    try:
        cases = load_eval_cases()
    except Exception as e:
        return (
            f"❌ {e}",
            0,
            {},
            {},
            "",
            "",
            "Progress: 0/0 reviewed",
            "API status: unknown"
        )

    reviewed = load_reviewed_case_ids()
    idx, case = pick_next_unreviewed(cases, reviewed, 0)

    ok, api_msg = api_health(api_base_url)
    api_status = f"API status: {api_msg}"

    if case is None:
        progress = f"✅ Progress: {len(reviewed)}/{len(cases)} reviewed (all done)"
        return (
            "✅ Loaded eval cases. No unreviewed cases left.",
            len(cases),
            {},
            {},
            "",
            "",
            progress,
            api_status
        )

    payload = {
        "age": case["age"],
        "gender": case["gender"],
        "symptoms": case["symptoms"],
        "severity": case["severity"],
        "duration": case["duration"]
    }

    if ok:
        success, out, err = api_recommend(api_base_url, payload)
        if not success:
            out = {"error": err}
    else:
        out = {"error": "API not reachable. Start FastAPI first (uvicorn app:app --reload)."}

    progress = f"Progress: {len(reviewed)}/{len(cases)} reviewed"

    return (
        "✅ Loaded eval cases.",
        len(cases),
        {"_idx": idx, **case},
        out,
        json.dumps(case, indent=2, ensure_ascii=False),
        json.dumps(out, indent=2, ensure_ascii=False),
        progress,
        api_status
    )


def next_case(api_base_url: str, current_state: dict) -> Tuple[dict, dict, str, str, str]:
    cases = load_eval_cases()
    reviewed = load_reviewed_case_ids()

    start_idx = 0
    if current_state and isinstance(current_state.get("_idx"), int):
        start_idx = (current_state["_idx"] + 1) % max(len(cases), 1)

    idx, case = pick_next_unreviewed(cases, reviewed, start_idx)
    if case is None:
        progress = f"✅ Progress: {len(reviewed)}/{len(cases)} reviewed (all done)"
        return {}, {}, "", "", progress

    payload = {
        "age": case["age"],
        "gender": case["gender"],
        "symptoms": case["symptoms"],
        "severity": case["severity"],
        "duration": case["duration"]
    }

    ok, _ = api_health(api_base_url)
    if ok:
        success, out, err = api_recommend(api_base_url, payload)
        if not success:
            out = {"error": err}
    else:
        out = {"error": "API not reachable. Start FastAPI first (uvicorn app:app --reload)."}

    progress = f"Progress: {len(reviewed)}/{len(cases)} reviewed"

    return (
        {"_idx": idx, **case},
        out,
        json.dumps(case, indent=2, ensure_ascii=False),
        json.dumps(out, indent=2, ensure_ascii=False),
        progress
    )


def set_vote_up() -> Tuple[str, gr.update]:
    # vote_state, correct_specialist_visibility
    return "up", gr.update(visible=False, value=None)


def set_vote_down() -> Tuple[str, gr.update]:
    return "down", gr.update(visible=True)


def save_feedback(
    reviewer_id: str,
    api_base_url: str,
    app_version: str,
    current_case: dict,
    current_output: dict,
    vote_state: str,
    correct_specialist: Optional[str],
    comment: str
) -> Tuple[str, str]:
    if not current_case or "case_id" not in current_case:
        return "❌ No case loaded.", ""

    if vote_state not in ("up", "down"):
        return "❌ Please choose 👍 or 👎 first.", ""

    if vote_state == "down":
        if not correct_specialist or correct_specialist.strip() == "":
            return "❌ For 👎 you must pick the correct specialist.", ""

    case_id = current_case["case_id"]

    # Build feedback entry
    feedback_id = f"{now_ist_iso()}_{case_id}_{reviewer_id}"
    entry = {
        "feedback_id": feedback_id,
        "timestamp_ist": now_ist_iso(),
        "case_id": case_id,
        "reviewer_id": reviewer_id,
        "vote": vote_state,
        "correct_specialist": (correct_specialist if vote_state == "down" else None),
        "comment": (comment.strip() if comment and comment.strip() else None),
        "model_output": {
            "recommended_specialist": current_output.get("recommended_specialist"),
            "confidence": current_output.get("confidence"),
            "model_label": current_output.get("model_label"),
            "normalized_symptoms": current_output.get("normalized_symptoms"),
            "raw": current_output,  # keep full response for debugging/versioning
        },
        "model_id": MODEL_ID,
        "api_base_url": api_base_url,
        "app_version": app_version,
        "input": {
            "age": current_case.get("age"),
            "gender": current_case.get("gender"),
            "symptoms": current_case.get("symptoms"),
            "severity": current_case.get("severity"),
            "duration": current_case.get("duration"),
        }
    }

    append_feedback(entry)
    return "✅ Feedback saved.", json.dumps(entry, indent=2, ensure_ascii=False)


# -------------------------
# UI
# -------------------------
with gr.Blocks(title="Specialist Recommender Feedback") as demo:
    gr.Markdown("# Specialist Recommender — Feedback App")

    with gr.Row():
        api_base_url = gr.Textbox(
            label="FastAPI Base URL",
            value=API_BASE_URL_DEFAULT,
            interactive=True
        )
        reviewer_id = gr.Dropdown(
            label="Reviewer",
            choices=["tester_1", "tester_2"],
            value="tester_1",
            interactive=True
        )
        app_version = gr.Textbox(label="App Version", value=APP_VERSION, interactive=True)

    status_md = gr.Markdown("")
    api_status_md = gr.Markdown("API status: unknown")

    with gr.Row():
        progress_md = gr.Markdown("Progress: -")
        total_cases = gr.Number(label="Total Cases", value=0, precision=0, interactive=False)

    current_case_state = gr.State({})
    current_output_state = gr.State({})
    vote_state = gr.State("")  # "up" or "down"

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Current Case (JSON)")
            case_json = gr.Code(label="Case", language="json", value="")
            with gr.Row():
                btn_next = gr.Button("Next Case")
                btn_skip = gr.Button("Skip (Next Unreviewed)")
        with gr.Column(scale=1):
            gr.Markdown("## Model Output (JSON)")
            out_json = gr.Code(label="Output", language="json", value="")

    gr.Markdown("## Feedback")

    with gr.Row():
        btn_up = gr.Button("👍 Thumbs Up")
        btn_down = gr.Button("👎 Thumbs Down")

    correct_specialist = gr.Dropdown(
        label="If 👎, choose the correct specialist",
        choices=SPECIALIST_CHOICES,
        value=None,
        visible=False,
        interactive=True
    )
    comment = gr.Textbox(label="Optional comment", placeholder="Why up/down? (optional)", lines=2)

    with gr.Row():
        btn_save = gr.Button("Save Feedback")
        save_status = gr.Markdown("")

    saved_entry_preview = gr.Code(label="Last saved entry (JSON)", language="json", value="")

    # Init
    btn_init = gr.Button("Load / Start")

    btn_init.click(
        fn=init_app,
        inputs=[api_base_url],
        outputs=[status_md, total_cases, current_case_state, current_output_state, case_json, out_json, progress_md, api_status_md]
    )

    # Next/Skip (same behavior: go to next unreviewed)
    btn_next.click(
        fn=next_case,
        inputs=[api_base_url, current_case_state],
        outputs=[current_case_state, current_output_state, case_json, out_json, progress_md]
    )
    btn_skip.click(
        fn=next_case,
        inputs=[api_base_url, current_case_state],
        outputs=[current_case_state, current_output_state, case_json, out_json, progress_md]
    )

    # Vote buttons
    btn_up.click(
        fn=set_vote_up,
        inputs=[],
        outputs=[vote_state, correct_specialist]
    )
    btn_down.click(
        fn=set_vote_down,
        inputs=[],
        outputs=[vote_state, correct_specialist]
    )

    # Save
    btn_save.click(
        fn=save_feedback,
        inputs=[
            reviewer_id,
            api_base_url,
            app_version,
            current_case_state,
            current_output_state,
            vote_state,
            correct_specialist,
            comment
        ],
        outputs=[save_status, saved_entry_preview]
    )

    gr.Markdown(
        "### Notes\n"
        "- Make sure your FastAPI server is running before you start reviewing:\n"
        "  - `uvicorn app:app --reload`\n"
        "- This app writes feedback to `feedback_log.jsonl` in the current folder.\n"
        "- Eval cases are loaded from `eval_cases.jsonl`.\n"
    )

if __name__ == "__main__":
    demo.launch()