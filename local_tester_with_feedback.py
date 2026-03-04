import os, json, uuid
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "OzzeY72/biobert-medical-specialities"
FEEDBACK_PATH = os.path.join("data", "feedback.jsonl")

SPECIALIST_LABELS_20 = [
    "General Physician / Family Medicine",
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
    "Infectious Disease Specialist",
    "Allergist / Immunologist",
    "Pediatrician",
]

MODEL_TO_20 = {
    "Cardiology": "Cardiologist",
    "Neurology": "Neurologist",
    "Respiratory": "Pulmonologist",
    "Gastroenterology": "Gastroenterologist",
    "Dermatology": "Dermatologist",
    "Otorhinolaryngology": "Otolaryngologist (Ear, Nose and Throat Specialist)",
    "Orthopedics": "Orthopedic Surgeon",
    "Urology": "Urologist",
    "Obstetrics": "Gynecologist / Obstetrician",
    "Gynecology": "Gynecologist / Obstetrician",
    "Psychiatry": "Psychiatrist",
    "Psychology": "Psychiatrist",
    "Ophthalmology": "Ophthalmologist (Eye Specialist)",
    "Endocrinology": "Endocrinologist",
    "Nephrology": "Nephrologist (Kidney Specialist)",
    "Oncology": "Oncologist",
    "Rheumatology": "Rheumatologist",
    "Hematology": "Hematologist",
    "Allergy": "Allergist / Immunologist",
    "Pediatrics": "Pediatrician",
    "Microbiology": "Infectious Disease Specialist",
}

CONF_THRESHOLD = 0.55  # below this -> General Physician, except pediatric override


def map_to_20(model_label: str) -> str:
    return MODEL_TO_20.get(model_label, "General Physician / Family Medicine")


def build_text(age, gender, symptoms, severity, duration):
    return (
        f"Age: {age} | Gender: {gender} | Severity: {severity}/10 | "
        f"Duration: {duration} | Symptoms: {symptoms}"
    )


def ensure_feedback_file():
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    if not os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            pass


def append_jsonl(obj: dict):
    obj = dict(obj)
    obj["ts_utc"] = datetime.now(timezone.utc).isoformat()
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def predict_topk(model, tokenizer, text: str, topk: int = 5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]

    top = torch.topk(probs, k=topk)
    rows = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        lbl = model.config.id2label[idx]
        rows.append({
            "model_label": lbl,
            "confidence": float(score),
            "mapped_specialist": map_to_20(lbl),
        })
    return rows


def choose_final(age: int, best):
    best_score = best["confidence"]
    best_mapped = best["mapped_specialist"]

    # Pediatric override: if child and model says Pediatrics -> choose Pediatrician
    if age < 16 and best_mapped == "Pediatrician":
        return "Pediatrician", "pediatric_override"

    if best_score >= CONF_THRESHOLD:
        return best_mapped, "confidence_ok"

    return "General Physician / Family Medicine", "low_confidence_fallback"


def prompt_int(prompt, min_v=None, max_v=None):
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if min_v is not None and v < min_v:
                print(f"Enter >= {min_v}")
                continue
            if max_v is not None and v > max_v:
                print(f"Enter <= {max_v}")
                continue
            return v
        except:
            print("Enter a valid number.")


def prompt_nonempty(prompt):
    while True:
        s = input(prompt).strip()
        if s:
            return s
        print("Cannot be empty.")


def normalize_corrected_label(user_text: str):
    """
    Attempts to map user input to one of SPECIALIST_LABELS_20:
    - Exact match
    - Case-insensitive match
    - Simple contains-based match
    """
    if not user_text:
        return None

    user_text_stripped = user_text.strip()

    # Exact
    if user_text_stripped in SPECIALIST_LABELS_20:
        return user_text_stripped

    # Case-insensitive exact
    lower_map = {x.lower(): x for x in SPECIALIST_LABELS_20}
    if user_text_stripped.lower() in lower_map:
        return lower_map[user_text_stripped.lower()]

    # Contains match (very simple)
    u = user_text_stripped.lower()
    for label in SPECIALIST_LABELS_20:
        if u in label.lower() or label.lower() in u:
            return label

    return user_text_stripped  # keep as-is (we’ll warn)


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("Model loaded.\n")

    ensure_feedback_file()

    print("Specialist Recommender (local tester)")
    print("Type 'exit' at Symptoms to stop. (Ctrl+C also exits)\n")

    try:
        while True:
            age = prompt_int("Age: ", 0, 120)
            gender = prompt_nonempty("Gender: ")
            symptoms = input("Symptoms: ").strip()
            if symptoms.lower() == "exit":
                break
            severity = prompt_int("Severity (1-10): ", 1, 10)
            duration = prompt_nonempty("Duration (e.g., 2 hours / 3 days): ")

            text = build_text(age, gender, symptoms, severity, duration)
            top5 = predict_topk(model, tokenizer, text, topk=5)

            best = top5[0]
            final, rule = choose_final(age, best)

            request_id = str(uuid.uuid4())

            print("\n" + "=" * 90)
            print("INPUT:", text)
            print(f"\nFINAL RECOMMENDATION: {final}")
            print(f"Best: {best['model_label']}  conf={best['confidence']:.3f}  -> {best['mapped_specialist']}  (rule={rule})")

            print("\nTop-5:")
            for i, row in enumerate(top5, 1):
                print(f"{i}. {row['model_label']:<25} conf={row['confidence']:.3f} -> {row['mapped_specialist']}")

            append_jsonl({
                "type": "prediction",
                "request_id": request_id,
                "input": {"age": age, "gender": gender, "symptoms": symptoms, "severity": severity, "duration": duration},
                "text": text,
                "final_recommendation": final,
                "rule": rule,
                "top5": top5,
            })

            fb = input("\nFeedback? (u=thumbs up, d=thumbs down, s=skip): ").strip().lower()
            if fb in ("u", "d"):
                thumbs = 1 if fb == "u" else 0
                corrected = None

                if thumbs == 0:
                    raw_corrected = input(
                        "If wrong, enter corrected specialist (optional, exact label): "
                    ).strip()
                    corrected = normalize_corrected_label(raw_corrected) if raw_corrected else None

                    if corrected and corrected not in SPECIALIST_LABELS_20:
                        print("\n⚠️ Corrected label is not in the official 20-label set.")
                        print("Please use one of these labels next time:\n")
                        for x in SPECIALIST_LABELS_20:
                            print(" -", x)

                append_jsonl({
                    "type": "feedback",
                    "request_id": request_id,
                    "thumbs": thumbs,
                    "corrected_specialist": corrected
                })
                print("✅ Feedback saved.\n")
            else:
                print("⏭️ Skipped feedback.\n")

    except KeyboardInterrupt:
        print("\n🛑 Exiting (Ctrl+C).")

    print("\nDone. Feedback file:", FEEDBACK_PATH)


if __name__ == "__main__":
    main()
