import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "OzzeY72/biobert-medical-specialities"

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

def map_to_20(model_label: str) -> str:
    return MODEL_TO_20.get(model_label, "General Physician / Family Medicine")

def build_text(age, gender, symptoms, severity, duration):
    return (
        f"Age: {age} | Gender: {gender} | Severity: {severity}/10 | "
        f"Duration: {duration} | Symptoms: {symptoms}"
    )

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.\n")

def predict(text: str, topk: int = 5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]

    top = torch.topk(probs, k=topk)
    results = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        label = model.config.id2label[idx]
        results.append((label, score, map_to_20(label)))
    return results

# ---- Quick test set ----
tests = [
    dict(age=34, gender="male", symptoms="chest tightness and shortness of breath", severity=8, duration="2 hours"),
    dict(age=28, gender="female", symptoms="itchy red rash on arms for 3 days", severity=4, duration="3 days"),
    dict(age=45, gender="male", symptoms="sudden severe headache, vomiting, confusion", severity=9, duration="1 hour"),
    dict(age=30, gender="male", symptoms="ear pain and discharge", severity=6, duration="2 days"),
    dict(age=5, gender="male", symptoms="fever and cough", severity=6, duration="1 day"),
]

CONF_THRESHOLD = 0.55  # if below this, recommend GP

for t in tests:
    text = build_text(**t)
    print("=" * 90)
    print("INPUT:", text)
    preds = predict(text, topk=5)

    best_label, best_score, best_mapped = preds[0]
    # Pediatric override
    if t["age"] < 16 and best_mapped == "Pediatrician":
        final = "Pediatrician"
    else:
        final = best_mapped if best_score >= CONF_THRESHOLD else "General Physician / Family Medicine"


    print(f"\nFINAL RECOMMENDATION: {final}  (best_conf={best_score:.3f}, model_label={best_label})")
    print("\nTop-5:")
    for i, (lbl, sc, mapped) in enumerate(preds, 1):
        print(f"{i}. {lbl:25s}  conf={sc:.3f}  -> {mapped}")
