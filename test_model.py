import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- Model ----
MODEL_ID = "OzzeY72/biobert-medical-specialities"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.\n")

# ---- Your 20-label mapping ----
MODEL_TO_20 = {
    "Cardiology": "Cardiologist",
    "Neurology": "Neurologist",
    "Respiratory": "Pulmonologist",
    "Gastroenterology": "Gastroenterologist",
    "Dermatology": "Dermatologist",
    "Otorhinolaryngology": "Otolaryngologist (Ear, Nose and Throat Specialist)",
    "Orthopedics": "Orthopedic Surgeon",
    "Urology": "Urologist",
    "Obstetrics": "Gynecology / Obstetrician",
    "Gynecology": "Gynecology / Obstetrician",
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
}

def map_to_20(model_label):
    return MODEL_TO_20.get(model_label, "General Physician / Family Medicine")

# ---- Test Input ----
age = 34
gender = "male"
symptoms = "pain in left knee joint and back pain, headache"
severity = 8
duration = "2 hours"

# Format input text
text = (
    f"Age: {age} | Gender: {gender} | "
    f"Severity: {severity}/10 | Duration: {duration} | "
    f"Symptoms: {symptoms}"
)

print("Input:")
print(text)
print("\nRunning inference...\n")

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    confidence, pred_id = torch.max(probs, dim=1)

model_label = model.config.id2label[pred_id.item()]
mapped_label = map_to_20(model_label)

print("Raw Model Label:", model_label)
print("Mapped Specialist:", mapped_label)
print("Confidence:", round(confidence.item(), 4))
