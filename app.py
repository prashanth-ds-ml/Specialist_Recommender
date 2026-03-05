import re
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "OzzeY72/biobert-medical-specialities"
MAX_LEN = 128

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
}

def map_to_specialist(model_label: str) -> str:
    return MODEL_TO_20.get(model_label, "General Physician / Family Medicine")

CANONICAL_TERMS = {
    "sinusitis", "nasal congestion", "runny nose", "sore throat", "ear pain", "ear discharge",
    "shortness of breath", "wheezing", "cough",
    "headache", "dizziness", "vertigo", "seizure", "numbness", "weakness",
    "abdominal pain", "diarrhea", "constipation", "vomiting", "nausea", "acid reflux",
    "frequent urination", "burning urination", "blood in urine",
    "rash", "itching",
    "joint pain", "back pain",
    "anxiety", "depression", "insomnia",
    "high blood sugar", "low blood sugar", "thyroid problem",
    "fever", "fatigue"
}

ALIASES_TO_CANONICAL = {
    "loose motions": "diarrhea",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "giddiness": "dizziness",
    "gas": "bloating",
    "acidity": "acid reflux",
    "heart burn": "acid reflux",
    "heartburn": "acid reflux",
    "bp": "high blood pressure",
    "sugar": "high blood sugar",
    "uti": "urinary tract infection",
    "urine infection": "urinary tract infection",
    "breathlessness": "shortness of breath",
    "breathless": "shortness of breath",
    "blocked nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "running nose": "runny nose",
    "cold": "runny nose",
    "coughing": "cough",
    "vomit": "vomiting",
    "throwing up": "vomiting",
    "pimples": "acne",
    "itchy": "itching",
    "pain while urinating": "burning urination",
    "burning urine": "burning urination",
    "pee frequently": "frequent urination",
    "urinating frequently": "frequent urination",
    "earache": "ear pain",
    "sinus": "sinusitis"
}

_ALIAS_KEYS = sorted(ALIASES_TO_CANONICAL.keys(), key=len, reverse=True)
_ALIAS_PATTERNS = [
    (re.compile(rf"\b{re.escape(k)}\b", flags=re.IGNORECASE), ALIASES_TO_CANONICAL[k])
    for k in _ALIAS_KEYS
]

PHRASE_PATTERNS = [
    (re.compile(r"\bpain(ful)?\s+urination\b", re.IGNORECASE), "burning urination"),
    (re.compile(r"\bburning\s+(while\s+)?urinating\b", re.IGNORECASE), "burning urination"),
    (re.compile(r"\bfrequent\s+urination\b", re.IGNORECASE), "frequent urination"),
    (re.compile(r"\bshort\s*ness\s+of\s+breath\b", re.IGNORECASE), "shortness of breath"),
    (re.compile(r"\bdifficulty\s+breathing\b", re.IGNORECASE), "shortness of breath"),
    (re.compile(r"\bpost\s+nasal\s+drip\b", re.IGNORECASE), "post nasal drip"),
    (re.compile(r"\bstomach\s+burn(ing)?\b", re.IGNORECASE), "acid reflux"),
]

def _basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s/.-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_symptoms(symptoms: str) -> str:
    s = _basic_clean(symptoms)

    for pat, repl in _ALIAS_PATTERNS:
        s = pat.sub(repl, s)

    for pat, repl in PHRASE_PATTERNS:
        s = pat.sub(repl, s)

    if re.search(r"\bsinus\b", symptoms.lower()) or re.search(r"\bsinusitis\b", s):
        if "nasal congestion" not in s:
            s += " nasal congestion"
        if "facial pressure" not in s:
            s += " facial pressure"

    return s.strip()

def build_text(age, gender, symptoms, severity, duration):
    return (
        f"Age: {age} | Gender: {gender} | Severity: {severity}/10 | "
        f"Duration: {duration} | Symptoms: {symptoms}"
    )

app = FastAPI(title="Specialist Recommender API", version="1.0.0")

# ✅ ADD THIS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., min_length=1, max_length=32)
    symptoms: str = Field(..., min_length=1, max_length=2000)
    severity: int = Field(..., ge=1, le=10)
    duration: str = Field(..., min_length=1, max_length=64)

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, v):
        return v.strip().lower()

    @field_validator("symptoms", "duration")
    @classmethod
    def strip_and_validate_nonempty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty")
        return v

class RecommendResponse(BaseModel):
    recommended_specialist: str
    confidence: float
    model_label: str
    normalized_symptoms: str

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")

DEVICE = "cpu"
model.to(DEVICE)

@app.get("/")
def root():
    return {
        "message": "Specialist Recommender API is running",
        "endpoints": {
            "health": "GET /health",
            "recommend": "POST /recommend"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        norm_symptoms = normalize_symptoms(req.symptoms)
        text = build_text(req.age, req.gender, norm_symptoms, req.severity, req.duration)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)[0]
            conf, pred_id = torch.max(probs, dim=0)

        model_label = model.config.id2label[int(pred_id.item())]
        specialist = map_to_specialist(model_label)

        if req.age < 16 and specialist == "Pediatrician":
            specialist = "Pediatrician"

        return RecommendResponse(
            recommended_specialist=specialist,
            confidence=float(conf.item()),
            model_label=model_label,
            normalized_symptoms=norm_symptoms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))