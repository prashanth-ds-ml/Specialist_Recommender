import re
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "OzzeY72/biobert-medical-specialities"
MAX_LEN = 128

# -------------------------
# Label Mapping (Model -> Your Specialist Names)
# -------------------------
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

def map_to_specialist(model_label: str) -> str:
    return MODEL_TO_20.get(model_label, "General Physician / Family Medicine")

# -------------------------
# Layered Symptom Normalization (3 Layers)
# -------------------------

# LAYER 1: Canonical terms (keep this list small & meaningful; expand over time)
CANONICAL_TERMS = {
    # ENT / URTI
    "sinusitis", "nasal congestion", "runny nose", "sore throat", "ear pain", "ear discharge",
    # Respiratory
    "shortness of breath", "wheezing", "cough",
    # Neuro
    "headache", "dizziness", "vertigo", "seizure", "numbness", "weakness",
    # GI
    "abdominal pain", "diarrhea", "constipation", "vomiting", "nausea", "acid reflux",
    # Urinary
    "frequent urination", "burning urination", "blood in urine",
    # Skin
    "rash", "itching",
    # MSK
    "joint pain", "back pain",
    # Mental health
    "anxiety", "depression", "insomnia",
    # Endocrine
    "high blood sugar", "low blood sugar", "thyroid problem",
    # General
    "fever", "fatigue"
}

# LAYER 2: Aliases/Synonyms -> canonical term
# Use lowercase keys. Keep values as canonical terms or strong medical phrases.
ALIASES_TO_CANONICAL = {
    # India-common phrasing
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
    # Key one: sinus (ambiguous) → we handle with Layer 3 too, but alias helps
    "sinus": "sinusitis"
}

# Precompile alias regex for whole-word/phrase replacement
# We replace longer phrases first (avoid partial replacement issues)
_ALIAS_KEYS = sorted(ALIASES_TO_CANONICAL.keys(), key=len, reverse=True)
_ALIAS_PATTERNS = [(re.compile(rf"\b{re.escape(k)}\b", flags=re.IGNORECASE), ALIASES_TO_CANONICAL[k]) for k in _ALIAS_KEYS]

# LAYER 3: Phrase patterns (regex normalization for common free-text structures)
PHRASE_PATTERNS = [
    # Urinary
    (re.compile(r"\bpain(ful)?\s+urination\b", re.IGNORECASE), "burning urination"),
    (re.compile(r"\bburning\s+(while\s+)?urinating\b", re.IGNORECASE), "burning urination"),
    (re.compile(r"\bfrequent\s+urination\b", re.IGNORECASE), "frequent urination"),
    # Respiratory
    (re.compile(r"\bshort\s*ness\s+of\s+breath\b", re.IGNORECASE), "shortness of breath"),
    (re.compile(r"\bdifficulty\s+breathing\b", re.IGNORECASE), "shortness of breath"),
    # ENT
    (re.compile(r"\bpost\s+nasal\s+drip\b", re.IGNORECASE), "post nasal drip"),
    # GI
    (re.compile(r"\bstomach\s+burn(ing)?\b", re.IGNORECASE), "acid reflux"),
]

def _basic_clean(text: str) -> str:
    # Lowercase + normalize punctuation to spaces
    text = text.lower()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s/.-]+", " ", text)  # keep simple separators
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_symptoms(symptoms: str) -> str:
    """
    3-layer normalization:
    1) Basic cleaning
    2) Alias replacement -> canonical phrases
    3) Phrase-pattern normalization
    4) Light enrichment for ambiguous terms (e.g., sinus)
    """
    s = _basic_clean(symptoms)

    # Layer 2: aliases -> canonical
    for pat, repl in _ALIAS_PATTERNS:
        s = pat.sub(repl, s)

    # Layer 3: phrase patterns
    for pat, repl in PHRASE_PATTERNS:
        s = pat.sub(repl, s)

    # Extra enrichment (still "general", not specialist-specific):
    # If user says sinus but not explicit ENT features, add common related phrases
    if re.search(r"\bsinus\b", symptoms.lower()) or re.search(r"\bsinusitis\b", s):
        # Helps the classifier lean ENT without hardcoding ENT
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

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="Specialist Recommender API", version="1.0.0")

# -------------------------
# Request / Response Models
# -------------------------
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
    normalized_symptoms: str  # helpful for debugging and QA

# -------------------------
# Load Model Once
# -------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")

DEVICE = "cpu"
model.to(DEVICE)

# -------------------------
# Routes
# -------------------------
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
        # 3-layer normalization applied here
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

        # Pediatric override (keep as your existing rule)
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