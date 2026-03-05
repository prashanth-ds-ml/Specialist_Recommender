import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "OzzeY72/biobert-medical-specialities"
MAX_LEN = 128

# -------------------------
# Label Mapping
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
    def strip_text(cls, v):
        return v.strip()

class RecommendResponse(BaseModel):
    recommended_specialist: str
    confidence: float
    model_label: str

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
        text = build_text(req.age, req.gender, req.symptoms, req.severity, req.duration)

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

        # Pediatric override
        if req.age < 16 and specialist == "Pediatrician":
            specialist = "Pediatrician"

        return RecommendResponse(
            recommended_specialist=specialist,
            confidence=float(conf.item()),
            model_label=model_label
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
