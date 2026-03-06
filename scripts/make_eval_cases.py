import json
import random
from pathlib import Path

OUT_PATH = Path("eval_cases.jsonl")
NUM_CASES = 100
SEED = 42

random.seed(SEED)

DURATIONS = [
    "1 day", "2 days", "3 days", "4 days", "5 days", "1 week", "10 days",
    "2 weeks", "3 weeks", "1 month", "2 months"
]

# Specialists (same strings your API outputs via map_to_specialist)
SPECIALISTS = [
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
]

# Symptom phrase pools per specialist (kept simple & realistic)
SYMPTOMS_BY_SPECIALIST = {
    "Cardiologist": [
        "chest pain on exertion, shortness of breath, sweating",
        "palpitations, dizziness, chest tightness",
        "shortness of breath, swelling in legs, fatigue",
        "chest discomfort, left arm pain, nausea"
    ],
    "Neurologist": [
        "severe headache, nausea, sensitivity to light",
        "dizziness, vertigo, imbalance while walking",
        "numbness in one arm, weakness, difficulty speaking",
        "seizure episode, confusion afterwards"
    ],
    "Pulmonologist": [
        "persistent cough, wheezing, shortness of breath",
        "shortness of breath on walking, dry cough, fatigue",
        "cough with phlegm, fever, chest congestion",
        "wheezing, chest tightness, breathlessness at night"
    ],
    "Gastroenterologist": [
        "abdominal pain, acidity, nausea after meals",
        "loose motions, abdominal cramps, dehydration",
        "constipation, abdominal discomfort, bloating",
        "vomiting, nausea, stomach pain"
    ],
    "Dermatologist": [
        "itchy rash on arms, redness, dry skin",
        "acne on face, painful pimples, oily skin",
        "skin allergy, itching, hives after food",
        "scaly patches on scalp, itching"
    ],
    "Otolaryngologist (Ear, Nose and Throat Specialist)": [
        "sinus, headache, nasal congestion, facial pressure",
        "sore throat, fever, difficulty swallowing",
        "ear pain, ear discharge, reduced hearing",
        "runny nose, blocked nose, sneezing"
    ],
    "Orthopedic Surgeon": [
        "knee pain, swelling after walking, stiffness",
        "back pain, shooting pain down leg, numbness",
        "shoulder pain, reduced range of motion",
        "joint pain, stiffness in morning, swelling"
    ],
    "Urologist": [
        "burning urination, frequent urination, lower abdominal pain",
        "blood in urine, flank pain, nausea",
        "difficulty urinating, weak urine stream, frequent urination at night",
        "pain while urinating, urgency, fever"
    ],
    "Gynecologist / Obstetrician": [
        "irregular periods, lower abdominal pain, mood changes",
        "heavy menstrual bleeding, fatigue, dizziness",
        "missed period, nausea, vomiting, breast tenderness",
        "vaginal itching, discharge, burning sensation"
    ],
    "Psychiatrist": [
        "anxiety, palpitations, restlessness, insomnia",
        "depression, low mood, loss of interest, fatigue",
        "insomnia, racing thoughts, irritability",
        "panic attacks, sweating, fear, shortness of breath"
    ],
    "Ophthalmologist (Eye Specialist)": [
        "eye pain, redness, watery eyes, sensitivity to light",
        "blurred vision, headache, eye strain while reading",
        "itchy eyes, watering, sneezing",
        "sudden vision changes, floaters, flashes"
    ],
    "Endocrinologist": [
        "high blood sugar, frequent urination, increased thirst",
        "low blood sugar, sweating, dizziness, shaking",
        "thyroid problem, weight gain, fatigue, constipation",
        "thyroid problem, weight loss, palpitations, anxiety"
    ],
    "Nephrologist (Kidney Specialist)": [
        "swelling in legs, fatigue, decreased urine output",
        "high blood pressure, frothy urine, weakness",
        "flank pain, fever, urinary tract infection symptoms",
        "blood in urine, fatigue, loss of appetite"
    ],
    "Oncologist": [
        "unexplained weight loss, persistent fatigue, loss of appetite",
        "lump in neck, night sweats, fever",
        "persistent pain, unexplained bleeding, weakness",
        "chronic cough, weight loss, chest discomfort"
    ],
    "Rheumatologist": [
        "joint pain, stiffness in morning, swelling in fingers",
        "back pain, stiffness, improves with movement",
        "joint pain, fatigue, rash, mouth ulcers",
        "multiple joint pain, swelling, fever"
    ],
    "Hematologist": [
        "fatigue, pale skin, shortness of breath on exertion",
        "easy bruising, bleeding gums, weakness",
        "fever, night sweats, swollen lymph nodes",
        "recurrent infections, fatigue, weight loss"
    ],
    "Allergist / Immunologist": [
        "sneezing, runny nose, itchy eyes, dust allergy",
        "skin hives, itching after food, swelling",
        "wheezing, shortness of breath after exposure to smoke",
        "seasonal allergy, nasal congestion, cough"
    ],
    "Pediatrician": [
        "child has fever, cough, runny nose",
        "child has vomiting, loose motions, dehydration",
        "child has rash, fever, itching",
        "baby has poor feeding, fever, irritability"
    ],
    "General Physician / Family Medicine": [
        "fever, fatigue, body aches",
        "headache, mild fever, sore throat",
        "nausea, fatigue, loss of appetite",
        "dizziness, weakness, tiredness"
    ],
}

def pick_gender_for_specialist(spec: str) -> str:
    # Keep it simple: mostly balanced; gyn tends to female
    if spec == "Gynecologist / Obstetrician":
        return random.choice(["f", "female", "woman"])
    return random.choice(["m", "male", "f", "female"])

def pick_age_for_specialist(spec: str) -> int:
    if spec == "Pediatrician":
        return random.randint(1, 15)
    if spec == "Gynecologist / Obstetrician":
        return random.randint(18, 45)
    if spec in ["Cardiologist", "Nephrologist (Kidney Specialist)", "Oncologist", "Hematologist"]:
        return random.randint(35, 75)
    return random.randint(16, 70)

def pick_severity(spec: str) -> int:
    # mild/moderate by default; some specialties skew higher
    if spec in ["Cardiologist", "Oncologist", "Hematologist"]:
        return random.randint(5, 9)
    return random.randint(2, 8)

def make_case(case_idx: int, spec: str) -> dict:
    case_id = f"CASE_{case_idx:04d}"
    age = pick_age_for_specialist(spec)
    gender = pick_gender_for_specialist(spec)
    symptoms = random.choice(SYMPTOMS_BY_SPECIALIST[spec])
    severity = pick_severity(spec)
    duration = random.choice(DURATIONS)

    # Notes are helpful to testers but optional
    notes = f"Designed to resemble a typical {spec} presentation."

    return {
        "case_id": case_id,
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "severity": severity,
        "duration": duration,
    }

def main():
    # Create a balanced list: repeat specialists to reach NUM_CASES
    specs = []
    while len(specs) < NUM_CASES:
        specs.extend(SPECIALISTS)
    specs = specs[:NUM_CASES]
    random.shuffle(specs)

    # Write JSONL
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, spec in enumerate(specs, start=1):
            record = make_case(i, spec)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {NUM_CASES} cases to {OUT_PATH.resolve()} (seed={SEED})")

if __name__ == "__main__":
    main()