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
    "Pediatrician"
]

# Model labels (from the model card)
# None, Cardiology, Hematology, Oncology, Endocrinology, Respiratory, Allergy, Dermatology,
# Nephrology, Gastroenterology, Rheumatology, Otorhinolaryngology, ... Psychiatry, ...
# Obstetrics, Gynecology, ... Orthopedics, Neurology, Urology, ... Ophthalmology, Pediatrics, ...
# (See model card for full list) :contentReference[oaicite:1]{index=1}

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

    # Closest available mapping (model doesn't have explicit "Infectious Disease")
    "Microbiology": "Infectious Disease Specialist",
}

def map_to_20(model_label: str) -> str:
    return MODEL_TO_20.get(model_label, "General Physician / Family Medicine")
