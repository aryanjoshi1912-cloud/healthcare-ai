from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "datasets"

DEMO_DISEASES = {
    "common cold": {
        "keywords": {"cough", "sneezing", "runny nose", "throat", "mild fever", "congestion"},
        "description": "A usually mild viral upper-respiratory infection that commonly improves with rest and fluids.",
        "precautions": ["Rest", "Drink warm fluids", "Use steam inhalation", "Seek care if fever or breathing worsens"],
    },
    "migraine": {
        "keywords": {"headache", "nausea", "light sensitivity", "vomiting", "blurred vision"},
        "description": "A neurological headache pattern that may include nausea and sensitivity to light or sound.",
        "precautions": ["Rest in a dark room", "Hydrate", "Avoid known triggers", "Consult a clinician for recurrent attacks"],
    },
    "gastritis": {
        "keywords": {"stomach pain", "acidity", "nausea", "vomiting", "burning", "indigestion"},
        "description": "Irritation of the stomach lining that can cause upper abdominal pain, nausea, or burning.",
        "precautions": ["Avoid spicy food", "Eat smaller meals", "Avoid alcohol", "Discuss medication options with a doctor"],
    },
    "influenza": {
        "keywords": {"high fever", "body pain", "fatigue", "cough", "chills", "weakness"},
        "description": "A contagious respiratory viral illness that can cause fever, cough, body pain, and fatigue.",
        "precautions": ["Rest", "Hydrate", "Isolate while febrile", "Seek urgent care for breathing difficulty"],
    },
}


def _load_pickle(name: str):
    path = MODEL_DIR / name
    return joblib.load(path) if path.exists() else None


def load_description_data() -> tuple[dict[str, str], dict[str, list[str]]]:
    descriptions: dict[str, str] = {}
    precautions: dict[str, list[str]] = {}

    desc_path = DATASET_DIR / "symptom_Description.csv"
    precaution_path = DATASET_DIR / "symptom_precaution.csv"

    if desc_path.exists():
        df = pd.read_csv(desc_path)
        disease_col, desc_col = df.columns[:2]
        descriptions = dict(zip(df[disease_col].astype(str).str.lower(), df[desc_col].astype(str)))

    if precaution_path.exists():
        df = pd.read_csv(precaution_path)
        disease_col = df.columns[0]
        for _, row in df.iterrows():
            values = [str(v) for v in row.iloc[1:].dropna().tolist() if str(v).strip()]
            precautions[str(row[disease_col]).lower()] = values

    for disease, data in DEMO_DISEASES.items():
        descriptions.setdefault(disease, data["description"])
        precautions.setdefault(disease, data["precautions"])
    return descriptions, precautions


def urgency_from_confidence(confidence: float, symptoms: str) -> str:
    red_flags = {"chest pain", "breathing", "unconscious", "seizure", "stroke", "severe bleeding"}
    if any(flag in symptoms.lower() for flag in red_flags):
        return "Emergency"
    if confidence >= 0.75:
        return "Moderate"
    return "Low"


def predict_disease(symptoms: str) -> dict:
    symptoms = symptoms.strip()
    descriptions, precautions = load_description_data()

    model = _load_pickle("disease_model.pkl")
    vectorizer = _load_pickle("tfidf_vectorizer.pkl")
    encoder = _load_pickle("label_encoder.pkl")

    if model and vectorizer and encoder:
        vector = vectorizer.transform([symptoms])
        pred = model.predict(vector)[0]
        disease = str(encoder.inverse_transform([pred])[0])
        proba = max(model.predict_proba(vector)[0]) if hasattr(model, "predict_proba") else 0.7
    else:
        words = set(symptoms.lower().replace(",", " ").split())
        scores = {}
        for disease, data in DEMO_DISEASES.items():
            keyword_hits = sum(1 for keyword in data["keywords"] if keyword in symptoms.lower() or keyword in words)
            scores[disease] = keyword_hits
        disease = max(scores, key=scores.get) if any(scores.values()) else "common cold"
        proba = min(0.92, 0.48 + (scores.get(disease, 0) * 0.13))

    disease_key = disease.lower()
    return {
        "disease": disease.title(),
        "confidence": round(float(proba), 2),
        "urgency": urgency_from_confidence(float(proba), symptoms),
        "description": descriptions.get(disease_key, "Description is not available yet. Add Kaggle description data to enrich this result."),
        "precautions": precautions.get(disease_key, ["Consult a qualified healthcare professional for personalized advice."]),
        "is_demo": not (model and vectorizer and encoder),
    }
