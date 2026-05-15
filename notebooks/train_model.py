from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "datasets" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"


def row_to_text(row: pd.Series) -> str:
    symptoms = [str(value).replace("_", " ").strip() for value in row.dropna().tolist()[1:] if str(value).strip()]
    return " ".join(symptoms)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Place Kaggle dataset.csv in healthcare-ai/datasets before training.")

    df = pd.read_csv(DATASET_PATH)
    disease_col = df.columns[0]
    X_text = df.apply(row_to_text, axis=1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[disease_col].astype(str))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(X_text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test), target_names=encoder.classes_))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "disease_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")
    print("Models saved to healthcare-ai/models.")


if __name__ == "__main__":
    main()
