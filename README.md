# MediMind AI - Healthcare Diagnosis & Recommendation System

MediMind AI is a merged Streamlit healthcare assistant that combines the Medinova-style medical UI with symptom prediction, xAI Grok API chatbot, PDF knowledge retrieval, SQLite history, analytics, and downloadable medical summary reports.

## Features

- Medinova-inspired UI with teal branding, white top navigation, navy headings, medical imagery, service cards, and clean page sections
- 9 Streamlit pages: Home, About, Services, Symptom Checker, AI Chatbot, Knowledge Base, History, Analytics, Contact
- Session-state navigation using `st.session_state.active_page` and top navigation buttons
- Disease prediction with Random Forest + TF-IDF after training
- Demo prediction fallback when model files are not available
- Chatbot integration powered by the xAI Grok API (configured with `grok-2-1212`)
- LangChain + ChromaDB RAG pipeline for medical PDFs
- SQLite diagnosis and chat history
- Plotly analytics
- ReportLab PDF report generation

## Setup

```powershell
cd healthcare-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the app at `http://localhost:8501`.

## Train the ML model

Place Kaggle files in `datasets/`:

- `dataset.csv`
- `symptom_Description.csv`
- `symptom_precaution.csv`
- `Symptom-severity.csv`

Then run:

```powershell
python notebooks/train_model.py
```

This creates:

- `models/disease_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/label_encoder.pkl`

## Grok API Setup

To use the AI Chatbot feature, you need an API key from xAI.

1. Create an account at the [xAI Console](https://console.x.ai/) and generate an API key.
2. Set the `XAI_API_KEY` environment variable before running the application:

```powershell
# PowerShell (Windows)
$env:XAI_API_KEY="your-api-key-here"

# CMD (Windows)
set XAI_API_KEY=your-api-key-here

# Bash (Linux/macOS)
export XAI_API_KEY="your-api-key-here"
```

## RAG PDFs

Put medical PDFs in `rag/medical_docs/`, then open the Knowledge Base page and click **Build / Refresh Vector Database**.

## Disclaimer

This project is for educational support only. It is not a medical diagnosis tool and should not replace professional medical care.
