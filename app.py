from __future__ import annotations

import sys
# Conditional patch to bypass older sqlite3 version on Streamlit Cloud (Linux)
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import pandas as pd
import plotly.express as px
import streamlit as st

from database.db_handler import get_chat_history, get_diagnoses, init_db, save_chat, save_diagnosis
from utils.grok_chat import ask_grok, get_model_name
from utils.prediction import predict_disease
from utils.rag_pipeline import build_vector_store, retrieve_context
from utils.report_generator import generate_report

st.set_page_config(page_title="MediMind AI", page_icon="+", layout="wide")
init_db()

PAGES = ["Home", "About", "Services", "Symptom Checker", "AI Chatbot", "Knowledge Base", "History", "Analytics", "Contact"]


def go_to(page_name: str) -> None:
    st.session_state.active_page = page_name
    st.rerun()


if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

st.markdown(
    """
    <style>
    :root {
        --teal:#22c6d8;
        --teal-dark:#12aabd;
        --navy:#172449;
        --text:#6f7782;
        --soft:#f3fcfd;
        --line:#e7edf2;
    }
    .stApp { background:#ffffff; color:var(--text); }
    .main .block-container { max-width:1180px; padding-top:1.2rem; padding-bottom:2rem; }
    section[data-testid="stSidebar"] { display:none; }
    [data-testid="collapsedControl"] { display:none; }
    h1, h2, h3 { color:var(--navy); letter-spacing:0 !important; line-height:1.16; }
    h1 { font-size:clamp(2.4rem, 6vw, 4.6rem) !important; }
    h2 { font-size:clamp(2rem, 4vw, 3.2rem) !important; }
    .topbar {
        background:#fff;
        border-bottom:1px solid var(--line);
        box-shadow:0 2px 18px rgba(23,36,73,.06);
        padding:.45rem 0 .25rem;
        margin:-1.2rem calc(50% - 50vw) 2.5rem;
    }
    .topbar-inner {
        width:min(1120px, calc(100% - 32px));
        min-height:84px;
        margin:0 auto;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:1.5rem;
    }
    .brand {
        display:flex;
        align-items:center;
        gap:.72rem;
        color:var(--teal);
        font-size:2.08rem;
        font-weight:900;
    }
    .brand-mark {
        width:42px;
        height:42px;
        border:4px solid var(--teal);
        border-radius:6px 6px 14px 14px;
        display:grid;
        place-items:center;
        transform:rotate(45deg);
        color:var(--teal);
    }
    .brand-mark span { transform:rotate(-45deg); font-size:1.6rem; line-height:1; }
    .nav-status { color:var(--navy); font-weight:700; font-size:.9rem; text-align:right; }
    .nav-row div[data-testid="column"] { width:auto !important; flex:0 0 auto !important; }
    .nav-row button {
        border:0 !important;
        border-radius:0 !important;
        background:transparent !important;
        color:#151a33 !important;
        box-shadow:none !important;
        font-weight:900 !important;
        padding:1rem .15rem 1.3rem !important;
        min-height:3rem;
    }
    .nav-row button:hover, .nav-row button[kind="primary"] {
        color:var(--teal) !important;
        border-bottom:5px solid var(--teal) !important;
    }
    .hero { padding:2.5rem 0 1rem; border-bottom:1px solid #eef3f6; }
    .eyebrow {
        color:var(--teal);
        font-weight:900;
        text-transform:uppercase;
        border-bottom:4px solid #d7f3f6;
        display:inline-block;
        margin:0 0 .75rem;
    }
    .lead { font-size:1.08rem; line-height:1.75; color:var(--text); }
    .metric-card, .service-card, .info-card {
        border:1px solid var(--line);
        border-radius:8px;
        padding:1.25rem;
        background:#fff;
        min-height:145px;
    }
    .service-card { transition:transform 180ms ease, box-shadow 180ms ease; }
    .service-card:hover { transform:translateY(-4px); box-shadow:0 18px 40px rgba(23,36,73,.08); }
    .number { color:var(--teal); font-weight:900; font-size:2rem; margin-bottom:.2rem; }
    .notice {
        background:#f0fbfd;
        border-left:5px solid var(--teal);
        padding:1rem;
        border-radius:6px;
        color:#25445a;
        margin-bottom:1rem;
    }
    .band {
        background:#f8fdfe;
        border:1px solid var(--line);
        border-radius:8px;
        padding:2rem;
        margin:1.5rem 0;
    }
    div.stButton > button[kind="primary"], div.stDownloadButton > button {
        background:var(--teal) !important;
        border-color:var(--teal) !important;
        color:white !important;
        border-radius:6px !important;
        font-weight:900 !important;
    }
    div.stButton > button[kind="secondary"] {
        border-color:#bdebf0 !important;
        color:var(--navy) !important;
        border-radius:6px !important;
        font-weight:800 !important;
    }
    .footer {
        background:var(--navy);
        color:#c7d4dc;
        padding:1.5rem 2rem;
        border-radius:8px;
        margin-top:2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand"><div class="brand-mark"><span>+</span></div><span>MEDIMIND</span></div>
        <div class="nav-status">AI healthcare diagnosis system<br>Grok model: {get_model_name()}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="nav-row">', unsafe_allow_html=True)
nav_cols = st.columns([1, 1, 1, 1.45, 1.1, 1.3, 1, 1, 1])
for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(page, key=f"nav_{page}", type="primary" if st.session_state.active_page == page else "secondary"):
            go_to(page)
st.markdown("</div>", unsafe_allow_html=True)


def page_home() -> None:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="hero">', unsafe_allow_html=True)
        st.markdown('<p class="eyebrow">MediMind AI</p>', unsafe_allow_html=True)
        st.title("Best AI Medical Guidance For Early Symptom Awareness")
        st.write(
            "Check symptoms, talk to a Grok AI assistant, search your medical PDFs, save diagnosis history, "
            "and generate clean PDF summaries from one Streamlit dashboard."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", "9")
        c2.metric("Storage", "SQLite")
        c3.metric("LLM", "Local")
        st.button("Start Symptom Check", on_click=go_to, args=("Symptom Checker",), type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.image("https://images.unsplash.com/photo-1581056771107-24ca5f033842?auto=format&fit=crop&w=900&q=80", use_column_width=True)

    st.subheader("Core Capabilities")
    cols = st.columns(4)
    items = [
        ("Disease Prediction", "Random Forest + TF-IDF with confidence and urgency."),
        ("AI Chatbot", "Grok replies using the xAI Grok API."),
        ("RAG Knowledge", "Index PDFs with ChromaDB and retrieve medical context."),
        ("Reports", "Download educational PDF summaries with precautions."),
    ]
    for col, (title, text) in zip(cols, items):
        col.markdown(f'<div class="metric-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="band">', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">How It Works</p>', unsafe_allow_html=True)
    st.subheader("From symptoms to a clear next step")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="info-card"><div class="number">01</div><h3>Enter Symptoms</h3><p>Use quick chips or natural language to describe what the patient is feeling.</p></div>', unsafe_allow_html=True)
    s2.markdown('<div class="info-card"><div class="number">02</div><h3>Review AI Result</h3><p>The system estimates a possible condition, confidence, urgency, and precautions.</p></div>', unsafe_allow_html=True)
    s3.markdown('<div class="info-card"><div class="number">03</div><h3>Save Report</h3><p>History, analytics, chat context, and PDF reports keep follow-up organized.</p></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_about() -> None:
    left, right = st.columns([0.9, 1.1])
    with left:
        st.image("https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=900&q=80", use_column_width=True)
    with right:
        st.markdown('<p class="eyebrow">About Us</p>', unsafe_allow_html=True)
        st.title("Best Medical Care For Yourself and Your Family")
        st.write(
            "MediMind AI is a healthcare diagnosis and recommendation system built as one merged project. "
            "It combines the Medinova-style interface with a working Streamlit app, machine learning, local AI chat, "
            "RAG document search, history, analytics, and PDF reports."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown('<div class="info-card"><div class="number">ML</div><b>Random Forest</b></div>', unsafe_allow_html=True)
        c2.markdown('<div class="info-card"><div class="number">AI</div><b>Grok</b></div>', unsafe_allow_html=True)
        c3.markdown('<div class="info-card"><div class="number">DB</div><b>SQLite</b></div>', unsafe_allow_html=True)
        c4.markdown('<div class="info-card"><div class="number">PDF</div><b>Reports</b></div>', unsafe_allow_html=True)


def page_services() -> None:
    st.markdown('<p class="eyebrow">Services</p>', unsafe_allow_html=True)
    st.title("Complete AI Healthcare Toolkit")
    st.write("Nine core modules cover the full MediMind workflow from first symptoms to saved reports.")
    services = [
        ("01", "Symptom Checker", "Interactive symptom entry with confidence and urgency levels."),
        ("02", "Disease Prediction", "Random Forest model trained on Kaggle symptom-to-disease data."),
        ("03", "AI Chatbot", "Replies powered by the xAI Grok API."),
        ("04", "Medical RAG", "PDF upload, chunking, embeddings, ChromaDB storage, and retrieval."),
        ("05", "Diagnosis History", "SQLite persistence for previous checks and conversations."),
        ("06", "Analytics Dashboard", "Plotly charts for conditions, confidence, and usage trends."),
        ("07", "PDF Reports", "Downloadable medical summary reports generated with ReportLab."),
        ("08", "Precautions", "Disease descriptions and safety recommendations from CSV data."),
        ("09", "Local Privacy", "Designed around local models and local SQLite storage."),
    ]
    for row in range(0, len(services), 3):
        cols = st.columns(3)
        for col, (num, title, text) in zip(cols, services[row : row + 3]):
            col.markdown(f'<div class="service-card"><div class="number">{num}</div><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)


def page_symptom_checker() -> None:
    st.title("Symptom Checker")
    st.markdown('<div class="notice">This tool supports awareness only. It does not replace a licensed clinician.</div>', unsafe_allow_html=True)
    suggestions = ["fever", "cough", "headache", "nausea", "fatigue", "stomach pain", "sneezing", "body pain", "chest pain"]
    selected = st.multiselect("Quick symptom chips", suggestions)
    typed = st.text_area("Describe symptoms", value=", ".join(selected), height=130)
    if st.button("Analyze Symptoms", type="primary", disabled=not typed.strip()):
        result = predict_disease(typed)
        save_diagnosis(
            typed,
            result["disease"],
            result["confidence"],
            result["urgency"],
            result["description"],
            "; ".join(result["precautions"]),
        )
        st.session_state.latest_result = result
        st.session_state.latest_symptoms = typed

    result = st.session_state.get("latest_result")
    if result:
        a, b, c = st.columns(3)
        a.metric("Possible condition", result["disease"])
        b.metric("Confidence", f"{int(result['confidence'] * 100)}%")
        c.metric("Urgency", result["urgency"])
        if result.get("is_demo"):
            st.info("Demo mode is active because trained model files were not found. Run notebooks/train_model.py after adding Kaggle data.")
        st.write(result["description"])
        st.subheader("Recommended Precautions")
        for item in result["precautions"]:
            st.write(f"- {item}")
        pdf = generate_report(result, st.session_state.get("latest_symptoms", typed))
        st.download_button("Download PDF Report", data=pdf, file_name="medimind-report.pdf", mime="application/pdf")


def page_chatbot() -> None:
    st.title("AI Chatbot")
    history = get_chat_history()
    for item in history[-12:]:
        with st.chat_message(item["role"]):
            st.write(item["message"])

    prompt = st.chat_input("Ask about symptoms, precautions, or your medical documents")
    if prompt:
        save_chat("user", prompt)
        context = retrieve_context(prompt)
        answer = ask_grok(prompt, context=context, history=history[-8:])
        save_chat("assistant", answer)
        st.rerun()


def page_knowledge_base() -> None:
    st.title("Knowledge Base")
    st.write("Add medical PDF files to `rag/medical_docs`, then build the vector database for RAG answers.")
    if st.button("Build / Refresh Vector Database", type="primary"):
        with st.spinner("Indexing PDF documents..."):
            st.success(build_vector_store())
    query = st.text_input("Test retrieval")
    if query:
        context = retrieve_context(query)
        st.text_area("Retrieved context", context or "No context found yet.", height=220)


def page_history() -> None:
    st.title("History")
    records = get_diagnoses(100)
    if not records:
        st.info("No diagnosis history yet.")
        return
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


def page_analytics() -> None:
    st.title("Analytics Dashboard")
    records = get_diagnoses(200)
    if not records:
        st.info("Analytics will appear after a few symptom checks.")
        return
    df = pd.DataFrame(records)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total checks", len(df))
    c2.metric("Average confidence", f"{df['confidence'].mean() * 100:.0f}%")
    c3.metric("Most common", df["disease"].mode().iloc[0])
    st.plotly_chart(px.histogram(df, x="disease", color="urgency", title="Predicted Conditions"), use_container_width=True)
    st.plotly_chart(px.line(df.sort_values("created_at"), x="created_at", y="confidence", title="Confidence Over Time"), use_container_width=True)


def page_contact() -> None:
    st.markdown('<p class="eyebrow">Contact</p>', unsafe_allow_html=True)
    st.title("Build, Test, and Present MediMind AI")
    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown(
            """
            <div class="info-card">
            <h3>Project Stack</h3>
            <p>Streamlit, Scikit-learn, Grok API, LangChain, ChromaDB, SQLite, Plotly, ReportLab, Pandas, and NumPy.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        with st.form("contact_form"):
            name = st.text_input("Your name")
            email = st.text_input("Email address")
            topic = st.selectbox("Topic", ["Project demo", "Dataset setup", "Grok support", "RAG pipeline"])
            message = st.text_area("Message")
            sent = st.form_submit_button("Send Message", type="primary")
        if sent:
            st.success(f"Thanks {name or 'there'}, your {topic.lower()} message was saved in demo mode.")


ROUTES = {
    "Home": page_home,
    "About": page_about,
    "Services": page_services,
    "Symptom Checker": page_symptom_checker,
    "AI Chatbot": page_chatbot,
    "Knowledge Base": page_knowledge_base,
    "History": page_history,
    "Analytics": page_analytics,
    "Contact": page_contact,
}

ROUTES[st.session_state.active_page]()
st.markdown('<div class="footer">MediMind AI is an educational healthcare project and is not a replacement for professional diagnosis.</div>', unsafe_allow_html=True)
