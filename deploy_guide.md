# Streamlit Cloud Deployment & Grok Setup Guide

This guide details the step-by-step process to deploy your **MediMind AI** application to Streamlit Community Cloud and integrate xAI's Grok API for chatbot functionality.

---

## Step 1: Obtain your Grok API Key

To use the chatbot feature, you must get an API key from the official xAI console.

1. Go to the [xAI Console](https://console.x.ai/) and sign in or register.
2. Navigate to **API Keys** and generate a new key.
3. Copy the key value (it starts with `xai-`). Save it securely; you will need it for local configuration and deployment.

---

## Step 2: Push your Project to GitHub

Streamlit Community Cloud deploys apps directly from GitHub repositories.

1. **Initialize Git** (if not already done):
   ```bash
   git init
   ```
2. **Add and Commit** all files:
   ```bash
   git add .
   git commit -m "Configure Grok API and prepare Streamlit deployment"
   ```
3. **Create a GitHub Repository**:
   - Go to [GitHub](https://github.com/) and create a new public or private repository named `healthcare-ai` or similar.
4. **Push the Code**:
   ```bash
   git remote add origin https://github.com/yourusername/healthcare-ai.git
   git branch -M main
   git push -u origin main
   ```

---

## Step 3: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click the **"New app"** button in the top right.
3. Configure the deployment details:
   - **Repository**: Select your `healthcare-ai` repository from the dropdown.
   - **Branch**: Select `main`.
   - **Main file path**: Set this to `app.py`.
4. Click on **"Advanced settings..."** (before deploying).
5. In the **Secrets** text area, paste your Grok API Key in TOML format:
   ```toml
   XAI_API_KEY = "your-grok-api-key-here"
   ```
   *(Optional)* If you want to use a specific Grok model (such as `grok-2-mini` instead of the default `grok-2-1212`), you can also add:
   ```toml
   XAI_MODEL = "grok-2-mini"
   ```
6. Click **"Save"**.
7. Click **"Deploy!"**. Streamlit will set up the container, install packages listed in `requirements.txt` (including the `pysqlite3-binary` patch), and launch your app.

---

## Step 4: Local Configuration (Optional)

To run the application locally with Grok API:

1. **Set the environment variable** before starting the app:
   - **Windows PowerShell**:
     ```powershell
     $env:XAI_API_KEY="your-grok-api-key-here"
     ```
   - **Windows Command Prompt (CMD)**:
     ```cmd
     set XAI_API_KEY=your-grok-api-key-here
     ```
   - **Bash / macOS / Linux**:
     ```bash
     export XAI_API_KEY="your-grok-api-key-here"
     ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Important Architectural Notes for Streamlit Cloud

### 1. File & Database Persistence
Streamlit Community Cloud operates on **ephemeral containers**. Any data saved during user sessions will eventually be reset when the container is rebuilt or goes to sleep:
*   **Diagnosis & Chat History (SQLite)**: Changes are stored in the local SQLite file. This history will reset whenever the application container restarts.
*   **Knowledge Base PDFs (RAG)**: If you upload new PDFs or build the vector store dynamically on the deployed website, those files will vanish on container restart. 

> [!TIP]
> **Best Practice for RAG**: Place your initial target medical PDFs inside the `rag/medical_docs/` folder *before* pushing your repository to GitHub. During the build phase on Streamlit Cloud, they will be bundled, and you can pre-index them or let the user click **Build / Refresh Vector Database** to index them dynamically.

### 2. SQLite Version Override
ChromaDB requires SQLite version 3.35.0 or higher. Since Streamlit Community Cloud uses an older system version of SQLite, our codebase includes a conditional auto-patch at the top of [app.py](file:///C:/healthcare-ai/app.py):
```python
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
```
This requires `pysqlite3-binary` to be installed on Linux hosts, which we configured using environment markers in [requirements.txt](file:///C:/healthcare-ai/requirements.txt):
```text
pysqlite3-binary; sys_platform == 'linux'
```
This is fully automated and requires no additional action from you!
