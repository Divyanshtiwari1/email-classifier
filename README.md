Email Classifier

Lightweight Python project to classify emails and update email status; supports a classical ML backend and an LLM backend (LangChain + Ollama).

Files

email_classification_div.py — classification runner

email_status_div.py — status updater

streamlit_div.py — Streamlit UI

Usage

Use email_classification_div.py to generate category predictions; choose backend "sklearn" or "llm" (Ollama).

Use email_status_div.py to compute or update status labels.

Use streamlit_div.py to interact via a web UI.

Data format
CSV with columns: email_id, subject, body, sender, date

Maintainer: Divyansh Tiwari


output ss - (https://github.com/user-attachments/assets/95895e70-279c-4d7f-8ba2-d21c3567db9c)

