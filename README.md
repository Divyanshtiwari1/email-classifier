# Email Classifier

A lightweight Python project for classifying emails into categories and tracking email status, with a Streamlit UI — now with an example integration using **LangChain + Ollama** for structured LLM-driven classification.

---

## Overview

This repo contains scripts to classify emails and update their status, and a Streamlit app to interact with the system. The project can use a classical ML model (scikit-learn) or a modern LLM-based classifier (LangChain + Ollama) for higher-quality, promptable classification.

This README now includes ready-to-copy contents for:

* `requirements.txt`
* `data/example_emails.csv` (sample dataset)
* Badge & screenshot suggestions for the GitHub `README.md`

---

## Repo structure

* `email_classification_div.py` — performs email classification. Can call either a saved classical ML model or an LLM-based classifier.
* `email_status_div.py` — computes/updates email status labels.
* `streamlit_div.py` — Streamlit app for upload, preview, and running the classifier.
* `models/` (suggested) — place trained classical models or serialized artifacts here.
* `data/` (suggested) — example datasets and templates.
* `requirements.txt` — dependency list (content shown below).
* `data/example_emails.csv` — small example dataset (content shown below).
* `docs/demo_screenshot.png` — demo screenshot (copy the file into your repo at `docs/demo_screenshot.png`).

---

## requirements.txt (copy this file into the repo)

```
langchain>=0.0.0
langchain-ollama>=0.0.0
ollama>=0.0.0
pydantic>=1.10
pandas>=2.0
scikit-learn>=1.2
joblib>=1.2
streamlit>=1.20
python-dotenv>=1.0
```

> Note: Replace version pins with the exact versions you use. `langchain-ollama` package name may vary depending on the distribution — if you installed via `pip install langchain-ollama` use that, otherwise adapt.

---

## Example dataset: `data/example_emails.csv`

Create a folder `data/` and add the following CSV as `example_emails.csv`.

```
email_id,subject,body,sender,date
1,Welcome to our service,Hi there — welcome! We're glad to have you.,noreply@example.com,2025-09-01
2,Your invoice is ready,Please find your invoice attached for August.,billing@example.com,2025-08-30
3,Discounts this weekend,Flash sale — up to 50% off on selected items.,promos@example.com,2025-09-05
4,Meeting rescheduled,Can we move our 3pm sync to 4pm? Regards, manager@example.com,2025-09-06
5,Account suspicious activity,We detected a sign-in from a new device — please verify.,security@example.com,2025-09-07
```

---

## LangChain + Ollama example (structured JSON output)

Below is an example showing how to call an Ollama model through LangChain's `ChatOllama`, instruct it to return JSON, and parse that with a Pydantic schema using `PydanticOutputParser`.

```python
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser

# 1) Define the Pydantic schema for structured output
class EmailClassification(BaseModel):
    category: str = Field(..., description="Predicted email category, e.g., spam, promotion, important")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence as a float between 0.0 and 1.0")
    suggested_action: str = Field(..., description="Short suggested action for the inbox owner")

# 2) Create the Pydantic-based parser
parser = PydanticOutputParser(pydantic_object=EmailClassification)

# 3) Build a prompt template that asks for JSON matching the Pydantic model
prompt = ChatPromptTemplate.from_template(
    """
You are an email triage assistant. Given the email subject and body, respond with a JSON object matching this Pydantic schema:
{schema}

Example input:
Subject: {subject}
Body: {body}

Return only the JSON object.
""".format(schema=parser.get_format_instructions())
)

# 4) Initialize the Ollama LLM
llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0.3,
    num_ctx=8000,
    format="json"
)

# 5) Call the model and parse the output
chat_input = prompt.format_prompt(subject="Meeting update", body="Please join the sync at 3pm...")
response = llm(chat_input.to_messages())
# The parser will validate and convert the LLM output into the Pydantic model
parsed = parser.parse(response.content)
print(parsed)  # instance of EmailClassification
```

**Notes:**

* `format="json"` and `temperature=0.3` together encourage stable JSON outputs.
* `PydanticOutputParser` verifies types and raises meaningful errors on schema mismatch — helpful for production pipelines.

---

## Streamlit quickstart (example `streamlit_div.py` pattern)

A minimal Streamlit flow you can plug into `streamlit_div.py`:

```python
import streamlit as st
import pandas as pd
from your_package import classify_with_llm, classify_with_sklearn

st.title("Email Classifier")
backend = st.selectbox("Backend", ["sklearn", "llm"])

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

if st.button("Run classification"):
    st.spinner("Classifying...")
    if backend == "llm":
        results = [classify_with_llm(r.subject, r.body) for r in df.itertuples()]
    else:
        results = [classify_with_sklearn(r.subject, r.body) for r in df.itertuples()]
    st.write(results)
```

Replace `classify_with_llm` and `classify_with_sklearn` with the functions you implement.

---

## Demo screenshot (add to your repo)

To show the dashboard screenshot in your GitHub README, place the image at `docs/demo_screenshot.png` and insert the following Markdown under the badges near the top of the README:

```markdown
![Email Classification Dashboard](docs/demo_screenshot.png)
*Dashboard: Email classification overview and status charts.*
```

I created a resized demo screenshot for you at `/mnt/data/docs/demo_screenshot.png`. Download it and place it in your repo at `docs/demo_screenshot.png` (or push it directly if you clone and commit locally).

Download link: [Download demo\_screenshot.png](sandbox:/mnt/data/docs/demo_screenshot.png)

---

## Suggested `README.md` badges (copy into top of README)

```markdown
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-orange)](https://streamlit.io)
```

You can add a demo screenshot below the badges using standard Markdown:

```markdown
![demo screenshot](docs/demo_screenshot.png)
```

---

## Example `git` commands to add files

```bash
mkdir data
echo "(paste CSV content)" > data/example_emails.csv
echo "(paste requirements from above)" > requirements.txt
git add data/example_emails.csv requirements.txt docs/demo_screenshot.png README.md
git commit -m "Add requirements, example data, and demo screenshot"
git push origin main
```

---

## Next steps I can do for you (I already updated the README in the canvas):

* Extract CLI flags from your scripts (`email_classification_div.py`, `email_status_div.py`, `streamlit_div.py`) and insert exact usage examples into this README. (If you'd like this, paste the top \~40 lines of each script and I will parse them.)
* Create a `requirements.txt` and `data/example_emails.csv` files directly in the repo (I can generate file content here; to actually commit to GitHub you'll need to add/push them locally or give me a link with write access).
* Add a ready-made `streamlit_div.py` minimal app file to the canvas so you can copy it into the repo.

Tell me which of those to perform next or paste the script headers and I will proceed. (If you prefer, I can proceed and add the minimal `streamlit_div.py` and CLI examples to the canvas now.)

---

Maintainer: Divyansh Tiwari

*README now includes requirements.txt, a sample CSV, badge suggestions, Streamlit quickstart, a demo screenshot snippet, and next-step options.*
