# Project 3: Resume Parser & Candidate Matcher

**Module:** Natural Language Processing (CDAC)  
**Manual Reference:** `projects/guides/Project_3_Resume_Parser_Candidate_Matcher_Learning_Notes.pdf`  
**Model Footprint:** ~12 MB (`en_core_web_sm` from spaCy, lightweight and offline-ready)

---

## 1. Problem Statement
Recruitment workflows require parsing unstructured candidate resumes (varying formats, contact details, experiences, and skill sets) and matching them against specific job descriptions. This project builds an automated end-to-end NLP pipeline to extract structured candidate information and compute skill coverage percentages.

---

## 2. System Architecture

```
[Candidate Resumes] (Raw Text / .txt files)
         |
         +---> [spaCy NER (en_core_web_sm)] ----> Person Names, Organizations
         |
         +---> [Regular Expressions] -----------> Emails, Phone Numbers
         |
         +---> [Vocabulary & Boundary Regex] ---> Canonical Skills, Education
         |
         v
[Parsed Candidate Profile]
         |
         v
[Skill Coverage Engine] <--- [Job Description Skills]
         |
         v
[Ranked Candidates Table & Match Percentages]
         |
         +---> Streamlit UI (app.py)
         +---> FastAPI REST Endpoints (api.py)
         +---> Interactive Notebook (resume_parser_interactive.ipynb)
         +---> Elasticsearch Storage (elasticsearch_store.py)
```

### Key Techniques
1. **Named Entity Recognition (NER):** Uses spaCy's `en_core_web_sm` statistical model to detect `PERSON` entities (candidate names) and `ORG` entities (companies and institutions).
2. **Regex Contact Extraction:** Validates standard email patterns (`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`) and international phone numbers.
3. **Skill Vocabulary & Boundary Matching:** Prevents partial-word false positives using regex lookaround boundaries:
   ```python
   pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
   ```
   Ensures words like `java` are not falsely detected inside `javascript`.
4. **Coverage Scoring:**
   $$\text{Match Score} = \frac{|S_{\text{candidate}} \cap S_{\text{job}}|}{|S_{\text{job}}|}$$

---

## 3. Directory Structure

```
projects/project-03-resume-parser-matcher/
|-- app.py                           # Streamlit web dashboard
|-- parser.py                        # Core extraction & matching logic + CLI demo
|-- api.py                           # Optional FastAPI REST service
|-- elasticsearch_store.py           # Optional Elasticsearch persistence module
|-- sample_job_description.txt       # Benchmark job description
|-- sample_resumes.json              # Benchmark candidate profiles
|-- resume_parser_interactive.ipynb  # Interactive Jupyter Notebook with live outputs
|-- requirements.txt                 # Dependencies
`-- README.md                        # Documentation and viva notes
```

---

## 4. Execution Guide

### Option A: Command-Line Verification (Fastest)
```powershell
python projects/project-03-resume-parser-matcher/parser.py --demo
```

### Option B: Interactive Streamlit Web Dashboard
```powershell
cd projects/project-03-resume-parser-matcher
streamlit run app.py
```
Open browser at `http://localhost:8501`.

### Option C: FastAPI REST API
```powershell
cd projects/project-03-resume-parser-matcher
uvicorn api:app --reload
```
Interactive Swagger docs available at `http://127.0.0.1:8000/docs`.

### Option D: Interactive Jupyter Notebook
Open and run `projects/project-03-resume-parser-matcher/resume_parser_interactive.ipynb`.

---

## 5. Benchmark Verification Results

Using the benchmark job description (8 required skills: `docker`, `fastapi`, `machine learning`, `natural language processing`, `pandas`, `python`, `scikit-learn`, `sql`):

| Rank | Candidate ID | Detected Name | Match Score | Matched Skills | Missing Skills |
|:---:|:---:|:---:|:---:|:---|:---|
| 1 | Candidate_A | Ananya Rao | **87.5%** (7/8) | fastapi, machine learning, natural language processing, pandas, python, scikit-learn, sql | docker |
| 2 | Candidate_C | Meera Nair | **75.0%** (6/8) | docker, fastapi, machine learning, natural language processing, python, scikit-learn | pandas, sql |
| 3 | Candidate_B | Rahul Kumar | **50.0%** (4/8) | docker, pandas, python, sql | fastapi, machine learning, natural language processing, scikit-learn |

---

## 6. Viva Questions & Model Answers

### Q1: Why use spaCy instead of pure regular expressions for candidate names?
**Answer:** Person names do not follow predictable syntactical structures or character patterns. spaCy utilizes statistical transition-based Named Entity Recognition trained on large corpora, taking grammatical context and part-of-speech dependencies into account to accurately identify `PERSON` spans.

### Q2: What is the purpose of negative lookaround in skill extraction?
**Answer:** Without boundaries, simple substring search matches short technical tokens inside unrelated words (e.g. `c` inside `clean`, `r` inside `read`, or `java` inside `javascript`). The pattern `(?<![a-z0-9])phrase(?![a-z0-9])` ensures that matches only trigger on standalone word tokens.

### Q3: How are aliases handled in skill extraction?
**Answer:** The `SKILL_ALIASES` dictionary maps multiple colloquial or shorthand representations (e.g. `['ml', 'machine learning']` or `['sklearn', 'scikit-learn']`) to a single canonical skill key, ensuring consistent matching against job criteria.

### Q4: Does a higher match score guarantee hiring suitability?
**Answer:** No. The score measures keyword overlap against the configured vocabulary. It does not measure years of experience, depth of knowledge, code quality, communication skills, or soft skills.

