"""Streamlit demo for resume parsing and job-skill matching.

Run from this folder with:
    streamlit run app.py
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
from parser import match_candidates, parse_resume, extract_skills

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Resume Parser and Candidate Matcher", layout="wide")
st.title("Resume Parser & Candidate Matcher")
st.write("Extract candidate details with spaCy and regular expressions, then rank candidates by required-skill coverage.")

SAMPLE_JOB = """We are looking for a Python developer with experience in machine learning, natural language processing, SQL, 
pandas, and scikit-learn. Experience with FastAPI and Docker is an advantage."""

SAMPLE_CANDIDATES = [
    {
        "candidate_id": "Candidate_A",
        "resume_text": """Ananya Rao
Email: ananya.rao@example.com
Phone: +91 98765 43210
Education: B.Tech Computer Science
Skills: Python, machine learning, natural language processing, SQL, pandas, scikit-learn, FastAPI.
Experience: Data Science Intern at ABC Technologies.""",
    },
    {
        "candidate_id": "Candidate_B",
        "resume_text": """Rahul Kumar
Email: rahul.kumar@example.com
Phone: 9876543210
Education: B.E. Information Science
Skills: Python, SQL, pandas, Docker, Java, data analysis.
Experience: Software Engineer at Bright Systems.""",
    },
    {
        "candidate_id": "Candidate_C",
        "resume_text": """Meera Nair
Email: meera.nair@example.com
Phone: +91-9988776655
Education: M.Sc. Data Science
Skills: Python, machine learning, scikit-learn, natural language processing, FastAPI, Docker.
Experience: Research Assistant at Data Research Lab.""",
    },
]

job_description = st.text_area("Job description", value=SAMPLE_JOB, height=130)

uploaded_files = st.file_uploader(
    "Upload candidate resumes as .txt files (optional). When none are uploaded, the three sample resumes are used.",
    type=["txt"], accept_multiple_files=True,
)

candidate_pairs = []
if uploaded_files:
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error(f"Could not read {file.name}. Save the file as UTF-8 text and upload again.")
            continue
        candidate_pairs.append((file.name, content))
else:
    candidate_pairs = [(c["candidate_id"], c["resume_text"]) for c in SAMPLE_CANDIDATES]
    st.info("Using built-in sample resumes for the first demonstration.")

if st.button("Parse resumes and rank candidates", type="primary"):
    try:
        required_skills = extract_skills(job_description)
        if not required_skills:
            st.error("No configured skills were detected in the job description. Add terms such as Python, SQL, or machine learning.")
        else:
            st.subheader("Required skills detected")
            st.write(", ".join(required_skills))
            ranked = match_candidates(job_description, candidate_pairs)
            table_rows = [{
                "Rank": index + 1,
                "Candidate": row["candidate_id"],
                "Detected name": row["candidate_name"],
                "Match score": f"{row['match_score'] * 100:.1f}%",
                "Matched skills": ", ".join(row["matched_skills"]),
                "Missing skills": ", ".join(row["missing_skills"]),
            } for index, row in enumerate(ranked)]
            st.subheader("Candidate ranking")
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            st.subheader("Extracted resume information")
            for row in ranked:
                with st.expander(f"{row['candidate_id']} - {row['candidate_name']} ({row['match_score'] * 100:.1f}%)"):
                    st.json(row["parsed_information"])
    except Exception as exc:
        st.error(str(exc))
        st.info("Check that the spaCy model is installed in the active nlp-env environment.")

st.divider()
st.caption("Teaching note: this is a rule-assisted prototype. spaCy identifies general entities; the editable vocabulary identifies skills. It is not a validated hiring or employment decision system.")

