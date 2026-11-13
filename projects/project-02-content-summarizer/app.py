"""Streamlit interface comparing BART and TextRank summaries.

Run from this folder with:
    streamlit run app.py
"""

import streamlit as st
from summarizer import abstractive_summary, extractive_summary

st.set_page_config(page_title="NLP Content Summarizer", layout="wide")
st.title("Content Summarizer: Abstractive vs Extractive")
st.write(
    "Paste an article and compare a generated BART summary with sentences "
    "selected by the TextRank algorithm."
)

SAMPLE_ARTICLE = """Artificial intelligence is changing how organizations analyze information and make decisions. In healthcare, machine-learning systems can help clinicians review medical images, identify patterns in patient records, and prioritize cases that may need attention. These systems are intended to support professional judgement rather than replace doctors.

In education, intelligent learning platforms can adapt exercises to a learner's progress. They can identify topics that require additional practice and provide feedback quickly. However, teachers remain important because they understand the social and individual needs of students.

Businesses also use natural language processing to process customer messages, summarize reports, classify documents, and search large collections of text. Such tools can reduce repetitive work, but their outputs should be checked for accuracy, privacy, and bias. The quality of an AI system depends on its data, evaluation, and the way people use its results.

Researchers are therefore studying explainability, fairness, and responsible deployment. A successful AI application combines technical performance with human oversight, clear objectives, and continuous evaluation."""

article = st.text_area(
    "Article text",
    value=SAMPLE_ARTICLE,
    height=260,
    help="The BART implementation truncates input to a maximum of 1024 tokens, as specified in the project guide.",
)

col_a, col_b = st.columns(2)
with col_a:
    max_length = st.slider("BART maximum summary tokens", 40, 200, 150, 10)
    min_length = st.slider("BART minimum summary tokens", 5, 100, 50, 5)
with col_b:
    num_beams = st.selectbox("Beam search (num_beams)", [2, 4, 6, 8], index=1)
    sentence_count = st.slider("TextRank: number of source sentences", 1, 8, 3)

if st.button("Generate both summaries", type="primary", disabled=not article.strip()):
    with st.spinner("Generating BART summary. The first run downloads the model and may take some time..."):
        try:
            bart_result = abstractive_summary(
                article, max_length=max_length, min_length=min_length,
                num_beams=num_beams,
            )
            st.session_state["bart_result"] = bart_result
        except Exception as exc:
            st.error(f"BART summarization failed: {exc}")
            st.info("Check the internet connection on the first run and confirm that PyTorch and Transformers are installed.")

    try:
        textrank_result = extractive_summary(article, sentences_count=sentence_count)
        st.session_state["textrank_result"] = textrank_result
    except LookupError as exc:
        st.error(f"TextRank tokenizer data is missing: {exc}")
        st.code("python -m nltk.downloader punkt punkt_tab", language="powershell")
    except Exception as exc:
        st.error(f"TextRank summarization failed: {exc}")

if "bart_result" in st.session_state or "textrank_result" in st.session_state:
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("1. Abstractive summary (BART)")
        st.caption("BART generates a new sequence of words; it may paraphrase the article.")
        st.write(st.session_state.get("bart_result", "No BART result available."))
    with right:
        st.subheader("2. Extractive summary (TextRank)")
        st.caption("TextRank selects important sentences from the original article.")
        st.write(st.session_state.get("textrank_result", "No TextRank result available."))

st.divider()
st.caption("Classroom discussion: compare factual coverage, wording, readability, and whether each summary introduces or omits information.")
