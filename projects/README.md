# NLP Projects Curriculum & Roadmap

**Course:** Natural Language Processing & Computer Vision (PGCP-AI) -- CDAC  
**Batch:** August 2026  
**Primary Guides:**
- [`projects/guides/NLP Environment Setup & Duplicate Question Detector Guide.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/projects/guides/NLP%20Environment%20Setup%20%26%20Duplicate%20Question%20Detector%20Guide.pdf)
- [`projects/guides/Project_1_Duplicate_Question_Detection_Learning_Notes.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/projects/guides/Project_1_Duplicate_Question_Detection_Learning_Notes.pdf)
- [`projects/guides/Project_2_Content_Summarization_Learning_Notes.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/projects/guides/Project_2_Content_Summarization_Learning_Notes.pdf)
- [`projects/guides/Project_3_Resume_Parser_Candidate_Matcher_Learning_Notes.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/projects/guides/Project_3_Resume_Parser_Candidate_Matcher_Learning_Notes.pdf)

---

## 1. Projects Roadmap

| Project | Focus Area | Key Technologies & Models | Architecture / Deliverables | Status |
|---|---|---|---|---|
| **Project 1: Duplicate Question Detection** | Semantic Search & Vector Similarity | `sentence-transformers` (`all-MiniLM-L6-v2`), PyTorch, scikit-learn, FastAPI, PostgreSQL (`pgvector`) | Dense 384-dimensional vector embeddings, cosine similarity ranking, HNSW indexing, and interactive notebook. | Complete |
| **Project 2: Content Summarizer** | Abstractive vs. Extractive Summarization | `sumy` (TextRank), Streamlit, NLTK `punkt_tab` | Interactive comparison of extractive graph-based ranking and lightweight text summarization. | Complete |
| **Project 3: Resume Parser & Matcher** | Named Entity Recognition (NER) & Information Extraction | `spaCy` (`en_core_web_sm`), Regex, Python set operations, FastAPI, Elasticsearch | Structured extraction of candidate contact info, educational history, and skill matrices matched against job description criteria. | Complete |

---

## 2. Directory Structure

```text
projects/
|-- guides/
|   |-- NLP Environment Setup & Duplicate Question Detector Guide.pdf
|   |-- Project_1_Duplicate_Question_Detection_Learning_Notes.pdf
|   |-- Project_2_Content_Summarization_Learning_Notes.pdf
|   |-- Project_3_Resume_Parser_Candidate_Matcher_Learning_Notes.pdf
|   |-- content_summarization.zip
|   `-- duplicate_question_detector.zip
|-- project-01-duplicate-question-detector/
|   |-- app.py
|   |-- api.py
|   |-- pgvector_schema.sql
|   |-- pgvector_store.py
|   |-- duplicate_detector_interactive.ipynb
|   `-- README.md
|-- project-02-content-summarizer/
|   |-- app.py
|   |-- summarizer.py
|   |-- sample_article.txt
|   `-- requirements.txt
|-- project-03-resume-parser-matcher/
|   |-- app.py
|   |-- parser.py
|   |-- api.py
|   |-- elasticsearch_store.py
|   |-- sample_job_description.txt
|   |-- sample_resumes.json
|   |-- resume_parser_interactive.ipynb
|   |-- requirements.txt
|   `-- README.md
`-- README.md
```
