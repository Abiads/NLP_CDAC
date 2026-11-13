# NLP Projects Curriculum & Roadmap

**Course:** Natural Language Processing & Computer Vision (PGCP-AI) -- CDAC  
**Batch:** August 2026  
**Primary Guide 1:** [`projects/guides/NLP Environment Setup & Duplicate Question Detector Guide.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/projects/guides/NLP%20Environment%20Setup%20%26%20Duplicate%20Question%20Detector%20Guide.pdf)  
**Primary Guide 2:** [`projects/guides/Project_1_Duplicate_Question_Detection_Learning_Notes.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/projects/guides/Project_1_Duplicate_Question_Detection_Learning_Notes.pdf) (18-page comprehensive student manual)

---

## 1. Projects Roadmap

| Project | Focus Area | Key Technologies & Models | Architecture / Deliverables |
|---|---|---|---|
| **Project 1: Duplicate Question Detection** | Semantic Search & Vector Similarity | `sentence-transformers` (`all-MiniLM-L6-v2`), PyTorch, scikit-learn, FastAPI, PostgreSQL (`pgvector`) | Semantic duplicate question detection using dense 384-dimensional vector embeddings and cosine similarity ranking. |
| **Project 2: Content Summarizer** | Abstractive vs. Extractive Summarization | `facebook/bart-large-cnn`, `sumy` (TextRank), Streamlit | Interactive dashboard comparing deep learning abstractive summaries against statistical extractive text rankings. |
| **Project 3: Resume Parser & Matcher** | Named Entity Recognition (NER) & Information Extraction | `spaCy` (`en_core_web_sm`), Regex, Python set operations | Structured extraction of candidate contact info, educational history, and skill matrices matched against job description criteria. |

---

## 2. Directory Structure

```text
projects/
├── guides/
│   ├── NLP Environment Setup & Duplicate Question Detector Guide.pdf
│   └── Project_1_Duplicate_Question_Detection_Learning_Notes.pdf
├── project-01-duplicate-question-detector/
│   ├── app.py
│   ├── duplicate_detector_interactive.ipynb
│   └── README.md
└── README.md
```

