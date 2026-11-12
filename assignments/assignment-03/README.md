# Assignment 03 - Lexical Resources, Grammar Systems & Deep Semantic Parsing

**Course:** Natural Language Processing and Computer Vision (PGCP-AI)  
**Session:** Session 4 - Deep Parsing and Tools for NLP  
**Batch:** August 2026  
**Primary Reference 1:** Official Assignment Document `Day_3 Assignment.docx`  
**Primary Reference 2:** Lecture Slides 32 - 36 (`notes/day-03/slides/Session4_Deep_Parsing_and_Tools_for_NLP_2026.pptx`)  
**Submission Date:** 13/11/2026  
**Executable Notebook:** [`assignment_03.ipynb`](assignment_03.ipynb)  
**Official Assignment Document:** [`Day_3 Assignment.docx`](Day_3%20Assignment.docx)

---

## Deliverables Architecture

### Part 1: Official Assignment 3 Tasks (`Day_3 Assignment.docx`)

| Task | Topic | Tools & Techniques | Deliverable / Verification |
|---|---|---|---|
| **Task 1** | Lexicon, Dictionary, and Thesaurus-Based Text Analysis | NLTK, WordNet | Extracted synsets, definitions, synonyms, antonyms, hypernyms/hyponyms, and computed path and Wu-Palmer semantic conceptual similarity. |
| **Task 2** | Spell Checker using NLP Techniques | TextBlob, PySpellChecker | Benchmarked Levenshtein edit-distance candidate ranking vs. TextBlob language model probability correction on noisy text documents. |
| **Task 3** | Grammar Checker using NLP Techniques | LanguageTool, spaCy | Implemented automated grammatical correction combining LanguageTool rule matching with spaCy syntactic dependency subject-verb agreement diagnostics. |

---

### Part 2: Deep Parsing & Semantic Systems Tasks (Slide 36)

| Question | Topic | Tools & Techniques | Deliverable / Verification |
|---|---|---|---|
| **Task 4 (Q1a)** | CKY Dynamic Programming Parsing | Dynamic Programming Chart | Proved cell $T[0, n]$ in Cocke-Younger-Kasami chart proves the entire sentence span is grammatical ($S \in T[0, n]$). |
| **Task 4 (Q1b)** | Probabilistic CFG (PCFG) Disambiguation | NLTK ViterbiParser | Adjusted rule probabilities ($P(NP \rightarrow NP\ PP) = 0.5$) so Viterbi parser selects NP attachment ("the man has the telescope"). |
| **Task 5 (Q2)** | Dependency SVO Extraction & Failure Analysis | spaCy Dependency Parsing | Extracted Subject-Verb-Object triples and documented 2 failure modes: passive voice agent/patient inversion and clausal complements (`ccomp`). |
| **Task 6 (Q3)** | Aspect-Based Sentiment Mining | spaCy, Counter | Mined top $(noun, adjective)$ complaint pairs across 1,500 reviews across Paris, California, and Hong Kong; documented negation failure. |
| **Task 7 (Q4)** | TextRank Graph Summarization & ROUGE-1 | NetworkX, scikit-learn, PageRank | Extracted top salient review sentences via PageRank cosine similarity graph; evaluated unigram overlap vs. human reference summary. |
| **Task 8 (Q5)** | Winograd Schema Pronoun Resolution | Linguistic Analysis | Analyzed ambiguous pronoun resolution ("The trophy would not fit in the suitcase because it was too big/small") and common-sense physical containment knowledge requirements. |
| **Task 9 (Q6)** | Semantic Web RDF Ontology & SPARQL | rdflib, SPARQL 1.1 | Populated Disney theme park knowledge graph and executed multi-join queries with chronological ordering and visitor filtering. |

---

## Environment & Execution
The notebook is self-contained and pre-executed with all live outputs preserved. It supports both local execution and Google Colab (configured for `cdac74abhlak@gmail.com` with automated Google Drive mounting).
