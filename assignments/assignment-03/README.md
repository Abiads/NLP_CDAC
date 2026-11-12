# Assignment 03 - Deep Parsing, Review Mining, and Semantic Systems

**Course:** Natural Language Processing and Computer Vision (PGCP-AI)  
**Session:** Session 4 - Deep Parsing and Tools for NLP  
**Batch:** August 2026  
**Assignment Reference:** Lecture Slides 32-36 (`notes/day-03/slides/Session4_Deep_Parsing_and_Tools_for_NLP_2026.pptx`)  
**Executable Notebook:** [`assignment_03.ipynb`](assignment_03.ipynb)

---

## Deliverables Summary (Slide 36)

| Question | Topic | Primary Deliverable | Output / Verification |
|---|---|---|---|
| **Q1a** | CKY Chart Parsing | Grammaticality proof via CKY dynamic programming cell | Proved: Cell $T[0, n]$ must contain start symbol $S$. |
| **Q1b** | Probabilistic CFG (PCFG) | Modify rule weights to favor NP attachment ("the man has the telescope") | $P(NP \rightarrow NP\ PP)$ adjusted from 0.3 to 0.5; Viterbi parser selects NP modification tree. |
| **Q2** | Dependency SVO Extraction | Rule-based SVO extractor and two failure cases with linguistic analysis | Documented failure in passive voice (misidentifies patient as actor) and clausal complements (`ccomp`). |
| **Q3** | Aspect-Based Sentiment Mining | Aspect-opinion extraction across 1,500 reviews by park branch | Extracted top complaint pairs for Paris, California, and Hong Kong; documented negation failure error. |
| **Q4** | TextRank Summarization | Extractive summarization of longest review and ROUGE-1 evaluation | Compared against human reference summary; achieved solid ROUGE-1 unigram overlap. |
| **Q5** | Coreference Resolution | Winograd Schema failure analysis (the trophy example) | Analyzed physical containment world knowledge required for singular neuter pronouns. |
| **Q6** | Semantic Web and SPARQL | RDF knowledge graph construction and multi-join SPARQL query | Queried parks opened after 1980 with attraction and visitor metrics, sorted chronologically. |

---

## Environment & Execution
The notebook supports both local execution and Google Colab (configured for `cdac74abhlak@gmail.com` with automatic Google Drive mounting).

