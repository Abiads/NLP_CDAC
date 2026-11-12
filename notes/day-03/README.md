# Day 03 -- Deep Parsing and Tools for NLP

**Course:** NLP & Computer Vision (PGCP-AI) -- CDAC  
**Session covered:** Session 4: Deep Parsing and Tools for NLP  
**Textbook Reference:** *Natural Language Processing* by Prof. Pushpak Bhattacharyya & Aditya Joshi (Wiley India, 2023), Chapters 4 & 11.  
**Google Colab Support:** Direct Google Drive mount integration configured for `cdac74abhlak@gmail.com`.

---

## 1. Syllabus & Laboratory Experiments (from `Session4_NLP_Lab_Programs.pdf`)

| Exp # | Official Experiment Title | Key Concepts & Algorithms | Local Python File / Notebook |
| :--- | :--- | :--- | :--- |
| **Exp 1 & 4** | CFG Constituency Parsing | Phrase Structure Grammars, Tree Generation | [`code/01. Context Free Grammar Constituency Parser.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/01.%20Context%20Free%20Grammar%20Constituency%20Parser.py) |
| **Exp 2** | Recursive Descent Parser | Top-down depth-first parsing search | Included in Exp 1 & [`code/Day 3/cfg_constituency_parser.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/cfg_constituency_parser.py) |
| **Exp 3** | Dependency Parsing with spaCy | Directed head-dependent relations, ROOT, nsubj | [`code/03. Dependency Parsing with spaCy.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/03.%20Dependency%20Parsing%20with%20spaCy.py) |
| **Exp 5** | Constituency vs Dependency Comparison | Phrase brackets vs typed binary arcs, PP-attachment | [`code/04. Constituency vs Dependency Parsing Comparison.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/04.%20Constituency%20vs%20Dependency%20Parsing%20Comparison.py) |
| **Exp 6** | CYK (Cocke-Younger-Kasami) Algorithm | Dynamic programming chart parsing, Chomsky Normal Form | [`code/02. CYK Dynamic Programming Parser.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/02.%20CYK%20Dynamic%20Programming%20Parser.py) |
| **Exp 7** | Semantic Role Labeling (SRL) | Predicate-argument structure (Agent, Action, Patient, Instrument) | [`code/05. Semantic Role Labeling with spaCy.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/05.%20Semantic%20Role%20Labeling%20with%20spaCy.py) |
| **Exp 8** | Semantic Parsing | Natural language to logical representations via Lambda Calculus | [`code/06. Semantic Parsing using Feature Grammar.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/06.%20Semantic%20Parsing%20using%20Feature%20Grammar.py) |
| **Exp 9** | Information Extraction (NER) | Extracting entities (Persons, Orgs, Locations) with spaCy | [`code/07. Information Extraction using NER with spaCy.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/07.%20Information%20Extraction%20using%20NER%20with%20spaCy.py) |
| **Exp 10** | Relation Extraction | Entity pairs (Person-Org, Org-Location) | Slide 19-20 & Slide 36 Q3 |
| **Exp 11** | Automatic Text Summarization | Extractive summarization using TextRank & ROUGE | [`code/11. Text Summarization using TextRank.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/11.%20Text%20Summarization%20using%20TextRank.py) |
| **Exp 12** | Anaphora & Coreference Resolution | Resolving pronouns to antecedents | [`code/08. Anaphora and Coreference Resolution using spaCy.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/08.%20Anaphora%20and%20Coreference%20Resolution%20using%20spaCy.py) |
| **Exp 13** | Pragmatic Analysis | Speech acts, speaker intent, context | [`code/09. Pragmatic Analysis and Speech Act Classification.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/09.%20Pragmatic%20Analysis%20and%20Speech%20Act%20Classification.py) |
| **Exp 14** | Discourse Analysis & Coherence | Coherence relations, discourse markers | [`code/10. Discourse Analysis and Coherence Modeling.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/10.%20Discourse%20Analysis%20and%20Coherence%20Modeling.py) |
| **Exp 15** | Ontology & Semantic Web (RDF / SPARQL) | Triples, OWL knowledge graphs, SPARQL queries | [`code/12. Semantic Web Ontology and SPARQL Querying.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/12.%20Semantic%20Web%20Ontology%20and%20SPARQL%20Querying.py) |

---

## 2. Interactive Master Walkthrough Notebook
- **Notebook:** [`day_03_deep_parsing_walkthrough.ipynb`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/day_03_deep_parsing_walkthrough.ipynb)
- **Contents:**
  1. Google Colab Drive Mount cell (`/content/drive/MyDrive/NLP_CDAC/notes/day-03/`).
  2. CFG Constituency & Recursive Descent Parsing (NLTK).
  3. Universal Dependencies Parsing with spaCy & DisplaCy visualization.
  4. Constituency vs. Dependency Comparison (Prepositional Phrase Attachment Ambiguity).
  5. CYK Dynamic Programming Chart Parsing (Upper-triangular span matrix).
  6. Semantic Role Labeling (SRL) System mapping syntactic dependencies to Agent, Action, Patient, Instrument, Recipient, and Location.
  7. Real-world event extraction on authentic customer review data.

---

## 3. Teacher Reference Files Synced from CloudFoundry (`code/Day 3/`)
- [`code/Day 3/cfg_constituency_parser.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/cfg_constituency_parser.py): Context-Free Grammar parser using NLTK `RecursiveDescentParser`.
- [`code/Day 3/context_free_grammer_parser.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/context_free_grammer_parser.py): Jupyter-compatible CFG tree visualizer with `svgling`.
- [`code/Day 3/cyk_parser.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/cyk_parser.py): Dynamic programming CYK chart parser implementing Chomsky Normal Form (CNF).
- [`code/Day 3/dependency_parsing_spacy.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/dependency_parsing_spacy.py): Dependency parser using spaCy's pretrained `en_core_web_sm`.
- [`code/Day 3/compare_parsing.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/compare_parsing.py): Side-by-side comparison between Constituency and Dependency representations.
- [`code/Day 3/srl_system.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/srl_system.py): Rule-based Semantic Role Labeling identifying Agent, Action, Patient, and Instrument.
- [`code/Day 3/semantic_parsing.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/semantic_parsing.py): Semantic parsing converting natural language to First-Order Logic predicates via NLTK `FeatureGrammar`.
- [`code/Day 3/information_extraction.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/information_extraction.py): Information Extraction using Named Entity Recognition (NER) with spaCy.
- [`code/Day 3/anaphora_resolution_updated.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/anaphora_resolution_updated.py) & [`code/Day 3/anaphora_coreference.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/anaphora_coreference.py): Anaphora and pronoun-antecedent coreference matching.
- [`code/Day 3/pragmatic_analysis.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/pragmatic_analysis.py): Pragmatic intent and Searle's speech act classification.
- [`code/Day 3/discourse_analysis.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/discourse_analysis.py): Rhetorical discourse connectives and consecutive sentence coherence modeling.
- [`code/Day 3/text_summarization.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/text_summarization.py): Extractive text summarization using NetworkX PageRank (TextRank).
- [`code/Day 3/ontology_rdf_sparql.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/code/Day%203/ontology_rdf_sparql.py): RDF University knowledge graph and multi-join SPARQL queries.

---

## 4. Lecture Slides & PDF Guides
- **Slides:** [`slides/Session4_Deep_Parsing_and_Tools_for_NLP_2026.pptx`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/slides/Session4_Deep_Parsing_and_Tools_for_NLP_2026.pptx) (38 slides)
- **PDF Guide:** [`pdfs/Session4_NLP_Lab_Programs.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-03/pdfs/Session4_NLP_Lab_Programs.pdf) (15-experiment laboratory syllabus)
