# Day 02 -- Shallow Parsing and Tools for NLP

**Course:** NLP & Computer Vision (PGCP-AI) -- CDAC  
**Sessions covered:**
- Session 3: Shallow Parsing and Tools for NLP
- Session 4: Morphological Synthesis, WSD & Laboratory Applications

---

## 1. Syllabus & Laboratory Experiment Mapping

| Exp # | Official Experiment Title | Key Tools | Local Python File |
| :--- | :--- | :--- | :--- |
| **Exp 1** | Tokenization of Text Data | `nltk.tokenize` | [`code/Tokenization_of_Text_Data.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/Tokenization_of_Text_Data.py) |
| **Exp 2** | Stop Word Removal & Text Preprocessing | `nltk.corpus.stopwords` | [`code/Stop_Word_Removal_and_Text_Preprocessing.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/Stop_Word_Removal_and_Text_Preprocessing.py) |
| **Exp 3** | Part-of-Speech (POS) Tagging | `nltk.pos_tag` | [`code/Part_of_Speech_Tagging.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/Part_of_Speech_Tagging.py) |
| **Exp 4** | Stemming using Porter Stemmer | `nltk.stem.PorterStemmer` | [`code/4. stemming using porter stemmer.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/4.%20stemming%20using%20porter%20stemmer.py) |
| **Exp 5** | Lemmatization using WordNet | `nltk.stem.WordNetLemmatizer` | [`code/05. Lemmatization using wordnet.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/05.%20Lemmatization%20using%20wordnet.py) |
| **Exp 6** | Morphological Analysis of Words | `nltk.corpus.wordnet` | [`code/06. Morphological Synthesis Analysis of words.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/06.%20Morphological%20Synthesis%20Analysis%20of%20words.py) |
| **Exp 7** | Chunking for Shallow Parsing | `nltk.chunk.RegexpParser` | [`code/07. Chunking for Shallow Parsing RegexpParser.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/07.%20Chunking%20for%20Shallow%20Parsing%20RegexpParser.py) |
| **Exp 8** | Multi-Word Expression (MWE) Identification | `nltk.tokenize.MWETokenizer` | [`code/08. Multi Word Expressions.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/08.%20Multi%20Word%20Expressions.py) |
| **Exp 9** | Named Entity Recognition (NER) | `nltk.ne_chunk`, `spacy` | [`code/09. Named Entity Recognition.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/09.%20Named%20Entity%20Recognition.py) |
| **Exp 10** | Morphological Synthesis | Rule-Based Python Engine | [`code/10. Morphological Synthesis and Inflection Generator.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/10.%20Morphological%20Synthesis%20and%20Inflection%20Generator.py) |
| **Exp 11** | Word Sense Disambiguation (WSD) | NLTK Lesk Algorithm | [`code/11. Word Sense Disambiguation using Lesk.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/11.%20Word%20Sense%20Disambiguation%20using%20Lesk.py) |

---

## 2. Notebooks (in `notes/day-02/`)
- [`day_02_lab_walkthrough.ipynb`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/day_02_lab_walkthrough.ipynb): Master interactive walkthrough notebook containing all 11 experiments with full cell execution outputs and Google Colab linking (`cdac74abhlak@gmail.com`).
- [`Session3_Lab_Disneyland_Reviews.ipynb`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/Session3_Lab_Disneyland_Reviews.ipynb): Full dataset preprocessing pipeline on the 42,656 reviews Kaggle dataset.

---

## 3. Teacher Reference Code from CloudFoundry (`code/DAY 2/`)
Exact raw lecture files synced from CDAC CloudFoundry:
- [`code/DAY 2/Tokenization_of_Text_Data.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Tokenization_of_Text_Data.py)
- [`code/DAY 2/Stop_Word_Removal_and_Text_Preprocessing.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Stop_Word_Removal_and_Text_Preprocessing.py)
- [`code/DAY 2/Part_of_Speech_Tagging.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Part_of_Speech_Tagging.py)
- [`code/DAY 2/Stemming_using_Porter_Stemmer.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Stemming_using_Porter_Stemmer.py)
- [`code/DAY 2/Lemmatization_using_WordNet.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Lemmatization_using_WordNet.py)
- [`code/DAY 2/morphological_analysis_noun_prefix.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/morphological_analysis_noun_prefix.py)
- [`code/DAY 2/Chunking_for_Shallow_Parsing.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Chunking_for_Shallow_Parsing.py)
- [`code/DAY 2/Multi_Word_Expression_Identification.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Multi_Word_Expression_Identification.py) & [`code/DAY 2/mwe.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/mwe.py)
- [`code/DAY 2/Named_Entity_Recognition.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Named_Entity_Recognition.py)
- [`code/DAY 2/Morphologica_Synthesis.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Morphologica_Synthesis.py)
- [`code/DAY 2/Word_Sense_Disambiguation.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/Word_Sense_Disambiguation.py) & [`code/DAY 2/wsd.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/wsd.py)
- [`code/DAY 2/mock_disney_read_csv.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/code/DAY%202/mock_disney_read_csv.py)

---

## 4. Lecture Slides & Guides
- **Slides:** [`slides/Session3_Shallow_Parsing_and_Tools_for_NLP-2026.pptx`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/slides/Session3_Shallow_Parsing_and_Tools_for_NLP-2026.pptx)
- **PDF Guide:** [`pdfs/Session 3 Shallow Parsing and Tools for NLP.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/pdfs/Session%203%20Shallow%20Parsing%20and%20Tools%20for%20NLP.pdf)
- **Dataset:** [`DisneylandReviews.csv`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-02/DisneylandReviews.csv) (42,656 reviews)

