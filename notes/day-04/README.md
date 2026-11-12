# Day 04 -- Statistical Approaches in NLP

**Course:** NLP & Computer Vision (PGCP-AI) -- CDAC  
**Session covered:** Session 5: Statistical Approaches (Theory & Laboratory)  
**Batch:** August 2026  
**Reference Textbook:** *Natural Language Processing* by Prof. Pushpak Bhattacharyya & Aditya Joshi (Wiley India, 2023), Chapters 2, 3, 5 & 6.  
**Google Colab Support:** Direct Google Drive mount integration configured for `cdac74abhlak@gmail.com`.

---

## 1. Syllabus & Lecture Modules (from `Session5_Statistical_Approaches_Simplified 10 slides each.pptx`)

The Session 5 lecture deck contains **82 slides organized into 7 fundamental statistical topics**:

| Module # | Topic Title | Key Concepts Covered |
| :--- | :--- | :--- |
| **Topic 1** | **Probability Theory in NLP** | Relative frequency estimation, joint probability tables, conditional probability $P(w_i \mid w_{i-1})$, independence, chain rule, Bayes' theorem, Zipf's law, generative vs. discriminative modeling. |
| **Topic 2** | **Discrete-Time Models & Markov Chains** | Text as a discrete stochastic process, first-order Markov assumption, transition probability matrices, random walk sentence generation, grammatical state sequences (POS chains). |
| **Topic 3** | **Markov Models & Entropy Models** | N-gram language models (unigram, bigram, trigram), data sparsity and the zero-frequency problem, Laplace smoothing, Hidden Markov Models (HMM), information theory, entropy $H(X)$, cross-entropy, and perplexity. |
| **Topic 4** | **Statistical Parsing (PCFGs)** | Probabilistic Context-Free Grammars, rule probability estimation $\sum_{\beta} P(A \rightarrow \beta) = 1$, joint parse tree probability computation, disambiguating prepositional phrase (PP) attachment, Probabilistic CYK. |
| **Topic 5** | **Text Classification & Clustering** | Vector Space Models, Bag-of-Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), Multinomial Naive Bayes, evaluation metrics (Accuracy, Precision, Recall, F1), K-Means clustering. |
| **Topic 6** | **Support Vector Machines (SVM)** | Maximum margin hyperplanes, support vectors, soft-margin regularization ($C$), linear vs. non-linear kernel trick, high-dimensional sparsity resilience in text categorization. |
| **Topic 7** | **Centroid-Based Classification** | Prototype vector computation (Rocchio algorithm), cosine angular similarity vs. Euclidean distance, multi-class assignment, comparison of Centroid vs. k-NN vs. Linear SVM. |

---

## 2. Laboratory Curriculum (from `Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf`)

The official laboratory manual contains **11 progressive laboratory experiments divided into two core parts**:

### Part A: Statistical Language Processing (Experiments 1 - 5)
1. **Experiment 1: Probability Theory and Language Models**
   - Implement unigram, bigram, and trigram language models.
   - Calculate sentence likelihood via sequential multiplication using the chain rule.
   - Student implementation: [`code/01. Probability Theory and N-gram Language Models.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/01.%20Probability%20Theory%20and%20N-gram%20Language%20Models.py)
   - Teacher reference code: [`code/Day 4/Probability_Theory_and_Language_Model.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/Probability_Theory_and_Language_Model.py) & [`code/Day 4/Probability_Theory_and_Language_Model1.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/Probability_Theory_and_Language_Model1.py)
2. **Experiment 2: Discrete-Time Language Models (N-gram Prediction)**
   - Implement an N-gram next-word prediction engine given an antecedent sequence of words.
   - Student implementation: [`code/02. N-gram Prediction and Next-Word Generation.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/02.%20N-gram%20Prediction%20and%20Next-Word%20Generation.py)
   - Teacher reference code: [`code/Day 4/ngram_prediction.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/ngram_prediction.py) & [`code/Day 4/ngram_model_updated.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/ngram_model_updated.py)
   - Guides: [`pdfs/2. ngram prediction.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/pdfs/2.%20ngram%20prediction.pdf) & [`pdfs/2.Generalized_Ngram_Model_Execution.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/pdfs/2.Generalized_Ngram_Model_Execution.pdf)
3. **Experiment 3: Markov Models for Text Processing**
   - Construct state-transition matrices and simulate sequence generation across textual states.
   - Student implementation: [`code/03. Markov Models for Text Processing.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/03.%20Markov%20Models%20for%20Text%20Processing.py)
   - Teacher reference code: [`code/Day 4/markov_model.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/markov_model.py) & [`code/Day 4/markov_model2.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/markov_model2.py)
   - Guides: [`pdfs/3. markov model.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/pdfs/3.%20markov%20model.pdf) & [`pdfs/3.Markov_Models2_for_Text_Processing_Explanation.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/pdfs/3.Markov_Models2_for_Text_Processing_Explanation.pdf)
4. **Experiment 4: Entropy and Information Theory in NLP**
   - Calculate corpus entropy, cross-entropy, and perplexity ($PP(W) = 2^{H(W)}$) to measure language model uncertainty.
   - Student implementation: [`code/04. Entropy and Information Theory in NLP.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/04.%20Entropy%20and%20Information%20Theory%20in%20NLP.py)
   - Teacher reference code: [`code/Day 4/entropy_nlp.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/entropy_nlp.py) & [`code/Day 4/entropy_nlp2.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/entropy_nlp2.py)
   - Guides: [`pdfs/4. entropy nlp.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/pdfs/4.%20entropy%20nlp.pdf) & [`pdfs/4.Entropy_and_Information_Theory_in_NLP_Explanation2.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/pdfs/4.Entropy_and_Information_Theory_in_NLP_Explanation2.pdf)
5. **Experiment 5: Statistical Parsing Using Probabilistic Context-Free Grammar (PCFG)**
   - Construct probabilistic grammar rules and determine the most probable parse tree for syntactically ambiguous sentences.
   - Student implementation: [`code/05. Statistical Parsing Using Probabilistic Context-Free Grammar.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/05.%20Statistical%20Parsing%20Using%20Probabilistic%20Context-Free%20Grammar.py)
   - Teacher reference code: [`code/Day 4/pcfg_parser.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/pcfg_parser.py) & [`code/Day 4/pcfg_parser2.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/code/Day%204/pcfg_parser2.py)

### Part B: Statistical Text Classification (Experiments 6 - 11)
6. **Experiment 6: Text Categorization Using Machine Learning**
   - Extract Bag-of-Words and TF-IDF feature matrices from document collections.
7. **Experiment 7: Text Classification Using Naive Bayes**
   - Implement Multinomial Naive Bayes for binary classification (Spam/Ham, Positive/Negative sentiment).
8. **Experiment 8: Text Classification Using Support Vector Machine (SVM)**
   - Implement linear SVM with TF-IDF features and evaluate with confusion matrices, Precision, Recall, and F1-score.
9. **Experiment 9: Text Clustering Using K-Means**
   - Unsupervised grouping of document vector representations into cohesive thematic clusters.
10. **Experiment 10: Centroid-Based Text Classification**
    - Compute normalized prototype centroids for each document category and classify unseen text by cosine proximity.
11. **Experiment 11: Comparative Performance Analysis**
    - Head-to-head empirical benchmarking of Naive Bayes, Linear SVM, and Centroid-based classifiers on the same dataset.

---

## 3. Recommended Datasets & Tooling

| Resource | Recommended Libraries / Files |
| :--- | :--- |
| **Python Libraries** | `nltk`, `scikit-learn`, `numpy`, `spacy`, `pandas`, `matplotlib`, `seaborn` |
| **Recommended Datasets** | SMS Spam Collection, 20 Newsgroups, Disneyland Reviews (`DisneylandReviews.csv`), IMDB Movie Reviews |
| **Slide Deck** | [`slides/Session5_Statistical_Approaches_Simplified 10 slides each.pptx`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/slides/Session5_Statistical_Approaches_Simplified%2010%20slides%20each.pptx) |
| **Laboratory Manual** | [`pdfs/Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-04/pdfs/Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf) |

