# Day 05 -- NLP with Machine Learning and Deep Learning

**Course:** Natural Language Processing & Computer Vision (PGCP-AI) -- CDAC  
**Session covered:** Session 6: NLP with Machine Learning and Deep Learning (Theory & Laboratory)  
**Batch:** August 2026  
**Reference Textbook:** *Natural Language Processing* by Prof. Pushpak Bhattacharyya & Aditya Joshi (Wiley India, 2023), §1.4 Generations of NLP, §5.4 Vector Space Models, §12.2 Recurrent Neural Networks, §12.3 Long Short-Term Memory.  
**Google Colab Support:** Direct Google Drive mount integration configured for `cdac74abhlak@gmail.com`.

---

## 1. Syllabus & Lecture Modules (from `Session6_NLP_with_ML_and_DL_2026.pptx`)

The Session 6 lecture deck contains **34 slides organized into 4 core thematic modules**:

| Module # | Topic Title | Key Concepts Covered |
| :--- | :--- | :--- |
| **Module I** | **Machine Learning & Deep Learning for NLP** | Three generations of NLP (Rules -> Statistical ML -> Deep Learning); feature engineering vs representation learning; sparse one-hot encodings vs dense embeddings; geometry of meaning and vector arithmetic. |
| **Module II** | **ML vs DL: The Algorithms** | Classical ML (Naive Bayes, Logistic Regression, Linear SVM, Decision Trees, Random Forests) vs Deep Learning (Feedforward ANN, CNN, RNN, LSTM, GRU, Transformers); empirical performance comparisons on sentiment classification; decision criteria for ML vs DL. |
| **Module III** | **Linear Algebra Review for Neural NLP** | Scalars, vectors, matrices, tensors; dot product, Euclidean norm, and cosine similarity; neural dense layers as matrix-vector products $h = f(Wx + b)$; activation functions (Sigmoid, Tanh, ReLU) and Softmax; backpropagation and gradient descent optimization $w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$. |
| **Module IV** | **From Neurons to Long Short-Term Memory (LSTM)** | Feedforward neural network limitations with sequence data; Recurrent Neural Networks (RNN) and hidden state recurrence $h_t = \tanh(W x_t + U h_{t-1} + b)$; the vanishing and exploding gradient problem; LSTM internal architecture (Cell State $c_t$, Forget Gate $f_t$, Input Gate $i_t$, Candidate $\tilde{c}_t$, Output Gate $o_t$); parameter counting ($4 \times (d_{hidden} \times d_{in} + d_{hidden}^2 + d_{hidden})$); hand-worked LSTM step; stacked and bidirectional LSTMs; transition to Transformers and attention mechanisms. |

---

## 2. Laboratory Curriculum (from `Session_6_NLP_ML_DL_Lab_Programs.pdf`)

The official laboratory manual specifies **10 progressive laboratory programs**:

| Program # | Laboratory Program Title | Concepts / Algorithms Covered | Suggested Dataset | Python Libraries |
| :--- | :--- | :--- | :--- | :--- |
| **Program 1** | Text Classification using Machine Learning | Text preprocessing, Bag-of-Words, TF-IDF vectorization, feature extraction | SMS Spam Collection | `nltk`, `scikit-learn` |
| **Program 2** | Naive Bayes Classifier for Spam Email Detection | Multinomial Naive Bayes, Laplace smoothing, probabilistic text classification | SMS Spam Collection | `pandas`, `scikit-learn` |
| **Program 3** | Sentiment Analysis using Logistic Regression & SVM | Logistic Regression, Support Vector Machine, TF-IDF, binary sentiment polarity | IMDb / [`sentiment_dataset.csv`](sentiment_dataset.csv) | `nltk`, `scikit-learn` |

- **Program 3 Reference Implementation:**
  - Teacher Code: [`code/Day 5/sentiment_analysis.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/code/Day%205/sentiment_analysis.py) & [`code/Day 5/sentiment_analysis2.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/code/Day%205/sentiment_analysis2.py)
  - Dataset: [`sentiment_dataset.csv`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/sentiment_dataset.csv)
  - Official Lab Guide: [`pdfs/1.Sentiment_Analysis_Logistic_Regression_SVM.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/pdfs/1.Sentiment_Analysis_Logistic_Regression_SVM.pdf)
| **Program 4** | Comparison of Machine Learning Algorithms | Naive Bayes, Decision Trees, Random Forests, Logistic Regression; Accuracy, Precision, Recall, F1 | 20 Newsgroups / [`sentiment_dataset.csv`](sentiment_dataset.csv) | `scikit-learn`, `matplotlib` |

- **Program 4 Reference Implementation:**
  - Teacher Code: [`code/Day 5/compare_nlp_algorithms.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/code/Day%205/compare_nlp_algorithms.py)
  - Official Lab Guide: [`pdfs/2.Comparison_of_ML_Algorithms_for_NLP_Tasks.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/pdfs/2.Comparison_of_ML_Algorithms_for_NLP_Tasks.pdf)
| **Program 5** | Word Embeddings for NLP using Word2Vec | One-hot encoding, Continuous Bag-of-Words (CBOW), Skip-gram, semantic cosine similarity | Text Corpus / [`sentiment_dataset.csv`](sentiment_dataset.csv) | `gensim`, `pandas`, `numpy` |

- **Program 5 Reference Implementation:**
  - Teacher Code: [`code/Day 5/word2vec_implementation.py`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/code/Day%205/word2vec_implementation.py)
  - Official Lab Guide: [`pdfs/3.Word2Vec_NLP_Lab.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/pdfs/3.Word2Vec_NLP_Lab.pdf)
| **Program 6** | Neural Network for Text Classification | Artificial Neural Network (ANN), dense layers, ReLU/Softmax, binary cross-entropy, backpropagation | IMDb Movie Reviews | `tensorflow`, `keras` |
| **Program 7** | Recurrent Neural Network (RNN) for Sequence Processing | RNN hidden state recurrence, sequential token ingestion, vanishing gradient analysis | Shakespeare Text Dataset | `tensorflow`, `keras` |
| **Program 8** | Long Short-Term Memory (LSTM) for Sentiment Analysis | LSTM cell state gating (forget, input, output), sequence padding, bidirectional processing | IMDb Movie Reviews | `tensorflow`, `keras` |
| **Program 9** | Next Word Prediction using LSTM Networks | Tokenization, sliding window sequence generation, embedding layers, LSTM text generation | Shakespeare Text Dataset | `tensorflow`, `keras` |
| **Program 10** | Comparative Analysis of ML and DL Models for NLP | Naive Bayes vs SVM vs ANN vs RNN vs LSTM; comprehensive empirical evaluation and metric visualization | IMDb Movie Reviews | `scikit-learn`, `tensorflow`, `matplotlib` |

---

## 3. Self-Learning Conceptual Self-Check (Slide 33)

1. **Question 1:** Name three ML and three DL algorithms and one NLP task for each.
   - *ML:* Multinomial Naive Bayes (Spam Filtering), Support Vector Machine (Topic Categorization), Logistic Regression (Sentiment Analysis).
   - *DL:* LSTM (Next-Word Generation), Bidirectional LSTM (Named Entity Recognition), Transformer / BERT (Contextual Question Answering).
2. **Question 2:** What is the cosine of $(1, 0)$ and $(0, 1)$? Of $(1, 1)$ and $(2, 2)$?
   - $\cos((1, 0), (0, 1)) = \frac{1 \cdot 0 + 0 \cdot 1}{1 \times 1} = 0.0$ (Orthogonal / Unrelated).
   - $\cos((1, 1), (2, 2)) = \frac{1 \cdot 2 + 1 \cdot 2}{\sqrt{2} \times \sqrt{8}} = \frac{4}{4} = 1.0$ (Parallel / Identical orientation).
3. **Question 3:** A neural layer maps 100-d input to 32 units. How many parameters?
   - Weight Matrix $W$: $32 \times 100 = 3,200$ weights.
   - Bias Vector $b$: $32$ biases.
   - Total Parameters: $3,200 + 32 = 3,232$ parameters.
4. **Question 4:** Why does a simple RNN forget the start of a long sentence?
   - Due to repeated multiplication of the transition weight matrix $U$ across $T$ time steps during Backpropagation Through Time (BPTT). If the largest eigenvalue of $U$ is $< 1$, gradients diminish exponentially ($\frac{\partial h_T}{\partial h_1} \to 0$), causing the vanishing gradient problem.
5. **Question 5:** Which LSTM gate decides what to erase from memory?
   - The **Forget Gate** ($f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$) outputs a vector with values between 0 (completely erase) and 1 (completely preserve) that modulates previous cell state $c_{t-1}$.
6. **Question 6:** Why do pre-trained embeddings help the LSTM so much?
   - Pre-trained embeddings (Word2Vec, GloVe, FastText) initialize word representations with semantic and syntactic geometry trained on billions of tokens, preventing the LSTM from overfitting on sparse small datasets and allowing unseen or rare synonym generalization.

---

## 4. Primary References

- **Slide Deck:** [`slides/Session6_NLP_with_ML_and_DL_2026.pptx`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/slides/Session6_NLP_with_ML_and_DL_2026.pptx)
- **Laboratory Manual:** [`pdfs/Session_6_NLP_ML_DL_Lab_Programs.pdf`](file:///c:/Users/conne/Downloads/NLP_CDAC/notes/day-05/pdfs/Session_6_NLP_ML_DL_Lab_Programs.pdf)

