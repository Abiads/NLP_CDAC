"""
Script to construct and execute the Day 4 Master Lab Walkthrough Notebook:
notes/day-04/day_04_statistical_approaches_walkthrough.ipynb
Covers Experiments 1 to 5 of Session 5 Statistical Approaches.
Strict Zero-Emoji Policy.
"""

import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Title Markdown
cells.append(nbf.v4.new_markdown_cell(r"""# Day 04 Master Lab Walkthrough - Statistical Approaches in NLP
**Course:** Natural Language Processing & Computer Vision (PGCP-AI)  
**Session:** Session 5 - Statistical Approaches (Theory & Laboratory)  
**Batch:** August 2026  
**Reference Textbook:** *Natural Language Processing* by Prof. Pushpak Bhattacharyya & Aditya Joshi (Wiley India, 2023), Chapters 2, 3 & 5.  
**Google Colab Support:** Automatic Google Drive mount integration configured for `cdac74abhlak@gmail.com`.

---

## Laboratory Overview & Objectives:
This notebook provides a unified, end-to-end practical walkthrough of the foundational statistical techniques in modern NLP:
1. **Experiment 1:** Probability Theory and N-gram Language Models (Unigram, Bigram, Trigram probabilities).
2. **Experiment 2:** Discrete-Time Language Models and N-gram Next-Word Prediction.
3. **Experiment 3:** Markov Models for Text Processing & State Transition Matrices.
4. **Experiment 4:** Entropy, Cross-Entropy, Model Perplexity & Information Theory.
5. **Experiment 5:** Statistical Parsing using Probabilistic Context-Free Grammar (PCFG) and the Viterbi Algorithm.
"""))

# Cell 1: Environment Setup
cells.append(nbf.v4.new_code_cell(r"""# [Google Colab & Local Environment Setup]
import os
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    WORKSPACE_DIR = '/content/drive/MyDrive/NLP_CDAC/notes/day-04'
    if os.path.exists(WORKSPACE_DIR):
        os.chdir(WORKSPACE_DIR)
        print(f"[Colab] Switched working directory to: {WORKSPACE_DIR}")
    else:
        print(f"[Colab] Executing in standard Colab directory: {os.getcwd()}")
    !pip install -q nltk scikit-learn numpy pandas
else:
    print(f"[Local] Executing in local repository environment: {os.getcwd()}")
"""))

# Cell 2: Experiment 1 Theory
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 1: Probability Theory and N-gram Language Models
**Lab Manual Reference:** Experiment 1 (`Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf`)

### Mathematical Principles:
1. **Maximum Likelihood Estimation (MLE):**
   - **Unigram Probability:**
     $$P(w_i) = \frac{C(w_i)}{N}$$
   - **Bigram Conditional Probability:**
     $$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$
   - **Trigram Conditional Probability:**
     $$P(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}$$

2. **Sentence Likelihood under Chain Rule & Markov Assumptions:**
   - **Unigram (Bag-of-Words):**
     $$P(W) = \prod_{i=1}^n P(w_i)$$
   - **Bigram (1st-order Markov):**
     $$P(W) = \prod_{i=1}^{n+1} P(w_i \mid w_{i-1}) \quad \text{where } w_0 = \text{<s>}, w_{n+1} = \text{</s>}$$
   - **Trigram (2nd-order Markov):**
     $$P(W) = \prod_{i=1}^{n+1} P(w_i \mid w_{i-2}, w_{i-1}) \quad \text{where } w_{-1} = w_0 = \text{<s>}$$
"""))

# Cell 3: Experiment 1 Code
cells.append(nbf.v4.new_code_cell(r"""from collections import Counter
import pandas as pd

# Training Corpus
corpus = [
    "I love machine learning",
    "I love deep learning",
    "I study machine learning",
    "I study artificial intelligence",
    "Machine learning is interesting"
]

# Vocabulary and frequency extraction
words = []
for sent in corpus:
    words.extend(sent.lower().split())

total_words = len(words)
unigram_counts = Counter(words)
unigram_probs = {w: count / total_words for w, count in unigram_counts.items()}

# Construct Bigrams with boundary tokens
bigram_corpus = []
for s in corpus:
    tokens = ["<s>"] + s.lower().split() + ["</s>"]
    bigram_corpus.extend([(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)])

bigram_counts = Counter(bigram_corpus)
context_counts = Counter([b[0] for b in bigram_corpus])
bigram_probs = {bg: count / context_counts[bg[0]] for bg, count in bigram_counts.items()}

# Function to compute sentence probabilities
def compute_sentence_probabilities(sentence):
    tokens = sentence.lower().split()
    
    # 1. Unigram
    p_uni = 1.0
    for w in tokens:
        p_uni *= unigram_probs.get(w, 0.0)
        
    # 2. Bigram
    b_tokens = ["<s>"] + tokens + ["</s>"]
    p_bi = 1.0
    for i in range(len(b_tokens) - 1):
        bg = (b_tokens[i], b_tokens[i+1])
        p_bi *= bigram_probs.get(bg, 0.0)
        
    return p_uni, p_bi

benchmark_sentences = [
    "I love machine learning",
    "I study machine learning",
    "Machine learning is interesting",
    "I love deep learning",
    "Learning machine love I"
]

eval_rows = []
for sent in benchmark_sentences:
    p_u, p_b = compute_sentence_probabilities(sent)
    eval_rows.append({
        "Sentence": sent,
        "Unigram P(W)": f"{p_u:.4e}",
        "Bigram P(W)": f"{p_b:.4e}"
    })

df_exp1 = pd.DataFrame(eval_rows)
print("EXPERIMENT 1: N-GRAM SENTENCE PROBABILITY EVALUATION")
display(df_exp1)
"""))

# Cell 4: Experiment 2 Theory
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 2: Discrete-Time Language Models (N-gram Next-Word Prediction)
**Lab Manual Reference:** Experiment 2 (`Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf`)

### Objective:
Predict the most probable following word $w_t$ given an antecedent history $w_{t-1}$ using discrete conditional probability lookups:
$$\hat{w}_t = \arg\max_{w \in V} P(w \mid w_{t-1})$$
"""))

# Cell 5: Experiment 2 Code (use single quote block to avoid string conflict)
cell_5_code = '''from collections import defaultdict
import pandas as pd

text_data = (
    "the cat is sitting on the mat "
    "the cat is playing with a ball "
    "the dog is playing in the garden "
    "the dog is sitting near the door "
    "the cat is sleeping on the sofa"
)

tokens = text_data.lower().split()

# Build Bigram transition mapping
bigram_transitions = defaultdict(Counter)
for i in range(len(tokens) - 1):
    curr_w = tokens[i]
    next_w = tokens[i + 1]
    bigram_transitions[curr_w][next_w] += 1

def predict_next(word):
    word = word.lower()
    if word in bigram_transitions:
        best_candidate, count = bigram_transitions[word].most_common(1)[0]
        total = sum(bigram_transitions[word].values())
        prob = count / total
        return best_candidate, prob
    return None, 0.0

test_words = ["the", "cat", "is", "sitting", "dog", "playing"]
pred_rows = []
for tw in test_words:
    pred, p = predict_next(tw)
    pred_rows.append({
        "Antecedent Word": tw,
        "Predicted Next Word": pred,
        "Transition Probability": f"{p:.2f} ({p*100:.0f}%)"
    })

df_exp2 = pd.DataFrame(pred_rows)
print("EXPERIMENT 2: NEXT-WORD DISCRETE TRANSITION PREDICTIONS")
display(df_exp2)
'''
cells.append(nbf.v4.new_code_cell(cell_5_code))

# Cell 6: Experiment 3 Theory
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 3: Markov Models for Text Processing & State Transitions
**Lab Manual Reference:** Experiment 3 (`Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf`)

### Principles of First-Order Markov Text Modeling:
- **Markov Property:** Future state depends exclusively on the present state:
  $$P(S_{t+1} = s \mid S_t = s_t, S_{t-1} = s_{t-1}, \dots, S_0 = s_0) = P(S_{t+1} = s \mid S_t = s_t)$$
- We estimate the complete state-transition matrix $T_{ij} = P(S_j \mid S_i)$ and perform greedy trajectory walks.
"""))

# Cell 7: Experiment 3 Code
cell_7_code = '''from collections import defaultdict, Counter

# Build Markov Chain transition table
markov_counts = defaultdict(Counter)
corpus_markov = [
    "I love machine learning",
    "I love deep learning",
    "I study machine learning",
    "I study artificial intelligence",
    "Machine learning is interesting"
]

for s in corpus_markov:
    w_list = s.lower().split()
    for i in range(len(w_list) - 1):
        markov_counts[w_list[i]][w_list[i+1]] += 1

# Calculate normalized transition probabilities
markov_probs = {}
for s_curr, next_counts in markov_counts.items():
    tot = sum(next_counts.values())
    markov_probs[s_curr] = {nxt: c / tot for nxt, c in next_counts.items()}

# Simulate Markov chain walk / sequence generation
def generate_markov_trajectory(start_state, max_steps=6):
    trajectory = [start_state.lower()]
    curr = start_state.lower()
    for _ in range(max_steps):
        if curr not in markov_probs:
            break
        next_s = max(markov_probs[curr], key=markov_probs[curr].get)
        trajectory.append(next_s)
        curr = next_s
    return " ".join(trajectory)

print("EXPERIMENT 3: MARKOVIAN SEQUENCE TRAJECTORIES")
for seed in ["i", "love", "study", "machine"]:
    print(f"  Seed State '{seed:<8}' -> State Trajectory: '{generate_markov_trajectory(seed)}'")
'''
cells.append(nbf.v4.new_code_cell(cell_7_code))

# Cell 8: Experiment 4 Theory
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 4: Entropy, Cross-Entropy & Information Theory in NLP
**Lab Manual Reference:** Experiment 4 (`Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf`)

### Information-Theoretic Metrics:
1. **Shannon Entropy (Bits of Uncertainty):**
   $$H(P) = -\sum_{w \in V} P(w) \log_2 P(w)$$
2. **Cross-Entropy between Empirical P and Model Q:**
   $$H(P, Q) = -\sum_{w \in V} P(w) \log_2 Q(w)$$
3. **Model Perplexity (Effective Branching Factor):**
   $$PP(W) = 2^{H(P, Q)}$$
4. **Kullback-Leibler (KL) Divergence:**
   $$D_{KL}(P \parallel Q) = H(P, Q) - H(P)$$
"""))

# Cell 9: Experiment 4 Code
cell_9_code = '''import math

corpus_P = [
    "data science is useful",
    "data science is interesting",
    "machine learning is useful"
]

corpus_Q = [
    "data science is useful",
    "machine learning is useful",
    "data science is useful",
    "machine learning is interesting"
]

words_P = " ".join(corpus_P).lower().split()
words_Q = " ".join(corpus_Q).lower().split()

vocab = sorted(set(words_P) | set(words_Q))
P = {w: words_P.count(w) / len(words_P) for w in vocab}
Q = {w: words_Q.count(w) / len(words_Q) for w in vocab}

# Calculate Shannon Entropy of P
entropy_P = -sum(P[w] * math.log2(P[w]) for w in vocab if P[w] > 0)

# Calculate Cross-Entropy H(P, Q) with epsilon smoothing
eps = 1e-5
cross_entropy_PQ = -sum(P[w] * math.log2(Q[w] if Q[w] > 0 else eps) for w in vocab if P[w] > 0)
perplexity_PQ = math.pow(2, cross_entropy_PQ)
kl_div = cross_entropy_PQ - entropy_P

print("EXPERIMENT 4: INFORMATION THEORY EVALUATION")
print(f"  - Shannon Entropy H(P)       : {entropy_P:.4f} bits/token")
print(f"  - Cross-Entropy H(P, Q)      : {cross_entropy_PQ:.4f} bits/token")
print(f"  - Model Perplexity 2^H(P, Q) : {perplexity_PQ:.4f}")
print(f"  - KL Divergence D_KL(P || Q) : {kl_div:.4f} bits")
'''
cells.append(nbf.v4.new_code_cell(cell_9_code))

# Cell 10: Experiment 5 Theory
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 5: Statistical Parsing Using Probabilistic Context-Free Grammar (PCFG)
**Lab Manual Reference:** Experiment 5 (`Session_5_Statistical_Approaches_NLP_Lab_Programs.pdf`)

### Principles of PCFG & Viterbi Parsing:
- A PCFG assigns probability $P(A \rightarrow \beta)$ to each production such that $\sum_{\beta} P(A \rightarrow \beta) = 1.0$.
- The joint derivation probability of tree $T$ is $P(T) = \prod_{r \in T} P(r)$.
- The **Viterbi parser** computes $\arg\max_T P(T, S)$ via dynamic programming to resolve structural ambiguity.
"""))

# Cell 11: Experiment 5 Code
cell_11_code = '''import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser

# Define standard PCFG grammar
grammar_rules = "\\n".join([
    "S -> NP VP [1.0]",
    "NP -> Det N [0.7] | NP PP [0.3]",
    "VP -> V NP [0.7] | VP PP [0.3]",
    "PP -> P NP [1.0]",
    "Det -> 'the' [0.6] | 'a' [0.4]",
    "N -> 'man' [0.5] | 'telescope' [0.5]",
    "V -> 'saw' [1.0]",
    "P -> 'with' [1.0]"
])
grammar = PCFG.fromstring(grammar_rules)

parser = ViterbiParser(grammar)
sentence = "the man saw the man with a telescope".split()

print("EXPERIMENT 5: VITERBI PCFG PARSING")
print(f"Input Sentence: {' '.join(sentence)}\\n")

parses = list(parser.parse(sentence))
if parses:
    best_tree = parses[0]
    print("Most Probable Parse Tree (Viterbi Optimum):")
    best_tree.pretty_print()
    print(f"\\nJoint Parse Tree Probability P(T): {best_tree.prob():.6e}")
else:
    print("No parse tree derived.")
'''
cells.append(nbf.v4.new_code_cell(cell_11_code))

# Cell 12: Summary Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---

## Summary of Day 04 Statistical Approaches
1. **N-gram Models:** Model local word probability distributions via Markov assumptions, capturing syntactic patterns while balancing vocabulary sparsity.
2. **Discrete Markov Chains:** Represent language as a sequence of state transitions; allow greedy next-token prediction and autoregressive sampling.
3. **Information Theory Metrics:** Shannon entropy measures inherent uncertainty, cross-entropy penalizes divergence from test distributions, and perplexity quantifies predictive branching factor.
4. **Statistical Parsing (PCFG):** Assigns probabilistic weights to context-free productions, enabling the Viterbi algorithm to mathematically resolve structural ambiguities (e.g. PP attachment).
"""))

nb.cells = cells

out_notebook_path = "notes/day-04/day_04_statistical_approaches_walkthrough.ipynb"
with open(out_notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[Done] Generated Day 4 Walkthrough Notebook at: {out_notebook_path} ({len(cells)} cells)")

print("[Executing] Running day_04_statistical_approaches_walkthrough.ipynb via nbclient...")
client = NotebookClient(nb, timeout=120, kernel_name='python3')
client.execute()

with open(out_notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[Verified] Day 4 Walkthrough Notebook successfully executed and updated with live outputs at: {out_notebook_path}")
