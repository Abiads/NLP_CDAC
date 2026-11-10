"""
Builder script to generate and execute notes/day-03/day_03_deep_parsing_walkthrough.ipynb
Strict Zero-Emoji Policy.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Header
cells.append(nbf.v4.new_markdown_cell("""# Day 03 Master Lab Walkthrough - Deep Parsing and Tools for NLP
**Course:** Natural Language Processing & Computer Vision (PGCP-AI)  
**Session:** Session 4 - Deep Parsing, Constituency vs Dependency, CYK Algorithm & Semantic Role Labeling  
**Batch:** August 2026  

---

### Module Overview & Objectives
In Session 4, we transition from surface-level lexical and morphological processing (Session 2 & 3) to **syntactic and semantic deep parsing**:
1. **Context-Free Grammars (CFG) & Recursive Descent Parsing (Exp 1 & 2):** Formal grammar specification, phrase structure rules, avoiding left-recursion, and top-down recursive descent tree generation.
2. **Dependency Parsing with spaCy (Exp 3):** Directed head-dependent relations, Universal Dependencies (UD) tagset, and tree traversal.
3. **Constituency vs. Dependency Head-to-Head Comparison (Exp 5):** Resolving structural prepositional phrase (PP) attachment ambiguity ("she ate the cake with a fork").
4. **CYK Parsing Algorithm (Exp 6):** $O(n^3)$ dynamic programming tabular parsing for Chomsky Normal Form (CNF) grammars.
5. **Semantic Role Labeling (SRL) System (Exp 7):** Predicate-argument classification mapping surface syntax to Agent, Action, Patient, Recipient, Location, and Instrument.
6. **Semantic Parsing with Feature Grammar (Exp 8):** Translating natural language to First-Order Logic (FOL) using Lambda Calculus and compositional $\beta$-reduction.
7. **Information Extraction using NER (Exp 9):** Entity boundary detection and classification (PERSON, ORG, GPE, DATE, MONEY) with DisplaCy visualization."""))

# Cell 1: Colab Setup
cells.append(nbf.v4.new_code_cell("""# [Google Colab & Environment Setup]
# Recommended execution environment: Python 3.10+ with spaCy and NLTK
import os
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Path configuration for student Google Drive (cdac74abhlak@gmail.com)
    WORKSPACE_DIR = '/content/drive/MyDrive/NLP_CDAC/notes/day-03'
    if os.path.exists(WORKSPACE_DIR):
        os.chdir(WORKSPACE_DIR)
        print(f"[Colab] Switched working directory to: {WORKSPACE_DIR}")
    else:
        print(f"[Colab] Drive mounted. Workspace directory: {os.getcwd()}")
        
    # Install required dependencies if missing
    !pip install -q nltk spacy
    !python -m spacy download en_core_web_sm
else:
    print(f"[Local] Executing in local environment: {os.getcwd()}")
"""))

# Cell 2: Exp 1 & 2 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 1 & 2: Context-Free Grammar (CFG) & Recursive Descent Parser
**Aims:**
- Define formal phrase structure rules using Chomsky's Context-Free Grammar (CFG) 4-tuple $G = (V, \Sigma, R, S)$.
- Implement top-down Recursive Descent Parsing to validate sentence syntax and generate constituency parse trees.

### Theory:
- **CFG Definition:** A 4-tuple where:
  - $V$: Non-terminal symbols (e.g., $S, NP, VP, PP, Det, N, V$)
  - $\Sigma$: Terminal symbols / vocabulary words (e.g., 'the', 'student', 'reads')
  - $R$: Production rules of form $A \rightarrow \alpha$ ($A \in V, \alpha \in (V \cup \Sigma)^*$)
  - $S$: Start symbol ($S \in V$)
- **Recursive Descent:** A top-down parser that begins at start symbol $S$ and recursively expands non-terminals by matching production rules against input tokens.
- **Left-Recursion Caveat:** Top-down parsers enter an infinite loop on left-recursive rules (e.g. $VP \rightarrow VP\ PP$). To eliminate this, we formulate rules with non-left-recursive productions ($VP \rightarrow V\ NP\ PP \mid V\ NP \mid V$)."""))

# Cell 3: Exp 1 & 2 Code
cells.append(nbf.v4.new_code_cell("""import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser

# Step 1: Define Context-Free Grammar Rules (Non-Left-Recursive for Recursive Descent)
cfg_rules = \"\"\"
    S -> NP VP
    NP -> Det N | Det Adj N | 'John' | 'Alice'
    VP -> V NP PP | V NP | V
    PP -> P NP
    Det -> 'the' | 'a' | 'an'
    Adj -> 'big' | 'smart' | 'magical' | 'curious'
    N -> 'student' | 'book' | 'park' | 'cat' | 'rollercoaster' | 'library'
    V -> 'reads' | 'visits' | 'rides' | 'sees' | 'enjoys'
    P -> 'in' | 'at' | 'with'
\"\"\"

grammar = CFG.fromstring(cfg_rules)

print("Context-Free Grammar Production Rules:")
print("-" * 50)
for prod in grammar.productions():
    print(f"  {prod}")

# Step 2: Initialize Recursive Descent Parser
rd_parser = RecursiveDescentParser(grammar)

# Step 3: Test valid and invalid sentences
test_sentences = [
    "the student reads a book",
    "Alice visits the magical park",
    "the curious cat sees a book in the library",
    "the student reads at book"  # syntactically invalid under current CFG
]

print("\\n" + "=" * 60)
print("TESTING SENTENCES WITH RECURSIVE DESCENT PARSER")
print("=" * 60)

for sent in test_sentences:
    tokens = sent.split()
    print(f"\\nInput: '{sent}'")
    try:
        trees = list(rd_parser.parse(tokens))
        if trees:
            print(f"Status: VALID - Parsed successfully ({len(trees)} tree(s) found)")
            for i, tree in enumerate(trees, 1):
                print(f"Parse Tree #{i}:")
                tree.pretty_print()
        else:
            print("Status: INVALID - No valid parse tree under grammar")
    except ValueError as e:
        print(f"Status: REJECTED (Vocabulary error: {e})")
"""))

# Cell 4: Exp 3 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 3: Dependency Parsing with spaCy
**Aim:** Implement dependency parsing to extract direct head-to-dependent syntactic relations and visualize parse trees using spaCy's `displacy`.

### Theory:
- **Dependency Grammar:** Replaces phrase hierarchies with direct directed binary relations between lexical words:
  $$\text{Head (Governor)} \xrightarrow{\text{relation}} \text{Dependent (Modifier)}$$
- **Universal Dependencies (UD) Key Tags:**
  - `ROOT`: Central organizing predicate of the clause (usually the main verb).
  - `nsubj`: Nominal subject.
  - `dobj` / `obj`: Direct object.
  - `prep`: Prepositional modifier.
  - `pobj`: Object of preposition.
  - `amod`: Adjectival modifier.
  - `compound`: Multiword nominal compound."""))

# Cell 5: Exp 3 Code
cells.append(nbf.v4.new_code_cell("""import spacy
from spacy import displacy
import pandas as pd

# Step 1: Load English Model
nlp = spacy.load("en_core_web_sm")

sample_sentence = "The excited visitors ride the Space Mountain roller coaster at Disneyland."
doc = nlp(sample_sentence)

# Step 2: Tabulate Dependency Relations
dep_data = []
for token in doc:
    dep_data.append({
        "Token": token.text,
        "POS": token.pos_,
        "Tag": token.tag_,
        "Dep Relation": token.dep_,
        "Head Word": token.head.text,
        "Head POS": token.head.pos_,
        "Children": [child.text for child in token.children]
    })

df_dep = pd.DataFrame(dep_data)
print("Dependency Parse Table:")
display(df_dep)

# Step 3: Subtree Extraction (Noun Chunks & Verb Arguments)
print("\\nSubtree Navigation (Semantic Phrases):")
for token in doc:
    if token.dep_ in ("nsubj", "dobj", "ROOT"):
        phrase = " ".join([w.text for w in token.subtree])
        print(f"  Role: {token.dep_:<8} | Head: {token.text:<12} | Full Subtree: '{phrase}'")

# Step 4: Render Dependency Graph
displacy.render(doc, style="dep", jupyter=True, options={"distance": 110, "compact": False})
"""))

# Cell 6: Exp 5 Markdown
cells.append(nbf.v4.new_markdown_cell("""---

## Experiment 5: Constituency Parsing vs. Dependency Parsing
**Aim:** Compare constituency and dependency parsing paradigms on structural Prepositional Phrase (PP) attachment ambiguity and analyze structural differences.

### The Canonical Ambiguity Case:
Consider the classic sentence:
> **"She ate the cake with a fork."**

- **Interpretation 1 (Instrument):** She used a fork to eat the cake.
  - In Constituency: `[VP ate [NP the cake] [PP with a fork]]` (PP attaches to VP).
  - In Dependency: `with -> ate` (PP modifies verb `ate`).
- **Interpretation 2 (Modifier):** The cake had a fork inside / on it.
  - In Constituency: `[VP ate [NP the cake [PP with a fork]]]` (PP attaches to NP).
  - In Dependency: `with -> cake` (PP modifies noun `cake`)."""))

# Cell 7: Exp 5 Code
cells.append(nbf.v4.new_code_cell("""from nltk.parse import ChartParser

# Ambiguous grammar modeling PP attachment to both VP and NP
ambig_grammar_str = \"\"\"
    S -> NP VP
    NP -> Pro | Det N | NP PP
    VP -> V NP | VP PP | V NP PP
    PP -> P NP
    Pro -> 'she' | 'I'
    Det -> 'the' | 'a'
    N -> 'cake' | 'fork' | 'man' | 'telescope'
    V -> 'ate' | 'saw'
    P -> 'with'
\"\"\"

ambig_grammar = CFG.fromstring(ambig_grammar_str)
chart_parser = ChartParser(ambig_grammar)

ambig_sentence = "she ate the cake with a fork"
ambig_tokens = ambig_sentence.split()

print("=" * 70)
print(f"Target Ambiguous Sentence: '{ambig_sentence}'")
print("=" * 70)

# 1. Constituency Parsing (All Possible Derivations via Chart Parsing)
c_trees = list(chart_parser.parse(ambig_tokens))
print(f"\\n[A] CONSTITUENCY PARSING (Total Valid Parses: {len(c_trees)})")
for idx, tree in enumerate(c_trees, 1):
    print(f"\\nDerivation #{idx}:")
    tree.pretty_print()

# 2. Dependency Parsing (Statistical Disambiguation)
print("\\n[B] SPACY DEPENDENCY PARSING (Statistical Single Best Parse)")
ambig_doc = nlp(ambig_sentence)
for token in ambig_doc:
    print(f"  {token.text:<8} --({token.dep_})--> {token.head.text}")

print("\\nDisplaCy Dependency Visualization:")
displacy.render(ambig_doc, style="dep", jupyter=True, options={"distance": 100})
"""))

# Cell 8: Comparison Table Markdown
cells.append(nbf.v4.new_markdown_cell("""### Structural Comparison Table: Constituency vs. Dependency

| Dimension | Constituency (Phrase Structure) Parsing | Dependency Parsing |
| :--- | :--- | :--- |
| **Theoretical Foundation** | Chomsky's Generative Grammar (1957) | Tesnière's Structural Syntax (1959) |
| **Node Types** | Non-terminals (Phrasal tags) + Terminal words | Terminal words only (no phrasal nodes) |
| **Edge Meaning** | Part-whole constituency (is-a-component-of) | Functional relation (governor-to-dependent) |
| **Computational Complexity** | $O(n^3)$ via CYK / Chart Parsing | $O(n)$ via Shift-Reduce / Transition-based |
| **Free Word Order Languages** | Difficult (leads to non-continuous phrases) | Highly natural (models non-projective dependencies) |
| **Downstream Applications** | Formal language theory, syntax tree transformations | Information extraction, semantic role labeling, question answering |"""))

# Cell 9: Exp 6 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 6: CYK (Cocke-Younger-Kasami) Parsing Algorithm
**Aim:** Implement the CYK dynamic programming algorithm to determine whether a sentence is generated by a grammar in Chomsky Normal Form (CNF) and construct its span chart table.

### Theory:
- **Chomsky Normal Form (CNF):** Every production rule must be strictly in one of two forms:
  1. $A \rightarrow BC$ (Non-terminal generates exactly two non-terminals)
  2. $A \rightarrow a$ (Non-terminal generates exactly one terminal word)
- **Dynamic Programming Table:**
  - Let $P[i, j, A]$ be true if non-terminal $A$ derives the substring $w_i \dots w_j$ (span of length $j - i + 1$).
  - Base case (span length 1): Match terminal rules $A \rightarrow w_i$.
  - Recursive case (span length $l \ge 2$): Check all split points $k$ such that $A \rightarrow BC$ with $B$ spanning $i \dots k$ and $C$ spanning $k+1 \dots j$.
  - Membership check: The sentence belongs to the grammar if start symbol $S \in P[0, n-1]$.
  - Time Complexity: $O(n^3 \cdot |R|)$ where $n$ is sentence length and $|R|$ is rule count."""))

# Cell 10: Exp 6 Code
cells.append(nbf.v4.new_code_cell("""from typing import List, Dict, Set

# Step 1: Define Grammar in strict Chomsky Normal Form (CNF)
cnf_grammar = {
    # Non-terminal binary rules (A -> B C)
    "S":  [("NP", "VP")],
    "VP": [("V", "NP"), ("V", "PP")],
    "NP": [("Det", "N"), ("NP", "PP")],
    "PP": [("P", "NP")],
    # Terminal lexical rules (A -> a)
    "Det": ["the", "a"],
    "N":   ["student", "book", "library", "teacher", "park"],
    "V":   ["reads", "likes", "visits"],
    "P":   ["in", "at"]
}

def cyk_parse(sentence: str, grammar: Dict) -> bool:
    tokens = sentence.lower().split()
    n = len(tokens)
    
    # Initialize 2D table of sets: table[i][j] stores non-terminals for span [i..j]
    table = [[set() for _ in range(n)] for _ in range(n)]
    
    # Step 1: Base Case - Fill diagonal for single words (span length 1)
    for i, word in enumerate(tokens):
        for lhs, productions in grammar.items():
            for prod in productions:
                if isinstance(prod, str) and prod == word:
                    table[i][i].add(lhs)
                    
    # Step 2: Inductive Case - Fill spans of length 2 to n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):  # split point
                # Check all binary grammar rules A -> B C
                for lhs, productions in grammar.items():
                    for prod in productions:
                        if isinstance(prod, tuple) and len(prod) == 2:
                            b, c = prod
                            if b in table[i][k] and c in table[k + 1][j]:
                                table[i][j].add(lhs)
                                
    # Step 3: Print formatted upper-triangular CYK chart
    print(f"\\nInput Sentence: '{sentence}'")
    print(f"Token Length  : {n}")
    print("-" * 65)
    print("Span [i..j]           | Span Text                      | Non-Terminals")
    print("-" * 65)
    for i in range(n):
        for j in range(i, n):
            span_text = " ".join(tokens[i:j+1])
            non_terms = sorted(list(table[i][j])) if table[i][j] else "{}"
            print(f"Span [{i}..{j}]             | {span_text:<30} | {non_terms}")
    print("-" * 65)
    
    # Sentence accepted if start symbol 'S' is in root cell table[0][n-1]
    is_accepted = "S" in table[0][n - 1]
    print(f"CYK Decision: {'ACCEPTED (Grammatically Valid)' if is_accepted else 'REJECTED (Ungrammatical)'}")
    return is_accepted

# Run CYK Verification
cyk_parse("the student reads a book", cnf_grammar)
cyk_parse("reads the student a book", cnf_grammar)  # ungrammatical word order
"""))

# Cell 11: Exp 7 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 7: Semantic Role Labeling (SRL) System
**Aim:** Implement a rule-based Semantic Role Labeling system that identifies verbal predicates and extracts semantic arguments:
- **Agent (ARG0):** The entity causing or performing the action.
- **Action / Predicate:** The central event or verb.
- **Patient / Theme (ARG1):** The entity affected by the action.
- **Recipient (ARG2):** The entity receiving the action or object.
- **Instrument:** The tool or means used to perform the action.
- **Location:** The spatial setting where the event occurs.

### Theoretical Mapping from Dependencies to Semantic Roles:
- Active Subject (`nsubj`) $\rightarrow$ Agent
- Passive Agent (`agent` / `prep(by)` $\rightarrow$ `pobj`) $\rightarrow$ Agent
- Passive Subject (`nsubjpass`) $\rightarrow$ Patient / Theme
- Direct Object (`dobj` / `obj`) $\rightarrow$ Patient / Theme
- Prepositional modifier with 'with' (`prep(with)` $\rightarrow$ `pobj`) $\rightarrow$ Instrument
- Prepositional modifier with 'in'/'at' (`prep(in/at)` $\rightarrow$ `pobj`) $\rightarrow$ Location"""))

# Cell 12: Exp 7 Code
cells.append(nbf.v4.new_code_cell("""from typing import List, Dict, Any

def extract_semantic_roles(sentence: str) -> List[Dict[str, Any]]:
    doc = nlp(sentence)
    roles_list = []
    
    for token in doc:
        # Verbal predicates
        if token.pos_ in ("VERB", "AUX") and token.dep_ in ("ROOT", "advcl", "relcl", "conj", "xcomp"):
            frame = {
                "predicate": token.lemma_,
                "predicate_text": token.text,
                "agent": [],
                "patient": [],
                "recipient": [],
                "instrument": [],
                "location": []
            }
            
            for child in token.children:
                # Active Agent
                if child.dep_ == "nsubj":
                    frame["agent"].append(" ".join([w.text for w in child.subtree]))
                
                # Passive Agent (spaCy tags 'by' as 'agent')
                elif child.dep_ == "agent":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            frame["agent"].append(" ".join([w.text for w in pobj.subtree]))
                
                # Passive Patient (nsubjpass)
                elif child.dep_ == "nsubjpass":
                    frame["patient"].append(" ".join([w.text for w in child.subtree]))
                    
                # Direct Object (Patient)
                elif child.dep_ in ("dobj", "obj"):
                    frame["patient"].append(" ".join([w.text for w in child.subtree]))
                    
                # Recipient (Dative)
                elif child.dep_ == "dative":
                    frame["recipient"].append(" ".join([w.text for w in child.subtree]))
                    
                # Prepositional Arguments
                elif child.dep_ == "prep":
                    prep = child.text.lower()
                    if prep == "with":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                frame["instrument"].append(" ".join([w.text for w in pobj.subtree]))
                    elif prep in ("in", "at", "on", "into", "inside"):
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                frame["location"].append(" ".join([w.text for w in pobj.subtree]))
                    elif prep == "by":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                frame["agent"].append(" ".join([w.text for w in pobj.subtree]))
                                
            roles_list.append(frame)
            
    return roles_list

# Test SRL across diverse sentence structures
srl_test_set = [
    "John writes a letter.",
    "Mary cuts the fresh vegetables with a sharp knife.",
    "The boy kicks the soccer ball into the net.",
    "The teacher explains the complex NLP lesson in the auditorium.",
    "Alice gave Bob a book with a smiling face.",
    "The car was repaired by the mechanic with a wrench."
]

srl_records = []
for sent in srl_test_set:
    frames = extract_semantic_roles(sent)
    for f in frames:
        srl_records.append({
            "Sentence": sent,
            "Action (Predicate)": f["predicate_text"],
            "Agent (ARG0)": ", ".join(f["agent"]) if f["agent"] else "[Passive / None]",
            "Patient / Theme (ARG1)": ", ".join(f["patient"]) if f["patient"] else "[None]",
            "Instrument": ", ".join(f["instrument"]) if f["instrument"] else "-",
            "Recipient (ARG2)": ", ".join(f["recipient"]) if f["recipient"] else "-",
            "Location (ARGM-LOC)": ", ".join(f["location"]) if f["location"] else "-"
        })

df_srl = pd.DataFrame(srl_records)
print("Semantic Role Labeling Summary Table:")
display(df_srl)
"""))

# Cell 13: Exp 8 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 8: Semantic Parsing (Natural Language to First-Order Logic)
**Aim:** Implement a compositional Semantic Parsing system that converts natural language sentences into formal logical representations (First-Order Logic) using NLTK `FeatureGrammar` and Lambda Calculus.

### Theory:
- **Principle of Compositionality (Frege's Principle):** The meaning of a complex expression is determined by the meanings of its syntactic parts and the syntactic rules used to combine them:
  $$\text{Meaning}(S) = \text{Meaning}(VP)(\text{Meaning}(NP))$$
- **Lambda Calculus ($\lambda$-calculus) & $\beta$-reduction:**
  - One-place Intransitive Verbs: $\lambda x. \text{walk}(x)$ applied to $\text{john} \Rightarrow \text{walk}(\text{john})$.
  - Two-place Transitive Verbs: $\lambda y. \lambda x. \text{love}(x, y)$ applied to object $\text{mary} \Rightarrow \lambda x. \text{love}(x, \text{mary})$; applied to subject $\text{john} \Rightarrow \text{love}(\text{john}, \text{mary})$."""))

# Cell 14: Exp 8 Code
cells.append(nbf.v4.new_code_cell(r"""try:
    from nltk import FeatureGrammar
except ImportError:
    from nltk.grammar import FeatureGrammar
from nltk.parse import FeatureChartParser

# Define Feature Grammar with Compositional Semantics
sem_grammar_text = r'''
% start S

# Sentence rule combining Subject (NP) and Predicate (VP)
S[SEM=<?vp(?np)>] -> NP[SEM=?np] VP[SEM=?vp]

# Intransitive Verb Phrase (1-place predicate)
VP[SEM=?v] -> IV[SEM=?v]

# Transitive Verb Phrase (2-place predicate applied to object)
VP[SEM=<?tv(?obj)>] -> TV[SEM=?tv] NP[SEM=?obj]

# Proper Noun / Entity Semantics
NP[SEM=<john>] -> 'John'
NP[SEM=<mary>] -> 'Mary'
NP[SEM=<alice>] -> 'Alice'
NP[SEM=<the_dog>] -> 'the' 'dog'

# Intransitive Verbs (\x. P(x))
IV[SEM=<\x.walk(x)>] -> 'walks'
IV[SEM=<\x.run(x)>] -> 'runs'
IV[SEM=<\x.bark(x)>] -> 'barks'

# Transitive Verbs (\y x. P(x, y))
TV[SEM=<\y x.love(x,y)>] -> 'loves'
TV[SEM=<\y x.feed(x,y)>] -> 'feeds'
TV[SEM=<\y x.chase(x,y)>] -> 'chases'
'''

sem_grammar = FeatureGrammar.fromstring(sem_grammar_text)
sem_parser = FeatureChartParser(sem_grammar)

sem_test_sentences = [
    "John walks",
    "Mary runs",
    "the dog barks",
    "John loves Mary",
    "Mary feeds the dog"
]

sem_records = []
for sent in sem_test_sentences:
    tokens = sent.split()
    parses = list(sem_parser.parse(tokens))
    if parses:
        for p in parses:
            sem_records.append({
                "Sentence": sent,
                "Logical Representation (FOL)": str(p.label()["SEM"]),
                "Derivation Tree": str(p).replace('\n', ' ')[:60] + "..."
            })

df_sem = pd.DataFrame(sem_records)
print("Semantic Parsing (Logical Form Representations):")
display(df_sem)
"""))

# Cell 15: Exp 9 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---

## Experiment 9: Information Extraction using Named Entity Recognition (NER)
**Aim:** Implement an Information Extraction (IE) system using spaCy's Named Entity Recognition (NER) pipeline to identify entities (Persons, Organizations, Locations, Dates, Money) and visualize entity spans.

### Theory:
- **Information Extraction (IE):** Translating unstructured free text into structured knowledge tuples.
- **Named Entity Recognition (NER):** Detecting and classifying named entities into ontologies:
  - `PERSON`: Real and fictional people (e.g. *Dr. A.P.J. Abdul Kalam*, *Bill Gates*)
  - `ORG`: Organizations, agencies, corporations (e.g. *ISRO*, *Microsoft*)
  - `GPE`: Geo-political entities, nations, cities (e.g. *India*, *Bengaluru*)
  - `DATE`: Temporal expressions (e.g. *1975*, *15 August 2026*)
  - `MONEY`: Currency and numerical amounts (e.g. *$5 million*)"""))

# Cell 16: Exp 9 Code
cells.append(nbf.v4.new_code_cell('''ie_text = """
Dr. A.P.J. Abdul Kalam worked at ISRO in India.
Microsoft was founded by Bill Gates in 1975.
The AI Summit conference will be held in Bengaluru on 15 August 2026.
Alphabet invested $5 million in quantum computing research in Zurich.
"""

ie_doc = nlp(ie_text)

ie_records = []
for ent in ie_doc.ents:
    ie_records.append({
        "Entity Text": ent.text,
        "Entity Label": ent.label_,
        "Span": f"[{ent.start_char}..{ent.end_char}]",
        "Description": spacy.explain(ent.label_)
    })

df_ie = pd.DataFrame(ie_records)
print("Information Extraction (Named Entities Table):")
display(df_ie)

print("\\nDisplaCy Named Entity Visualizer:")
displacy.render(ie_doc, style="ent", jupyter=True)
'''))

# Cell 13: Real-World Domain Pipeline Markdown
cells.append(nbf.v4.new_markdown_cell("""---

## Real-World Pipeline Application: Structured Event & Knowledge Extraction
**Aim:** Apply our dependency parser and Semantic Role Labeling system to authentic customer reviews (Disneyland Reviews Dataset) to automatically extract actionable events, guest feedback, and interactions."""))

# Cell 14: Real-World Domain Code
cells.append(nbf.v4.new_code_cell("""# Sample real Disneyland customer review snippets
reviews = [
    "My family visited the Magic Kingdom park during our summer holiday.",
    "The friendly staff served delicious snacks with a welcoming smile.",
    "We rode the Space Mountain coaster in the evening.",
    "The security guards checked all guest bags at the entrance."
]

extracted_events = []
for review in reviews:
    frames = extract_semantic_roles(review)
    for f in frames:
        extracted_events.append({
            "Source Review": review,
            "Action": f["predicate"],
            "Subject (Agent)": ", ".join(f["agent"]) or "Unspecified",
            "Object (Target)": ", ".join(f["patient"]) or "Unspecified",
            "Context / Location": ", ".join(f["location"]) or "Unspecified",
            "Means / Instrument": ", ".join(f["instrument"]) or "Unspecified"
        })

df_events = pd.DataFrame(extracted_events)
print("Automated Event Extraction from Guest Reviews:")
display(df_events)
"""))

# Cell 15: Conclusion & Summary
cells.append(nbf.v4.new_markdown_cell(r"""---

## Summary & Key Takeaways
1. **Constituency Parsing:** Breaks sentences down into hierarchical phrase structures ($S \rightarrow NP + VP$). Ideal for generative grammar verification, but suffers from exponential search spaces without chart memoization.
2. **Dependency Parsing:** Maps directed relationships between words directly. Far more computationally efficient ($O(n)$) and resilient to free word order languages.
3. **CYK Parsing:** Employs dynamic programming to parse Chomsky Normal Form grammars in guaranteed $O(n^3)$ time, systematically building spans in an upper-triangular chart.
4. **Semantic Role Labeling (SRL):** Connects syntax to semantics, answering "Who did What to Whom, Where, When, and How". Enables robust downstream information extraction, knowledge graph population, and event detection.
5. **Semantic Parsing:** Translates natural language queries and statements directly into formal logic representations (FOL) via Lambda Calculus and compositional $\beta$-reduction, enabling automated question answering and database querying."""))

nb.cells = cells

out_path = r"notes/day-03/day_03_deep_parsing_walkthrough.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[Done] Created notebook at: {out_path} ({len(cells)} cells)")

# Now execute notebook to verify full correctness
print("[Executing] Running notebook with nbclient to verify all cells...")
client = NotebookClient(nb, timeout=60, kernel_name='python3')
client.execute()

with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[Verified] Notebook successfully executed and updated with live outputs at: {out_path}")
