"""
=============================================================================
Course: NLP & Computer Vision (PGCP-AI) - Day 3 / Session 4
Topic:  Experiment 7: Semantic Role Labeling (SRL) with spaCy
File:   05. Semantic Role Labeling with spaCy.py
=============================================================================

WHAT IS SEMANTIC ROLE LABELING (SRL)?
-------------------------------------
Semantic Role Labeling (SRL), often called shallow semantic parsing, identifies
the predicate (typically a verb) in a sentence and determines "who did what to
whom, where, when, and how".

Unlike syntactic parsing (which outputs grammatical categories like NP, VP,
nsubj, dobj), SRL assigns semantic roles / thematic relations:
  - Agent (ARG0)      : The entity intentionally initiating or performing the action.
  - Action/Predicate  : The central event, state, or relation (the verb).
  - Patient/Theme(ARG1: The entity undergoing or affected by the action.
  - Instrument        : The tool, medium, or means used to perform the action.
  - Recipient (ARG2)  : The entity receiving the action or object.
  - Location (ARGM-LOC): Where the action takes place.
  - Temporal (ARGM-TMP): When the action occurs.

In this implementation, we map dependency parse relations (via spaCy) to
semantic roles:
  - Subject (nsubj, nsubjpass)      -> Agent
  - Main Verb (ROOT / VERB)         -> Action / Predicate
  - Direct Object (dobj, obj)       -> Patient / Theme
  - Indirect Object (dative)        -> Recipient
  - Prepositional modifier 'with'   -> Instrument
  - Prepositional modifier 'in/at'  -> Location
=============================================================================
"""

import spacy
from typing import Dict, List, Any

# Step 1: Load English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[Error] spaCy model 'en_core_web_sm' is not installed.")
    print("Run: python -m spacy download en_core_web_sm")
    exit(1)


def extract_semantic_roles(sentence: str) -> List[Dict[str, Any]]:
    """
    Analyzes a sentence and extracts predicate-argument semantic roles.
    
    Args:
        sentence (str): Input text string.
        
    Returns:
        List[Dict[str, Any]]: List of extracted role structures per predicate.
    """
    doc = nlp(sentence)
    results = []
    
    for token in doc:
        # Identify verbal predicates
        if token.pos_ in ("VERB", "AUX") and token.dep_ in ("ROOT", "advcl", "relcl", "conj", "xcomp"):
            role_dict = {
                "predicate": token.lemma_,
                "predicate_text": token.text,
                "agent": [],
                "patient": [],
                "instrument": [],
                "recipient": [],
                "location": []
            }
            
            for child in token.children:
                # Active Subject -> Agent
                if child.dep_ == "nsubj":
                    agent_phrase = " ".join([w.text for w in child.subtree])
                    role_dict["agent"].append(agent_phrase)
                
                # Passive Agent (spaCy tags 'by' as 'agent')
                elif child.dep_ == "agent":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            agent_phrase = " ".join([w.text for w in pobj.subtree])
                            role_dict["agent"].append(agent_phrase)
                
                # Passive Subject -> Patient / Theme
                elif child.dep_ == "nsubjpass":
                    patient_phrase = " ".join([w.text for w in child.subtree])
                    role_dict["patient"].append(patient_phrase)
                
                # Direct Object -> Patient / Theme
                elif child.dep_ in ("dobj", "obj"):
                    patient_phrase = " ".join([w.text for w in child.subtree])
                    role_dict["patient"].append(patient_phrase)
                
                # Recipient (Dative / Indirect Object)
                elif child.dep_ == "dative":
                    recipient_phrase = " ".join([w.text for w in child.subtree])
                    role_dict["recipient"].append(recipient_phrase)
                
                # Prepositional phrases
                elif child.dep_ == "prep":
                    prep_text = child.text.lower()
                    # Passive Agent fallback if tagged as prep
                    if prep_text == "by":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                agent_phrase = " ".join([w.text for w in pobj.subtree])
                                role_dict["agent"].append(agent_phrase)
                    # Instrument typically introduced by 'with'
                    elif prep_text == "with":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                instr_phrase = " ".join([w.text for w in pobj.subtree])
                                role_dict["instrument"].append(instr_phrase)
                    # Location typically introduced by 'in', 'at', 'on'
                    elif prep_text in ("in", "at", "on", "inside"):
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                loc_phrase = " ".join([w.text for w in pobj.subtree])
                                role_dict["location"].append(loc_phrase)
            
            results.append(role_dict)
            
    return results


def display_srl_results(sentence: str, roles: List[Dict[str, Any]]):
    """
    Formats and prints the SRL extraction results clearly.
    """
    print("-" * 75)
    print(f"Sentence: \"{sentence}\"")
    print("-" * 75)
    
    if not roles:
        print("  [No verbal predicate identified]")
        return
        
    for idx, r in enumerate(roles, 1):
        print(f"Predicate #{idx}: {r['predicate_text'].upper()} (Lemma: '{r['predicate']}')")
        print(f"  Agent      : {', '.join(r['agent']) if r['agent'] else '[None / Passive]'}")
        print(f"  Action     : {r['predicate_text']}")
        print(f"  Patient    : {', '.join(r['patient']) if r['patient'] else '[None]'}")
        if r["instrument"]:
            print(f"  Instrument : {', '.join(r['instrument'])}")
        if r["recipient"]:
            print(f"  Recipient  : {', '.join(r['recipient'])}")
        if r["location"]:
            print(f"  Location   : {', '.join(r['location'])}")
        print()


def main():
    print("=" * 75)
    print("EXPERIMENT 7: SEMANTIC ROLE LABELING (SRL) USING SPACY")
    print("Course: CDAC PGCP-AI - Session 4 Deep Parsing & Semantic Analysis")
    print("=" * 75)
    
    test_sentences = [
        "John writes a letter.",
        "Mary cuts the fresh vegetables with a sharp knife.",
        "The boy kicks the ball into the net.",
        "The teacher explains the complex lesson in the classroom.",
        "Alice gave Bob a book with a smiling face.",
        "The car was repaired by the mechanic with a wrench."
    ]
    
    for sentence in test_sentences:
        roles = extract_semantic_roles(sentence)
        display_srl_results(sentence, roles)

    print("=" * 75)
    print("[Summary] SRL maps surface syntactic dependencies to semantic arguments:")
    print("  - 'Who' (Agent) -> nsubj / nsubjpass")
    print("  - 'Did what' (Action) -> ROOT verb")
    print("  - 'To whom/what' (Patient) -> dobj / obj")
    print("  - 'Using what' (Instrument) -> prep('with') -> pobj")
    print("  - 'Where' (Location) -> prep('in'/'at'/'on') -> pobj")
    print("=" * 75)


if __name__ == "__main__":
    main()
