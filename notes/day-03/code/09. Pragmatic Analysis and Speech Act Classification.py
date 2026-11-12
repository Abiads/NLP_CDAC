"""
Experiment 09: Pragmatic Analysis and Speech Act Classification
Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools
Reference: notes/day-03/code/Day 3/pragmatic_analysis.py

Key Objectives:
1. Classify utterances into Searle's taxonomic speech act categories:
   - Assertives (statements of fact/belief)
   - Directives (commands, requests, inquiries)
   - Commissives (promises, commitments)
   - Expressives (gratitude, apologies, greetings)
   - Declaratives (institutional state changes)
2. Infer speaker communicative intentions beyond literal surface syntax.
3. Handle indirect speech acts and interrogatives.
Strict Zero-Emoji Policy.
"""

import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def classify_speech_act(sentence):
    s_clean = sentence.strip()
    s_lower = s_clean.lower()
    tokens = word_tokenize(s_clean)

    # Directive: polite request or question
    if s_lower.startswith(("please ", "could you ", "would you ", "can you ", "kindly ")):
        speech_act = "Directive (Request)"
        intention = "Speaker requests the addressee to perform an action."
        context = "Polite interpersonal instruction."
    elif s_lower.startswith(("open ", "close ", "stop ", "go ", "come ", "submit ", "halt ")):
        speech_act = "Directive (Command)"
        intention = "Speaker issues an authoritative imperative command."
        context = "Direct instructional instruction."
    elif s_clean.endswith("?"):
        speech_act = "Directive (Inquiry)"
        intention = "Speaker requests information from the addressee."
        context = "Question eliciting a factual response."
    # Commissive: promise or commitment
    elif any(phrase in s_lower for phrase in ["i promise", "i will", "i shall", "i vow", "we guarantee"]):
        speech_act = "Commissive"
        intention = "Speaker commits themselves to a future course of action."
        context = "Obligation or assurance."
    # Expressive: psychological state or emotion
    elif any(phrase in s_lower for phrase in ["thank you", "thanks", "i am sorry", "i apologize", "congratulations", "well done"]):
        speech_act = "Expressive"
        intention = "Speaker expresses a psychological state or social attitude."
        context = "Social pleasantry, gratitude, or apology."
    # Declarative: institutional enactment
    elif any(phrase in s_lower for phrase in ["i declare", "hereby declare", "we announce", "i pronounce"]):
        speech_act = "Declarative"
        intention = "Speaker brings about a change in external reality via the utterance."
        context = "Institutional or formal decree."
    else:
        # Default Assertive
        speech_act = "Assertive"
        intention = "Speaker commits to the truth of the expressed proposition."
        context = "Factual declaration or informative statement."

    return {
        "tokens": tokens,
        "speech_act": speech_act,
        "intention": intention,
        "context": context
    }


def run_benchmark():
    test_utterances = [
        "The symposium commences promptly at nine o'clock.",
        "Could you please share the computational linguistics syllabus?",
        "Submit your laboratory notebook before Friday evening.",
        "Are these dependency representations projective or non-projective?",
        "I promise to complete the syntactic analysis tonight.",
        "Thank you for reviewing the parse tree derivations.",
        "I hereby declare the annual research summit officially open."
    ]

    print("=" * 80)
    print("EXPERIMENT 09: PRAGMATIC ANALYSIS & SPEECH ACT CLASSIFICATION")
    print("Course: CDAC PGCP-AI - Session 4 Deep Parsing & NLP Tools")
    print("=" * 80)

    for idx, u in enumerate(test_utterances, 1):
        res = classify_speech_act(u)
        print(f"\n[{idx}] Utterance : \"{u}\"")
        print(f"    Speech Act: {res['speech_act']}")
        print(f"    Intention : {res['intention']}")
        print(f"    Context   : {res['context']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_benchmark()
