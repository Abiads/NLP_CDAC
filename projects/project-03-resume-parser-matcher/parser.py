"""Resume information extraction and candidate-to-job skill matching.

The initial classroom implementation uses spaCy's en_core_web_sm model as
installed by the supplied environment guide. Set SPACY_MODEL=en_core_web_trf
when the transformer model is installed and you want to use it instead.

Usage:
    python parser.py --demo
"""

import argparse
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# Editable classroom vocabulary. A skill is extracted only when it appears in
# this list; add domain-specific skills before using the project with a class.
SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++"],
    "c#": ["c#"],
    "sql": ["sql"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision"],
    "data analysis": ["data analysis", "data analytics"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow"],
    "spacy": ["spacy", "spaCy"],
    "transformers": ["transformers", "hugging face"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript"],
    "react": ["react"],
    "git": ["git"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "power bi": ["power bi"],
    "tableau": ["tableau"],
    "opencv": ["opencv", "open cv"],
}

EDUCATION_TERMS = [
    "b.e.", "b.e", "b.tech", "btech", "m.e.", "m.e", "m.tech", "mtech",
    "b.sc", "bsc", "m.sc", "msc", "mba", "ph.d", "phd", "bachelor",
    "master", "diploma",
]


@lru_cache(maxsize=1)
def load_nlp():
    """Load and cache the spaCy pipeline."""
    import spacy
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL}' is not installed. Run: "
            f"python -m spacy download {SPACY_MODEL}"
        ) from exc


def _contains_phrase(text_lower: str, phrase: str) -> bool:
    """Match a phrase without matching it inside a longer alphanumeric word."""
    escaped = re.escape(phrase.lower())
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


def extract_skills(text: str) -> List[str]:
    """Extract canonical skills identified from SKILL_ALIASES."""
    text_lower = text.lower()
    found = []
    for canonical_skill, aliases in SKILL_ALIASES.items():
        if any(_contains_phrase(text_lower, alias) for alias in aliases):
            found.append(canonical_skill)
    return sorted(found)


def parse_resume(resume_text: str) -> Dict[str, Any]:
    """Extract candidate names, organizations, contact info, skills, and education."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be blank.")

    nlp = load_nlp()
    doc = nlp(resume_text)
    names = []
    organizations = []
    for entity in doc.ents:
        if entity.label_ == "PERSON" and entity.text not in names:
            names.append(entity.text)
        elif entity.label_ == "ORG" and entity.text not in organizations:
            organizations.append(entity.text)

    emails = sorted(set(re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        resume_text,
    )))
    # Deliberately permissive for classroom examples; validate country-specific
    # formats separately before using phone numbers in a real application.
    phone_candidates = re.findall(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)", resume_text)
    phones = sorted({re.sub(r"\s+", " ", value).strip() for value in phone_candidates})

    lower_text = resume_text.lower()
    education = [term for term in EDUCATION_TERMS if _contains_phrase(lower_text, term)]

    return {
        "names": names,
        "emails": emails,
        "phones": phones,
        "skills": extract_skills(resume_text),
        "experiences": organizations,
        "education": education,
    }


def match_candidates(job_description: str, candidate_resumes: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Rank (candidate_id, resume_text) pairs by required-skill coverage."""
    required_skills = extract_skills(job_description)
    if not required_skills:
        raise ValueError(
            "No skills from the configured skill vocabulary were found in the "
            "job description. Add the required terms to SKILL_ALIASES or edit the job description."
        )

    results = []
    for candidate_id, resume_text in candidate_resumes:
        parsed = parse_resume(resume_text)
        matched = sorted(set(parsed["skills"]) & set(required_skills))
        score = len(matched) / len(required_skills)
        results.append({
            "candidate_id": candidate_id,
            "candidate_name": parsed["names"][0] if parsed["names"] else "Not detected",
            "match_score": round(score, 4),
            "matched_skills": matched,
            "missing_skills": sorted(set(required_skills) - set(matched)),
            "parsed_information": parsed,
        })

    return sorted(results, key=lambda item: item["match_score"], reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description="Resume Parser and Candidate Matcher CLI verification tool."
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration with sample resumes")
    parser.add_argument("--job", type=str, default=None, help="Job description text")
    parser.add_argument("--resume", type=str, default=None, help="Resume text to parse")
    args = parser.parse_args()

    sample_job = (
        "We are looking for a Python developer with experience in machine learning, "
        "natural language processing, SQL, pandas, and scikit-learn. Experience with "
        "FastAPI and Docker is an advantage."
    )

    sample_candidates = [
        (
            "Candidate_A",
            "Ananya Rao\nEmail: ananya.rao@example.com\nPhone: +91 98765 43210\n"
            "Education: B.Tech Computer Science\n"
            "Skills: Python, machine learning, natural language processing, SQL, pandas, scikit-learn, FastAPI.\n"
            "Experience: Data Science Intern at ABC Technologies.",
        ),
        (
            "Candidate_B",
            "Rahul Kumar\nEmail: rahul.kumar@example.com\nPhone: 9876543210\n"
            "Education: B.E. Information Science\n"
            "Skills: Python, SQL, pandas, Docker, Java, data analysis.\n"
            "Experience: Software Engineer at Bright Systems.",
        ),
        (
            "Candidate_C",
            "Meera Nair\nEmail: meera.nair@example.com\nPhone: +91-9988776655\n"
            "Education: M.Sc. Data Science\n"
            "Skills: Python, machine learning, scikit-learn, natural language processing, FastAPI, Docker.\n"
            "Experience: Research Assistant at Data Research Lab.",
        ),
    ]

    print("=" * 80)
    print("Project 3: Resume Parser & Candidate Matcher")
    print(f"Backend NLP Model: spaCy ({SPACY_MODEL})")
    print("=" * 80)

    if args.resume:
        print("\nParsing single resume input...")
        parsed = parse_resume(args.resume)
        print(json.dumps(parsed, indent=2))
        return

    job_text = args.job if args.job else sample_job
    print(f"\nJob Description:\n{job_text}\n")

    required_skills = extract_skills(job_text)
    print(f"Required Skills ({len(required_skills)}): {', '.join(required_skills)}")

    ranked = match_candidates(job_text, sample_candidates)

    print("\n" + "=" * 80)
    print(f"{'Rank':<5} | {'Candidate ID':<13} | {'Detected Name':<15} | {'Score':<8} | {'Matched Skills'}")
    print("-" * 80)
    for rank, cand in enumerate(ranked, start=1):
        score_pct = f"{cand['match_score'] * 100:.1f}%"
        matched_str = ", ".join(cand["matched_skills"])
        print(f"{rank:<5} | {cand['candidate_id']:<13} | {cand['candidate_name']:<15} | {score_pct:<8} | {matched_str}")

    print("=" * 80)
    print("\nCandidate Detailed Extraction:")
    for cand in ranked:
        print(f"\n[{cand['candidate_id']}] {cand['candidate_name']} (Match: {cand['match_score']*100:.1f}%)")
        print(f"  Emails:      {cand['parsed_information']['emails']}")
        print(f"  Phones:      {cand['parsed_information']['phones']}")
        print(f"  Education:   {cand['parsed_information']['education']}")
        print(f"  Experiences: {cand['parsed_information']['experiences']}")
        print(f"  Skills:      {cand['parsed_information']['skills']}")
        print(f"  Missing:     {cand['missing_skills']}")


if __name__ == "__main__":
    main()

