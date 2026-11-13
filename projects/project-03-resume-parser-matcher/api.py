"""FastAPI backend for resume parsing and candidate ranking.

Run from this folder with:
    uvicorn api:app --reload
Open http://127.0.0.1:8000/docs
"""

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from parser import parse_resume, match_candidates

app = FastAPI(title="Resume Parser and Candidate Matcher API", version="1.0")


class ResumeRequest(BaseModel):
    resume_text: str = Field(min_length=5, description="Full text of the resume")


class CandidateInput(BaseModel):
    candidate_id: str
    resume_text: str = Field(min_length=5)


class RankingRequest(BaseModel):
    job_description: str = Field(min_length=5)
    candidates: List[CandidateInput]


class IndexRequest(BaseModel):
    candidate_id: str = Field(min_length=1)
    resume_text: str = Field(min_length=5)


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Resume Parser & Candidate Matcher API",
        "endpoints": ["/docs", "/parse", "/rank", "/index"],
    }


@app.post("/parse")
def parse(request: ResumeRequest):
    try:
        return parse_resume(request.resume_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rank")
def rank(request: RankingRequest):
    pairs = [(item.candidate_id, item.resume_text) for item in request.candidates]
    try:
        return {"ranked_candidates": match_candidates(request.job_description, pairs)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/index")
def index_resume(request: IndexRequest):
    """Optional Elasticsearch indexing endpoint; requires configured Elasticsearch."""
    try:
        from elasticsearch_store import index_candidate
        parsed = parse_resume(request.resume_text)
        index_candidate(request.candidate_id, request.resume_text, parsed)
        return {"indexed": True, "candidate_id": request.candidate_id, "parsed": parsed}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

