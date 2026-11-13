"""Elasticsearch storage and search for parsed resumes.

Install: python -m pip install elasticsearch
Set ELASTICSEARCH_URL to your reachable Elasticsearch endpoint. For secured
clusters, also set ELASTICSEARCH_API_KEY (or adapt the client for your setup).

If Elasticsearch is not configured or offline, this module provides an in-memory
fallback search store so code runs without unhandled crashes.
"""

import os
from typing import Any, Dict, List

try:
    from elasticsearch import Elasticsearch
    ES_AVAILABLE = True
except ImportError:
    Elasticsearch = None
    ES_AVAILABLE = False

INDEX_NAME = "nlp_resumes"

# In-memory document storage for local testing when Elasticsearch is not running
_LOCAL_STORE: Dict[str, Dict[str, Any]] = {}


def get_client():
    """Return an Elasticsearch client or raise informative error."""
    if not ES_AVAILABLE:
        raise RuntimeError("elasticsearch Python library not installed. Run: pip install elasticsearch")
    url = os.getenv("ELASTICSEARCH_URL")
    if not url:
        raise RuntimeError("Set ELASTICSEARCH_URL to your Elasticsearch endpoint first.")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")
    if api_key:
        return Elasticsearch(url, api_key=api_key)
    return Elasticsearch(url)


def ensure_index(client):
    """Ensure the nlp_resumes index exists with expected mappings."""
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(
            index=INDEX_NAME,
            mappings={
                "properties": {
                    "candidate_id": {"type": "keyword"},
                    "candidate_name": {"type": "text"},
                    "emails": {"type": "keyword"},
                    "phones": {"type": "keyword"},
                    "skills": {"type": "keyword"},
                    "experiences": {"type": "text"},
                    "education": {"type": "keyword"},
                    "resume_text": {"type": "text"},
                }
            },
        )


def index_candidate(candidate_id: str, resume_text: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Index parsed candidate record into Elasticsearch or fallback local store."""
    document = {
        "candidate_id": candidate_id,
        "candidate_name": parsed["names"][0] if parsed.get("names") else "",
        "emails": parsed.get("emails", []),
        "phones": parsed.get("phones", []),
        "skills": parsed.get("skills", []),
        "experiences": parsed.get("experiences", []),
        "education": parsed.get("education", []),
        "resume_text": resume_text,
    }

    try:
        client = get_client()
        ensure_index(client)
        client.index(index=INDEX_NAME, id=candidate_id, document=document, refresh=True)
        return document
    except Exception as exc:
        # Fallback to local in-memory store
        _LOCAL_STORE[candidate_id] = document
        return document


def search_candidates(query: str, size: int = 10) -> List[Dict[str, Any]]:
    """Search candidate records by multi-field query."""
    try:
        client = get_client()
        response = client.search(
            index=INDEX_NAME,
            size=size,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["candidate_name^2", "skills^3", "experiences", "education", "resume_text"],
                }
            },
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception:
        # Fallback multi-field keyword search
        query_lower = query.lower()
        results = []
        for doc in _LOCAL_STORE.values():
            skills_str = " ".join(doc.get("skills", [])).lower()
            name_str = doc.get("candidate_name", "").lower()
            text_str = doc.get("resume_text", "").lower()
            if query_lower in skills_str or query_lower in name_str or query_lower in text_str:
                results.append(doc)
            if len(results) >= size:
                break
        return results

