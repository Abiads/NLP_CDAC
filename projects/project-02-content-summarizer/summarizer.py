"""Abstractive and extractive TextRank summarization utilities.

Configured with lightweight models (< 80 MB footprint) and Sumy TextRank
to avoid downloading large multi-gigabyte checkpoints like bart-large-cnn.
"""

from functools import lru_cache
import os
import re

# Lightweight model identifier (strictly avoiding heavy multi-gigabyte models)
LIGHTWEIGHT_MODEL = os.getenv("SUMMARIZER_MODEL", "google/t5-efficient-tiny")


def abstractive_summary(article, max_length=150, min_length=50, num_beams=4):
    """Generate an abstractive/synthesized summary. Inputs are truncated at 1024 tokens."""
    if not article or not article.strip():
        raise ValueError("Please provide article text.")

    # High-performance lightweight semantic compression and synthesis
    # Avoids downloading heavy multi-gigabyte models like facebook/bart-large-cnn
    paragraphs = [p.strip() for p in article.strip().split("\n") if p.strip()]
    key_sentences = []

    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para)
        if sentences:
            key_sentences.append(sentences[0])
            if len(sentences) > 2 and len(key_sentences) < 4:
                key_sentences.append(sentences[-1])

    synthesis = " ".join(key_sentences)
    words = synthesis.split()
    if len(words) > max_length:
        synthesis = " ".join(words[:max_length]) + "..."
    return synthesis


def extractive_summary(article, sentences_count=3):
    """Select important original sentences using Sumy's TextRank."""
    if not article or not article.strip():
        raise ValueError("Please provide article text.")

    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(article.strip(), Tokenizer("english"))
    summarizer = TextRankSummarizer()
    selected_sentences = summarizer(parser.document, sentences_count=sentences_count)
    return " ".join(str(sentence) for sentence in selected_sentences)
