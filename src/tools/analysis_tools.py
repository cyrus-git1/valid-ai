"""
src/tools/analysis_tools.py
----------------------------
LangChain tools for sentiment analysis.

Supports documents stored in Supabase Storage (PDF, DOCX, WebVTT).
Text is extracted and chunked via src.processing.tokenization, then each
chunk is scored with a RoBERTa sentiment model and results are averaged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from langchain_core.tools import tool
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from src.processing.tokenization import document_bytes_to_chunks
from src.services.ingest_service import IngestService
from src.supabase.supabase_client import get_supabase

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
_tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
_model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)


def _score_text(text: str) -> dict:
    """Run sentiment on a single string, chunking to fit the 512-token window."""
    tokens = _tokenizer.encode(text, add_special_tokens=False)

    chunk_size = 510  # leave room for [CLS] and [SEP]
    token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    all_scores = []
    for chunk in token_chunks:
        chunk_with_special = [_tokenizer.cls_token_id] + chunk + [_tokenizer.sep_token_id]
        inputs = torch.tensor([chunk_with_special])

        with torch.no_grad():
            outputs = _model(inputs)

        scores = F.softmax(outputs.logits, dim=1).squeeze()
        all_scores.append(scores)

    avg_scores = torch.stack(all_scores).mean(dim=0)

    return {
        "negative": round(avg_scores[0].item(), 2),
        "neutral":  round(avg_scores[1].item(), 2),
        "positive": round(avg_scores[2].item(), 2),
    }


def _extract_text_from_uri(source_uri: str) -> str:
    """Download a document from Supabase Storage and return its full text.

    Supports PDF, DOCX, and WebVTT files. Uses ``document_bytes_to_chunks``
    from tokenization.py for extraction, then joins the chunk texts.
    """
    svc = IngestService(supabase=get_supabase())
    file_bytes, file_type, _bucket, _path = svc.download_from_storage(source_uri)
    chunks = document_bytes_to_chunks(file_bytes, file_type=file_type)
    return " ".join(c["text"] for c in chunks)


@tool
def sentiment_analysis_single(source_uri: str) -> dict:
    """Analyze the sentiment of a document stored in Supabase Storage.

    Args:
        source_uri: Storage URI in the form "bucket:<bucket>/<path>",
                    e.g. "bucket:pdf/transcripts/call.vtt".
                    Supported file types: PDF, DOCX, WebVTT.

    Returns:
        Dict with negative, neutral, and positive scores (0-1).
    """
    text = _extract_text_from_uri(source_uri)
    return _score_text(text)


@tool
def sentiment_analysis_batch(source_uris: list[str]) -> dict:
    """Analyze the sentiment of multiple documents stored in Supabase Storage.

    Args:
        source_uris: List of storage URIs (see sentiment_analysis_single).

    Returns:
        Dict with aggregate scores and per-document results.
    """
    sentiments = []
    for uri in source_uris:
        sentiment = sentiment_analysis_single.invoke(uri)
        sentiment["source_uri"] = uri
        sentiments.append(sentiment)

    n = len(sentiments)
    aggregate = {
        "negative": round(sum(s["negative"] for s in sentiments) / n, 2),
        "neutral":  round(sum(s["neutral"]  for s in sentiments) / n, 2),
        "positive": round(sum(s["positive"] for s in sentiments) / n, 2),
        "results":  sentiments,
    }
    return aggregate
