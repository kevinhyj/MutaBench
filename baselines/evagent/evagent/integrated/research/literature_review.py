"""Passage-level literature admission. Raw papers and reviews stay in quarantine.

Reviews are explicit, text-hash-bound curation, not model self-certification.
Keyword matches flag review; they no longer discard a whole mechanism paper.
"""
from __future__ import annotations

import re

from ..gateway import digest

POLICY = "passage_review_v2"
VARIANT = re.compile(r"\b[ACGTU]\s*\d+\s*[ACGTU]\b", re.I)
DIRECT_RESULT = re.compile(
    r"DMS[ _-]?score|fitness\s*(?:=|:)|experimental[ _]score|"
    r"(?:mutant|variant|substitution)\b.{0,50}\b(?:increased|decreased|abolished|enhanced|reduced)\b|"
    r"(?:\d+(?:\.\d+)?\s*(?:%|[- ]fold))\s*(?:activity|fitness|cleavage)", re.I)
REVIEW_FLAG = re.compile(r"mutation|mutagenesis|mutant|variant|deletion|substitution", re.I)


def norm_doi(value):
    return re.sub(r"^https?://(?:dx\.)?doi\.org/|^doi:\s*", "", value.strip().lower())


def norm_title(value):
    return re.sub(r"\W", "", value.lower())


def check_background_query(query):
    if not isinstance(query, str) or not 3 <= len(query) <= 400:
        raise ValueError("Invalid literature background query")
    if VARIANT.search(query) or re.search(
        r"\b(?:DMS|fitness|rankings?|scores?|best mutant|mutation effects?|mutant activity|variant effects?)\b",
        query, re.I,
    ):
        raise ValueError("Candidate effects and ranking queries remain prohibited")


def source_block(record, denied_dois=(), denied_titles=()):
    if norm_doi(record.get("doi", "")) in {norm_doi(d) for d in denied_dois}:
        return "benchmark_or_related_DOI"
    if norm_title(record.get("title", "")) in {norm_title(t) for t in denied_titles}:
        return "benchmark_or_related_title"
    if any(record.get(k) for k in ("target_experiment", "benchmark_derived", "target_supplement")):
        return "benchmark_source_or_derivative"
    if VARIANT.search(record.get("title", "")) or DIRECT_RESULT.search(record.get("title", "")):
        return "direct_effect_in_citation_title"
    if not all(record.get(k) for k in ("doi", "title", "text", "location", "date")):
        return "missing_traceable_source"
    return None


def screen_document(record, denied_dois=(), denied_titles=()):
    """Return only approved literal spans plus an exhaustive disposition audit."""
    source_hash = digest(record.get("text", ""))
    reason = source_block(record, denied_dois, denied_titles)
    if reason:
        return [], [{"decision": "quarantine_document", "reason": reason,
                     "doi": record.get("doi"), "source_sha256": source_hash}]
    text, approved, decisions = record["text"], [], []
    review = record.get("passage_review", {})
    if (review.get("source_sha256") != source_hash or not review.get("reviewer")
            or not review.get("reviewed_at") or review.get("policy") != POLICY):
        return [], [{"decision": "pending_review", "reason": "missing_or_stale_passage_review",
                     "source_sha256": source_hash, "doi": record["doi"]}]
    previous = 0
    for span in review.get("spans", []):
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or not previous <= start < end <= len(text):
            raise ValueError("Review spans must be ordered, disjoint, exact source offsets")
        if previous < start:
            decisions.append({"decision": "pending_review", "start": previous, "end": start})
        previous = end
        excerpt = text[start:end]
        scope = span.get("relevance")
        classification = span.get("classification")
        reason = None
        if classification not in {"mechanism", "background"}:
            reason = "direct_effect_or_unresolved_passage"
        elif VARIANT.search(excerpt) or DIRECT_RESULT.search(excerpt):
            reason = "explicit_variant_or_measured_effect"
        elif scope not in {"construct", "family", "background"} or not span.get("applicability"):
            reason = "missing_applicability"
        elif not span.get("reason") or not span.get("limitations"):
            reason = "missing_review_rationale_or_limitations"
        decision = {"decision": "quarantine_passage" if reason else "include", "reason": reason,
                    "start": start, "end": end, "keyword_flag": bool(REVIEW_FLAG.search(excerpt)),
                    "source_sha256": source_hash, "doi": record["doi"]}
        decisions.append(decision)
        if reason:
            continue
        public = {"id": "passage-" + digest([record["doi"], source_hash, start, end])[:20],
                  "doi": norm_doi(record["doi"]), "title": record["title"], "text": excerpt,
                  "location": record["location"], "date": record["date"], "relevance": scope,
                  "applicability": span["applicability"], "limitations": span["limitations"],
                  "source_sha256": source_hash, "source_start": start, "source_end": end,
                  "review": {"policy": POLICY, "reviewer": review["reviewer"],
                             "reviewed_at": review["reviewed_at"], "classification": classification,
                             "reason": span["reason"]}}
        public["admission_sha256"] = digest(public)
        approved.append(public)
    if previous < len(text):
        decisions.append({"decision": "pending_review", "start": previous, "end": len(text)})
    return approved, decisions


def validate_passage(record, denied_dois=(), denied_titles=()):
    if source_block(record, denied_dois, denied_titles):
        raise ValueError("Quarantined or untraceable literature source")
    if record.get("admission_sha256") != digest({k: v for k, v in record.items() if k != "admission_sha256"}):
        raise ValueError("Reviewed passage changed after admission")
    if record.get("review", {}).get("policy") != POLICY:
        raise ValueError("Missing passage review policy")
    if VARIANT.search(record["text"]) or DIRECT_RESULT.search(record["text"]):
        raise ValueError("Direct variant effects cannot enter prediction context")
    if record.get("source_end", 0) - record.get("source_start", 0) != len(record["text"]):
        raise ValueError("Invalid original passage offsets")
    if record.get("relevance") not in {"construct", "family", "background"}:
        raise ValueError("Missing literature applicability")


SPECIALIST_GUIDANCE = """
Literature v2: distinguish construct, family, and general background evidence.
A supported mechanism or family background finding is valid without candidate or
position mapping: use candidate_ids=[] and positions=[], and explain its scope in
limitations. Do not invent a candidate-specific effect from a family mechanism.
Cite retrieved literal passages, not a paper title or the fact a paper exists.
Mutation terminology alone does not make a mechanism passage unusable; trust only
the reviewed passages returned by tools. Empty literature means a single explicit
abstention, not repeated searches for candidate effects. Reserve a final report call.
"""
