"""PaperQA retrieval and single-attempt evidence summaries through EVAgent's ledger."""

from __future__ import annotations

import asyncio
import re

from ..gateway import digest

# Deliberately conservative: these documents are quarantined before model ingestion.
CONTAMINATION = re.compile(
    r"\b[ACGU]\d+[ACGU]\b|single[- ]base change|base substitution|nucleotide substitution|truncation|deletion|deep mutational|mutagenesis|mutation|variants?|mutant(?:s)?|DMS[ _-]?score|fitness landscape|supplementary (?:table|data).*variant",
    re.I,
)
MUTATION_QUERY = re.compile(
    r"\b[ACGU]\d+[ACGU]\b|\b(?:variants?|mutants?|fitness|rankings?|DMS|mutations?)\b", re.I
)


def check_query(query):
    if not isinstance(query, str) or not 3 <= len(query) <= 400 or MUTATION_QUERY.search(query):
        raise ValueError("Literature query must concern task background, not candidate effects")


def audit_document(record, denied_dois, denied_titles=()):
    doi = record.get("doi", "").lower().removeprefix("https://doi.org/")
    title = re.sub(r"\W", "", record.get("title", "").lower())
    if doi in {d.lower().removeprefix("https://doi.org/") for d in denied_dois}:
        return "benchmark_or_related_DOI"
    if title in {re.sub(r"\W", "", t.lower()) for t in denied_titles}:
        return "benchmark_or_related_title"
    if CONTAMINATION.search(record.get("title", "") + "\n" + record.get("text", "")):
        return "mutation_effect_or_supplement_contamination"
    if not record.get("text") or not record.get("doi") or not record.get("location"):
        return "missing_traceable_text"
    if record.get("relevance") not in {"construct", "family", "background"}:
        return "unreviewed_applicability"
    return None


def relevance_gate(query, document):
    """Require the actual family/entity in the title, not a generic RNA keyword."""
    q = query.lower()
    title = re.sub(r"<[^>]+>", " ", document.get("title", "")).lower()
    if re.search(r"clostridioides|virulence|antibiotic|antimicrobial", title):
        return "different_task_or_pathogen_focused_application"
    anchors = next(
        (
            patterns
            for key, patterns in (
                ("glms", [r"glms"]),
                ("cpeb3", [r"cpeb3"]),
                ("saccharomyces", [r"trna", r"yeast|saccharomyces"]),
                ("azoarcus", [r"azoarcus"]),
                ("tetrahymena", [r"tetrahymena"]),
                ("twister sister", [r"twister.?sister"]),
                ("twister", [r"twister"]),
                ("aminoacyl", [r"aminoacyl|amino.?acid", r"rna|ribozyme"]),
                ("triplet", [r"polymerase", r"ribozyme"]),
                ("class ii", [r"class.?ii", r"ligase"]),
                ("gfp", [r"gfp|green fluorescent protein", r"aptamer"]),
                ("nelf", [r"nelf|negative elongation", r"rna|recognition"]),
            )
            if key in q
        ),
        None,
    )
    if anchors is None or not all(re.search(pattern, title) for pattern in anchors):
        return "no_specific_family_or_entity_in_title"
    return None


class GatewaySummaryModel:
    def __init__(self, bounded):
        self.bounded = bounded

    async def call_single(self, messages, **kwargs):
        from aviary.core import Message
        from lmi import LLMResult

        system = "\n".join(m.content for m in messages if m.role == "system")
        user = "\n".join(m.content for m in messages if m.role != "system")
        text = self.bounded.generate(system, user, phase="paperqa_evidence")
        return LLMResult(
            text=text,
            model=self.bounded.gateway.model_config["model"],
            messages=[Message(role="assistant", content=text)],
        )


class PaperEvidence:
    def __init__(self, documents, *, passage_policy=False):
        self.documents = documents
        self.docs = None
        self.passage_policy = passage_policy

    async def build(self):
        from lmi import SparseEmbeddingModel
        from paperqa import Docs, Settings
        from paperqa.types import Doc, Text

        self.embedding = SparseEmbeddingModel()
        self.settings = Settings(
            llm="gemini-3.8-flash",
            summary_llm="gemini-3.8-flash",
            embedding="sparse",
            parsing={"multimodal": False, "use_doc_details": False},
            answer={
                "max_concurrent_requests": 1,
                "evidence_k": 2,
                "evidence_text_only_fallback": False,
            },
        )
        self.docs = Docs()
        self.lookup = {}
        for record in self.documents:
            doc = Doc(
                docname=record["id"],
                dockey=record["id"],
                citation=record["doi"] + " " + record["title"],
            )
            # Keep auditable paragraph offsets. No model metadata inference or answer index.
            chunks = []
            for i in range(0, len(record["text"]), 1800):
                name = record["id"] + f"-char{i}"
                chunks.append(Text(name=name, text=record["text"][i : i + 1800], doc=doc))
                self.lookup[name] = {
                    "id": name,
                    "doi": record["doi"],
                    "title": record["title"],
                    "location": record["location"],
                    "offset": i,
                    "relevance": record["relevance"],
                    "source_sha256": digest(record["text"]),
                }
            await self.docs.aadd_texts(
                chunks, doc, settings=self.settings, embedding_model=self.embedding
            )

    async def asearch(self, query, bounded):
        from paperqa.core import _map_fxn_summary, llm_parse_json

        if self.passage_policy:
            from .literature_review import check_background_query, validate_passage
            check_background_query(query)
            for document in self.documents:
                validate_passage(document)
        else:
            check_query(query)
        if not self.documents:
            return {
                "status": "unavailable",
                "reason": "No reviewed task-relevant literature",
                "evidence": [],
            }
        if self.docs is None:
            await self.build()
        count = 2
        if self.passage_policy:
            count = min(4, bounded.maximum - len(bounded.records) - 1)
            if count <= 0:
                return {"status": "unavailable", "reason": "Reserve final report allowance", "evidence": []}
        matches = await self.docs.retrieve_texts(query, count, settings=self.settings, embedding_model=self.embedding)
        evidence = []
        for text in matches:
            if self.passage_policy and bounded.maximum - len(bounded.records) <= 1:
                break
            # Public map_fxn_summary retries malformed contexts. The pinned single-attempt
            # primitive is intentional; acceptance tests detect upstream API changes.
            context, _ = await _map_fxn_summary(
                text=text,
                question=query,
                summary_llm_model=GatewaySummaryModel(bounded),
                prompt_templates=(
                    self.settings.prompts.summary_json,
                    self.settings.prompts.summary_json_system,
                ),
                extra_prompt_data={
                    "summary_length": "at most 100 words",
                    "citation": text.doc.citation,
                },
                parser=llm_parse_json,
                evidence_text_only_fallback=False,
            )
            if context.score > 0:
                item = {
                        **self.lookup[text.name],
                        "excerpt": text.text,
                        "summary": context.context,
                        "relevance_rating": context.score,
                    }
                if self.passage_policy:
                    document = next(d for d in self.documents if text.name.startswith(d["id"] + "-char"))
                    offset = item["offset"]
                    if document["text"][offset:offset + len(text.text)] != text.text:
                        raise ValueError("Retrieved excerpt does not match admitted passage")
                    item.update(source_sha256=document["source_sha256"],
                                source_start=document["source_start"] + offset,
                                source_end=document["source_start"] + offset + len(text.text),
                                applicability=document["applicability"], limitations=document["limitations"])
                evidence.append(item)
        evidence.sort(key=lambda row: (-row["relevance_rating"], row["id"]))
        return {"status": "ok", "evidence": evidence}

    def search(self, query, bounded):
        return asyncio.run(self.asearch(query, bounded))
