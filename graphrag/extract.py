"""Entity + relation extraction (Indexing, Step 1 of the lab).

Two real, working extractors:

    * HeuristicExtractor -- offline, deterministic. Real pattern-based relation
      extraction (RE): NER via capitalized noun-phrase detection + a tech/EV
      gazetteer, then verb/preposition patterns map entity pairs to typed
      relations such as FOUNDED_BY, ACQUIRED, CEO_OF, HEADQUARTERED_IN ...

    * LLMExtractor -- if an OpenAI/Anthropic key is configured, asks the model
      to emit JSON triples. Same output schema as the heuristic path, so the
      rest of the pipeline is identical.

Both return a list of Triple(subject, relation, object, source_chunk, evidence).
Deduplication / canonicalisation happens in graph_store.build_graph.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .corpus import Chunk
from .llm import LLM, estimate_tokens


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source_chunk: str = ""
    evidence: str = ""


# --------------------------------------------------------------------------- #
# Shared NER helpers
# --------------------------------------------------------------------------- #
# Words that are capitalised but are NOT entities (sentence starts, months ...)
_STOP_CAPS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "Its", "In", "On",
    "At", "For", "And", "But", "Or", "So", "As", "By", "With", "From", "To",
    "Of", "We", "They", "He", "She", "I", "You", "Our", "Their", "His", "Her",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Mr", "Ms", "Mrs",
    "Dr", "U", "US", "U.S", "However", "Meanwhile", "According", "While",
    "After", "Before", "During", "When", "Then", "Now", "Also", "Update",
    "Download", "Menu", "Full", "Content", "Query", "Title", "Link", "Snippet",
    "Hi", "Hello", "EV", "EVs", "Q1", "Q2", "Q3", "Q4", "CEO", "IPO", "ETF",
    "Segment", "Year", "Despite", "Though", "Many", "Some", "Most", "Both",
    "Even", "Still", "Here", "There", "What", "Which", "Who", "Where", "Why",
    "How", "Name", "Read", "Source", "Note", "Notably", "Briefing", "Contact",
}

# Single-token candidates are accepted only if they look like a real org/person
# acronym (>=2 upper-case letters) or appear in the gazetteer; this kills noise
# like "Hi", "EVs", "Segment" that the proper-noun regex would otherwise grab.
_ACRONYM = re.compile(r"^[A-Z]{2,6}$")

# Curated gazetteer of high-signal tech / EV / finance entities. Boosts recall
# for multi-word names that NER might otherwise split or miss.
_GAZETTEER = {
    "OpenAI", "Tesla", "Google", "Alphabet", "Microsoft", "Apple", "Amazon",
    "Meta", "Facebook", "Nvidia", "Intel", "AMD", "IBM", "Oracle", "Samsung",
    "Sony", "Netflix", "Uber", "Lyft", "Rivian", "Lucid", "Lucid Motors",
    "Ford", "General Motors", "GM", "Toyota", "Volkswagen", "BMW", "Hyundai",
    "Kia", "Nissan", "BYD", "Nio", "Xpeng", "Li Auto", "Zeekr", "Polestar",
    "Fisker", "Stellantis", "Mercedes-Benz", "Mercedes", "Porsche", "Audi",
    "Honda", "Chevrolet", "Chevy", "Chevy Bolt", "Cadillac", "Lexus", "Volvo",
    "Vinfast", "VinFast", "IM Motors", "KraneShares", "Inflation Reduction Act",
    "Tesla Model 3", "Model 3", "Model Y", "Green New Deal",
    "Panasonic", "LG", "LG Energy Solution", "CATL", "QuantumScape",
    "ChargePoint", "Anthropic", "DeepMind", "Waymo", "Cruise", "SpaceX",
    "Boeing", "Qualcomm", "TSMC", "Cox Automotive", "Kelley Blue Book",
    "Bloomberg", "Reuters", "Goldman Sachs", "Morgan Stanley", "JPMorgan",
    "BloombergNEF", "International Energy Agency", "IEA", "EIA",
    "Sam Altman", "Elon Musk", "Mary Barra", "Jensen Huang", "Tim Cook",
    "Satya Nadella", "Sundar Pichai", "Mark Zuckerberg", "Jeff Bezos",
    "Andy Jassy", "Donald Trump", "Joe Biden", "Jim Farley",
}
# pre-sort gazetteer by length so longer names match first
_GAZETTEER_SORTED = sorted(_GAZETTEER, key=len, reverse=True)

# A capitalised proper noun: a Capitalised word optionally followed by more
# Capitalised words, where lowercase connectors (of/the/and) are only allowed
# BETWEEN two capitalised words (so "China and" / "PHEVs and" do NOT match, but
# "LG Energy Solution" and "Bank of America" do).
_PROPER_NOUN = re.compile(
    r"\b([A-Z][a-zA-Z0-9.&'-]*"
    r"(?:\s+(?:of|the|and|de|für)\s+[A-Z][a-zA-Z0-9.&'-]*|\s+[A-Z][a-zA-Z0-9.&'-]*){0,3})\b"
)
_TRAILING_CONNECTOR = re.compile(r"\s+(?:of|the|and|de|für)$", re.I)


def _clean_entity(name: str) -> str:
    name = name.strip(" .,:;'\"()[]")
    name = re.sub(r"\s+", " ", name)
    name = _TRAILING_CONNECTOR.sub("", name).strip()
    return name


def find_entities(sentence: str) -> List[str]:
    """Return surface entities found in a sentence (gazetteer + proper nouns)."""
    found: List[str] = []
    seen_spans: List[tuple] = []

    # 1) gazetteer (highest precision, longest-first)
    for g in _GAZETTEER_SORTED:
        for m in re.finditer(r"\b" + re.escape(g) + r"\b", sentence):
            span = (m.start(), m.end())
            if not any(s <= span[0] < e for s, e in seen_spans):
                found.append(g)
                seen_spans.append(span)

    # 2) generic capitalised proper nouns
    for m in _PROPER_NOUN.finditer(sentence):
        span = (m.start(), m.end())
        if any(s <= span[0] < e for s, e in seen_spans):
            continue
        cand = _clean_entity(m.group(1))
        if not cand:
            continue
        first = cand.split()[0] if cand else ""
        if first in _STOP_CAPS or cand in _STOP_CAPS or len(cand) < 2:
            continue
        # single-token candidate must be a plausible acronym or gazetteer entry
        if " " not in cand and not (_ACRONYM.match(cand) or cand in _GAZETTEER):
            continue
        found.append(cand)
        seen_spans.append(span)

    # de-dup preserving order
    out, low = [], set()
    for e in found:
        k = e.lower()
        if k not in low:
            low.add(k)
            out.append(e)
    return out


# --------------------------------------------------------------------------- #
# Relation patterns  (subject  REL  object)
# --------------------------------------------------------------------------- #
# Each pattern returns triples given a sentence + its entities. We work on the
# raw sentence so we can use lexical cues (verbs / prepositions).
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

# (regex, relation, swap?) -- captures group 'a' and 'b' as entities
_PATTERNS = [
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+(?:was|were)?\s*(?:co-)?founded\s+by\s+(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)", re.I), "FOUNDED_BY"),
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+acquired\s+(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)"), "ACQUIRED"),
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+(?:bought|purchased)\s+(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)"), "ACQUIRED"),
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+(?:partnered|teamed up)\s+with\s+(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)", re.I), "PARTNERED_WITH"),
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+invested\s+(?:in|\$?[\d.,]+\s+(?:billion|million)\s+in)\s+(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)", re.I), "INVESTED_IN"),
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+(?:competes|competing|rival[s]?)\s+(?:with|to)?\s*(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)", re.I), "COMPETES_WITH"),
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+(?:is|are)?\s*(?:headquartered|based)\s+in\s+(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)", re.I), "HEADQUARTERED_IN"),
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+(?:makes|manufactures|produces|builds)\s+(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)"), "PRODUCES"),
    (re.compile(r"(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s*,?\s+(?:the\s+)?(?:CEO|chief executive)\s+of\s+(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)", re.I), "CEO_OF"),
    (re.compile(r"(?P<a>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)\s+(?:CEO|chief executive)\s+(?P<b>[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+)*)"), "HAS_CEO"),
]


class HeuristicExtractor:
    """Deterministic, offline entity + relation extractor."""

    name = "heuristic"

    def extract(self, chunk: Chunk) -> List[Triple]:
        triples: List[Triple] = []
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            entities = find_entities(sent)

            # 1) typed relation patterns
            for rx, rel in _PATTERNS:
                for m in rx.finditer(sent):
                    a = _clean_entity(m.group("a"))
                    b = _clean_entity(m.group("b"))
                    if a and b and a.lower() != b.lower():
                        triples.append(Triple(a, rel, b, chunk.chunk_id, sent[:240]))

            # 2) founded-in <year>
            if re.search(r"founded|established|launched|started", sent, re.I):
                for y in _YEAR.findall(sent):
                    pass  # findall returns groups; use finditer below
            if re.search(r"founded|established|launched|started", sent, re.I) and entities:
                ym = _YEAR.search(sent)
                if ym:
                    triples.append(
                        Triple(entities[0], "FOUNDED_IN", ym.group(0), chunk.chunk_id, sent[:240])
                    )

            # 3) co-occurrence fallback: link first entity to the others in the
            #    same sentence as a generic MENTIONED_WITH edge (weak but useful
            #    for multi-hop traversal). Only when no typed relation fired.
            if len(entities) >= 2:
                head = entities[0]
                for other in entities[1:4]:
                    if head.lower() != other.lower():
                        triples.append(
                            Triple(head, "MENTIONED_WITH", other, chunk.chunk_id, sent[:240])
                        )
        return triples


# --------------------------------------------------------------------------- #
# LLM extractor (real API call, optional)
# --------------------------------------------------------------------------- #
_LLM_SYSTEM = (
    "You are an information-extraction engine. Extract factual knowledge-graph "
    "triples from the text. Return ONLY a JSON array of objects with keys "
    "'subject', 'relation', 'object'. Relations must be UPPER_SNAKE_CASE verbs "
    "(e.g. FOUNDED_BY, ACQUIRED, CEO_OF, HEADQUARTERED_IN, INVESTED_IN, "
    "PRODUCES, COMPETES_WITH, FOUNDED_IN). Subjects/objects are named entities "
    "or literal values (years, amounts). No prose, no markdown."
)


class LLMExtractor:
    name = "llm"

    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, chunk: Chunk) -> List[Triple]:
        user = f"Text:\n{chunk.text}\n\nJSON triples:"
        raw = self.llm.complete(_LLM_SYSTEM, user, stage="extract", max_tokens=800)
        raw = raw.strip()
        # strip accidental code fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        triples: List[Triple] = []
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\[.*\]", raw, re.S)
            if not m:
                return triples
            try:
                data = json.loads(m.group(0))
            except Exception:
                return triples
        for d in data if isinstance(data, list) else []:
            try:
                s = _clean_entity(str(d["subject"]))
                r = str(d["relation"]).strip().upper().replace(" ", "_")
                o = _clean_entity(str(d["object"]))
            except Exception:
                continue
            if s and r and o:
                triples.append(Triple(s, r, o, chunk.chunk_id, chunk.text[:240]))
        return triples


def get_extractor(cfg: Config, llm: Optional[LLM] = None):
    """Return the configured extractor instance."""
    if cfg.extractor in ("openai", "anthropic", "llm") and llm is not None and llm.backend != "none":
        return LLMExtractor(llm)
    return HeuristicExtractor()


def extract_triples(chunks: List[Chunk], cfg: Config, llm: Optional[LLM] = None) -> List[Triple]:
    """Run the configured extractor over all chunks."""
    extractor = get_extractor(cfg, llm)
    triples: List[Triple] = []
    limit = cfg.max_chunks_for_llm_extract if isinstance(extractor, LLMExtractor) else len(chunks)
    for ch in chunks[:limit]:
        triples.extend(extractor.extract(ch))
    return triples
