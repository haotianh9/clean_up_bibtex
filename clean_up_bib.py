#!/usr/bin/env python3
"""
clean_bib_preprints_v4.py

Improvements vs v3:
- Adds OpenAlex search (excellent coverage for conference proceedings + older works).
- Treats conference proceedings (ENTRYTYPE=inproceedings) as "keep if unverified" (goes to --review).
- Treats manuals/specs/standards/RFCs as gray literature; keep if URL reachable (or RFC url reachable).
- Much more conservative "real discard": only discard if entry is information-poor AND unfound everywhere.
- Keeps arXiv/bioRxiv/medRxiv always; still tries to upgrade to published.

Outputs:
- cleaned bib: output.bib
- real discards (hard discard only): --discarded discarded_real.bib
- kept but unverified (review): --review needs_review.bib
- warnings: --warnings warnings.txt

Deps:
  pip install bibtexparser requests tqdm
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter
from tqdm.auto import tqdm


# -----------------------------
# Config
# -----------------------------
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "without",
    "via",
    "from",
    "by",
    "at",
    "as",
    "is",
    "are",
    "be",
    "being",
    "been",
    "we",
    "our",
    "their",
    "its",
    "into",
    "toward",
    "towards",
    "over",
    "under",
    "between",
    "within",
}

# Matching thresholds (recall-biased)
TITLE_SCORE_STRONG = 0.90  # accept if >= this regardless of author signals
TITLE_SCORE_MIN = 0.82  # minimum title score to consider
SCORE_ACCEPT = 0.78  # combined accept for non-strong titles

# Sources
SS_BASE = "https://api.semanticscholar.org/graph/v1"
SS_SEARCH = f"{SS_BASE}/paper/search"
SS_PAPER = f"{SS_BASE}/paper"
SS_FIELDS = ",".join(
    [
        "title",
        "authors",
        "year",
        "venue",
        "url",
        "externalIds",
        "doi",
        "journal",
        "publicationTypes",
        "publicationDate",
    ]
)

CR_BASE = "https://api.crossref.org/works"

# OpenAlex
OA_BASE = "https://api.openalex.org"
OA_WORKS = f"{OA_BASE}/works"

# arXiv API
ARXIV_API_QUERY = "http://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/"

# bioRxiv API
BIORXIV_API = "https://api.biorxiv.org"

USER_AGENT = "clean_bib_preprints_v4.py (mailto:you@example.com)"

SUBMITTED_MARKERS = [
    "submitted",
    "(submitted)",
    "in preparation",
    "(in preparation)",
    "in prepareation",
    "(in prepareation)",
    "in prep",
    "(in prep)",
]

GRAY_KEYWORDS = {
    "manual",
    "specification",
    "datasheet",
    "whitepaper",
    "technical report",
    "user guide",
    "documentation",
}


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class MatchResult:
    ok: bool
    score: float
    source: str
    data: Dict[str, Any]
    debug: Dict[str, Any]


# -----------------------------
# HTTP helpers
# -----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def get_json(
    sess: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 25,
    tries: int = 5,
    base_sleep: float = 0.8,
) -> Optional[Dict[str, Any]]:
    for i in range(tries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(base_sleep * (2**i))
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            time.sleep(base_sleep * (2**i))
    return None


def get_text(
    sess: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 25,
    tries: int = 5,
    base_sleep: float = 0.8,
) -> Optional[str]:
    for i in range(tries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(base_sleep * (2**i))
                continue
            if r.status_code != 200:
                return None
            return r.text
        except Exception:
            time.sleep(base_sleep * (2**i))
    return None


def url_reachable(sess: requests.Session, url: str, timeout: int = 15) -> bool:
    if not url:
        return False
    try:
        r = sess.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code < 400:
            return True
        r2 = sess.get(url, allow_redirects=True, timeout=timeout, stream=True)
        return r2.status_code < 400
    except Exception:
        return False


# -----------------------------
# Normalization
# -----------------------------
def _strip_outer_braces(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        return s[1:-1].strip()
    return s


def latex_to_text(s: str) -> str:
    if not s:
        return ""
    s = _strip_outer_braces(s)
    s = re.sub(r"\\[a-zA-Z]+\*?\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace(r"\&", "&")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_title(title: str) -> str:
    t = latex_to_text(title).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_words(title: str) -> List[str]:
    t = normalize_title(title)
    return [w for w in t.split() if w and w not in STOPWORDS]


def title_similarity_seq(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def title_similarity_token(a: str, b: str) -> float:
    wa, wb = set(title_words(a)), set(title_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def title_similarity_containment(a: str, b: str) -> float:
    wa, wb = set(title_words(a)), set(title_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def title_score(a: str, b: str) -> float:
    # take the maximum of multiple metrics to reduce punctuation/formatting sensitivity
    an = normalize_title(a)
    bn = normalize_title(b)
    return max(
        title_similarity_seq(an, bn),
        title_similarity_token(a, b),
        title_similarity_containment(a, b),
    )


def split_authors(author_field: str) -> List[str]:
    if not author_field:
        return []
    return [a.strip() for a in author_field.split(" and ") if a.strip()]


def author_last_name(author: str) -> str:
    a = latex_to_text(author).strip()
    if "," in a:
        last = a.split(",", 1)[0].strip()
    else:
        toks = a.split()
        last = toks[-1].strip() if toks else ""
    last = re.sub(r"[^a-zA-Z0-9]", "", last).lower()
    return last


def normalize_author_list(author_field: str) -> List[str]:
    lasts = [author_last_name(a) for a in split_authors(author_field)]
    return [x for x in lasts if x]


def author_containment(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    denom = max(1, min(len(sa), len(sb)))
    return inter / denom


def is_submitted_or_inprep(entry: Dict[str, Any]) -> bool:
    hay = " ".join(
        str(entry.get(k, "")).lower()
        for k in ("note", "annote", "howpublished", "status", "title")
    )
    return any(m in hay for m in SUBMITTED_MARKERS)


def parse_year(entry: Dict[str, Any]) -> Optional[int]:
    y = str(entry.get("year", "") or "").strip()
    return int(y) if y.isdigit() else None


# -----------------------------
# Keying / dedup
# -----------------------------
def first_significant_title_word(title: str) -> str:
    words = title_words(title)
    return words[0] if words else "untitled"


def slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def google_scholar_like_key(entry: Dict[str, Any]) -> str:
    authors = normalize_author_list(entry.get("author", ""))
    first_last = authors[0] if authors else "anon"
    year = re.sub(r"[^0-9]", "", str(entry.get("year", "") or ""))
    year = year if len(year) == 4 else "nd"
    tw = first_significant_title_word(entry.get("title", ""))
    return slug(f"{first_last}{year}{tw}")


def signature_for_dedup(entry: Dict[str, Any]) -> str:
    t = normalize_title(entry.get("title", ""))
    a = normalize_author_list(entry.get("author", ""))
    return f"{t}||{'|'.join(a)}"


def pick_better_entry(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    def is_preprint_doi(doi: str) -> bool:
        d = (doi or "").lower().strip()
        return (
            d.startswith("10.1101/")
            or d.startswith("10.21203/")
            or d.startswith("10.48550/arxiv.")
        )

    def score(e: Dict[str, Any]) -> Tuple[int, int, int, int]:
        doi = (e.get("doi") or "").strip()
        has_doi = 1 if doi else 0
        has_nonpreprint = 1 if (doi and not is_preprint_doi(doi)) else 0
        has_url = 1 if e.get("url") else 0
        nfields = len(e.keys())
        return (has_nonpreprint, has_doi, has_url, nfields)

    return a if score(a) >= score(b) else b


# -----------------------------
# Preprint detection (arXiv strict)
# -----------------------------
_ARXIV_NEW_RE = re.compile(r"^(?P<yymm>\d{4})\.(?P<num>\d{4,5})(v\d+)?$", re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)


def is_valid_arxiv_id(arx: str) -> bool:
    arx = (arx or "").strip()
    if not arx:
        return False
    m = _ARXIV_NEW_RE.match(arx)
    if m:
        mm = int(m.group("yymm")[2:4])
        return 1 <= mm <= 12
    return _ARXIV_OLD_RE.match(arx) is not None


def extract_arxiv_id(entry: Dict[str, Any]) -> Optional[str]:
    ap = str(entry.get("archiveprefix", "") or "").lower()
    eprint = str(entry.get("eprint", "") or "").strip()
    if "arxiv" in ap and is_valid_arxiv_id(eprint):
        return eprint

    url = str(entry.get("url", "") or "").strip()
    m = re.search(r"arxiv\.org/(abs|pdf)/(?P<id>[^?#\s/]+)", url, re.IGNORECASE)
    if m:
        arx = m.group("id").replace(".pdf", "")
        arx = re.sub(r"v\d+$", "", arx, flags=re.IGNORECASE)
        if is_valid_arxiv_id(arx):
            return arx

    for k in ("note", "howpublished"):
        v = str(entry.get(k, "") or "")
        m2 = re.search(
            r"arxiv:\s*(?P<id>[0-9]{4}\.[0-9]{4,5}(v\d+)?)", v, re.IGNORECASE
        )
        if m2:
            arx = m2.group("id")
            arx = re.sub(r"v\d+$", "", arx, flags=re.IGNORECASE)
            if is_valid_arxiv_id(arx):
                return arx

    doi = str(entry.get("doi", "") or "").strip()
    m3 = re.search(
        r"10\.48550/arxiv\.(?P<id>[0-9]{4}\.[0-9]{4,5}(v\d+)?)", doi, re.IGNORECASE
    )
    if m3:
        arx = m3.group("id")
        arx = re.sub(r"v\d+$", "", arx, flags=re.IGNORECASE)
        if is_valid_arxiv_id(arx):
            return arx

    return None


def is_arxiv_preprint(entry: Dict[str, Any]) -> bool:
    if "arxiv" in str(entry.get("archiveprefix", "") or "").lower():
        return True
    if extract_arxiv_id(entry):
        return True
    doi = str(entry.get("doi", "") or "").lower()
    return doi.startswith("10.48550/arxiv.")


def is_biorxiv_like(entry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    doi = str(entry.get("doi", "") or "").strip()
    url = str(entry.get("url", "") or "").lower()
    note = str(entry.get("note", "") or "").lower()
    if doi.lower().startswith("10.1101/"):
        return True, doi
    if "biorxiv.org" in url or "medrxiv.org" in url:
        return True, doi or None
    if "biorxiv" in note or "medrxiv" in note:
        return True, doi or None
    return False, None


def is_preprint(entry: Dict[str, Any]) -> bool:
    return is_arxiv_preprint(entry) or is_biorxiv_like(entry)[0]


# -----------------------------
# Gray literature (manual/spec/RFC)
# -----------------------------
def is_rfc_entry(entry: Dict[str, Any]) -> Optional[str]:
    key = str(entry.get("ID", "") or "").lower()
    title = normalize_title(entry.get("title", "") or "")
    m1 = re.search(r"\brfc\s*(\d{3,5})\b", key)
    if m1:
        return m1.group(1)
    m2 = re.search(r"\brfc\s*(\d{3,5})\b", title)
    if m2:
        return m2.group(1)
    return None


def guess_rfc_url(rfc_num: str) -> str:
    return f"https://www.rfc-editor.org/rfc/rfc{int(rfc_num)}.html"


def is_gray_literature(entry: Dict[str, Any]) -> bool:
    et = (entry.get("ENTRYTYPE") or "").lower()
    if et in ("manual", "misc", "techreport", "report", "standard"):
        return True
    rfc = is_rfc_entry(entry)
    if rfc:
        return True
    t = normalize_title(entry.get("title", "") or "")
    return any(k in t for k in GRAY_KEYWORDS)


def is_conference_proceeding(entry: Dict[str, Any]) -> bool:
    et = (entry.get("ENTRYTYPE") or "").lower()
    if et == "inproceedings":
        return True
    bt = normalize_title(entry.get("booktitle", "") or "")
    return ("proceedings" in bt) or ("conference" in bt) or ("symposium" in bt)


# -----------------------------
# URL helper
# -----------------------------
def ensure_url(entry: Dict[str, Any]) -> None:
    doi = (entry.get("doi") or "").strip()
    if doi and not entry.get("url"):
        entry["url"] = f"https://doi.org/{doi}"
        return
    if not entry.get("url"):
        arx = extract_arxiv_id(entry)
        if arx:
            entry["url"] = f"{ARXIV_ABS}{arx}"
            return
    if not entry.get("url"):
        is_bio, bio_doi = is_biorxiv_like(entry)
        if is_bio and bio_doi:
            entry["url"] = f"https://doi.org/{bio_doi}"


def append_annote(entry: Dict[str, Any], text: str) -> None:
    ann = (entry.get("annote") or "").strip()
    if text.lower() in ann.lower():
        return
    entry["annote"] = (ann + "; " + text).strip("; ").strip()


# -----------------------------
# Crossref enrichment
# -----------------------------
def crossref_get_by_doi(sess: requests.Session, doi: str) -> Optional[Dict[str, Any]]:
    doi = (doi or "").strip()
    if not doi:
        return None
    payload = get_json(sess, f"{CR_BASE}/{quote(doi, safe='')}")
    return (payload or {}).get("message") if payload else None


def crossref_search(
    sess: requests.Session, title: str, author_last: Optional[str], rows: int = 50
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"rows": rows, "query.title": title}
    if author_last:
        params["query.author"] = author_last
    payload = get_json(sess, CR_BASE, params=params)
    if not payload:
        return []
    return ((payload.get("message") or {}).get("items")) or []


def enrich_from_crossref_message(entry: Dict[str, Any], msg: Dict[str, Any]) -> None:
    if not msg:
        return
    # title
    tl = msg.get("title") or []
    if tl:
        entry["title"] = tl[0]

    # year
    if not entry.get("year"):
        issued = msg.get("issued") or {}
        parts = issued.get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            entry["year"] = str(parts[0][0])

    # DOI / URL
    doi = (msg.get("DOI") or "").strip()
    if doi:
        entry["doi"] = doi
        entry["url"] = f"https://doi.org/{doi}"

    # venue
    cont = msg.get("container-title") or []
    if cont:
        # heuristic: if already looks like proceedings
        if is_conference_proceeding(entry):
            entry.setdefault("booktitle", cont[0])
            entry.setdefault("ENTRYTYPE", "inproceedings")
        else:
            entry.setdefault("journal", cont[0])
            entry.setdefault("ENTRYTYPE", "article")

    # pages/volume/issue
    if msg.get("volume") and not entry.get("volume"):
        entry["volume"] = str(msg["volume"])
    if msg.get("issue") and not entry.get("number"):
        entry["number"] = str(msg["issue"])
    if msg.get("page") and not entry.get("pages"):
        entry["pages"] = str(msg["page"])


# -----------------------------
# Semantic Scholar matching
# -----------------------------
def ss_search(
    sess: requests.Session, query: str, limit: int = 100
) -> List[Dict[str, Any]]:
    payload = get_json(
        sess, SS_SEARCH, params={"query": query, "limit": limit, "fields": SS_FIELDS}
    )
    if not payload:
        return []
    return payload.get("data", []) or []


def ss_get_by_id(sess: requests.Session, paper_id: str) -> Optional[Dict[str, Any]]:
    return get_json(
        sess, f"{SS_PAPER}/{quote(paper_id, safe='')}", params={"fields": SS_FIELDS}
    )


def ss_enrich(entry: Dict[str, Any], data: Dict[str, Any]) -> None:
    if not data:
        return
    if data.get("title"):
        entry["title"] = data["title"]
    if data.get("year") and not entry.get("year"):
        entry["year"] = str(data["year"])

    doi = (data.get("doi") or "").strip()
    if not doi:
        ext = data.get("externalIds") or {}
        doi = (ext.get("DOI") or "").strip()
    if doi:
        entry["doi"] = doi
        entry["url"] = f"https://doi.org/{doi}"
    else:
        if data.get("url") and not entry.get("url"):
            entry["url"] = data["url"]

    venue = data.get("venue")
    if venue:
        if is_conference_proceeding(entry):
            entry.setdefault("booktitle", venue)
            entry.setdefault("ENTRYTYPE", "inproceedings")
        else:
            entry.setdefault("journal", venue)


def ss_best_match(
    sess: requests.Session, entry: Dict[str, Any]
) -> Optional[MatchResult]:
    title = entry.get("title", "") or ""
    if not title.strip():
        return None

    authors = normalize_author_list(entry.get("author", ""))
    first_last = authors[0] if authors else ""

    # Multi-query; include booktitle for conference items
    bt = entry.get("booktitle", "") or ""
    q_raw = latex_to_text(title)
    q_simp = " ".join(title_words(title)[:20])
    queries = [q_raw, q_simp]
    if first_last:
        queries += [f"{q_simp} {first_last}", f"{q_raw} {first_last}"]
    if bt and is_conference_proceeding(entry):
        queries += [
            f"{q_simp} {latex_to_text(bt)}",
            f"{q_simp} {first_last} {latex_to_text(bt)}".strip(),
        ]

    # unique, non-empty
    qs: List[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in qs:
            qs.append(q)

    best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None
    for q in qs:
        cands = ss_search(sess, q, limit=100)
        for c in cands:
            ctitle = c.get("title") or ""
            ts = title_score(title, ctitle)

            cauth = [
                author_last_name(a.get("name", "")) for a in (c.get("authors") or [])
            ]
            cauth = [x for x in cauth if x]
            contain = author_containment(authors, cauth)
            first_match = 1.0 if (authors and cauth and authors[0] == cauth[0]) else 0.0

            score = 0.86 * ts + 0.12 * contain + 0.02 * first_match
            dbg = {
                "title_score": ts,
                "author_contain": contain,
                "first_match": first_match,
                "query": q,
                "cand_title": ctitle,
            }
            if best is None or score > best[0]:
                best = (score, c, dbg)

    if best is None:
        return None

    score, data, dbg = best
    ts = float(dbg["title_score"])
    ok = (ts >= TITLE_SCORE_STRONG) or (ts >= TITLE_SCORE_MIN and score >= SCORE_ACCEPT)
    return MatchResult(
        ok=ok, score=score, source="semanticscholar", data=data, debug=dbg
    )


# -----------------------------
# OpenAlex matching (high-value for conferences/older works)
# -----------------------------
def oa_search(
    sess: requests.Session, query: str, per_page: int = 50
) -> List[Dict[str, Any]]:
    # Using "search" is simple and robust; OpenAlex also supports filters but not needed here
    payload = get_json(sess, OA_WORKS, params={"search": query, "per-page": per_page})
    if not payload:
        return []
    return payload.get("results", []) or []


def oa_best_match(
    sess: requests.Session, entry: Dict[str, Any]
) -> Optional[MatchResult]:
    title = entry.get("title", "") or ""
    if not title.strip():
        return None

    authors = normalize_author_list(entry.get("author", ""))
    first_last = authors[0] if authors else ""
    q1 = latex_to_text(title)
    q2 = " ".join(title_words(title)[:20])
    queries = [q1, q2]
    if first_last:
        queries.append(f"{q2} {first_last}".strip())

    best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None
    for q in queries:
        cands = oa_search(sess, q, per_page=50)
        for c in cands:
            ctitle = c.get("title") or ""
            ts = title_score(title, ctitle)

            # author last names from OpenAlex
            cauth = []
            for au in c.get("authorships") or []:
                a = (au.get("author") or {}).get("display_name") or ""
                cauth.append(author_last_name(a))
            cauth = [x for x in cauth if x]
            contain = author_containment(authors, cauth)
            first_match = 1.0 if (authors and cauth and authors[0] == cauth[0]) else 0.0

            score = 0.86 * ts + 0.12 * contain + 0.02 * first_match
            dbg = {
                "title_score": ts,
                "author_contain": contain,
                "first_match": first_match,
                "query": q,
                "cand_title": ctitle,
            }
            if best is None or score > best[0]:
                best = (score, c, dbg)

    if best is None:
        return None
    score, data, dbg = best
    ts = float(dbg["title_score"])
    ok = (ts >= TITLE_SCORE_STRONG) or (ts >= TITLE_SCORE_MIN and score >= SCORE_ACCEPT)
    return MatchResult(ok=ok, score=score, source="openalex", data=data, debug=dbg)


def oa_enrich(entry: Dict[str, Any], data: Dict[str, Any]) -> None:
    if not data:
        return
    if data.get("title"):
        entry["title"] = data["title"]
    if data.get("publication_year") and not entry.get("year"):
        entry["year"] = str(data["publication_year"])

    # OpenAlex provides DOI in "doi" as https://doi.org/...
    doi_url = (data.get("doi") or "").strip()
    if doi_url.startswith("https://doi.org/"):
        doi = doi_url.replace("https://doi.org/", "").strip()
        entry["doi"] = doi
        entry["url"] = doi_url

    # venue
    host = data.get("host_venue") or {}
    venue = host.get("display_name")
    if venue:
        if is_conference_proceeding(entry):
            entry.setdefault("booktitle", venue)
            entry.setdefault("ENTRYTYPE", "inproceedings")
        else:
            entry.setdefault("journal", venue)
            entry.setdefault("ENTRYTYPE", "article")


# -----------------------------
# arXiv title fallback
# -----------------------------
def arxiv_title_search(
    sess: requests.Session, title: str, max_results: int = 6
) -> Optional[str]:
    if not title.strip():
        return None
    q = f'ti:"{latex_to_text(title)}"'
    params = {"search_query": q, "start": 0, "max_results": max_results}
    txt = get_text(sess, ARXIV_API_QUERY, params=params)
    if not txt:
        return None
    try:
        root = ET.fromstring(txt)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return None
        best_id = None
        best_ts = 0.0
        for e in entries:
            tnode = e.find("atom:title", ns)
            if tnode is None:
                continue
            ts = title_score(title, tnode.text or "")
            if ts > best_ts:
                best_ts = ts
                idnode = e.find("atom:id", ns)
                if idnode is not None and idnode.text:
                    m = re.search(r"/abs/([^/]+)$", idnode.text.strip())
                    if m:
                        arx = re.sub(r"v\d+$", "", m.group(1), flags=re.IGNORECASE)
                        if is_valid_arxiv_id(arx):
                            best_id = arx
        return best_id if (best_id and best_ts >= 0.88) else None
    except Exception:
        return None


# -----------------------------
# bioRxiv upgrade
# -----------------------------
def biorxiv_pubs(sess: requests.Session, doi: str) -> Optional[Dict[str, Any]]:
    if not doi:
        return None
    for server in ("biorxiv", "medrxiv"):
        payload = get_json(
            sess, f"{BIORXIV_API}/pubs/{server}/{quote(doi, safe='')}/na/json"
        )
        if payload:
            coll = payload.get("collection") or []
            if coll:
                return {"server": server, "pubs": coll[0]}
    return None


def try_upgrade_preprint(
    sess: requests.Session, entry: Dict[str, Any], cache: Dict[str, Any], sleep_s: float
) -> None:
    # arXiv -> Semantic Scholar by arXiv id -> DOI -> Crossref
    arx = extract_arxiv_id(entry)
    if arx:
        ck = f"ss_arxiv::{arx}"
        data = cache.get(ck)
        if data is None:
            data = ss_get_by_id(sess, f"arXiv:{arx}") or {}
            cache[ck] = data
            time.sleep(sleep_s)
        if isinstance(data, dict) and data:
            ss_enrich(entry, data)
            doi = (entry.get("doi") or "").strip()
            if doi:
                msg = crossref_get_by_doi(sess, doi)
                if msg:
                    enrich_from_crossref_message(entry, msg)

    # bioRxiv -> published DOI
    is_bio, bio_doi = is_biorxiv_like(entry)
    if is_bio and bio_doi:
        pk = f"biorxiv_pubs::{bio_doi}"
        pubs = cache.get(pk)
        if pubs is None:
            pubs = biorxiv_pubs(sess, bio_doi) or {}
            cache[pk] = pubs
            time.sleep(sleep_s)
        if isinstance(pubs, dict) and pubs.get("pubs"):
            p = pubs["pubs"]
            published_doi = (p.get("published_doi") or "").strip()
            if published_doi and published_doi.lower() != bio_doi.lower():
                msg = crossref_get_by_doi(sess, published_doi)
                if msg:
                    enrich_from_crossref_message(entry, msg)
                else:
                    entry["doi"] = published_doi
                    entry["url"] = f"https://doi.org/{published_doi}"

    ensure_url(entry)


# -----------------------------
# Validation
# -----------------------------
def crossref_get_by_doi(sess: requests.Session, doi: str) -> Optional[Dict[str, Any]]:
    ck = f"{doi}".strip()
    if not ck:
        return None
    return crossref_get_by_doi_raw(sess, doi)


def crossref_get_by_doi_raw(
    sess: requests.Session, doi: str
) -> Optional[Dict[str, Any]]:
    doi = (doi or "").strip()
    if not doi:
        return None
    payload = get_json(sess, f"{CR_BASE}/{quote(doi, safe='')}")
    return (payload or {}).get("message") if payload else None


def validate_and_enrich_entry(
    sess: requests.Session, entry: Dict[str, Any], cache: Dict[str, Any], sleep_s: float
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns (verified, reason, debug).
    """
    title = (entry.get("title") or "").strip()
    if not title:
        return False, "missing title", {}

    # 1) DOI -> Crossref
    doi = (entry.get("doi") or "").strip()
    if doi:
        msg = crossref_get_by_doi_raw(sess, doi)
        if msg:
            # verify by title score
            cr_title = ((msg.get("title") or [""])[0]) if msg.get("title") else ""
            ts = title_score(title, cr_title)
            dbg = {"path": "crossref_doi", "title_score": ts, "cr_title": cr_title}
            if ts >= 0.84:
                enrich_from_crossref_message(entry, msg)
                ensure_url(entry)
                return True, "verified via Crossref DOI", dbg

    # 2) Semantic Scholar
    ms = ss_best_match(sess, entry)
    if ms and ms.ok:
        ss_enrich(entry, ms.data or {})
        doi2 = (entry.get("doi") or "").strip()
        if doi2:
            msg2 = crossref_get_by_doi_raw(sess, doi2)
            if msg2:
                enrich_from_crossref_message(entry, msg2)
        ensure_url(entry)
        return True, "verified via Semantic Scholar", ms.debug

    # 3) OpenAlex
    mo = oa_best_match(sess, entry)
    if mo and mo.ok:
        oa_enrich(entry, mo.data or {})
        doi3 = (entry.get("doi") or "").strip()
        if doi3:
            msg3 = crossref_get_by_doi_raw(sess, doi3)
            if msg3:
                enrich_from_crossref_message(entry, msg3)
        ensure_url(entry)
        return True, "verified via OpenAlex", mo.debug

    # 4) Crossref title search fallback
    authors = normalize_author_list(entry.get("author", ""))
    first_last = authors[0] if authors else None
    items = crossref_search(
        sess, title=" ".join(title_words(title)[:20]), author_last=first_last, rows=50
    )
    best_ts = 0.0
    best_it = None
    for it in items:
        ct = ((it.get("title") or [""])[0]) if it.get("title") else ""
        ts = title_score(title, ct)
        if ts > best_ts:
            best_ts = ts
            best_it = it
    if best_it and best_ts >= 0.88:
        doi4 = (best_it.get("DOI") or "").strip()
        dbg = {
            "path": "crossref_title",
            "title_score": best_ts,
            "doi": doi4,
            "cand_title": (best_it.get("title") or [""])[0],
        }
        if doi4:
            msg4 = crossref_get_by_doi_raw(sess, doi4)
            if msg4:
                enrich_from_crossref_message(entry, msg4)
            else:
                entry["doi"] = doi4
                entry["url"] = f"https://doi.org/{doi4}"
        ensure_url(entry)
        return True, "verified via Crossref title search", dbg

    return (
        False,
        "unverified (no acceptable match on Semantic Scholar/OpenAlex/Crossref)",
        (ms.debug if ms else {}),
    )


def should_hard_discard(entry: Dict[str, Any], found_anywhere: bool) -> bool:
    """
    Conservative "real discard":
    Only discard if information-poor AND not found anywhere.
    """
    if found_anywhere:
        return False
    if is_preprint(entry) or is_submitted_or_inprep(entry):
        return False
    if is_conference_proceeding(entry):
        return False
    if is_gray_literature(entry):
        return False

    title = (entry.get("title") or "").strip()
    if not title:
        return True

    authors = (entry.get("author") or "").strip()
    year = (entry.get("year") or "").strip()
    doi = (entry.get("doi") or "").strip()
    url = (entry.get("url") or "").strip()

    # information-poor criteria
    poor = (not doi) and (not url) and (not authors) and (not year)
    return poor


def validate_and_enrich_all(
    entries: List[Dict[str, Any]], cache: Dict[str, Any], sleep_s: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    sess = make_session()
    kept: List[Dict[str, Any]] = []
    discarded_real: List[Dict[str, Any]] = []
    kept_review: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for e in tqdm(entries, desc="Validating/enriching", unit="entry"):
        title = (e.get("title") or "").strip()
        if not title:
            d = dict(e)
            append_annote(d, "DISCARDED: missing title")
            discarded_real.append(d)
            warnings.append(f"DROP (missing title): {e.get('ID','<noid>')}")
            continue

        if is_submitted_or_inprep(e):
            ensure_url(e)
            kept.append(e)
            continue

        if is_preprint(e):
            try_upgrade_preprint(sess, e, cache, sleep_s)
            ensure_url(e)
            kept.append(e)
            continue

        # Gray literature: keep if URL reachable or RFC reachable
        if is_gray_literature(e):
            rfc_num = is_rfc_entry(e)
            if rfc_num and not e.get("url"):
                e["url"] = guess_rfc_url(rfc_num)
            if e.get("url") and url_reachable(sess, str(e["url"]), timeout=12):
                ensure_url(e)
                kept.append(e)
                continue
            # Even if not reachable now, keep in review instead of discarding
            append_annote(
                e, "KEPT_UNVERIFIED_GRAY: not indexed/URL not reachable during run"
            )
            ensure_url(e)
            kept.append(e)
            kept_review.append(dict(e))
            warnings.append(
                f"KEEP (gray literature): {e.get('ID','<noid>')} | title={latex_to_text(title)[:140]}"
            )
            continue

        # Validate via bibliographic sources
        verified, reason, dbg = validate_and_enrich_entry(sess, e, cache, sleep_s)

        if verified:
            kept.append(e)
            continue

        # arXiv title fallback (covers arXiv even if BibTeX lacks arXiv id)
        arx = arxiv_title_search(sess, title)
        if arx:
            e["archiveprefix"] = "arXiv"
            e["eprint"] = arx
            e["url"] = f"{ARXIV_ABS}{arx}"
            append_annote(e, "KEPT: found via arXiv title search")
            kept.append(e)
            kept_review.append(dict(e))
            warnings.append(
                f"KEEP (arXiv title match): {e.get('ID','<noid>')} | arXiv:{arx}"
            )
            continue

        # Conference proceedings: keep (review) instead of discard
        if is_conference_proceeding(e):
            append_annote(e, f"KEPT_UNVERIFIED_PROCEEDINGS: {reason}")
            if dbg:
                append_annote(e, f"DEBUG: {json.dumps(dbg, ensure_ascii=False)}")
            ensure_url(e)
            kept.append(e)
            kept_review.append(dict(e))
            warnings.append(
                f"KEEP (unverified proceedings): {e.get('ID','<noid>')} | title={latex_to_text(title)[:140]}"
            )
            continue

        # Decide: hard discard or keep in review
        found_anywhere = False  # at this point, none of the methods found a match
        if should_hard_discard(e, found_anywhere=found_anywhere):
            d = dict(e)
            append_annote(d, f"DISCARDED: {reason}")
            if dbg:
                append_annote(d, f"DEBUG: {json.dumps(dbg, ensure_ascii=False)}")
            discarded_real.append(d)
            warnings.append(
                f"DROP ({reason}): {e.get('ID','<noid>')} | title={latex_to_text(title)[:140]}"
            )
        else:
            append_annote(e, f"KEPT_UNVERIFIED: {reason}")
            if dbg:
                append_annote(e, f"DEBUG: {json.dumps(dbg, ensure_ascii=False)}")
            ensure_url(e)
            kept.append(e)
            kept_review.append(dict(e))
            warnings.append(
                f"KEEP (unverified): {e.get('ID','<noid>')} | title={latex_to_text(title)[:140]}"
            )

    return kept, discarded_real, kept_review, warnings


# -----------------------------
# Dedup / rekey / IO
# -----------------------------
def deduplicate(
    entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    by_sig: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        sig = signature_for_dedup(e)
        if sig in by_sig:
            kept = pick_better_entry(by_sig[sig], e)
            dropped = e if kept is by_sig[sig] else by_sig[sig]
            by_sig[sig] = kept
            warnings.append(
                f"DEDUP: dropped {dropped.get('ID','<noid>')} (kept {kept.get('ID','<noid>')})"
            )
        else:
            by_sig[sig] = e
    return list(by_sig.values()), warnings


def rekey(entries: List[Dict[str, Any]]) -> None:
    used: Dict[str, int] = {}
    for e in entries:
        base = google_scholar_like_key(e)
        if base in used:
            used[base] += 1
            idx = used[base]
            suffix = chr(ord("a") + idx - 1) if idx <= 26 else str(idx)
            e["ID"] = f"{base}{suffix}"
        else:
            used[base] = 0
            e["ID"] = base


def load_cache(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(path: Optional[str], cache: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def write_bib(entries: List[Dict[str, Any]], out_path: str) -> None:
    db = BibDatabase()
    db.entries = entries
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = ("ID",)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(writer.write(db))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_bib")
    ap.add_argument("output_bib")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--warnings", default=None)
    ap.add_argument(
        "--discarded", default=None, help="Real discards only (not dedup drops)"
    )
    ap.add_argument("--review", default=None, help="Kept but unverified")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    cache = load_cache(args.cache)

    with open(args.input_bib, "r", encoding="utf-8") as f:
        bib_db = bibtexparser.load(f)
    raw_entries = bib_db.entries

    deduped, w_d1 = deduplicate(raw_entries)
    kept, discarded_real, kept_review, w_v = validate_and_enrich_all(
        deduped, cache=cache, sleep_s=args.sleep
    )
    final_entries, w_d2 = deduplicate(kept)

    rekey(final_entries)
    rekey(discarded_real)
    rekey(kept_review)

    write_bib(final_entries, args.output_bib)
    if args.discarded:
        write_bib(discarded_real, args.discarded)
    if args.review:
        write_bib(kept_review, args.review)

    save_cache(args.cache, cache)

    warnings = w_d1 + w_v + w_d2
    for w in warnings:
        print(w, file=sys.stderr)
    if args.warnings:
        with open(args.warnings, "w", encoding="utf-8") as f:
            for w in warnings:
                f.write(w + "\n")

    print(
        f"Wrote {len(final_entries)} entries to {args.output_bib}. "
        f"Real discards: {len(discarded_real)}. "
        f"Review: {len(kept_review)}. "
        f"Warnings: {len(warnings)}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
