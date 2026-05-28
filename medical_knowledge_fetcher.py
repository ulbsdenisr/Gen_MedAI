"""
medical_knowledge_fetcher.py
============================
Fetches medical information for a disease from Wikipedia and Mayo Clinic.
Results are cached locally in medical_cache/ as JSON files.

Usage:
    from medical_knowledge_fetcher import get_disease_info
    info = get_disease_info("anxiety")
    # Returns: {overview, causes, treatment, prevention, when_to_see_a_doctor, sources}
"""

from __future__ import annotations

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = Path("medical_cache")
CACHE_DIR.mkdir(exist_ok=True)

REQUEST_TIMEOUT = 10
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MedAI-Research-Bot/1.0)"
}

# Sectiuni Wikipedia relevante
WIKI_SECTIONS_OVERVIEW   = ["signs and symptoms", "symptoms", "presentation", "description", "overview"]
WIKI_SECTIONS_CAUSES     = ["causes", "cause", "etiology", "pathophysiology", "risk factors"]
WIKI_SECTIONS_TREATMENT  = ["treatment", "management", "therapy", "therapies", "treatments",
                            "intervention", "medication", "pharmacotherapy", "antibiotics",
                            "medical treatment", "clinical management"]
WIKI_SECTIONS_PREVENTION = ["prevention", "prognosis and prevention"]
WIKI_SECTIONS_DOCTOR     = ["complications", "when to see a doctor", "diagnosis", "prognosis"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(disease: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", disease.strip().lower())


def _cache_path(disease: str) -> Path:
    return CACHE_DIR / f"{_cache_key(disease)}.json"


def _load_cache(disease: str) -> Optional[Dict]:
    p = _cache_path(disease)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf8"))
        except Exception:
            pass
    return None


def _save_cache(disease: str, data: Dict):
    try:
        _cache_path(disease).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf8"
        )
    except Exception as e:
        print(f"[Fetcher] Cache write failed: {e}")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)          # remove [1], [2] citations
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(".", 1)
    return (cut[0] + ".").strip() if len(cut) > 1 else text[:max_chars].strip() + "..."


# ---------------------------------------------------------------------------
# Wikipedia fetcher
# ---------------------------------------------------------------------------

def _wiki_search_url(disease: str) -> str:
    q = disease.replace(" ", "+")
    return f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit=3"


def _wiki_page_url(title: str) -> str:
    t = title.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{t}"


def _fetch_wikipedia(disease: str) -> Dict[str, Any]:
    result = {"overview": "", "causes": "", "treatment": "", "prevention": "",
              "when_to_see_a_doctor": "", "source_title": "", "source_url": ""}
    try:
        # 1. Search for best matching article title
        search_resp = requests.get(
            _wiki_search_url(disease),
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        search_data = search_resp.json()
        hits = search_data.get("query", {}).get("search", [])
        if not hits:
            return result

        # Pick best title match
        disease_lower = disease.lower()
        page_title = None
        for hit in hits:
            title_lower = hit["title"].lower()
            disease_words = set(disease_lower.split())
            title_words = set(title_lower.split())
            if len(disease_words & title_words) >= max(1, len(disease_words) - 1):
                page_title = hit["title"]
                break
        if not page_title:
            page_title = hits[0]["title"]

        page_url = _wiki_page_url(page_title)
        result["source_title"] = f"Wikipedia: {page_title}"
        result["source_url"] = page_url

        # 2. Fetch sections via Wikipedia REST API (returns clean JSON)
        encoded_title = page_title.replace(" ", "_")
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{encoded_title}"
        api_resp = requests.get(api_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)

        if api_resp.status_code != 200:
            # Fallback to HTML scraping
            return _fetch_wikipedia_html(disease, page_title, page_url)

        data = api_resp.json()

        # 3. Collect all sections as (title, text) pairs
        sections = []

        # Lead section (overview)
        lead = data.get("lead", {})
        lead_sections = lead.get("sections", [])
        if lead_sections:
            lead_text = BeautifulSoup(
                lead_sections[0].get("text", ""), "html.parser"
            ).get_text()
            lead_text = _clean(lead_text)
            if lead_text:
                sections.append(("overview", lead_text))
                result["overview"] = _truncate(lead_text)

        # Remaining sections
        remaining = data.get("remaining", {}).get("sections", [])
        for sec in remaining:
            title = sec.get("title", "").strip().lower()
            text_html = sec.get("text", "")
            text = _clean(BeautifulSoup(text_html, "html.parser").get_text())
            if text and len(text) > 30:
                sections.append((title, text))

        # 4. RAG-style semantic matching for each field
        if sections:
            try:
                from sentence_transformers import SentenceTransformer, util
                import torch

                model = SentenceTransformer("all-MiniLM-L6-v2")

                section_texts = [f"{t}: {c[:300]}" for t, c in sections]
                section_embs = model.encode(section_texts, convert_to_tensor=True, show_progress_bar=False)

                def find_best_section(query: str, exclude_titles: list = None) -> str:
                    q_emb = model.encode(query, convert_to_tensor=True)
                    scores = util.cos_sim(q_emb, section_embs)[0]

                    # Sort by score
                    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)

                    for idx, score in ranked:
                        sec_title, sec_text = sections[idx]
                        if exclude_titles and any(e in sec_title for e in exclude_titles):
                            continue
                        if score > 0.25 and len(sec_text) > 40:
                            return _truncate(sec_text)
                    return ""

                result["causes"]               = find_best_section(
                    f"causes risk factors etiology of {disease}", exclude_titles=["overview"])
                result["treatment"]            = find_best_section(
                    f"treatment management therapy medication for {disease}", exclude_titles=["overview"])
                result["prevention"]           = find_best_section(
                    f"prevention how to prevent {disease}", exclude_titles=["overview"])
                result["when_to_see_a_doctor"] = find_best_section(
                    f"when to see a doctor complications prognosis {disease}", exclude_titles=["overview"])

            except ImportError:
                # Fallback to keyword matching if sentence_transformers not available
                field_keywords = {
                    "causes":               WIKI_SECTIONS_CAUSES,
                    "treatment":            WIKI_SECTIONS_TREATMENT,
                    "prevention":           WIKI_SECTIONS_PREVENTION,
                    "when_to_see_a_doctor": WIKI_SECTIONS_DOCTOR,
                }
                for field, keywords in field_keywords.items():
                    for title, text in sections:
                        if any(k in title for k in keywords):
                            result[field] = _truncate(text)
                            break

    except Exception as e:
        print(f"[Fetcher] Wikipedia error for '{disease}': {e}")

    return result


def _fetch_wikipedia_html(disease: str, page_title: str, page_url: str) -> Dict[str, Any]:
    """Fallback HTML scraping when REST API fails."""
    result = {"overview": "", "causes": "", "treatment": "", "prevention": "",
              "when_to_see_a_doctor": "", "source_title": f"Wikipedia: {page_title}",
              "source_url": page_url}
    try:
        page_resp = requests.get(page_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(page_resp.text, "html.parser")

        content_div = soup.find("div", {"id": "mw-content-text"})
        if content_div:
            lead_paras = content_div.find_all("p", recursive=False)[:3]
            overview_parts = [_clean(p.get_text()) for p in lead_paras if len(_clean(p.get_text())) > 40]
            result["overview"] = _truncate(" ".join(overview_parts))

        def extract_section(section_names):
            for heading in soup.find_all(["h2", "h3"]):
                heading_text = re.sub(r'\[.*?\]', '', heading.get_text()).strip().lower()
                if any(s in heading_text for s in section_names):
                    texts = []
                    for sibling in heading.find_next_siblings():
                        tag = sibling.name
                        if tag in ["h2", "h3"]:
                            break
                        if tag == "p":
                            t = _clean(sibling.get_text())
                            if len(t) > 20:
                                texts.append(t)
                        if sum(len(t) for t in texts) > 600:
                            break
                    if texts:
                        return _truncate(" ".join(texts))
            return ""

        result["causes"]               = extract_section(WIKI_SECTIONS_CAUSES)
        result["treatment"]            = extract_section(WIKI_SECTIONS_TREATMENT)
        result["prevention"]           = extract_section(WIKI_SECTIONS_PREVENTION)
        result["when_to_see_a_doctor"] = extract_section(WIKI_SECTIONS_DOCTOR)

    except Exception as e:
        print(f"[Fetcher] Wikipedia HTML fallback error: {e}")

    return result


# ---------------------------------------------------------------------------
# Mayo Clinic fetcher
# ---------------------------------------------------------------------------

def _mayo_search_url(disease: str) -> str:
    q = disease.replace(" ", "+")
    return f"https://www.mayoclinic.org/search/search-results?q={q}"


def _fetch_mayo_clinic(disease: str) -> Dict[str, Any]:
    result = {"overview": "", "causes": "", "treatment": "", "prevention": "",
              "when_to_see_a_doctor": "", "source_title": "", "source_url": ""}
    try:
        search_resp = requests.get(
            _mayo_search_url(disease),
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        soup = BeautifulSoup(search_resp.text, "html.parser")

        # Find first symptoms-causes link
        symptoms_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/diseases-conditions/" in href and "symptoms-causes" in href:
                symptoms_link = href if href.startswith("http") else "https://www.mayoclinic.org" + href
                break

        if not symptoms_link:
            return result

        result["source_title"] = f"Mayo Clinic: {disease.title()}"
        result["source_url"] = symptoms_link

        def get_section_text(page_soup, keywords):
            for h in page_soup.find_all(["h2", "h3"]):
                if any(k in h.get_text().lower() for k in keywords):
                    texts = []
                    for sib in h.next_siblings:
                        tag = getattr(sib, "name", None)
                        if tag in ["h2", "h3"]:
                            break
                        if tag == "p":
                            t = _clean(sib.get_text())
                            if len(t) > 20:
                                texts.append(t)
                        if sum(len(t) for t in texts) > 600:
                            break
                    if texts:
                        return _truncate(" ".join(texts))
            return ""

        # Fetch symptoms-causes page
        page_resp = requests.get(symptoms_link, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        soup1 = BeautifulSoup(page_resp.text, "html.parser")

        meta = soup1.find("meta", {"name": "description"})
        if meta and meta.get("content"):
            result["overview"] = _truncate(_clean(meta["content"]))
        else:
            result["overview"] = get_section_text(soup1, ["overview", "what is"])

        result["causes"]               = get_section_text(soup1, ["cause", "risk factor"])
        result["when_to_see_a_doctor"] = get_section_text(soup1, ["when to see", "see a doctor", "seek medical"])

        # Also fetch diagnosis-treatment page for treatment and prevention
        treatment_link = symptoms_link.replace("symptoms-causes", "diagnosis-treatment")
        try:
            time.sleep(0.2)
            treat_resp = requests.get(treatment_link, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            soup2 = BeautifulSoup(treat_resp.text, "html.parser")
            result["treatment"]  = get_section_text(soup2, ["treatment", "medication", "therapy", "management"])
            result["prevention"] = get_section_text(soup2, ["prevention", "prevent", "lifestyle"])
        except Exception:
            pass

        # Fallback prevention from symptoms page
        if not result["prevention"]:
            result["prevention"] = get_section_text(soup1, ["prevention", "prevent"])

    except Exception as e:
        print(f"[Fetcher] Mayo Clinic error for '{disease}': {e}")

    return result


# ---------------------------------------------------------------------------
# Merge results
# ---------------------------------------------------------------------------

def _merge(wiki: Dict, mayo: Dict) -> Dict[str, Any]:
    """Merge Wikipedia and Mayo Clinic results, preferring the longer/more complete."""

    def best(field: str) -> str:
        w = wiki.get(field, "")
        m = mayo.get(field, "")
        if not w and not m:
            return ""
        if not w:
            return m
        if not m:
            return w
        # Prefer the longer one, up to a point
        return w if len(w) >= len(m) else m

    sources = []
    if wiki.get("source_title"):
        sources.append({
            "title": wiki["source_title"],
            "url": wiki.get("source_url", ""),
            "journal": "Wikipedia"
        })
    if mayo.get("source_title"):
        sources.append({
            "title": mayo["source_title"],
            "url": mayo.get("source_url", ""),
            "journal": "Mayo Clinic"
        })

    # Pentru treatment folosim Wikipedia ca sursa primara (Mayo il pune pe pagina separata)
    def best_treatment():
        w = wiki.get("treatment", "")
        m = mayo.get("treatment", "")
        if w and len(w) > 30:
            return w
        if m and len(m) > 30:
            return m
        return ""

    return {
        "overview":             best("overview")             or "No information found.",
        "causes":               best("causes")               or "No information found.",
        "treatment":            best_treatment()             or "No information found.",
        "prevention":           best("prevention")           or "No information found.",
        "when_to_see_a_doctor": best("when_to_see_a_doctor") or "No information found.",
        "sources": sources,
        "_fetched_from": "wikipedia+mayo_clinic"
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_disease_info(disease: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get medical information for a disease.
    Returns cached result if available, otherwise fetches from Wikipedia + Mayo Clinic.

    Args:
        disease: disease name (e.g. "anxiety", "type 2 diabetes")
        force_refresh: ignore cache and re-fetch

    Returns:
        dict with keys: overview, causes, treatment, prevention,
                        when_to_see_a_doctor, sources
    """
    if not force_refresh:
        cached = _load_cache(disease)
        if cached:
            print(f"[Fetcher] Cache hit: {disease}")
            return cached

    print(f"[Fetcher] Fetching: {disease}")

    wiki = _fetch_wikipedia(disease)
    time.sleep(0.3)  # polite delay
    mayo = _fetch_mayo_clinic(disease)

    merged = _merge(wiki, mayo)
    _save_cache(disease, merged)

    return merged


def get_disease_info_safe(disease: str) -> Dict[str, Any]:
    """
    Safe wrapper — never raises, always returns a valid dict.
    Falls back to empty strings if fetch fails.
    """
    try:
        return get_disease_info(disease)
    except Exception as e:
        print(f"[Fetcher] Failed for '{disease}': {e}")
        return {
            "overview": "Information could not be retrieved at this time.",
            "causes": "",
            "treatment": "",
            "prevention": "",
            "when_to_see_a_doctor": "",
            "sources": []
        }


def clear_cache(disease: str = None):
    """Clear cache for a specific disease or all diseases."""
    if disease:
        p = _cache_path(disease)
        if p.exists():
            p.unlink()
            print(f"[Fetcher] Cache cleared for: {disease}")
    else:
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
        print("[Fetcher] All cache cleared")


if __name__ == "__main__":
    # Test
    import sys
    disease = sys.argv[1] if len(sys.argv) > 1 else "anxiety"
    info = get_disease_info(disease, force_refresh=True)
    print(f"\n{'='*60}")
    print(f"DISEASE: {disease.upper()}")
    print(f"{'='*60}")
    for k, v in info.items():
        if k != "sources":
            print(f"\n{k.upper()}:\n{v}")
    print(f"\nSOURCES: {info.get('sources', [])}")
