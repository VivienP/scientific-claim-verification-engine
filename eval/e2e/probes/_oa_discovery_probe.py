"""S2-P0 — OA discovery feasibility probe.

For each input DOI, query free OA discovery services and report whether an
open-access full-text path exists. This probe answers a single sprint question:
*before* we invest 350+ LoC building an Europe PMC + Semantic Scholar +
preprint client chain, what is the actual hit-rate on the 9 paywalled claims
that S2 is supposed to unblock?

The probe is read-only, jetable, and intentionally generic — it accepts any
list of DOIs so the same script can be re-pointed at a different benchmark
(per the user's reminder that the 25-claim lactate-ISF set is a sample).

Services queried (all free, no auth required):
    1. Europe PMC      - different OA mirrors than NCBI
    2. Semantic Scholar - openAccessPdf and abstract fields
    3. OpenAlex        - oa_url and is_oa fields
    4. Unpaywall       - canonical OA discovery (email param required)

Deliberately NOT queried:
    - CORE      - requires API key
    - BASE      - requires registration
    - Sci-Hub   - illegal

Outputs:
    * stdout: a markdown summary table with one row per DOI x service
    * eval/e2e/probes/oa_discovery_results.json: full structured payload
      so downstream design decisions are reproducible.

Usage::

    python eval/e2e/probes/_oa_discovery_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

_OUT_PATH = Path(__file__).parent / "oa_discovery_results.json"
_TIMEOUT = 15.0
_USER_AGENT = "ScientificClaimVerifier/0.1 (OA-discovery-probe)"
_HEADERS = {"User-Agent": _USER_AGENT}


# Paywalled / problematic DOIs from the lactate-ISF benchmark that S2 is
# supposed to unblock. Each entry is `(claim_id, doi, expected_difficulty)`.
# expected_difficulty is purely informational — set when authoring this probe.
_DOIS_TO_PROBE: list[tuple[str, str, str]] = [
    # Wrong-pick claims that need multi-source aggregation; OA path useful too.
    ("004_Williams1992", "10.1080/02640419208729912", "T&F paywall"),
    ("005_Raa2020", "10.1186/s13049-020-00776-z", "BMC OA expected"),
    # Heikenfeld review — Nature Biotech paywall.
    ("008_009_Heikenfeld2019", "10.1038/s41587-019-0040-3", "Nature paywall"),
    # Birklein + Muller multi-source for claim 011.
    ("011_Birklein2000", "10.1212/WNL.55.8.1213", "AAN/Neurology paywall"),
    ("011_Muller1996", "10.1152/ajpendo.1996.271.6.E1003", "APS paywall"),
    # Jansson + Krogstad for claims 015 / 016 (catheter depth relation).
    ("015_Krogstad1996", "10.1046/j.1365-2133.1996.d01-893.x", "Wiley paywall"),
    ("016_Jansson1996", "10.1152/ajpendo.1996.271.1.E138", "APS paywall"),
    # Multi-source for claim 017 (muscle/blood lactate parallel patterns).
    ("017_Loellgen1980", "10.1249/00005768-198012050-00008", "LWW paywall, pre-internet"),
    # Ventrelli microneedle review for claim 022.
    ("022_Ventrelli2015", "10.1002/adhm.201500450", "Wiley paywall"),
    # Sanity-check controls: claims that ARE on PMC already (Bug A landed).
    ("003_Goodwin2007_control", "10.1177/193229680700100414", "PMC2769631 confirmed"),
    ("020_Kotwal2012_control", "10.4103/2230-8210.104052", "PMC3603039 confirmed"),
    ("013_Ming2022_control", "10.1136/bmjinnov-2021-000864", "PMC7618145 confirmed"),
]


@dataclass
class ServiceHit:
    service: str
    found: bool
    has_abstract: bool
    has_oa_url: bool
    pmcid: str | None
    abstract_preview: str | None
    error: str | None = None


def _safe_get(client: httpx.Client, url: str, **kwargs: object) -> dict[str, object] | None:
    try:
        response = client.get(url, headers=_HEADERS, timeout=_TIMEOUT, **kwargs)  # type: ignore[arg-type]
        response.raise_for_status()
        data: dict[str, object] = response.json()
        return data
    except (httpx.HTTPError, ValueError) as exc:
        print(f"    error on {url[:60]}: {type(exc).__name__}: {str(exc)[:80]}")
        return None


def _query_europe_pmc(client: httpx.Client, doi: str) -> ServiceHit:
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json&resultType=core"
    data = _safe_get(client, url)
    if data is None:
        return ServiceHit("europe_pmc", False, False, False, None, None, "request_failed")
    result_list_obj = data.get("resultList", {})
    result_list = result_list_obj if isinstance(result_list_obj, dict) else {}
    results_obj = result_list.get("result", [])
    results = results_obj if isinstance(results_obj, list) else []
    if not results:
        return ServiceHit("europe_pmc", False, False, False, None, None, None)
    item = results[0] if isinstance(results[0], dict) else {}
    pmcid_raw = item.get("pmcid")
    pmcid = str(pmcid_raw) if pmcid_raw else None
    abstract_raw = item.get("abstractText")
    abstract = str(abstract_raw) if abstract_raw else None
    is_oa_raw = item.get("isOpenAccess", "N")
    is_oa = is_oa_raw == "Y"
    has_oa = bool(pmcid) or is_oa
    return ServiceHit(
        "europe_pmc",
        True,
        bool(abstract),
        has_oa,
        pmcid,
        (abstract[:200] if abstract else None),
    )


def _query_semantic_scholar(client: httpx.Client, doi: str) -> ServiceHit:
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        "?fields=abstract,openAccessPdf,externalIds,isOpenAccess"
    )
    data = _safe_get(client, url)
    if data is None:
        return ServiceHit("semantic_scholar", False, False, False, None, None, "request_failed")
    abstract_raw = data.get("abstract")
    abstract = str(abstract_raw) if abstract_raw else None
    oa_obj = data.get("openAccessPdf")
    oa_pdf = oa_obj if isinstance(oa_obj, dict) else None
    has_oa = bool(oa_pdf and oa_pdf.get("url"))
    ext_obj = data.get("externalIds")
    ext_ids = ext_obj if isinstance(ext_obj, dict) else {}
    pmcid_raw = ext_ids.get("PubMedCentral")
    pmcid = f"PMC{pmcid_raw}" if pmcid_raw else None
    return ServiceHit(
        "semantic_scholar",
        True,
        bool(abstract),
        has_oa,
        pmcid,
        (abstract[:200] if abstract else None),
    )


def _query_openalex(client: httpx.Client, doi: str) -> ServiceHit:
    url = f"https://api.openalex.org/works/doi:{doi}"
    data = _safe_get(client, url)
    if data is None:
        return ServiceHit("openalex", False, False, False, None, None, "request_failed")
    oa_obj = data.get("open_access")
    oa = oa_obj if isinstance(oa_obj, dict) else {}
    is_oa = bool(oa.get("is_oa"))
    oa_url_raw = oa.get("oa_url")
    oa_url = str(oa_url_raw) if oa_url_raw else None
    abstract_inv_obj = data.get("abstract_inverted_index")
    abstract_inv = abstract_inv_obj if isinstance(abstract_inv_obj, dict) else {}
    has_abstract = bool(abstract_inv)
    abstract_preview: str | None = None
    if abstract_inv:
        # Reconstruct abstract from inverted index for preview.
        word_positions = []
        for word, positions in abstract_inv.items():
            if isinstance(positions, list):
                for pos in positions:
                    if isinstance(pos, int):
                        word_positions.append((pos, word))
        word_positions.sort()
        abstract_preview = " ".join(w for _, w in word_positions[:40])
    return ServiceHit(
        "openalex",
        True,
        has_abstract,
        is_oa or bool(oa_url),
        None,
        abstract_preview,
    )


def _query_unpaywall(client: httpx.Client, doi: str, email: str) -> ServiceHit:
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    data = _safe_get(client, url)
    if data is None:
        return ServiceHit("unpaywall", False, False, False, None, None, "request_failed")
    is_oa = bool(data.get("is_oa"))
    best_obj = data.get("best_oa_location")
    best = best_obj if isinstance(best_obj, dict) else None
    has_oa = bool(best and (best.get("url") or best.get("url_for_pdf")))
    return ServiceHit("unpaywall", True, False, is_oa or has_oa, None, None)


def _probe_one(client: httpx.Client, doi: str, email: str) -> dict[str, ServiceHit]:
    return {
        "europe_pmc": _query_europe_pmc(client, doi),
        "semantic_scholar": _query_semantic_scholar(client, doi),
        "openalex": _query_openalex(client, doi),
        "unpaywall": _query_unpaywall(client, doi, email),
    }


def main() -> int:
    email = os.environ.get("UNPAYWALL_EMAIL") or os.environ.get("PUBMED_EMAIL")
    if not email:
        print("ERROR: set UNPAYWALL_EMAIL in your env (.env) before running.", file=sys.stderr)
        return 1

    results: dict[str, dict[str, dict[str, object]]] = {}

    print(f"\nProbing {len(_DOIS_TO_PROBE)} DOIs across 4 OA discovery services\n")
    print(
        f"{'claim_id':<28} {'EuPMC':<12} {'S2':<12} {'OAlex':<12} {'Unpw':<12} {'note'}"
    )
    print("-" * 110)

    with httpx.Client() as client:
        for claim_id, doi, note in _DOIS_TO_PROBE:
            hits = _probe_one(client, doi, email)
            results[claim_id] = {
                "doi": doi,
                "note": note,
                "services": {k: asdict(v) for k, v in hits.items()},
            }

            def fmt(s: ServiceHit) -> str:
                if not s.found:
                    return "miss"
                tag = []
                if s.has_oa_url:
                    tag.append("OA")
                if s.has_abstract:
                    tag.append("abs")
                if s.pmcid:
                    tag.append(s.pmcid)
                return ",".join(tag) or "found"

            print(
                f"{claim_id:<28} "
                f"{fmt(hits['europe_pmc']):<12} "
                f"{fmt(hits['semantic_scholar']):<12} "
                f"{fmt(hits['openalex']):<12} "
                f"{fmt(hits['unpaywall']):<12} "
                f"{note}"
            )
            time.sleep(0.4)  # be polite to free APIs

    _OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull payload: {_OUT_PATH}")

    # Hit-rate summary: count claim_ids where ANY service exposed an OA path.
    any_oa = 0
    any_abstract = 0
    for _claim_id, payload in results.items():
        services_obj = payload["services"]
        services = services_obj if isinstance(services_obj, dict) else {}
        oa_flags = [bool(s.get("has_oa_url")) for s in services.values() if isinstance(s, dict)]
        abs_flags = [bool(s.get("has_abstract")) for s in services.values() if isinstance(s, dict)]
        if any(oa_flags):
            any_oa += 1
        if any(abs_flags):
            any_abstract += 1

    n = len(_DOIS_TO_PROBE)
    print(f"\n=== Hit-rate summary across {n} DOIs ===")
    print(f"  At least one OA URL discovered : {any_oa}/{n} ({100 * any_oa / n:.0f}%)")
    print(f"  At least one abstract available: {any_abstract}/{n} ({100 * any_abstract / n:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
