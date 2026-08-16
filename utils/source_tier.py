"""Source-quality classification, shared across any agent that rules on
evidence (originally lived in agents/synthesizer.py; moved here once a
second PM-path agent needed the same classification — see
agents/primary_research.py).

Provenance (a real source_url, DB constraints) is not the same thing as
source QUALITY — a run can have every claim traced to a real URL and
still have its "supported" verdict resting entirely on one person's
LinkedIn opinion post. source_tier(url) classifies each piece of
evidence by domain pattern (filing > analyst > news > blog > social)
computed on read from the already-stored source_url — not a stored
column, so no schema change/migration and no backfill question for rows
inserted before this existed.
"""

from urllib.parse import urlparse

# Ascending reliability. Domains not matched anywhere below default to
# "blog", not "news" — an unrecognized domain hasn't earned that credit.
TIER_RANK = {"social": 0, "blog": 1, "news": 2, "analyst": 3, "filing": 4}

_SOCIAL_DOMAINS = {
    "linkedin.com", "x.com", "twitter.com", "facebook.com", "instagram.com",
    "reddit.com", "youtube.com", "youtu.be", "tiktok.com", "threads.net",
}
_FILING_DOMAINS = {
    "sec.gov", "annualreports.com", "bseindia.com", "nseindia.com",
    "otcmarkets.com", "q4cdn.com",
}
_ANALYST_DOMAINS = {
    "gartner.com", "mckinsey.com", "bain.com", "bcg.com", "forrester.com",
    "moodys.com", "spglobal.com", "crisil.com", "icra.in", "statista.com",
    "ibisworld.com", "marketsandmarkets.com", "emergenresearch.com",
    "custommarketinsights.com", "redseer.com",
}
_NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "economictimes.indiatimes.com", "livemint.com", "moneycontrol.com",
    "business-standard.com", "thehindubusinessline.com", "cnbc.com",
    "forbes.com", "techcrunch.com", "entrackr.com", "inc42.com",
    "yourstory.com", "financialexpress.com", "hindustantimes.com",
    "ndtv.com", "indianexpress.com", "thehindu.com",
}
# Official regulator / government domains — a step above the FILING set
# above (which is mostly corporate/exchange filing platforms) but still
# classified as "filing" tier: the point of the tier system is "is this
# authoritative", and a regulator's own site is as authoritative as it gets.
_REGULATORY_DOMAINS = {
    "gov.in", "meity.gov.in", "prsindia.org", "eur-lex.europa.eu",
    "ftc.gov", "ico.org.uk", "gdpr.eu",
}


def _domain_matches(domain: str, domain_set: set[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in domain_set)


def source_tier(url: str | None) -> str:
    """Classify a source URL by domain pattern into filing / analyst /
    news / blog / social. Computed fresh from source_url every time it's
    needed (not persisted) — tier is a pure function of the URL, so
    there's nothing to migrate or backfill when this list is extended.
    An unrecognized domain defaults to "blog": absence of a bad signal
    isn't the same as presence of a good one.
    """
    if not url:
        return "blog"
    try:
        domain = urlparse(url).netloc.lower()
    except ValueError:
        return "blog"
    if domain.startswith("www."):
        domain = domain[4:]
    if not domain:
        return "blog"

    if _domain_matches(domain, _SOCIAL_DOMAINS):
        return "social"
    if (
        _domain_matches(domain, _FILING_DOMAINS)
        or _domain_matches(domain, _REGULATORY_DOMAINS)
        or "investor" in domain
        or domain.startswith("ir.")
        or domain.endswith(".gov")
        or ".gov." in domain
    ):
        return "filing"
    if _domain_matches(domain, _ANALYST_DOMAINS):
        return "analyst"
    if _domain_matches(domain, _NEWS_DOMAINS):
        return "news"
    return "blog"
