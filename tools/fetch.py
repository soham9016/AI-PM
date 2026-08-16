"""Page fetch tool — downloads a URL and extracts readable text.

Only ever hands back text worth feeding to an LLM: genuinely non-text
binary formats (.doc/.xls/.zip/...) are filtered before the request is
even made; .pdf URLs (or a response that declares Content-Type:
application/pdf) get a real extraction attempt via pypdf rather than
being skipped outright — business research leans heavily on PDFs
(reports, filings, whitepapers), so treating every PDF as unusable would
bias the evidence base toward weaker sources just because they're easier
to parse. Everything else goes through the HTML path: Content-Type is
checked before decoding, decoding uses the page's own declared charset
(falling back to a content-based guess, never a hardcoded one), and a
garbage gate after extraction catches whatever slips through in either
path (e.g. a scanned PDF with no text layer, or a server that mislabels
a PDF as text/html) before it reaches a caller. No OCR — a PDF with no
extractable text layer is skipped, not rasterized/read.
"""

import io
import logging

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pypdf import PdfReader

MAX_CHARS = 8000
TIMEOUT_SECONDS = 10
USER_AGENT = "Mozilla/5.0 (compatible; BusinessResearchCopilot/1.0)"

MIN_TEXT_CHARS = 200
MAX_REPLACEMENT_CHAR_RATIO = 0.05
MAX_PDF_PAGES = 15

# Genuinely non-text formats we never attempt to parse. .pdf is deliberately
# NOT here — it gets a real extraction attempt below instead of a blind skip.
NON_PDF_BINARY_EXTENSIONS = (".doc", ".docx", ".xls", ".xlsx", ".zip", ".ppt", ".pptx")
ALLOWED_CONTENT_TYPES = ("text/html", "text/plain")

logger = logging.getLogger("business_copilot.fetch")


def _skip(url: str, reason: str) -> dict:
    logger.info("Skipping %s: %s", url, reason)
    return {"url": url, "title": "", "text": "", "error": None, "skipped": True, "skip_reason": reason}


def _declared_charset(content_type_header: str) -> str | None:
    lowered = content_type_header.lower()
    if "charset=" not in lowered:
        return None
    return lowered.split("charset=", 1)[1].split(";")[0].strip() or None


def _garbage_check(text: str) -> str | None:
    """Returns a skip reason if `text` doesn't look like real content, else None."""
    replacement_ratio = (text.count("�") / len(text)) if text else 1.0
    if len(text) < MIN_TEXT_CHARS or replacement_ratio > MAX_REPLACEMENT_CHAR_RATIO:
        return f"garbage content after extraction ({len(text)} chars, {replacement_ratio:.0%} replacement characters)"
    return None


def _extract_pdf_text(content: bytes) -> tuple[str, str | None]:
    """Try to extract text from PDF bytes.

    Returns (text, skip_reason) — exactly one is populated. Every failure
    mode (corrupted, encrypted, no text layer) is a distinct, logged skip
    reason, never an exception — a bad PDF must degrade the same way a
    bad HTML page does, not crash the node.
    """
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001 — any parse failure must not crash the node
        return "", f"PDF could not be parsed (corrupted or invalid): {exc}"

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception:  # noqa: BLE001 — treat any decrypt failure as "can't read it"
            decrypt_result = 0
        if not decrypt_result:
            return "", "PDF is encrypted, cannot extract text"

    try:
        pages = reader.pages[:MAX_PDF_PAGES]
        text = " ".join(" ".join((page.extract_text() or "") for page in pages).split())
    except Exception as exc:  # noqa: BLE001 — any parse failure must not crash the node
        return "", f"PDF could not be parsed (corrupted or invalid): {exc}"

    if not text:
        return "", "PDF has no extractable text layer (likely scanned)"

    return text, None


@tool
def fetch_page(url: str) -> dict:
    """Fetch a URL and return its extracted text content (HTML or PDF).

    Returns a dict: {url, title, text, error, skipped, skip_reason}.
    - `error` is set on a hard failure (network/HTTP error) — `text` is empty.
    - `skipped` is True when the page was reachable but deliberately not
      used — `skip_reason` explains why (and every skip is logged), one of:
      a non-PDF binary file extension, a non-text/non-PDF Content-Type, a
      PDF parse failure ("PDF could not be parsed..."), an encrypted PDF
      ("PDF is encrypted..."), a PDF with no text layer ("PDF has no
      extractable text layer..."), or content that failed the post-
      extraction garbage gate (HTML or PDF).
    Callers should treat `error` or `skipped` the same way: don't use this page.
    """
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    if path.endswith(NON_PDF_BINARY_EXTENSIONS):
        ext = path.rsplit(".", 1)[-1]
        return _skip(url, f"URL path looks like a non-PDF binary file (.{ext})")

    is_pdf_url = path.endswith(".pdf")

    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"url": url, "title": "", "text": "", "error": str(exc), "skipped": False, "skip_reason": None}

    content_type_header = resp.headers.get("Content-Type", "")
    content_type = content_type_header.split(";")[0].strip().lower()

    if is_pdf_url or content_type == "application/pdf":
        text, pdf_skip_reason = _extract_pdf_text(resp.content)
        if pdf_skip_reason:
            return _skip(url, f"PDF: {pdf_skip_reason}")

        garbage_reason = _garbage_check(text)
        if garbage_reason:
            return _skip(url, f"PDF: {garbage_reason}")

        text = text[:MAX_CHARS]
        logger.info("Parsed PDF OK: %s (%d chars)", url, len(text))
        return {"url": url, "title": "", "text": text, "error": None, "skipped": False, "skip_reason": None}

    if not content_type.startswith(ALLOWED_CONTENT_TYPES):
        return _skip(url, f"unsupported Content-Type {content_type_header!r}")

    # Decode with the page's own declared charset; only guess from content
    # (never a hardcoded default) if the server didn't declare one.
    resp.encoding = _declared_charset(content_type_header) or resp.apparent_encoding
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = " ".join(soup.get_text(separator=" ").split())

    garbage_reason = _garbage_check(text)
    if garbage_reason:
        return _skip(url, garbage_reason)

    text = text[:MAX_CHARS]
    return {"url": url, "title": title, "text": text, "error": None, "skipped": False, "skip_reason": None}
