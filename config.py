"""
Central model configuration.

Every agent's model + provider is chosen here, in one place, instead of
being hardcoded inline in each agent file. Swap a model (or provider)
for an agent by editing one entry in MODELS.

MULTI-PROVIDER ROUTING: the real bottleneck on Groq's free tier is
tokens-per-minute (6,000 TPM), not request count — that's what threw a
413 on agents/competitive_audit.py's _classify_areas. Cerebras runs
~30,000 TPM / ~1M tokens/day, so the few LARGE-PAYLOAD calls
(competitive_audit's classify step, solution_framing,
funnel_decomposition) are routed there on gpt-oss-120b (Cerebras only
serves gpt-oss-120b, zai-glm-4.7, and gemma-4-31b — an earlier config
pointed at a Llama model name that doesn't exist on Cerebras and 404'd);
the many SMALL-PAYLOAD, HIGH-CALL-COUNT calls (evidence extraction,
stance classification, relevance filtering, and everything else not
explicitly listed) stay on Groq, where llama-3.1-8b-instant's 14,400
requests/day is the better fit.

NO HARD DEPENDENCY ON CEREBRAS, AT EITHER CONSTRUCTION OR INVOCATION
TIME: langchain_cerebras is imported inside a try/except (package not
installed silently falls back), get_llm() falls back to Groq if
CEREBRAS_API_KEY isn't set or ChatCerebras() construction itself raises,
and invoke_llm() below falls back to Groq if the CALL to Cerebras fails
(bad model name, quota/payment errors, any non-rate-limit 4xx/5xx) —
this last one is the gap that construction-time handling alone doesn't
cover: a model name Cerebras doesn't recognize, or a payment-required
account, both construct a ChatCerebras object fine and only fail once
invoked. Every path is: log clearly, retry once on that entry's
groq_fallback_model, never crash the run. Rate-limit errors are the one
exception — those stay on utils/retry.py's existing backoff decorator,
same as always, rather than triggering an immediate fallback (a rate
limit is transient and provider-agnostic; a 404/402 is not going to
succeed on a second identical attempt).

DISABLE_NON_GROQ_PROVIDERS: set this env var (any of "1"/"true"/"yes")
to force every agent onto Groq regardless of its MODELS entry — a single
switch for "Cerebras is having a bad day, just run pure-Groq."

validate_models() — CATCH A DEAD MODEL NAME AT LAUNCH, NOT MID-RUN: this
is the fourth distinct provider failure in three days (stale model name,
account quota, daily token ceiling, and now Groq deprecating both
llama-3.1-8b-instant and llama-3.3-70b-versatile outright). invoke_llm's
fallback covers a model failing AFTER a run has started; it does nothing
for "half of MODELS points at a model that no longer exists" being
discovered three agents into a 5-10 minute run. validate_models() fetches
each active provider's live model list ONCE per process (module-level
cache; graph.py's build_graph() calls it unconditionally, including from
the Streamlit app, which calls build_graph() on every click — the cache
is what keeps that cheap) and logs one clear error listing every
MODELS[*]['model']/['groq_fallback_model']/FALLBACK value that isn't on
it. Logs only, never raises: a provider outage during THIS check must not
block startup any more than a bad model should block the whole run --
this is an early warning, not a gate.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger("business_copilot.config")

try:
    from langchain_cerebras import ChatCerebras
except ImportError:
    ChatCerebras = None

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")

# One-line kill switch for all non-Groq providers -- see module docstring.
DISABLE_NON_GROQ_PROVIDERS = os.environ.get("DISABLE_NON_GROQ_PROVIDERS", "").strip().lower() in ("1", "true", "yes")

# Search provider selection (see tools/search.py) -- kept here alongside
# the other provider config rather than hardcoded in three agent files,
# since search providers change (Tavily was acquired by Nebius in Feb
# 2026; Google sued SerpAPI in Dec 2025) and the agents should never need
# to know which one is active.
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "tavily")

# Model (+ provider) assigned to each agent/call site. Bigger/slower
# models go on agents that need more reasoning (structuring, hypothesis
# generation, synthesis); smaller/faster models go on agents that mostly
# summarize or format. "groq_fallback_model" only matters for
# provider="cerebras" entries -- it's what gets used if Cerebras is
# unavailable for any reason.
MODELS: dict[str, dict] = {
    # Reasoning-heavy
    "structurer":                 {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "funnel_decomposition":       {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "solution_framing":           {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "solution_review":            {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "pm":                         {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "primary_research":   {"provider": "groq", "model": "openai/gpt-oss-20b"},
    "competitive_audit_classify": {"provider": "groq", "model": "openai/gpt-oss-120b"},

    # Mid — separate quota bucket, keeps 120b from being the only ceiling
    
    "critic":                     {"provider": "groq", "model": "qwen/qwen3.6-27b"},
    "synthesizer":                {"provider": "groq", "model": "qwen/qwen3.6-27b"},

    # High-volume, narrow — replaces llama-3.1-8b-instant
    "evidence_extractor": {"provider": "groq", "model": "openai/gpt-oss-20b"},
    "researcher":         {"provider": "groq", "model": "openai/gpt-oss-20b"},
    
    "competitive_audit":  {"provider": "groq", "model": "openai/gpt-oss-20b"},

    # Diagnostic mode only
    "hypothesis": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "analyst":    {"provider": "groq", "model": "qwen/qwen3.6-27b"},
}

# Used when an agent's preferred model is unavailable / repeatedly errors
# out (the retry decorator's use_fallback path) -- always Groq, regardless
# of the agent's normal provider. llama-3.1-8b-instant (the previous
# value) was deprecated off Groq entirely -- caught by validate_models()
# below, fixed here to the smallest/fastest model still live.
FALLBACK = "openai/gpt-oss-20b"

DEFAULT_TEMPERATURE = 0.2

# Ceilings to keep one full run testable on Groq's free tier. Hypothesis
# count drives issue-tree coverage (too low and disclosure swallows the
# recommendation — see agents/synthesizer.py); extraction calls, not
# hypothesis generation, are where the quota actually goes, so that's the
# knob to turn down when raising MAX_HYPOTHESES.
MAX_HYPOTHESES = 4
MAX_EVIDENCE_EXTRACTIONS = 6


def _json_kwargs(json_mode: bool) -> dict:
    return {"model_kwargs": {"response_format": {"type": "json_object"}}} if json_mode else {}


def _groq_llm(model: str, temperature: float, json_mode: bool):
    return ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=temperature, **_json_kwargs(json_mode))


def get_llm(
    agent_name: str,
    temperature: float = DEFAULT_TEMPERATURE,
    use_fallback: bool = False,
    json_mode: bool = False,
):
    """Return a configured chat model instance for the given agent name
    (or, for multi-call agents, a specific call-site key — see
    "competitive_audit_classify" above).

    Looks up MODELS[agent_name]; falls back to Groq's FALLBACK model if
    the agent isn't registered, if use_fallback=True is passed explicitly
    (e.g. by the retry decorator after repeated failures), if
    DISABLE_NON_GROQ_PROVIDERS is set, or if the entry's provider is
    "cerebras" but Cerebras isn't actually available (package not
    installed, or CEREBRAS_API_KEY unset) — logged, never raised.

    This only covers CONSTRUCTION-time unavailability. A Cerebras client
    that constructs fine but fails when actually INVOKED (bad model name,
    payment/quota errors, etc.) is a separate gap — see invoke_llm()
    below, which is what agents calling a "cerebras" entry should use
    instead of calling .invoke() on this return value directly.

    json_mode=True requests structured-output mode
    (response_format={"type": "json_object"}), which makes malformed JSON
    structurally impossible rather than merely discouraged by the prompt.
    Groq (and Cerebras, OpenAI-compatible) require the word "json" to
    appear somewhere in the prompt when this is set; every prompt using
    it already says "Respond with ONLY a JSON object", so that
    requirement is already met.
    """
    entry = MODELS.get(agent_name, {"provider": "groq", "model": FALLBACK})
    provider = entry.get("provider", "groq")
    model = entry.get("model", FALLBACK)

    if use_fallback:
        provider, model = "groq", FALLBACK

    if provider == "cerebras" and DISABLE_NON_GROQ_PROVIDERS:
        logger.info("DISABLE_NON_GROQ_PROVIDERS is set — using Groq for %r instead of Cerebras", agent_name)
        return _groq_llm(entry.get("groq_fallback_model", FALLBACK), temperature, json_mode)

    if provider == "cerebras":
        if ChatCerebras is not None and CEREBRAS_API_KEY:
            try:
                return ChatCerebras(model=model, api_key=CEREBRAS_API_KEY, temperature=temperature, **_json_kwargs(json_mode))
            except Exception as exc:  # noqa: BLE001 — a Cerebras construction failure must not crash the run
                logger.warning("ChatCerebras construction failed for %r (%s) — falling back to Groq", agent_name, exc)
        else:
            reason = "langchain-cerebras isn't installed" if ChatCerebras is None else "CEREBRAS_API_KEY is not set"
            logger.warning("Cerebras requested for %r but unavailable (%s) — falling back to Groq", agent_name, reason)
        model = entry.get("groq_fallback_model", FALLBACK)

    return _groq_llm(model, temperature, json_mode)


def invoke_llm(agent_name: str, messages: list, temperature: float = DEFAULT_TEMPERATURE, json_mode: bool = False):
    """Build a model via get_llm() and invoke it, additionally covering
    the INVOCATION-time gap get_llm() alone doesn't: a Cerebras client
    that constructed fine (valid package + API key) but fails when
    actually called — an unrecognized model name (404), a payment/quota
    error (402), or any other non-rate-limit 4xx/5xx. Those used to
    propagate straight out of the agent's @with_retry-wrapped _call_llm
    and kill the whole run, because utils/retry.py's RETRYABLE_EXCEPTIONS
    only recognized the groq SDK's exception classes — Cerebras runs on
    an OpenAI-compatible client, so its errors are openai.* exceptions
    the decorator never even looked for. Two real runs hit exactly this:
    a 404 model_not_found, then a 402 payment_required, both fatal.

    Rate-limit errors are deliberately NOT intercepted here — those are
    left to propagate up to the caller's @with_retry decorator, which
    already backs off and retries them (a rate limit is transient and
    provider-agnostic; a 404/402 is not going to succeed on a second
    identical attempt, so it gets one immediate retry on Groq instead).

    Only intercepts errors when the model actually in use is Cerebras --
    if get_llm() already downgraded to Groq (disabled/unavailable), this
    behaves like a plain .invoke() and ordinary Groq error handling
    applies unchanged.
    """
    llm = get_llm(agent_name, temperature=temperature, json_mode=json_mode)
    is_cerebras = ChatCerebras is not None and isinstance(llm, ChatCerebras)
    if not is_cerebras:
        return llm.invoke(messages)

    try:
        return llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001 — see docstring: only a non-rate-limit provider error reaches here
        if getattr(exc, "status_code", None) == 429:
            raise  # rate limit -- let @with_retry's backoff handle it, not a fallback
        entry = MODELS.get(agent_name, {})
        fallback_model = entry.get("groq_fallback_model", FALLBACK)
        logger.warning(
            "Cerebras invocation failed for %r (%s: %s) — retrying once on Groq fallback model %r",
            agent_name, type(exc).__name__, exc, fallback_model,
        )
        return _groq_llm(fallback_model, temperature, json_mode).invoke(messages)


_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_models_validated = False  # module-level cache -- see validate_models()


def _fetch_groq_model_ids() -> set[str] | None:
    """Live model IDs from Groq's own /models endpoint, or None if the
    fetch couldn't happen (no key, network/API error) -- never raises."""
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return {m.id for m in Groq(api_key=GROQ_API_KEY).models.list().data}
    except Exception as exc:  # noqa: BLE001 — a validation-only call must never block startup
        logger.warning("validate_models: Groq model-list fetch failed (%s: %s) — skipping model validation this run", type(exc).__name__, exc)
        return None


def _fetch_cerebras_model_ids() -> set[str] | None:
    """Live model IDs from Cerebras's OpenAI-compatible /models endpoint
    (langchain_cerebras has no models.list() of its own -- it wraps
    BaseChatOpenAI, so the plain openai client works directly against the
    same base_url), or None if the fetch couldn't happen."""
    if not CEREBRAS_API_KEY:
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=CEREBRAS_API_KEY, base_url=_CEREBRAS_BASE_URL)
        return {m.id for m in client.models.list().data}
    except Exception as exc:  # noqa: BLE001 — same rule: never block startup over this
        logger.warning("validate_models: Cerebras model-list fetch failed (%s: %s) — skipping Cerebras model validation this run", type(exc).__name__, exc)
        return None


def validate_models(force: bool = False) -> list[str]:
    """Fetch each active provider's live model list ONCE per process and
    log a single clear error listing every model name this config
    references that isn't actually live — every MODELS[*]['model'],
    every MODELS[*]['groq_fallback_model'], and FALLBACK. Cached after
    the first call (module-level flag) so graph.py's build_graph()
    calling this unconditionally on every invocation — including from
    the Streamlit app, which rebuilds the graph on every click — costs
    one network round-trip per process, not one per click. Pass
    force=True to bypass the cache (e.g. after editing MODELS at
    runtime).

    Returns the list of problem strings found (empty if everything
    checks out, or if a fetch failed and the check had to be skipped —
    check the log for which). Never raises: a provider outage during
    THIS check must not block startup any more than a bad model should
    block the whole run — this is an early warning, not a gate.
    """
    global _models_validated
    if _models_validated and not force:
        return []
    _models_validated = True

    groq_ids = _fetch_groq_model_ids()
    if groq_ids is None:
        logger.warning("validate_models: skipped (Groq model list unavailable) — a bad model name will only surface at invocation time now")
        return []

    problems: list[str] = []
    for agent_name, entry in MODELS.items():
        provider = entry.get("provider", "groq")
        model = entry.get("model")
        if provider == "groq" and model and model not in groq_ids:
            problems.append(f"MODELS[{agent_name!r}]['model'] = {model!r} is not a live Groq model")
        fallback_model = entry.get("groq_fallback_model")
        if fallback_model and fallback_model not in groq_ids:
            problems.append(f"MODELS[{agent_name!r}]['groq_fallback_model'] = {fallback_model!r} is not a live Groq model")

    if FALLBACK not in groq_ids:
        problems.append(f"FALLBACK = {FALLBACK!r} is not a live Groq model")

    # Only fetch Cerebras's list if something actually routes there and a
    # key is set -- no point paying for a network call for a provider
    # nothing in MODELS currently uses.
    cerebras_entries = {name: e for name, e in MODELS.items() if e.get("provider") == "cerebras"}
    if cerebras_entries and CEREBRAS_API_KEY:
        cerebras_ids = _fetch_cerebras_model_ids()
        if cerebras_ids is not None:
            for agent_name, entry in cerebras_entries.items():
                model = entry.get("model")
                if model and model not in cerebras_ids:
                    problems.append(f"MODELS[{agent_name!r}]['model'] = {model!r} is not a live Cerebras model")

    if problems:
        logger.error(
            "validate_models: %d stale/nonexistent model name(s) in config.py's MODELS — these WILL "
            "fail at invocation time, not just log a warning here:\n  %s",
            len(problems), "\n  ".join(problems),
        )
    else:
        logger.info("validate_models: every configured model name is live on its provider")

    return problems
