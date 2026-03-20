############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_scanner.py: Data Loss Prevention scanning logic
#
# Pure logic module — no DB imports. Contains regex, GLiNER,
# and LLM-based scanners for detecting sensitive data.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""DLP scanner: regex, GLiNER NER, and LLM-based sensitive data detection."""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScanFinding:
    """A single sensitive-data finding from any scanner."""
    scanner: str          # "regex", "gliner", or "llm"
    category: str         # e.g. "social security number", "credit card number"
    text: str             # the matched text snippet
    confidence: float     # 0.0–1.0
    start: int = 0        # character offset in source text
    end: int = 0          # character offset end


@dataclass
class ScanResult:
    """Aggregated result of a DLP scan across all scanners."""
    findings: List[ScanFinding] = field(default_factory=list)
    severity: str = "minor"
    scan_latency_ms: int = 0
    scanner: str = "regex"       # primary scanner that produced the result
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Built-in regex patterns (always available)
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS = [
    {"name": "SSN", "pattern": r"\b\d{3}-\d{2}-\d{4}\b", "category": "social security number", "severity": "major"},
    {"name": "Credit Card", "pattern": r"\b(?:\d[ -]*?){13,19}\b", "category": "credit card number", "severity": "major"},
    {"name": "Email Address", "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "category": "email", "severity": "minor"},
    {"name": "Phone (US)", "pattern": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "category": "phone number", "severity": "minor"},
    {"name": "Date of Birth", "pattern": r"\b(?:DOB|date of birth|born on)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "category": "date of birth", "severity": "moderate"},
]


# ---------------------------------------------------------------------------
# Regex / keyword scanner
# ---------------------------------------------------------------------------

def scan_regex(
    text: str,
    custom_patterns: Optional[List[Dict[str, str]]] = None,
    keywords: Optional[List[str]] = None,
) -> List[ScanFinding]:
    """Scan text with regex patterns and keyword matching.

    Returns a list of ScanFinding objects for each match.
    """
    findings: List[ScanFinding] = []

    all_patterns = list(_BUILTIN_PATTERNS)
    if custom_patterns:
        all_patterns.extend(custom_patterns)

    for pat in all_patterns:
        try:
            for m in re.finditer(pat["pattern"], text, re.IGNORECASE):
                findings.append(ScanFinding(
                    scanner="regex",
                    category=pat.get("category", pat.get("name", "unknown")),
                    text=m.group(),
                    confidence=1.0,
                    start=m.start(),
                    end=m.end(),
                ))
        except re.error:
            logger.warning("dlp_regex_invalid", pattern=pat.get("name", "?"))

    if keywords:
        text_lower = text.lower()
        for kw in keywords:
            if not kw:
                continue
            kw_lower = kw.strip().lower()
            if not kw_lower:
                continue
            idx = 0
            while True:
                pos = text_lower.find(kw_lower, idx)
                if pos == -1:
                    break
                findings.append(ScanFinding(
                    scanner="regex",
                    category="keyword",
                    text=text[pos:pos + len(kw_lower)],
                    confidence=0.9,
                    start=pos,
                    end=pos + len(kw_lower),
                ))
                idx = pos + 1

    return findings


# ---------------------------------------------------------------------------
# GLiNER NER scanner (lazy model load)
# ---------------------------------------------------------------------------

_gliner_model = None
_gliner_lock = asyncio.Lock()


async def _load_gliner():
    """Lazily load the GLiNER PII model. Thread-safe via asyncio lock."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    async with _gliner_lock:
        if _gliner_model is not None:
            return _gliner_model

        logger.info("dlp_gliner_loading", model="urchade/gliner_multi_pii-v1")
        t0 = time.monotonic()

        try:
            from gliner import GLiNER
            loop = asyncio.get_event_loop()
            _gliner_model = await loop.run_in_executor(
                None, GLiNER.from_pretrained, "urchade/gliner_multi_pii-v1"
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info("dlp_gliner_loaded", elapsed_ms=elapsed)
        except ImportError:
            logger.error("dlp_gliner_not_installed", hint="pip install gliner")
            raise
        except Exception:
            logger.exception("dlp_gliner_load_failed")
            raise

    return _gliner_model


async def scan_gliner(
    text: str,
    categories: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> List[ScanFinding]:
    """Scan text using GLiNER NER model for PII entities.

    Args:
        text: Text to scan.
        categories: Entity categories to detect.
        threshold: Minimum confidence threshold.

    Returns:
        List of ScanFinding objects.
    """
    if not categories:
        categories = [
            "person", "phone number", "email", "credit card number",
            "social security number", "date of birth", "driver license number",
            "passport number", "bank account number",
        ]

    try:
        model = await _load_gliner()
    except Exception:
        return []

    loop = asyncio.get_event_loop()

    def _predict():
        return model.predict_entities(text, categories, threshold=threshold)

    try:
        entities = await loop.run_in_executor(None, _predict)
    except Exception:
        logger.exception("dlp_gliner_predict_failed")
        return []

    findings = []
    for ent in entities:
        findings.append(ScanFinding(
            scanner="gliner",
            category=ent.get("label", "unknown"),
            text=ent.get("text", ""),
            confidence=ent.get("score", threshold),
            start=ent.get("start", 0),
            end=ent.get("end", 0),
        ))

    return findings


# ---------------------------------------------------------------------------
# LLM contextual scanner (self-routes through MindRouter)
# ---------------------------------------------------------------------------

async def scan_llm(
    text: str,
    system_prompt: str,
    model: str,
    api_key: str,
    base_url: str = "http://localhost:8000",
) -> List[ScanFinding]:
    """Scan text using an LLM for contextual sensitive data detection.

    Self-routes through MindRouter's own /v1/chat/completions endpoint.
    """
    import httpx

    # Truncate very long text to avoid excessive token usage
    max_chars = 8000
    scan_text = text[:max_chars] if len(text) > max_chars else text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this text for sensitive data:\n\n{scan_text}"},
    ]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.exception("dlp_llm_scan_failed", model=model)
        return []

    # Parse LLM response — expect JSON array of findings
    content = ""
    if "choices" in data and data["choices"]:
        content = data["choices"][0].get("message", {}).get("content", "")

    findings = []
    try:
        # Try to extract JSON from the response (may have markdown fences)
        json_str = content.strip()
        if json_str.startswith("```"):
            # Strip markdown code fences
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            for item in parsed:
                findings.append(ScanFinding(
                    scanner="llm",
                    category=item.get("category", "unknown"),
                    text=item.get("text", ""),
                    confidence=float(item.get("confidence", 0.7)),
                ))
    except (json.JSONDecodeError, ValueError):
        # If the LLM didn't return valid JSON, treat non-empty response as a single finding
        if content.strip() and content.strip() != "[]":
            logger.warning("dlp_llm_parse_failed", content_preview=content[:200])

    return findings


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"minor": 0, "moderate": 1, "major": 2}


def classify_severity(
    findings: List[ScanFinding],
    severity_rules: Optional[Dict[str, str]] = None,
) -> str:
    """Classify the overall severity from a list of findings.

    Uses severity_rules mapping (category → severity). The highest severity
    among all findings wins. Unknown categories default to "moderate".
    """
    if not findings:
        return "minor"

    if severity_rules is None:
        severity_rules = {}

    highest = "minor"
    for f in findings:
        cat_severity = severity_rules.get(f.category, "moderate")
        if _SEVERITY_ORDER.get(cat_severity, 1) > _SEVERITY_ORDER.get(highest, 0):
            highest = cat_severity

    return highest


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_scannable_text(
    messages: Optional[Any] = None,
    prompt: Optional[str] = None,
    response_content: Optional[str] = None,
    modality: Optional[str] = None,
) -> Optional[str]:
    """Extract scannable text from request/response data.

    Skips image/multimodal content. Concatenates all text parts.
    """
    parts: List[str] = []

    # Extract from chat messages
    if messages:
        msg_list = messages if isinstance(messages, list) else messages.get("messages", [])
        for msg in msg_list:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # Multipart content — only extract text parts
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif part.get("type") in ("image_url", "image"):
                                continue  # skip images

    # Extract from raw prompt
    if prompt:
        parts.append(prompt)

    # Extract from response
    if response_content:
        parts.append(response_content)

    combined = "\n".join(p for p in parts if p)
    return combined if combined.strip() else None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_dlp_scan(
    text: str,
    config: Dict[str, Any],
) -> Optional[ScanResult]:
    """Run all enabled DLP scanners on the given text.

    Args:
        text: The text to scan.
        config: DLP configuration dict with keys like:
            - regex.enabled, regex.patterns, regex.keywords
            - gliner.enabled, gliner.threshold, gliner.categories
            - llm.enabled, llm.model, llm.system_prompt, llm.api_key, llm.base_url
            - severity_rules

    Returns:
        ScanResult if any findings, None if clean.
    """
    t0 = time.monotonic()
    all_findings: List[ScanFinding] = []
    scanners_used: List[str] = []

    # --- Regex scanner (always fast, run first) ---
    if config.get("regex.enabled", True):
        regex_findings = scan_regex(
            text,
            custom_patterns=config.get("regex.patterns"),
            keywords=config.get("regex.keywords"),
        )
        all_findings.extend(regex_findings)
        if regex_findings:
            scanners_used.append("regex")

    # --- GLiNER NER scanner ---
    if config.get("gliner.enabled", False):
        gliner_findings = await scan_gliner(
            text,
            categories=config.get("gliner.categories"),
            threshold=config.get("gliner.threshold", 0.5),
        )
        all_findings.extend(gliner_findings)
        if gliner_findings:
            scanners_used.append("gliner")

    # --- LLM contextual scanner ---
    if config.get("llm.enabled", False):
        api_key = config.get("llm.api_key", "")
        if api_key:
            llm_findings = await scan_llm(
                text,
                system_prompt=config.get("llm.system_prompt", ""),
                model=config.get("llm.model", ""),
                api_key=api_key,
                base_url=config.get("llm.base_url", "http://localhost:8000"),
            )
            all_findings.extend(llm_findings)
            if llm_findings:
                scanners_used.append("llm")

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if not all_findings:
        return None

    severity = classify_severity(
        all_findings,
        severity_rules=config.get("severity_rules"),
    )

    # Build detail summary
    categories = list(set(f.category for f in all_findings))
    detail_parts = [f"{len(all_findings)} finding(s) across {', '.join(scanners_used)}"]
    detail_parts.append(f"Categories: {', '.join(categories)}")
    detail = "; ".join(detail_parts)

    primary_scanner = scanners_used[-1] if scanners_used else "regex"

    return ScanResult(
        findings=all_findings,
        severity=severity,
        scan_latency_ms=elapsed_ms,
        scanner=primary_scanner,
        detail=detail,
    )
