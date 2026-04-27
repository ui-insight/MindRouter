#!/usr/bin/env python3
"""
MindRouter Usage Audit — sample and categorize interactions.

Run inside the app container:
    docker exec -it mindrouter-app-1 python scripts/usage_audit.py

Or with options:
    docker exec -it mindrouter-app-1 python scripts/usage_audit.py \
        --samples 200 --include-archive --output /data/artifacts/usage_audit.md
"""

import argparse
import json
import os
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pymysql
import pymysql.cursors


# ---------------------------------------------------------------------------
# Category definitions — keyword/pattern heuristics for first-pass labeling
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Code & Software Engineering": {
        "keywords": [
            "code", "function", "class", "debug", "error", "bug", "python",
            "javascript", "java", "rust", "c++", "html", "css", "sql",
            "api", "endpoint", "docker", "git", "compile", "runtime",
            "refactor", "implement", "algorithm", "regex", "script",
            "npm", "pip", "import", "syntax", "variable", "loop",
            "array", "object", "json", "xml", "yaml", "config",
            "database", "query", "schema", "migrate", "deploy",
            "terraform", "kubernetes", "cicd", "pipeline", "test",
            "unit test", "integration", "stack trace", "exception",
        ],
        "patterns": [
            r"def\s+\w+\(", r"function\s+\w+", r"class\s+\w+",
            r"import\s+\w+", r"from\s+\w+\s+import",
            r"```\w*\n", r"\bif\s*\(", r"for\s*\(",
        ],
    },
    "Academic Research": {
        "keywords": [
            "research", "study", "hypothesis", "methodology", "literature",
            "citation", "peer review", "journal", "publication", "thesis",
            "dissertation", "experiment", "data analysis", "statistical",
            "p-value", "regression", "correlation", "sample size",
            "findings", "abstract", "manuscript", "scholarly",
            "dataset", "survey", "qualitative", "quantitative",
            "longitudinal", "cross-sectional", "meta-analysis",
            "IRB", "informed consent", "grant", "NSF", "NIH",
        ],
        "patterns": [
            r"et\s+al\.?", r"\(\d{4}\)", r"doi:",
        ],
    },
    "Writing & Editing": {
        "keywords": [
            "write", "essay", "paragraph", "proofread", "grammar",
            "rewrite", "summarize", "paraphrase", "tone", "draft",
            "outline", "edit", "revise", "rephrase", "wordsmith",
            "blog post", "article", "report", "memo", "letter",
            "proposal", "abstract", "introduction", "conclusion",
            "narrative", "persuasive", "argumentative", "expository",
        ],
        "patterns": [
            r"(rewrite|rephrase|edit)\s+(this|the|my)",
            r"(improve|fix)\s+(the\s+)?(writing|grammar|tone)",
        ],
    },
    "Teaching & Education": {
        "keywords": [
            "student", "assignment", "syllabus", "course", "lecture",
            "curriculum", "rubric", "grading", "exam", "quiz",
            "lesson plan", "learning objective", "classroom",
            "pedagogy", "Canvas", "LMS", "homework", "semester",
            "teaching", "instructor", "professor", "TA ",
            "learning outcome", "accreditation", "assessment",
        ],
        "patterns": [
            r"(create|design|write)\s+(a\s+)?(quiz|exam|assignment|rubric|syllabus)",
        ],
    },
    "Data & Analytics": {
        "keywords": [
            "data", "csv", "dataframe", "pandas", "numpy", "matplotlib",
            "visualization", "chart", "graph", "plot", "R ",
            "statistics", "machine learning", "model", "training",
            "prediction", "classification", "clustering", "neural",
            "deep learning", "tensor", "pytorch", "tensorflow",
            "feature", "accuracy", "precision", "recall", "f1",
            "anova", "t-test", "chi-square", "bayesian",
        ],
        "patterns": [
            r"(analyze|visualize)\s+(this|the|my)\s+data",
            r"(create|build|train)\s+(a\s+)?(model|classifier|predictor)",
        ],
    },
    "Administrative & Business": {
        "keywords": [
            "meeting", "agenda", "email", "budget", "policy",
            "procedure", "compliance", "HR", "onboarding",
            "performance review", "strategic plan", "stakeholder",
            "workflow", "process", "schedule", "deadline",
            "committee", "approval", "requisition", "procurement",
        ],
        "patterns": [
            r"(draft|write)\s+(an?\s+)?(email|memo|policy|agenda)",
        ],
    },
    "Creative & General": {
        "keywords": [
            "story", "poem", "creative", "imagine", "brainstorm",
            "idea", "fiction", "character", "dialogue", "scene",
            "recipe", "travel", "game", "fun", "joke", "riddle",
        ],
        "patterns": [],
    },
    "Science & Domain Knowledge": {
        "keywords": [
            "biology", "chemistry", "physics", "ecology", "genome",
            "protein", "molecule", "climate", "geological", "species",
            "evolution", "cell", "DNA", "RNA", "enzyme", "pathogen",
            "agriculture", "soil", "crop", "forestry", "wildlife",
            "hydrology", "watershed", "GIS", "remote sensing",
            "spectroscopy", "chromatography", "PCR", "sequencing",
        ],
        "patterns": [],
    },
    "Translation & Languages": {
        "keywords": [
            "translate", "translation", "Spanish", "French", "German",
            "Chinese", "Japanese", "Korean", "Arabic", "Russian",
            "language", "bilingual", "multilingual", "localize",
        ],
        "patterns": [
            r"translate\s+(this|the|from|to|into)",
        ],
    },
    "Math & Problem Solving": {
        "keywords": [
            "calculate", "equation", "solve", "integral", "derivative",
            "matrix", "vector", "probability", "combinatorics",
            "proof", "theorem", "lemma", "polynomial", "eigenvalue",
            "optimization", "linear algebra", "calculus", "geometry",
        ],
        "patterns": [
            r"\d+\s*[\+\-\*/\^]\s*\d+",
        ],
    },
    "IT & Systems Administration": {
        "keywords": [
            "server", "network", "firewall", "DNS", "SSL", "TLS",
            "certificate", "nginx", "apache", "systemd", "cron",
            "backup", "restore", "monitoring", "alert", "log",
            "permission", "sudo", "SSH", "VPN", "LDAP", "Active Directory",
            "Linux", "Ubuntu", "CentOS", "Windows Server",
        ],
        "patterns": [],
    },
}


def classify_prompt(text: str) -> list[str]:
    """Return ranked list of category labels for a prompt."""
    if not text:
        return ["Uncategorized"]
    lower = text.lower()
    scores: dict[str, float] = {}
    for cat, rules in CATEGORIES.items():
        score = 0.0
        for kw in rules["keywords"]:
            if kw.lower() in lower:
                score += 1.0
        for pat in rules.get("patterns", []):
            if re.search(pat, text, re.IGNORECASE):
                score += 2.0
        if score > 0:
            scores[cat] = score
    if not scores:
        return ["Uncategorized"]
    ranked = sorted(scores, key=scores.get, reverse=True)
    top = scores[ranked[0]]
    return [c for c in ranked if scores[c] >= top * 0.5][:3]


def extract_user_prompt(messages_json: str | None, prompt_text: str | None) -> str | None:
    """Extract the user's input from a request."""
    if messages_json:
        try:
            msgs = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
            if isinstance(msgs, list):
                for msg in reversed(msgs):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            parts = []
                            for block in content:
                                if isinstance(block, dict):
                                    if block.get("type") == "text":
                                        parts.append(block.get("text", ""))
                                    elif block.get("type") in ("image_url", "image"):
                                        parts.append("[image]")
                                elif isinstance(block, str):
                                    parts.append(block)
                            return " ".join(parts) if parts else None
                        if isinstance(content, str) and content.strip():
                            return content.strip()
        except (json.JSONDecodeError, TypeError):
            pass
    if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
        return prompt_text.strip()
    return None


def extract_system_prompt(messages_json: str | None) -> str | None:
    """Extract system prompt if present."""
    if not messages_json:
        return None
    try:
        msgs = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
        if isinstance(msgs, list):
            for msg in msgs:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def truncate(text: str, max_len: int = 300) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def connect_db(url: str) -> pymysql.Connection:
    """Parse a SQLAlchemy-style URL and return a pymysql connection."""
    # mysql+pymysql://user:pass@host:port/dbname
    m = re.match(
        r"mysql\+pymysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", url
    )
    if not m:
        raise ValueError(f"Cannot parse DATABASE_URL: {url}")
    return pymysql.connect(
        host=m.group(3),
        port=int(m.group(4)),
        user=m.group(1),
        password=m.group(2),
        database=m.group(5),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
        read_timeout=600,
    )


def get_overview_stats(conn, sample_size: int = 10000, seed: int = 42) -> dict:
    """Collect high-level statistics using PK sampling for speed.

    Heavy aggregate queries are estimated from random samples instead
    of scanning the full 5M+ row table.
    """
    stats = {}
    rng = random.Random(seed)

    with conn.cursor() as cur:
        # These are instant (PK index / table metadata)
        cur.execute("SELECT MIN(id) as mn, MAX(id) as mx FROM requests")
        bounds = cur.fetchone()
        min_id, max_id = bounds["mn"], bounds["mx"]

        cur.execute("SELECT TABLE_ROWS FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='requests'")
        stats["total_requests"] = cur.fetchone()["TABLE_ROWS"]

        # Get date range from first and last row (PK order)
        cur.execute("SELECT created_at FROM requests WHERE id = %s", (min_id,))
        date_min = cur.fetchone()["created_at"]
        cur.execute("SELECT created_at FROM requests WHERE id = %s", (max_id,))
        date_max = cur.fetchone()["created_at"]
        stats["date_range"] = (date_min, date_max)

        # User/model counts from small tables or fast queries
        cur.execute("SELECT COUNT(*) as cnt FROM users WHERE is_active = 1")
        stats["distinct_users"] = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(DISTINCT name) as cnt FROM models")
        stats["distinct_models"] = cur.fetchone()["cnt"]

    # Sample random rows by PK for aggregate estimates
    print(f"  Sampling {sample_size} rows for statistics...", file=sys.stderr)
    sample_rows = []
    probes = [rng.randint(min_id, max_id) for _ in range(sample_size * 2)]

    with conn.cursor() as cur:
        for probe_id in probes:
            if len(sample_rows) >= sample_size:
                break
            cur.execute("""
                SELECT id, user_id, model, modality, status, user_agent,
                       total_tokens, created_at
                FROM requests WHERE id = %s
            """, (probe_id,))
            row = cur.fetchone()
            if row:
                sample_rows.append(row)

    total_est = stats["total_requests"]
    scale = total_est / len(sample_rows) if sample_rows else 1

    # Modality counts (estimated)
    modality_counter = Counter(r["modality"] for r in sample_rows)
    stats["modality_counts"] = {k: round(v * scale) for k, v in modality_counter.most_common()}

    # Top models (estimated)
    model_counter = Counter(r["model"] for r in sample_rows if r["status"] == "completed")
    stats["top_models"] = [(m, round(c * scale)) for m, c in model_counter.most_common(20)]

    # Client breakdown (estimated)
    def classify_ua(ua):
        if not ua: return "Unknown"
        if "aiohttp" in ua: return "Chat UI"
        if "python-requests" in ua: return "Python SDK"
        if "claude-cli" in ua: return "Claude Code/CoWork"
        if "curl" in ua: return "curl"
        if "openai-python" in ua: return "OpenAI Python SDK"
        if "axios" in ua: return "JavaScript/Axios"
        return "Other"

    client_counter = Counter(classify_ua(r["user_agent"]) for r in sample_rows)
    stats["client_breakdown"] = [(c, round(n * scale)) for c, n in client_counter.most_common()]

    # Hourly distribution
    hourly = Counter(r["created_at"].hour for r in sample_rows if r["created_at"])
    stats["hourly"] = {h: round(c * scale) for h, c in sorted(hourly.items())}

    # Daily distribution
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = Counter(r["created_at"].strftime("%A") for r in sample_rows if r["created_at"])
    stats["daily"] = [(d, round(daily.get(d, 0) * scale)) for d in day_names]

    # Total tokens (estimated from sample)
    token_sum = sum(r["total_tokens"] for r in sample_rows if r["total_tokens"])
    stats["total_tokens"] = round(token_sum * scale)

    # Role and college breakdowns (from users table — small, fast)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT u.id, u.role, u.college
            FROM users u WHERE u.is_active = 1
        """)
        user_info_list = cur.fetchall()

    user_roles = {u["id"]: u["role"] for u in user_info_list}
    user_colleges = {u["id"]: u["college"] for u in user_info_list if u["college"]}

    # Count requests per user from sample
    user_req_counts = Counter(r["user_id"] for r in sample_rows)
    role_users = defaultdict(set)
    role_reqs = defaultdict(int)
    for uid, cnt in user_req_counts.items():
        role = user_roles.get(uid, "unknown")
        role_users[role].add(uid)
        role_reqs[role] += round(cnt * scale)
    stats["role_breakdown"] = sorted(
        [(role, len(role_users[role]), role_reqs[role]) for role in role_users],
        key=lambda x: -x[2]
    )

    college_users = defaultdict(set)
    college_reqs = defaultdict(int)
    for uid, cnt in user_req_counts.items():
        college = user_colleges.get(uid)
        if college:
            college_users[college].add(uid)
            college_reqs[college] += round(cnt * scale)
    stats["college_breakdown"] = sorted(
        [(col, len(college_users[col]), college_reqs[col]) for col in college_users],
        key=lambda x: -x[2]
    )

    return stats


def sample_requests(conn, n: int, seed: int = 42) -> list[dict]:
    """Random sample of completed requests with messages.

    Uses pure PK lookups for speed on large tables.
    """
    rng = random.Random(seed)

    with conn.cursor() as cur:
        cur.execute("SELECT MIN(id) as mn, MAX(id) as mx FROM requests")
        bounds = cur.fetchone()
        min_id, max_id = bounds["mn"], bounds["mx"]

        cur.execute("SELECT id, username, full_name, role, college, department, intended_use FROM users")
        users_map = {r["id"]: r for r in cur.fetchall()}

    samples = []
    seen_ids = set()
    attempts = 0
    max_attempts = n * 8

    print(f"  PK range {min_id}-{max_id}, probing...", file=sys.stderr)

    while len(samples) < n and attempts < max_attempts:
        # Batch probe 50 IDs at a time for efficiency
        batch_ids = [rng.randint(min_id, max_id) for _ in range(50)]
        batch_ids = [i for i in batch_ids if i not in seen_ids]
        if not batch_ids:
            attempts += 50
            continue

        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(batch_ids))
            cur.execute(f"""
                SELECT r.id, r.request_uuid, r.user_id, r.model, r.modality,
                       r.endpoint, r.messages, r.prompt, r.is_streaming,
                       r.prompt_tokens, r.completion_tokens, r.total_tokens,
                       r.total_time_ms, r.user_agent, r.created_at,
                       ak.name as key_name, ak.is_service, ak.service_name
                FROM requests r
                LEFT JOIN api_keys ak ON ak.id = r.api_key_id
                WHERE r.id IN ({placeholders})
                  AND r.status = 'completed' AND r.messages IS NOT NULL
            """, batch_ids)
            rows = cur.fetchall()

        for row in rows:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                u = users_map.get(row["user_id"], {})
                row["username"] = u.get("username", f"user_{row['user_id']}")
                row["full_name"] = u.get("full_name")
                row["role"] = u.get("role")
                row["college"] = u.get("college")
                row["department"] = u.get("department")
                row["intended_use"] = u.get("intended_use")
                samples.append(row)

        attempts += len(batch_ids)
        if len(samples) % 100 == 0 and len(samples) > 0:
            print(f"  ... {len(samples)}/{n} samples collected", file=sys.stderr)

    rng.shuffle(samples)
    return samples[:n]


def sample_archive_requests(conn, n: int, seed: int = 42) -> list[dict]:
    """Sample from the archive database using PK batch lookups."""
    rng = random.Random(seed)
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(id) as mn, MAX(id) as mx FROM archived_requests")
        bounds = cur.fetchone()
        if not bounds or not bounds["mn"]:
            return []

    min_id, max_id = bounds["mn"], bounds["mx"]
    samples = []
    seen_ids = set()
    attempts = 0

    while len(samples) < n and attempts < n * 10:
        batch_ids = [rng.randint(min_id, max_id) for _ in range(50)]
        batch_ids = [i for i in batch_ids if i not in seen_ids]
        if not batch_ids:
            attempts += 50
            continue

        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(batch_ids))
            cur.execute(f"""
                SELECT id, request_uuid, user_id, model, modality,
                       endpoint, messages, prompt, is_streaming,
                       prompt_tokens, completion_tokens, total_tokens,
                       total_time_ms, user_agent, created_at
                FROM archived_requests
                WHERE id IN ({placeholders}) AND status = 'completed' AND messages IS NOT NULL
            """, batch_ids)
            rows = cur.fetchall()

        for row in rows:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                row["username"] = f"user_{row['user_id']}"
                row["full_name"] = None
                row["role"] = None
                row["college"] = None
                row["department"] = None
                row["intended_use"] = None
                row["key_name"] = None
                row["is_service"] = None
                row["service_name"] = None
                samples.append(row)

        attempts += len(batch_ids)

    return samples[:n]


def analyze_samples(samples: list[dict]) -> dict:
    """Analyze sampled requests and produce categorized results."""
    categories = defaultdict(list)
    service_uses = []
    multimodal_uses = []
    embedding_uses = []
    uncategorized = []

    for s in samples:
        prompt = extract_user_prompt(s.get("messages"), s.get("prompt"))
        system = extract_system_prompt(s.get("messages"))
        if not prompt:
            continue

        cats = classify_prompt(prompt + (" " + system if system else ""))

        entry = {
            "prompt": prompt,
            "system_hint": truncate(system, 150) if system else None,
            "model": s.get("model"),
            "modality": s.get("modality"),
            "user": s.get("username"),
            "role": s.get("role"),
            "college": s.get("college"),
            "department": s.get("department"),
            "intended_use": s.get("intended_use"),
            "is_service": s.get("is_service"),
            "service_name": s.get("service_name"),
            "key_name": s.get("key_name"),
            "tokens": s.get("total_tokens"),
            "time_ms": s.get("total_time_ms"),
            "created_at": s.get("created_at"),
            "categories": cats,
        }

        if s.get("is_service"):
            service_uses.append(entry)
        if s.get("modality") in ("embedding", "multimodal"):
            (embedding_uses if s.get("modality") == "embedding" else multimodal_uses).append(entry)

        for cat in cats:
            categories[cat].append(entry)

    return {
        "by_category": dict(categories),
        "service_uses": service_uses,
        "multimodal_uses": multimodal_uses,
        "embedding_uses": embedding_uses,
    }


def format_bar(value: int, max_val: int, width: int = 30) -> str:
    if max_val == 0:
        return ""
    filled = round(width * value / max_val)
    return "█" * filled + "░" * (width - filled)


def generate_report(stats: dict, analysis: dict, samples: list[dict],
                    archive_stats: dict | None = None) -> str:
    """Generate a markdown report."""
    lines = []
    w = lines.append

    w("# MindRouter Usage Audit Report")
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("")

    # === Overview ===
    w("## Overview")
    w("")
    d0, d1 = stats["date_range"]
    days = max(1, (d1 - d0).days)
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Date range (main DB) | {d0.strftime('%Y-%m-%d')} → {d1.strftime('%Y-%m-%d')} ({days} days) |")
    w(f"| Total requests | {stats['total_requests']:,} |")
    w(f"| Avg requests/day | {stats['total_requests'] // days:,} |")
    w(f"| Distinct users | {stats['distinct_users']} |")
    w(f"| Distinct models used | {stats['distinct_models']} |")
    w(f"| Total tokens served | {stats['total_tokens']:,} |")
    if archive_stats:
        w(f"| Archived requests | {archive_stats.get('total', 0):,} |")
        ar = archive_stats.get("date_range")
        if ar and ar[0] and ar[1]:
            w(f"| Archive date range | {ar[0].strftime('%Y-%m-%d')} → {ar[1].strftime('%Y-%m-%d')} |")
    w("")

    # === Modality ===
    w("## Request Modality")
    w("")
    mc = stats["modality_counts"]
    max_mc = max(mc.values()) if mc else 1
    for mod, cnt in sorted(mc.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / stats["total_requests"]
        w(f"- **{mod}**: {cnt:,} ({pct:.1f}%) {format_bar(cnt, max_mc, 20)}")
    w("")

    # === Client breakdown ===
    w("## Client Types")
    w("")
    for client, cnt in stats["client_breakdown"]:
        pct = 100 * cnt / stats["total_requests"]
        w(f"- **{client}**: {cnt:,} ({pct:.1f}%)")
    w("")

    # === User roles ===
    w("## User Roles")
    w("")
    w("| Role | Users | Requests | Req/User |")
    w("|------|-------|----------|----------|")
    for role, users, reqs in stats["role_breakdown"]:
        w(f"| {role or 'unset'} | {users} | {reqs:,} | {reqs // max(1, users):,} |")
    w("")

    # === Colleges ===
    if stats["college_breakdown"]:
        w("## Colleges / Units")
        w("")
        w("| College | Users | Requests |")
        w("|---------|-------|----------|")
        for college, users, reqs in stats["college_breakdown"]:
            w(f"| {college} | {users} | {reqs:,} |")
        w("")

    # === Top models ===
    w("## Top Models")
    w("")
    max_m = stats["top_models"][0][1] if stats["top_models"] else 1
    for model, cnt in stats["top_models"][:15]:
        pct = 100 * cnt / stats["total_requests"]
        w(f"- **{model}**: {cnt:,} ({pct:.1f}%) {format_bar(cnt, max_m, 20)}")
    w("")

    # === Time patterns ===
    w("## Usage by Day of Week")
    w("")
    max_d = max(c for _, c in stats["daily"]) if stats["daily"] else 1
    for day, cnt in stats["daily"]:
        w(f"- **{day}**: {cnt:,} {format_bar(cnt, max_d, 20)}")
    w("")

    w("## Usage by Hour (UTC)")
    w("")
    max_h = max(stats["hourly"].values()) if stats["hourly"] else 1
    for hr in range(24):
        cnt = stats["hourly"].get(hr, 0)
        w(f"- **{hr:02d}:00**: {cnt:,} {format_bar(cnt, max_h, 20)}")
    w("")

    # === Category analysis ===
    w("## Usage Categories (from sampled interactions)")
    w(f"*Based on {len(samples)} sampled requests with keyword/pattern classification*")
    w("")

    by_cat = analysis["by_category"]
    cat_counts = {cat: len(entries) for cat, entries in by_cat.items()}
    max_cat = max(cat_counts.values()) if cat_counts else 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        w(f"- **{cat}**: {cnt} samples {format_bar(cnt, max_cat, 20)}")
    w("")

    # === Sample interactions per category ===
    w("---")
    w("## Sample Interactions by Category")
    w("")

    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        entries = by_cat[cat]
        w(f"### {cat} ({len(entries)} samples)")
        w("")

        # Show up to 8 diverse examples per category
        shown = set()
        display = []
        for e in entries:
            sig = (e["user"], truncate(e["prompt"], 80))
            if sig not in shown:
                shown.add(sig)
                display.append(e)
            if len(display) >= 8:
                break

        for i, e in enumerate(display, 1):
            prompt_display = truncate(e["prompt"], 400)
            meta_parts = []
            if e["role"]:
                meta_parts.append(e["role"])
            if e["college"]:
                meta_parts.append(e["college"])
            if e["department"]:
                meta_parts.append(e["department"])
            meta_parts.append(e["model"] or "?")
            if e["is_service"]:
                meta_parts.append(f"service: {e['service_name'] or e['key_name']}")
            meta = " · ".join(meta_parts)
            ts = e["created_at"].strftime("%Y-%m-%d %H:%M") if e["created_at"] else "?"

            w(f"**{i}.** [{ts}] ({meta})")
            if e.get("system_hint"):
                w(f"> System: {e['system_hint']}")
            w(f"> {prompt_display}")
            w("")
    w("")

    # === Service/API key usage ===
    if analysis["service_uses"]:
        w("---")
        w("## Service Account Usage")
        w("")
        svc_by_name = defaultdict(list)
        for e in analysis["service_uses"]:
            name = e.get("service_name") or e.get("key_name") or "unnamed"
            svc_by_name[name].append(e)

        for svc, entries in sorted(svc_by_name.items(), key=lambda x: -len(x[1])):
            w(f"### {svc} ({len(entries)} samples)")
            for e in entries[:5]:
                w(f"- [{e.get('model')}] {truncate(e['prompt'], 200)}")
            w("")

    # === Intended use from user profiles ===
    intended_uses = set()
    for s in samples:
        iu = s.get("intended_use")
        if iu and isinstance(iu, str) and iu.strip():
            intended_uses.add((s.get("username"), iu.strip()))

    if intended_uses:
        w("---")
        w("## Self-Reported Intended Uses (from user profiles)")
        w("")
        for username, use in sorted(intended_uses):
            w(f"- **{username}**: {truncate(use, 300)}")
        w("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MindRouter Usage Audit")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of requests to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--include-archive", action="store_true",
                        help="Also sample from the archive database")
    parser.add_argument("--archive-samples", type=int, default=50,
                        help="Samples from archive DB (default: 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write report to file (default: stdout)")
    parser.add_argument("--db-url", type=str,
                        default=os.environ.get("DATABASE_URL"),
                        help="Main database URL (default: $DATABASE_URL)")
    parser.add_argument("--archive-db-url", type=str,
                        default=os.environ.get("ARCHIVE_DATABASE_URL"),
                        help="Archive database URL (default: $ARCHIVE_DATABASE_URL)")
    args = parser.parse_args()

    if not args.db_url:
        print("Error: DATABASE_URL not set. Pass --db-url or set the env var.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to main database...", file=sys.stderr)
    conn = connect_db(args.db_url)

    print(f"Collecting overview statistics...", file=sys.stderr)
    stats = get_overview_stats(conn)
    print(f"  {stats['total_requests']:,} requests from {stats['distinct_users']} users", file=sys.stderr)

    print(f"Sampling {args.samples} requests (stratified by user)...", file=sys.stderr)
    samples = sample_requests(conn, args.samples, args.seed)
    print(f"  Got {len(samples)} samples", file=sys.stderr)

    archive_stats = None
    archive_samples = []
    if args.include_archive and args.archive_db_url:
        print(f"Connecting to archive database...", file=sys.stderr)
        aconn = connect_db(args.archive_db_url)
        with aconn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM archived_requests")
            total = cur.fetchone()["cnt"]
            cur.execute("SELECT MIN(created_at) as mn, MAX(created_at) as mx FROM archived_requests")
            row = cur.fetchone()
            archive_stats = {"total": total, "date_range": (row["mn"], row["mx"])}
        print(f"  {total:,} archived requests", file=sys.stderr)
        print(f"Sampling {args.archive_samples} from archive...", file=sys.stderr)
        archive_samples = sample_archive_requests(aconn, args.archive_samples, args.seed)
        print(f"  Got {len(archive_samples)} archive samples", file=sys.stderr)
        aconn.close()

    all_samples = samples + archive_samples

    print(f"Analyzing and categorizing {len(all_samples)} samples...", file=sys.stderr)
    analysis = analyze_samples(all_samples)

    print(f"Generating report...", file=sys.stderr)
    report = generate_report(stats, analysis, all_samples, archive_stats)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    conn.close()
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
