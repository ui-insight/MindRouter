#!/usr/bin/env python3
"""
MindRouter Per-User Usage Audit — proportional sampling across ALL active users.

Every user who made at least one request in the reporting period is included.
Each user's sample count is proportional to their total requests, with a
guaranteed minimum so rare users are never invisible.

Run inside the app container:
    docker exec -it mindrouter-app-1 python scripts/usage_audit_by_user.py

With options:
    docker exec -it mindrouter-app-1 python scripts/usage_audit_by_user.py \
        --days 30 --total-samples 300 --min-per-user 2 \
        --output /data/artifacts/user_audit.md
"""

import argparse
import json
import math
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
# Category definitions (shared with usage_audit.py)
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


def extract_user_prompt(messages_json, prompt_text):
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


def extract_system_prompt(messages_json):
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


def format_bar(value, max_val, width=30):
    if max_val == 0:
        return ""
    filled = round(width * value / max_val)
    return "█" * filled + "░" * (width - filled)


def connect_db(url: str) -> pymysql.Connection:
    m = re.match(r"mysql\+pymysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", url)
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


# ---------------------------------------------------------------------------
# Core logic: per-user proportional sampling
# ---------------------------------------------------------------------------

def get_users_map(conn) -> dict[int, dict]:
    """Load all users from the main DB (small table, fast)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, username, full_name, role, college,
                   department, intended_use
            FROM users
        """)
        return {r["id"]: r for r in cur.fetchall()}


def get_user_request_counts(conn, start_dt: datetime, end_dt: datetime) -> dict[int, dict]:
    """Per-user request counts from the main requests table.

    Returns {user_id: {request_count, total_tokens, first_request, last_request}}.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_id,
                   COUNT(*) AS request_count,
                   SUM(total_tokens) AS total_tokens,
                   MIN(created_at) AS first_request,
                   MAX(created_at) AS last_request
            FROM requests
            WHERE created_at >= %s AND created_at < %s
              AND status = 'completed'
            GROUP BY user_id
        """, (start_dt, end_dt))
        return {r["user_id"]: r for r in cur.fetchall()}


def get_archive_user_request_counts(aconn, start_dt: datetime,
                                     end_dt: datetime) -> dict[int, dict]:
    """Per-user request counts from the archive database."""
    with aconn.cursor() as cur:
        cur.execute("""
            SELECT user_id,
                   COUNT(*) AS request_count,
                   SUM(total_tokens) AS total_tokens,
                   MIN(created_at) AS first_request,
                   MAX(created_at) AS last_request
            FROM archived_requests
            WHERE created_at >= %s AND created_at < %s
              AND status = 'completed'
            GROUP BY user_id
        """, (start_dt, end_dt))
        return {r["user_id"]: r for r in cur.fetchall()}


def merge_user_counts(main_counts: dict[int, dict],
                      archive_counts: dict[int, dict],
                      users_map: dict[int, dict]) -> list[dict]:
    """Merge main + archive per-user counts into a single sorted list.

    Each entry includes user metadata from the users table and
    separate main_count / archive_count fields for proportional sampling.
    """
    all_uids = set(main_counts) | set(archive_counts)
    merged = []
    for uid in all_uids:
        m = main_counts.get(uid, {})
        a = archive_counts.get(uid, {})
        u = users_map.get(uid, {})
        mc = m.get("request_count", 0)
        ac = a.get("request_count", 0)

        dates = [d for d in [m.get("first_request"), a.get("first_request")] if d]
        first = min(dates) if dates else None
        dates = [d for d in [m.get("last_request"), a.get("last_request")] if d]
        last = max(dates) if dates else None

        merged.append({
            "user_id": uid,
            "username": u.get("username") or f"user_{uid}",
            "full_name": u.get("full_name"),
            "role": u.get("role"),
            "college": u.get("college"),
            "department": u.get("department"),
            "intended_use": u.get("intended_use"),
            "request_count": mc + ac,
            "main_count": mc,
            "archive_count": ac,
            "total_tokens": (m.get("total_tokens") or 0) + (a.get("total_tokens") or 0),
            "first_request": first,
            "last_request": last,
        })

    merged.sort(key=lambda x: -x["request_count"])
    return merged


def allocate_samples(user_counts: list[dict], total_budget: int,
                     min_per_user: int) -> dict[int, int]:
    """Allocate sample budget across users proportionally.

    Every user gets at least *min_per_user* samples (capped at their
    actual request count).  The remaining budget is distributed in
    proportion to each user's request volume.
    """
    allocation = {}
    remaining_budget = total_budget

    for u in user_counts:
        uid = u["user_id"]
        guaranteed = min(min_per_user, u["request_count"])
        allocation[uid] = guaranteed
        remaining_budget -= guaranteed

    if remaining_budget > 0:
        total_reqs = sum(
            max(0, u["request_count"] - allocation[u["user_id"]])
            for u in user_counts
        )
        if total_reqs > 0:
            for u in user_counts:
                uid = u["user_id"]
                eligible = u["request_count"] - allocation[uid]
                if eligible <= 0:
                    continue
                share = remaining_budget * (eligible / total_reqs)
                extra = min(int(share), eligible)
                allocation[uid] += extra

    # Distribute any leftover from rounding (give 1 each to largest users)
    used = sum(allocation.values())
    leftover = total_budget - used
    if leftover > 0:
        by_size = sorted(user_counts,
                         key=lambda u: u["request_count"], reverse=True)
        for u in by_size:
            if leftover <= 0:
                break
            uid = u["user_id"]
            room = u["request_count"] - allocation[uid]
            if room > 0:
                allocation[uid] += 1
                leftover -= 1

    return allocation


def sample_user_requests(conn, user_id: int, n: int,
                         start_dt: datetime, end_dt: datetime,
                         rng: random.Random) -> list[dict]:
    """Sample up to *n* completed requests for a single user in the period."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM requests
            WHERE user_id = %s AND created_at >= %s AND created_at < %s
              AND status = 'completed' AND messages IS NOT NULL
        """, (user_id, start_dt, end_dt))
        ids = [row["id"] for row in cur.fetchall()]

    if not ids:
        return []

    chosen = rng.sample(ids, min(n, len(ids)))

    rows = []
    for batch_start in range(0, len(chosen), 50):
        batch = chosen[batch_start:batch_start + 50]
        placeholders = ",".join(["%s"] * len(batch))
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT r.id, r.request_uuid, r.user_id, r.model, r.modality,
                       r.endpoint, r.messages, r.prompt, r.is_streaming,
                       r.prompt_tokens, r.completion_tokens, r.total_tokens,
                       r.total_time_ms, r.user_agent, r.created_at,
                       ak.name AS key_name, ak.is_service, ak.service_name
                FROM requests r
                LEFT JOIN api_keys ak ON ak.id = r.api_key_id
                WHERE r.id IN ({placeholders})
            """, batch)
            for row in cur.fetchall():
                row["_source"] = "main"
                rows.append(row)
    return rows


def sample_user_archive_requests(aconn, user_id: int, n: int,
                                  start_dt: datetime, end_dt: datetime,
                                  rng: random.Random) -> list[dict]:
    """Sample up to *n* archived requests for a single user in the period."""
    with aconn.cursor() as cur:
        cur.execute("""
            SELECT id FROM archived_requests
            WHERE user_id = %s AND created_at >= %s AND created_at < %s
              AND status = 'completed' AND messages IS NOT NULL
        """, (user_id, start_dt, end_dt))
        ids = [row["id"] for row in cur.fetchall()]

    if not ids:
        return []

    chosen = rng.sample(ids, min(n, len(ids)))

    rows = []
    for batch_start in range(0, len(chosen), 50):
        batch = chosen[batch_start:batch_start + 50]
        placeholders = ",".join(["%s"] * len(batch))
        with aconn.cursor() as cur:
            cur.execute(f"""
                SELECT id, request_uuid, user_id, model, modality,
                       endpoint, messages, prompt, is_streaming,
                       prompt_tokens, completion_tokens, total_tokens,
                       total_time_ms, user_agent, created_at
                FROM archived_requests
                WHERE id IN ({placeholders})
            """, batch)
            for row in cur.fetchall():
                row["key_name"] = None
                row["is_service"] = None
                row["service_name"] = None
                row["_source"] = "archive"
                rows.append(row)
    return rows


def classify_ua(ua):
    if not ua:
        return "Unknown"
    if "aiohttp" in ua:
        return "Chat UI"
    if "python-requests" in ua:
        return "Python SDK"
    if "claude-cli" in ua:
        return "Claude Code/CoWork"
    if "curl" in ua:
        return "curl"
    if "openai-python" in ua:
        return "OpenAI Python SDK"
    if "axios" in ua:
        return "JavaScript/Axios"
    return "Other"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(user_counts, allocation, all_samples, user_samples_map,
                    start_dt, end_dt, include_archive=False) -> str:
    lines = []
    w = lines.append

    days = max(1, (end_dt - start_dt).days)
    total_reqs = sum(u["request_count"] for u in user_counts)
    total_main = sum(u.get("main_count", u["request_count"]) for u in user_counts)
    total_archive = sum(u.get("archive_count", 0) for u in user_counts)
    total_tokens = sum(u["total_tokens"] or 0 for u in user_counts)
    total_sampled = sum(allocation.values())

    w("# MindRouter Per-User Usage Audit")
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Period**: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} ({days} days)")
    if include_archive:
        w(f"**Sources**: main DB + archive DB")
    w("")

    # --- Overview ---
    w("## Overview")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Active users | {len(user_counts)} |")
    w(f"| Total requests | {total_reqs:,} |")
    if include_archive and total_archive > 0:
        w(f"|   ↳ main DB | {total_main:,} |")
        w(f"|   ↳ archive DB | {total_archive:,} |")
    w(f"| Avg requests/day | {total_reqs // days:,} |")
    w(f"| Total tokens | {total_tokens:,} |")
    w(f"| Samples collected | {len(all_samples)} (budget: {total_sampled}) |")
    w("")

    # --- User summary table ---
    w("## Active Users")
    w("")
    if include_archive and total_archive > 0:
        w("| User | Role | College | Requests (main+arch) | Tokens | Samples | First | Last |")
        w("|------|------|---------|----------------------|--------|---------|-------|------|")
        for u in user_counts:
            uid = u["user_id"]
            username = u["username"] or f"user_{uid}"
            role = u["role"] or "—"
            college = u["college"] or "—"
            mc = u.get("main_count", u["request_count"])
            ac = u.get("archive_count", 0)
            req_str = f"{u['request_count']:,}"
            if ac > 0:
                req_str += f" ({mc:,}+{ac:,})"
            tokens = u["total_tokens"] or 0
            sampled = allocation.get(uid, 0)
            first = u["first_request"].strftime("%m-%d") if u["first_request"] else "?"
            last = u["last_request"].strftime("%m-%d") if u["last_request"] else "?"
            w(f"| {username} | {role} | {college} | {req_str} | {tokens:,} | {sampled} | {first} | {last} |")
    else:
        w("| User | Role | College | Requests | Tokens | Samples | First | Last |")
        w("|------|------|---------|----------|--------|---------|-------|------|")
        for u in user_counts:
            uid = u["user_id"]
            username = u["username"] or f"user_{uid}"
            role = u["role"] or "—"
            college = u["college"] or "—"
            reqs = u["request_count"]
            tokens = u["total_tokens"] or 0
            sampled = allocation.get(uid, 0)
            first = u["first_request"].strftime("%m-%d") if u["first_request"] else "?"
            last = u["last_request"].strftime("%m-%d") if u["last_request"] else "?"
            w(f"| {username} | {role} | {college} | {reqs:,} | {tokens:,} | {sampled} | {first} | {last} |")
    w("")

    # --- Usage distribution ---
    w("## Usage Distribution")
    w("")
    max_reqs = user_counts[0]["request_count"] if user_counts else 1
    for u in user_counts[:30]:
        username = u["username"] or f"user_{u['user_id']}"
        pct = 100 * u["request_count"] / total_reqs if total_reqs else 0
        w(f"- **{username}**: {u['request_count']:,} ({pct:.1f}%) "
          f"{format_bar(u['request_count'], max_reqs, 25)}")
    if len(user_counts) > 30:
        w(f"- *...and {len(user_counts) - 30} more users*")
    w("")

    # --- Role breakdown ---
    role_users = defaultdict(set)
    role_reqs = defaultdict(int)
    role_tokens = defaultdict(int)
    for u in user_counts:
        role = u["role"] or "unset"
        role_users[role].add(u["user_id"])
        role_reqs[role] += u["request_count"]
        role_tokens[role] += u["total_tokens"] or 0

    w("## By Role")
    w("")
    w("| Role | Users | Requests | Tokens | Req/User |")
    w("|------|-------|----------|--------|----------|")
    for role in sorted(role_users, key=lambda r: -role_reqs[r]):
        n_users = len(role_users[role])
        w(f"| {role} | {n_users} | {role_reqs[role]:,} | "
          f"{role_tokens[role]:,} | {role_reqs[role] // max(1, n_users):,} |")
    w("")

    # --- College breakdown ---
    college_users = defaultdict(set)
    college_reqs = defaultdict(int)
    for u in user_counts:
        if u["college"]:
            college_users[u["college"]].add(u["user_id"])
            college_reqs[u["college"]] += u["request_count"]

    if college_users:
        w("## By College / Unit")
        w("")
        w("| College | Users | Requests |")
        w("|---------|-------|----------|")
        for col in sorted(college_users, key=lambda c: -college_reqs[c]):
            w(f"| {col} | {len(college_users[col])} | {college_reqs[col]:,} |")
        w("")

    # --- Model popularity ---
    model_counter = Counter()
    for s in all_samples:
        if s.get("model"):
            model_counter[s["model"]] += 1
    if model_counter:
        w("## Models Used (from samples)")
        w("")
        max_mc = model_counter.most_common(1)[0][1]
        for model, cnt in model_counter.most_common(15):
            w(f"- **{model}**: {cnt} samples {format_bar(cnt, max_mc, 20)}")
        w("")

    # --- Client types ---
    client_counter = Counter(classify_ua(s.get("user_agent")) for s in all_samples)
    if client_counter:
        w("## Client Types (from samples)")
        w("")
        for client, cnt in client_counter.most_common():
            pct = 100 * cnt / len(all_samples) if all_samples else 0
            w(f"- **{client}**: {cnt} ({pct:.1f}%)")
        w("")

    # --- Category analysis ---
    categories = defaultdict(list)
    for s in all_samples:
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
            "tokens": s.get("total_tokens"),
            "created_at": s.get("created_at"),
            "categories": cats,
        }
        for cat in cats:
            categories[cat].append(entry)

    w("## Usage Categories (from sampled interactions)")
    w(f"*Based on {len(all_samples)} samples across {len(user_counts)} users*")
    w("")
    cat_counts = {cat: len(entries) for cat, entries in categories.items()}
    max_cat = max(cat_counts.values()) if cat_counts else 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        w(f"- **{cat}**: {cnt} samples {format_bar(cnt, max_cat, 20)}")
    w("")

    # --- Per-user sample details ---
    w("---")
    w("## Sampled Interactions by User")
    w("")

    for u in user_counts:
        uid = u["user_id"]
        username = u["username"] or f"user_{uid}"
        samples = user_samples_map.get(uid, [])
        if not samples:
            continue

        role_str = u["role"] or "unset"
        college_str = u["college"] or ""
        meta = f"{role_str}"
        if college_str:
            meta += f" · {college_str}"
        if u["department"]:
            meta += f" · {u['department']}"

        w(f"### {username} ({meta}) — {u['request_count']:,} requests, "
          f"{len(samples)} sampled")
        w("")

        for i, s in enumerate(samples, 1):
            prompt = extract_user_prompt(s.get("messages"), s.get("prompt"))
            if not prompt:
                continue
            system = extract_system_prompt(s.get("messages"))
            cats = classify_prompt(prompt + (" " + system if system else ""))
            cat_str = ", ".join(cats)
            model = s.get("model") or "?"
            ts = s["created_at"].strftime("%Y-%m-%d %H:%M") if s.get("created_at") else "?"
            tokens = s.get("total_tokens") or 0

            w(f"**{i}.** [{ts}] {model} · {tokens:,} tok · _{cat_str}_")
            if system:
                w(f"> System: {truncate(system, 150)}")
            w(f"> {truncate(prompt, 400)}")
            w("")

    # --- Intended uses ---
    intended = [(u["username"], u["intended_use"])
                for u in user_counts
                if u.get("intended_use") and u["intended_use"].strip()]
    if intended:
        w("---")
        w("## Self-Reported Intended Uses")
        w("")
        for username, use in sorted(intended):
            w(f"- **{username}**: {truncate(use, 300)}")
        w("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="MindRouter per-user proportional usage audit")
    parser.add_argument("--days", type=int, default=30,
                        help="Look back N days from today (default: 30)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: now)")
    parser.add_argument("--total-samples", type=int, default=300,
                        help="Total sample budget across all users (default: 300)")
    parser.add_argument("--min-per-user", type=int, default=2,
                        help="Minimum samples per user (default: 2)")
    parser.add_argument("--include-archive", action="store_true",
                        help="Also sample from the archive database")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="Write report to file (default: stdout)")
    parser.add_argument("--db-url", type=str,
                        default=os.environ.get("DATABASE_URL"),
                        help="Database URL (default: $DATABASE_URL)")
    parser.add_argument("--archive-db-url", type=str,
                        default=os.environ.get("ARCHIVE_DATABASE_URL"),
                        help="Archive database URL (default: $ARCHIVE_DATABASE_URL)")
    args = parser.parse_args()

    if not args.db_url:
        print("Error: DATABASE_URL not set. Pass --db-url or set the env var.",
              file=sys.stderr)
        sys.exit(1)

    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_dt = datetime.now() - timedelta(days=args.days)
    end_dt = (datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)
              if args.end else datetime.now())

    rng = random.Random(args.seed)

    print(f"Period: {start_dt.strftime('%Y-%m-%d')} to "
          f"{end_dt.strftime('%Y-%m-%d')}", file=sys.stderr)

    print("Connecting to main database...", file=sys.stderr)
    conn = connect_db(args.db_url)

    # Load user metadata from main DB
    users_map = get_users_map(conn)

    # Per-user counts from main DB
    print("Finding active users in main DB...", file=sys.stderr)
    main_counts = get_user_request_counts(conn, start_dt, end_dt)
    print(f"  {len(main_counts)} users, "
          f"{sum(c['request_count'] for c in main_counts.values()):,} requests",
          file=sys.stderr)

    # Per-user counts from archive DB (optional)
    aconn = None
    archive_counts: dict[int, dict] = {}
    if args.include_archive and args.archive_db_url:
        print("Connecting to archive database...", file=sys.stderr)
        aconn = connect_db(args.archive_db_url)
        archive_counts = get_archive_user_request_counts(aconn, start_dt, end_dt)
        print(f"  {len(archive_counts)} users, "
              f"{sum(c['request_count'] for c in archive_counts.values()):,} "
              f"archived requests", file=sys.stderr)
    elif args.include_archive:
        print("Warning: --include-archive set but no ARCHIVE_DATABASE_URL",
              file=sys.stderr)

    # Merge into unified user list
    user_counts = merge_user_counts(main_counts, archive_counts, users_map)
    total_reqs = sum(u["request_count"] for u in user_counts)
    print(f"  {len(user_counts)} total active users, {total_reqs:,} combined requests",
          file=sys.stderr)

    if not user_counts:
        print("No requests found in the specified period.", file=sys.stderr)
        conn.close()
        if aconn:
            aconn.close()
        sys.exit(0)

    print(f"Allocating {args.total_samples} samples "
          f"(min {args.min_per_user}/user)...", file=sys.stderr)
    allocation = allocate_samples(user_counts, args.total_samples,
                                  args.min_per_user)
    print(f"  Budget: {sum(allocation.values())} across "
          f"{len(allocation)} users", file=sys.stderr)

    # For each user, split their allocation between main and archive
    # proportionally to how many requests come from each source.
    print("Sampling requests per user...", file=sys.stderr)
    user_samples_map: dict[int, list[dict]] = {}
    all_samples = []
    for i, u in enumerate(user_counts, 1):
        uid = u["user_id"]
        n = allocation.get(uid, 0)
        if n <= 0:
            continue

        mc = u.get("main_count", 0)
        ac = u.get("archive_count", 0)
        total = mc + ac

        # Split allocation proportionally between main and archive
        if total > 0 and ac > 0 and aconn:
            n_main = round(n * mc / total)
            n_archive = n - n_main
        else:
            n_main = n
            n_archive = 0

        samples = []
        if n_main > 0:
            samples.extend(
                sample_user_requests(conn, uid, n_main, start_dt, end_dt, rng))
        if n_archive > 0 and aconn:
            samples.extend(
                sample_user_archive_requests(aconn, uid, n_archive,
                                             start_dt, end_dt, rng))

        for s in samples:
            s["username"] = u["username"]
            s["role"] = u["role"]
            s["college"] = u["college"]
            s["department"] = u["department"]
            s["intended_use"] = u["intended_use"]
        user_samples_map[uid] = samples
        all_samples.extend(samples)
        if i % 20 == 0:
            print(f"  ... {i}/{len(user_counts)} users sampled",
                  file=sys.stderr)

    main_samples = sum(1 for s in all_samples if s.get("_source") == "main")
    arch_samples = sum(1 for s in all_samples if s.get("_source") == "archive")
    print(f"  {len(all_samples)} total samples collected"
          f" ({main_samples} main, {arch_samples} archive)",
          file=sys.stderr)

    print("Generating report...", file=sys.stderr)
    report = generate_report(user_counts, allocation, all_samples,
                             user_samples_map, start_dt, end_dt,
                             include_archive=bool(archive_counts))

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    conn.close()
    if aconn:
        aconn.close()
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
