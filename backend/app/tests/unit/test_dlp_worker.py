############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_worker.py: Unit tests for the DLP worker, admin
#     routes, and alert retention (2.9.9)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the DLP worker + admin surface (2.9.9).

Before 2.9.9 dlp_worker.py and every DLP route had zero test coverage, and
that untested code held a live bug: alert email called
``email_service.send_email()``, which does not exist, so notifications had
never fired in the product's life.

Covers:
- Credential removal: no plaintext key, no ensure_internal_api_key, no
  self-call; _internal_chat sends no Authorization header
- Email: the called function actually exists (AST cross-module contract),
  correct subject, no matched values in the body, flood guard
- Route validation: bad regex/JSON/threshold/email rejected before any write
- Filter normalization: empty query params must not become filters
- Retention: dlp_alerts wired into defaults, purge, cycle, and the UI

dlp_worker.py and dlp_routes.py are spec-loaded with their dependencies
pre-mocked in sys.modules, so the real functions execute — see MEMORY.md
"Import Chain Gotcha" (Fix A + Fix B).
"""

import ast
import re
import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_SERVICES_DIR = _APP_DIR / "services"
_DASHBOARD_DIR = _APP_DIR / "dashboard"
_DB_DIR = _APP_DIR / "db"

WORKER_SRC = (_SERVICES_DIR / "dlp_worker.py").read_text()
SCANNER_SRC = (_SERVICES_DIR / "dlp_scanner.py").read_text()
ROUTES_SRC = (_DASHBOARD_DIR / "dlp_routes.py").read_text()
EMAIL_SRC = (_SERVICES_DIR / "email_service.py").read_text()
RETENTION_SRC = (_SERVICES_DIR / "retention.py").read_text()
CRUD_SRC = (_DB_DIR / "crud.py").read_text()


def _spec_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def worker():
    """Spec-load dlp_worker.py; its module-level imports are only asyncio,
    time, and logging_config, so nothing DB-shaped is pulled in."""
    keys = ("backend", "backend.app", "backend.app.logging_config")
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules.setdefault("backend", MagicMock())
    sys.modules.setdefault("backend.app", MagicMock())
    sys.modules["backend.app.logging_config"] = MagicMock(
        get_logger=MagicMock(return_value=MagicMock())
    )
    try:
        yield _spec_load("dlp_worker_under_test", _SERVICES_DIR / "dlp_worker.py")
    finally:
        # Restore exactly, including popping keys we introduced: a leaked mock
        # of backend.app.* silently breaks unrelated modules later in the run.
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ===================================================================
# Credential removal (the 2.9.9 critical)
# ===================================================================

class TestCredentialRemoved:
    """The DLP LLM scanner must hold no credential anywhere.

    It used to mint a never-expiring API key owned by the bootstrap admin
    (so: admin rights) and store the RAW value in app_config — the only
    unhashed key in a system that otherwise persists Argon2 + SHA-256 only.
    """

    def test_no_raw_key_config_outside_migrations(self):
        """No code may read or write the credential config keys.

        Checks string literals via AST rather than raw text, so the historical
        note in dlp_worker.py naming the retired keys doesn't count as a use.
        """
        offenders = []
        for path in sorted(_APP_DIR.rglob("*.py")):
            if "migrations" in path.parts or path.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for needle in ("dlp.internal_api_key_raw", "dlp.internal_api_key_id"):
                        if needle in node.value:
                            offenders.append(f"{path.relative_to(_APP_DIR)}:{node.lineno} {needle}")
        assert not offenders, f"DLP credential config key still referenced: {offenders}"

    def test_ensure_internal_api_key_is_gone(self, worker):
        assert not hasattr(worker, "ensure_internal_api_key")
        assert "await ensure_internal_api_key" not in ROUTES_SRC

    def test_worker_never_generates_an_api_key(self):
        assert "generate_api_key" not in WORKER_SRC
        assert "create_api_key" not in WORKER_SRC

    def test_scanner_makes_no_gateway_self_call(self):
        assert "localhost:8000" not in WORKER_SRC
        assert "localhost:8000" not in SCANNER_SRC

    def test_migration_071_revokes_before_deleting(self):
        """Dropping the id row without revoking would orphan a live,
        never-expiring, admin-capable key with no pointer left to find it."""
        mig = (_DB_DIR / "migrations" / "versions"
               / "20260807_000000_071_dlp_key_removal_and_index.py").read_text()
        assert mig.index("UPDATE api_keys SET status = 'revoked'") < mig.index("DELETE FROM app_config")
        assert "ix_dlp_alerts_scanned_at" in mig
        assert 'revision = "071"' in mig and 'down_revision = "070"' in mig


class TestInternalChat:
    """_internal_chat dispatches straight to a backend — no key, no gateway."""

    def _registry(self, backends, available=True):
        reg = MagicMock()
        reg.resolve_alias = MagicMock(return_value=("resolved-model", None))

        async def _get(model, modality=None):
            return backends

        async def _avail(bid):
            return available

        reg.get_backends_with_model = _get
        reg.is_backend_available = _avail
        # The real picker, run against the stubs above (same selection logic).
        from backend.app.core.telemetry.registry import BackendRegistry

        async def _pick(model_name, *, engine=None):
            return await BackendRegistry.pick_available_backend(reg, model_name, engine=engine)

        reg.pick_available_backend = _pick
        return reg

    def _backend(self, engine="vllm", url="http://node1:8000"):
        b = MagicMock()
        b.id = 7
        b.url = url
        b.engine = engine
        return b

    async def _run(self, worker, registry, capture, model="m"):
        """Execute _internal_chat with the registry + httpx pre-mocked.

        Async on purpose: asyncio.run() would close the ambient event loop,
        which breaks sibling test modules that call get_event_loop().
        """
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "[]"}}]}

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kwargs):
                capture["url"] = url
                capture["kwargs"] = kwargs
                return _Resp()

        engines = MagicMock()
        engines.OLLAMA = "ollama"

        saved = {k: sys.modules.get(k) for k in
                 ("httpx", "backend.app.core.telemetry.registry", "backend.app.db.models",
                  "backend.app.core", "backend.app.core.telemetry", "backend.app.db")}
        sys.modules["httpx"] = MagicMock(AsyncClient=_Client, Timeout=MagicMock())
        sys.modules.setdefault("backend.app.core", MagicMock())
        sys.modules.setdefault("backend.app.core.telemetry", MagicMock())
        sys.modules.setdefault("backend.app.db", MagicMock())
        sys.modules["backend.app.core.telemetry.registry"] = MagicMock(
            get_registry=MagicMock(return_value=registry)
        )
        sys.modules["backend.app.db.models"] = MagicMock(BackendEngine=engines)
        try:
            return await worker._internal_chat(model, [{"role": "user", "content": "x"}])
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    @pytest.mark.asyncio
    async def test_sends_no_authorization_header(self, worker):
        capture = {}
        await self._run(worker, self._registry([self._backend()]), capture)
        headers = capture["kwargs"].get("headers") or {}
        assert "Authorization" not in headers
        assert not any(k.lower() == "authorization" for k in headers)

    @pytest.mark.asyncio
    async def test_posts_to_the_backend_not_the_gateway(self, worker):
        capture = {}
        await self._run(worker, self._registry([self._backend(url="http://node9:8000")]), capture)
        assert capture["url"] == "http://node9:8000/v1/chat/completions"
        assert "localhost:8000" not in capture["url"]

    @pytest.mark.asyncio
    async def test_disables_thinking_on_vllm(self, worker):
        """Dispatching directly skips the gateway's thinking-off policy, so a
        reasoning model would wrap its JSON answer in <think> and parse to
        zero findings."""
        capture = {}
        await self._run(worker, self._registry([self._backend(engine="vllm")]), capture)
        payload = capture["kwargs"]["json"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["stream"] is False
        assert payload["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_omits_thinking_kwarg_for_ollama(self, worker):
        capture = {}
        await self._run(worker, self._registry([self._backend(engine="ollama")]), capture)
        assert "chat_template_kwargs" not in capture["kwargs"]["json"]

    @pytest.mark.asyncio
    async def test_raises_when_no_backend_serves_the_model(self, worker):
        with pytest.raises(RuntimeError):
            await self._run(worker, self._registry([]), {})

    @pytest.mark.asyncio
    async def test_raises_when_circuit_is_open(self, worker):
        with pytest.raises(RuntimeError):
            await self._run(worker, self._registry([self._backend()], available=False), {})


# ===================================================================
# Email notifications
# ===================================================================

class TestEmailContract:
    """DLP called email_service.send_email() — a function that has never
    existed — so every alert email died as AttributeError inside the
    catch-all.  This guards the whole class of mistake."""

    def test_every_email_service_call_resolves(self):
        exported = set()
        for node in ast.parse(EMAIL_SRC).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                exported.add(node.name)

        called = set()
        for node in ast.walk(ast.parse(WORKER_SRC)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "email_service"):
                called.add(node.func.attr)

        assert called, "expected dlp_worker to call email_service"
        missing = called - exported
        assert not missing, f"dlp_worker calls nonexistent email_service functions: {missing}"

    def test_send_notification_email_is_defined(self):
        assert "async def send_notification_email(" in EMAIL_SRC

    def test_email_service_uses_stdlib_log_formatting(self):
        """email_service binds a STDLIB logger; structlog-style kwargs there
        raise TypeError at call time (the 2.9.6 lesson)."""
        tree = ast.parse(EMAIL_SRC)
        allowed = {"exc_info", "extra", "stack_info", "stacklevel"}
        levels = {"debug", "info", "warning", "warn", "error", "critical", "exception"}
        offenders = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in levels
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"):
                bad = [k.arg for k in node.keywords if k.arg not in allowed]
                if bad:
                    offenders.append(f"line {node.lineno}: {bad}")
        assert not offenders, f"structlog kwargs on a stdlib logger: {offenders}"


class TestEmailFloodGuard:
    def test_allows_burst_then_suppresses(self, worker):
        worker._email_budget.clear()
        allowed = [worker._email_allowed("major")[0] for _ in range(worker.EMAIL_BURST + 5)]
        assert all(allowed[: worker.EMAIL_BURST])
        assert not any(allowed[worker.EMAIL_BURST:])

    def test_counts_suppressed_alerts(self, worker):
        worker._email_budget.clear()
        for _ in range(worker.EMAIL_BURST):
            worker._email_allowed("minor")
        worker._email_allowed("minor")
        allowed, suppressed = worker._email_allowed("minor")
        assert not allowed and suppressed == 2

    def test_severities_have_independent_budgets(self, worker):
        worker._email_budget.clear()
        for _ in range(worker.EMAIL_BURST):
            worker._email_allowed("minor")
        assert worker._email_allowed("major")[0] is True

    def test_suppressed_count_survives_the_window_rollover(self, worker, monkeypatch):
        """The dropped-alert tally is only useful if it reaches the next email.
        Zeroing it at rollover made the 'N suppressed' note unreachable."""
        clock = {"t": 1000.0}
        monkeypatch.setattr(worker.time, "monotonic", lambda: clock["t"])
        worker._email_budget.clear()

        for _ in range(worker.EMAIL_BURST):
            assert worker._email_allowed("major")[0] is True
        for _ in range(7):
            worker._email_allowed("major")  # denied, tallied

        clock["t"] += worker.EMAIL_WINDOW_SECONDS + 1
        allowed, suppressed = worker._email_allowed("major")
        assert allowed is True
        assert suppressed == 7, "the next send must report what was dropped"

        # ...and reporting it clears the tally so it isn't counted twice.
        assert worker._email_allowed("major") == (True, 0)


class TestAlertEmailBody:
    """The alert email is metadata only — an inbox is the last place a DLP
    finding should reproduce the value it flagged."""

    async def _send(self, worker, findings, categories, recipients="ops@example.edu"):
        captured = {}

        async def _get_config_json(db, key, default=None):
            return recipients if key.endswith("_recipients") else default

        async def _send_notification_email(config, recipients, subject, body_html, base_url=""):
            captured["subject"] = subject
            captured["body"] = body_html
            captured["recipients"] = recipients
            return len(recipients)

        async def _get_smtp_config(db):
            return {"host": "smtp", "default_sender": "a@b.c"}

        async def _get_base_url(db):
            return "https://mindrouter.example.edu"

        email_service = MagicMock(
            get_smtp_config=_get_smtp_config,
            is_smtp_configured=MagicMock(return_value=True),
            get_base_url=_get_base_url,
            send_notification_email=_send_notification_email,
        )

        saved = {k: sys.modules.get(k) for k in
                 ("backend.app.db", "backend.app.services", "backend.app.db.crud")}
        # Fresh stub modules only — never set an attribute on a PRE-EXISTING
        # sys.modules entry: the finally below restores the keys, but an
        # attribute planted on someone else's module object survives it.
        # (`sys.modules["backend.app.db"].crud = ...` here once rebound the
        # crud that voice_api resolves at call time and broke 20 voice tests.)
        crud_stub = MagicMock(get_config_json=_get_config_json)
        sys.modules["backend.app.db"] = MagicMock(crud=crud_stub)
        sys.modules["backend.app.db.crud"] = crud_stub
        sys.modules["backend.app.services"] = MagicMock(email_service=email_service)
        try:
            alert = MagicMock(request_id=42, categories=categories)
            scan_result = MagicMock(
                severity="major", scanner="regex", findings=findings,
                scan_latency_ms=12, detail="detail text",
            )
            worker._email_budget.clear()
            await worker._maybe_send_email(MagicMock(), alert, scan_result)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        return captured

    @pytest.mark.asyncio
    async def test_subject_names_categories_not_their_characters(self, worker):
        """`', '.join(findings[0].category)` joined the characters of the
        category string: 's, o, c, i, a, l, ...'."""
        finding = MagicMock(category="social security number")
        cap = await self._send(worker, [finding], ["social security number"])
        assert "social security number" in cap["subject"]
        assert "s, o, c" not in cap["subject"]

    @pytest.mark.asyncio
    async def test_body_excludes_matched_values(self, worker):
        finding = MagicMock(category="social security number", text="123-45-6789")
        cap = await self._send(worker, [finding], ["social security number"])
        assert "123-45-6789" not in cap["body"]

    @pytest.mark.asyncio
    async def test_body_links_to_the_admin_page(self, worker):
        cap = await self._send(worker, [MagicMock(category="email")], ["email"])
        assert "/admin/dlp" in cap["body"]

    @pytest.mark.asyncio
    async def test_no_recipients_sends_nothing(self, worker):
        cap = await self._send(worker, [MagicMock(category="email")], ["email"], recipients="")
        assert cap == {}


class TestStoredEntities:
    """Alert rows store masked snippets, capped in number."""

    def test_process_one_masks_before_storing(self):
        assert "mask_snippet" in WORKER_SRC
        assert "f.text[:50]" not in WORKER_SRC, "raw truncated match must not be stored"

    def test_entity_count_is_capped(self, worker):
        assert worker.MAX_STORED_ENTITIES > 0
        assert "MAX_STORED_ENTITIES" in WORKER_SRC


# ===================================================================
# Admin routes
# ===================================================================

@pytest.fixture(scope="module")
def routes():
    """Spec-load dlp_routes.py with its dashboard/db dependencies mocked so
    the real route functions execute."""
    saved = {k: sys.modules.get(k) for k in (
        "backend", "backend.app", "backend.app.db", "backend.app.db.crud",
        "backend.app.db.session", "backend.app.dashboard", "backend.app.dashboard.routes",
        "backend.app.logging_config", "backend.app.services",
        "backend.app.services.dlp_scanner",
    )}
    sys.modules.setdefault("backend", MagicMock())
    sys.modules.setdefault("backend.app", MagicMock())
    sys.modules.setdefault("backend.app.db", MagicMock())
    sys.modules.setdefault("backend.app.dashboard", MagicMock())
    # The route parses the regex rule list with the REAL scanner helpers
    # (built-in definitions, Luhn validator lookup), resolved via importlib.
    sys.modules.setdefault("backend.app.services", MagicMock())
    sys.modules["backend.app.services.dlp_scanner"] = _spec_load(
        "dlp_scanner_for_routes", _SERVICES_DIR / "dlp_scanner.py"
    )
    sys.modules["backend.app.db.crud"] = MagicMock()
    sys.modules["backend.app.db.session"] = MagicMock(get_async_db=MagicMock())
    sys.modules["backend.app.dashboard.routes"] = MagicMock(
        get_client_ip=MagicMock(return_value="127.0.0.1"),
        get_session_user_id=MagicMock(return_value=1),
        _admin_masquerade_context=MagicMock(),
        templates=MagicMock(),
    )
    sys.modules["backend.app.logging_config"] = MagicMock(
        get_logger=MagicMock(return_value=MagicMock())
    )
    try:
        yield _spec_load("dlp_routes_under_test", _DASHBOARD_DIR / "dlp_routes.py")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _Form(dict):
    """Minimal starlette FormData stand-in."""

    def __init__(self, data, lists=None):
        super().__init__(data)
        self._lists = lists or {}

    def getlist(self, key):
        return self._lists.get(key, [])


async def _save(routes, form_data, lists=None):
    """Invoke save_dlp_config with an admin session and capture the result."""
    written = {}

    async def _set_config(db, key, value):
        written[key] = value

    async def _log_admin_action(db, *a, **kw):
        return None

    async def _get_user_by_id(db, uid):
        user = MagicMock()
        user.id = uid
        user.group.is_admin = True
        user.group.has_admin_read = True
        return user

    crud = sys.modules["backend.app.db.crud"]
    crud.set_config = _set_config
    crud.log_admin_action = _log_admin_action
    crud.get_user_by_id = _get_user_by_id
    routes.crud = crud

    request = MagicMock()

    async def _form():
        return _Form(form_data, lists)

    request.form = _form

    db = MagicMock()

    async def _commit():
        return None

    async def _rollback():
        return None

    db.commit = _commit
    db.rollback = _rollback

    resp = await routes.save_dlp_config(request, db)
    return resp, written


_VALID = {
    # A real browser submit: the page script serialized the JSON fields and
    # flipped _json_ready. Tests that drop it exercise the script-didn't-run path.
    "_json_ready": "1",
    "enabled": "on",
    "regex_enabled": "on",
    "gliner_threshold": "0.5",
    "llm_model": "qwen/qwen3.5-4b",
    "llm_system_prompt": "Return a JSON array of findings.",
    "severity_rules": '{"email": "minor"}',
    # One textarea: pattern lines are "Name | category | regex", bare lines
    # are keywords.  Built-ins are ordinary lines the admin may edit/remove.
    "regex_rules": "Vandal ID | student id | V\\d{8}\nconfidential\nproprietary",
    "email_minor": "",
    "email_moderate": "",
    "email_major": "ops@example.edu",
}


class TestConfigValidation:
    """Every field is validated BEFORE the first write: a half-applied config
    is worse than a rejected one, and the old route reported success while
    silently discarding malformed fields."""

    @pytest.mark.asyncio
    async def test_valid_config_saves(self, routes):
        resp, written = await _save(routes, dict(_VALID))
        assert "success" in resp.headers["location"]
        assert written["dlp.enabled"] is True
        assert written["dlp.gliner.threshold"] == 0.5
        assert written["dlp.regex.keywords"] == ["confidential", "proprietary"]
        assert written["dlp.regex.patterns"][0]["name"] == "Vandal ID"

    @pytest.mark.asyncio
    async def test_invalid_regex_is_rejected_with_no_writes(self, routes):
        form = dict(_VALID, regex_rules="Bad | bad | ([unclosed")
        resp, written = await _save(routes, form)
        assert "error" in resp.headers["location"]
        assert "Line" in resp.headers["location"]
        assert written == {}, "nothing may be written when validation fails"

    @pytest.mark.asyncio
    async def test_pattern_line_with_missing_field_rejected(self, routes):
        """'Name | regex' is ambiguous — require all three fields so a keyword
        containing a stray pipe can't silently become a half-formed pattern."""
        resp, written = await _save(routes, dict(_VALID, regex_rules="Only Two | \\d+"))
        assert "error" in resp.headers["location"]
        assert written == {}

    @pytest.mark.asyncio
    async def test_regex_may_contain_pipes(self, routes):
        """The regex is the LAST field, so alternation survives the split."""
        resp, written = await _save(routes, dict(_VALID, regex_rules="Secret | secret | \\b(foo|bar|baz)\\b"))
        assert "success" in resp.headers["location"]
        assert written["dlp.regex.patterns"] == [
            {"name": "Secret", "pattern": "\\b(foo|bar|baz)\\b", "category": "secret"}
        ]

    @pytest.mark.asyncio
    async def test_saved_list_is_authoritative_and_flagged(self, routes):
        """After a save the stored list IS the scanner's pattern set — the
        worker must stop prepending the implicit built-ins."""
        resp, written = await _save(routes, dict(_VALID))
        assert "success" in resp.headers["location"]
        assert written["dlp.regex.builtins_in_list"] is True
        assert [p["name"] for p in written["dlp.regex.patterns"]] == ["Vandal ID"]
        assert written["dlp.regex.keywords"] == ["confidential", "proprietary"]

    @pytest.mark.asyncio
    async def test_builtin_name_keeps_its_validator(self, routes):
        """Editing the card regex must not silently drop the Luhn check."""
        rules = "Credit Card | credit card number | \\b\\d{16}\\b\nSSN | social security number | \\b\\d{3}-\\d{2}-\\d{4}\\b"
        resp, written = await _save(routes, dict(_VALID, regex_rules=rules))
        assert "success" in resp.headers["location"]
        by_name = {p["name"]: p for p in written["dlp.regex.patterns"]}
        assert by_name["Credit Card"]["validator"] == "luhn"
        assert "validator" not in by_name["SSN"]

    @pytest.mark.asyncio
    async def test_comment_and_blank_lines_skipped(self, routes):
        resp, written = await _save(routes, dict(_VALID, regex_rules="# built-ins removed on purpose\n\nFERPA\n"))
        assert "success" in resp.headers["location"]
        assert written["dlp.regex.patterns"] == []
        assert written["dlp.regex.keywords"] == ["FERPA"]

    # --- 2.9.48: global scan window -----------------------------------

    @pytest.mark.asyncio
    async def test_max_scan_chars_default_saved(self, routes):
        resp, written = await _save(routes, dict(_VALID))
        assert "success" in resp.headers["location"]
        assert written["dlp.max_scan_chars"] == 200000

    @pytest.mark.asyncio
    async def test_max_scan_chars_unlimited_checkbox_stores_zero(self, routes):
        resp, written = await _save(routes, dict(_VALID, max_scan_chars="200000", max_scan_chars_unlimited="on"))
        assert "success" in resp.headers["location"]
        assert written["dlp.max_scan_chars"] == 0

    @pytest.mark.asyncio
    async def test_max_scan_chars_explicit_zero_is_unlimited(self, routes):
        resp, written = await _save(routes, dict(_VALID, max_scan_chars="0"))
        assert "success" in resp.headers["location"]
        assert written["dlp.max_scan_chars"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["500", "-1", "abc"])
    async def test_max_scan_chars_rejects_tiny_negative_garbage(self, routes, bad):
        resp, written = await _save(routes, dict(_VALID, max_scan_chars=bad))
        assert "error" in resp.headers["location"]
        assert written == {}

    @pytest.mark.asyncio
    async def test_max_scan_chars_positive_saved(self, routes):
        resp, written = await _save(routes, dict(_VALID, max_scan_chars="500000"))
        assert "success" in resp.headers["location"]
        assert written["dlp.max_scan_chars"] == 500000

    def test_worker_reads_the_global_scan_window(self):
        assert '"dlp.max_scan_chars", MAX_SCAN_CHARS' in WORKER_SRC
        assert 'config["max_scan_chars"]' in WORKER_SRC

    def test_template_offers_no_limit(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert 'name="max_scan_chars"' in html
        assert 'name="max_scan_chars_unlimited"' in html
        assert "scan the entire document" in html

    @pytest.mark.asyncio
    async def test_ignore_is_a_valid_rule_level(self, routes):
        resp, written = await _save(routes, dict(_VALID, severity_rules='{"email": "ignore", "person": "minor"}'))
        assert "success" in resp.headers["location"]
        assert written["dlp.severity_rules"] == {"email": "ignore", "person": "minor"}

    @pytest.mark.asyncio
    async def test_severity_rules_must_be_a_mapping(self, routes):
        """A list used to save fine and then 500 the page on next render."""
        resp, written = await _save(routes, dict(_VALID, severity_rules="[]"))
        assert "error" in resp.headers["location"]
        assert written == {}

    @pytest.mark.asyncio
    async def test_unknown_severity_level_rejected(self, routes):
        resp, _ = await _save(routes, dict(_VALID, severity_rules='{"email": "critical"}'))
        assert "error" in resp.headers["location"]

    @pytest.mark.asyncio
    async def test_unparseable_json_rejected_not_silently_skipped(self, routes):
        resp, written = await _save(routes, dict(_VALID, severity_rules="{not json"))
        assert "error" in resp.headers["location"]
        assert written == {}

    @pytest.mark.parametrize("bad", ["nan", "1.5", "-1", "0.05", "abc"])
    @pytest.mark.asyncio
    async def test_threshold_range_enforced(self, routes, bad):
        resp, _ = await _save(routes, dict(_VALID, gliner_threshold=bad))
        assert "error" in resp.headers["location"], f"threshold {bad!r} should be rejected"

    @pytest.mark.asyncio
    async def test_llm_enabled_requires_a_model(self, routes):
        resp, _ = await _save(routes, dict(_VALID, llm_enabled="on", llm_model="  "))
        assert "error" in resp.headers["location"]

    @pytest.mark.asyncio
    async def test_invalid_email_rejected(self, routes):
        resp, _ = await _save(routes, dict(_VALID, email_major="not-an-address"))
        assert "error" in resp.headers["location"]

    @pytest.mark.asyncio
    async def test_newline_in_recipients_cannot_inject_headers(self, routes):
        resp, _ = await _save(routes, dict(_VALID, email_major="a@b.co\nBcc: evil@x.com"))
        assert "error" in resp.headers["location"]

    @pytest.mark.asyncio
    async def test_single_char_keyword_rejected(self, routes):
        """A 1-char keyword matches on nearly every request and buries real
        findings under thousands of entities."""
        resp, _ = await _save(routes, dict(_VALID, regex_rules="a"))
        assert "error" in resp.headers["location"]

    @pytest.mark.asyncio
    async def test_gliner_categories_can_be_cleared(self, routes):
        """`if categories:` made an empty list indistinguishable from 'not
        submitted', so unchecking every box silently kept the old list."""
        _, written = await _save(routes, dict(_VALID), lists={"gliner_categories": []})
        assert written["dlp.gliner.categories"] == []

    @pytest.mark.asyncio
    async def test_gliner_categories_normalized(self, routes):
        _, written = await _save(
            routes, dict(_VALID),
            lists={"gliner_categories": ["  Person ", "person", "Email"]},
        )
        assert written["dlp.gliner.categories"] == ["email", "person"]

    @pytest.mark.asyncio
    async def test_error_messages_are_urlencoded(self, routes):
        """Assert it IS an error redirect as well as being encoded: the weaker
        `" " not in loc` alone is satisfied by the success URL too, so the test
        passed pre-fix without ever observing an error."""
        resp, _ = await _save(routes, dict(_VALID, email_major="not an address"))
        loc = resp.headers["location"]
        assert loc.startswith("/admin/dlp?error="), loc
        assert " " not in loc, loc
        assert "+" in loc, "a multi-word message must have its spaces encoded"

    @pytest.mark.asyncio
    async def test_oversized_repeat_count_is_rejected_not_a_500(self):
        """re.compile raises OverflowError (not re.error) on \\d{9999999999};
        an uncaught type escapes the route as a bare 500."""
        import re as _re

        with pytest.raises((OverflowError, _re.error, RecursionError, ValueError)):
            _re.compile(r"\d{9999999999}")
        assert "(re.error, OverflowError, RecursionError, ValueError)" in ROUTES_SRC

    @pytest.mark.asyncio
    async def test_json_fields_preserved_when_page_script_did_not_run(self, routes):
        """The hidden severity_rules/regex_patterns inputs are populated by page
        JavaScript.  If that script does not run the browser posts the empty
        defaults — writing them would wipe the admin's rules while reporting
        success, which is exactly what a dead {% block %} caused."""
        form = dict(_VALID, severity_rules="{}")
        form.pop("_json_ready", None)
        resp, written = await _save(routes, form)
        assert "success" in resp.headers["location"]
        assert "dlp.severity_rules" not in written
        # The regex rule list is a plain textarea — no script involved — so it
        # is always authoritative and still saves.
        assert written["dlp.regex.patterns"][0]["name"] == "Vandal ID"
        # everything else still saves
        assert written["dlp.enabled"] is True

    @pytest.mark.asyncio
    async def test_json_fields_written_when_script_ran(self, routes):
        resp, written = await _save(routes, dict(_VALID, _json_ready="1"))
        assert "success" in resp.headers["location"]
        assert written["dlp.severity_rules"] == {"email": "minor"}
        assert written["dlp.regex.patterns"][0]["name"] == "Vandal ID"

    def test_no_exception_text_reflected_to_the_user(self):
        """str(e) on a DBAPIError stringifies bound parameters."""
        assert "error={str(e)" not in ROUTES_SRC
        assert "str(e)[:100]" not in ROUTES_SRC


class TestAlertFilters:
    """The filter bar returned zero rows on any use: FastAPI binds a
    present-but-empty query param to "", and the crud guard tested
    `is not None`, producing WHERE severity = ''."""

    def test_crud_uses_truthiness(self):
        fn = CRUD_SRC[CRUD_SRC.index("async def get_dlp_alerts"):]
        fn = fn[: fn.index("async def get_dlp_alert_by_id")]
        assert "if severity:" in fn
        assert "if scanner:" in fn
        assert "if severity is not None:" not in fn
        assert "if scanner is not None:" not in fn

    def test_route_normalizes_empty_strings(self):
        assert 'severity = (severity or "").strip() or None' in ROUTES_SRC
        assert 'scanner = (scanner or "").strip() or None' in ROUTES_SRC

    def test_route_whitelists_filter_values(self):
        assert "VALID_SEVERITIES" in ROUTES_SRC and "VALID_SCANNERS" in ROUTES_SRC

    def test_page_is_clamped(self):
        """page=0 produced a negative SQL OFFSET and a 500."""
        assert "page = max(1, page)" in ROUTES_SRC

    def test_template_offers_a_clear_control(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        # Clear now lives in the enhanced filter bar (id on the anchor).
        assert 'id="dlpClear"' in html and '>Clear</a>' in html


class TestTemplateBlockNames:
    """A child template's {% block %} name must exist in its parent, or Jinja
    silently drops the whole block.

    dlp.html used `{% block scripts %}` while base.html defines `extra_js`, so
    every line of JavaScript on /admin/dlp — acknowledge, the pattern/severity/
    category builders, and the hidden-field serializer — was dead. Silent, so
    nothing surfaced it until a save started writing the unserialized defaults.
    """

    def _blocks(self, text):
        return set(re.findall(r"{%-?\s*block\s+(\w+)", text))

    def test_every_child_block_exists_in_its_parent(self):
        tmpl_dir = _DASHBOARD_DIR / "templates"
        offenders = []
        for path in sorted(tmpl_dir.rglob("*.html")):
            text = path.read_text()
            m = re.search(r'{%-?\s*extends\s+["\']([^"\']+)["\']', text)
            if not m:
                continue
            parent = tmpl_dir / m.group(1)
            if not parent.exists():
                offenders.append(f"{path.name}: extends missing {m.group(1)}")
                continue
            parent_blocks = self._blocks(parent.read_text())
            for blk in self._blocks(text) - parent_blocks:
                offenders.append(
                    f"{path.relative_to(tmpl_dir)}: {{% block {blk} %}} is not "
                    f"defined by {m.group(1)} — it will be silently dropped"
                )
        assert not offenders, "\n".join(offenders)

    def test_dlp_page_uses_the_real_js_block(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert "{% block extra_js %}" in html
        assert "{% block scripts %}" not in html


class TestConfigWipeGuard:
    """Fail-closed: the server must not overwrite stored rules with the empty
    defaults the browser posts when the page script did not run."""

    def test_route_gates_json_writes_on_the_ready_flag(self):
        assert 'form.get("_json_ready") == "1"' in ROUTES_SRC
        assert "if json_fields_authoritative:" in ROUTES_SRC

    def test_template_declares_and_sets_the_flag(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert 'name="_json_ready" id="dlpJsonReady" value="0"' in html
        assert "getElementById('dlpJsonReady').value = '1'" in html


class TestPublicDocsMatchTheCode:
    """documentation.html is the product's own public page; the repo convention
    is that it stays in sync with docs/index.md."""

    def _doc(self):
        return (_DASHBOARD_DIR / "templates" / "public" / "documentation.html").read_text()

    def test_no_internal_api_key_claim(self):
        doc = self._doc()
        assert "automatically creates a dedicated internal API key" not in doc
        assert "Self-Routing Loop Prevention" not in doc

    def test_dlp_alerts_listed_as_purgeable(self):
        doc = self._doc()
        assert "and DLP alerts" in doc
        assert "plus DLP alerts and the email log, which have no retention category" not in doc


class TestTemplateFixes:
    def test_custom_category_button_targets_a_real_element(self):
        """The old selector matched nothing, so the button silently no-opped."""
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert 'id="glinerCategoriesRow"' in html
        assert "getElementById('glinerCategoriesRow')" in html
        assert "#dlpForm .row .col-md-6:first-child .row" not in html

    def test_hidden_json_inputs_have_parseable_defaults(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert 'id="severity_rules_json" value="{}"' in html
        # The regex list is a real textarea now, not a script-filled hidden input.
        assert 'regex_patterns_json' not in html
        assert 'name="regex_rules"' in html and 'rows="8"' in html

    def test_masked_snippets_are_surfaced(self):
        # The alerts table moved into the AJAX partial; assert there.
        html = (_DASHBOARD_DIR / "templates" / "admin" / "_dlp_alerts_table.html").read_text()
        assert "<th>Matched</th>" in html
        assert "alert.entities" in html

    def test_llm_help_text_no_longer_claims_an_api_key(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert "internal API key" not in html


# ===================================================================
# Retention
# ===================================================================

class TestDlpAlertRetention:
    """dlp_alerts was the only content-bearing table with no retention
    category, no purge path, and no delete route — retained forever by
    omission while holding snippets of flagged content."""

    def test_default_is_keep_forever(self):
        """A nonzero default would mass-delete existing alert history on the
        first cycle after deploy, with no admin action."""
        assert '"retention.dlp_alerts_days": 0,' in RETENTION_SRC

    def test_purge_category_registered(self):
        block = RETENTION_SRC[RETENTION_SRC.index("PURGE_CATEGORIES = ("):]
        block = block[: block.index(")")]
        assert '"dlp_alerts"' in block

    def test_purge_dispatch_wired(self):
        assert 'elif category == "dlp_alerts":' in RETENTION_SRC
        assert "cleanup_expired_dlp_alerts" in RETENTION_SRC

    def test_cleanup_runs_in_the_retention_cycle(self):
        assert 'dlp_days = config.get("retention.dlp_alerts_days", 0)' in RETENTION_SRC
        assert 'summary["dlp_alerts"] = await cleanup_expired_dlp_alerts' in RETENTION_SRC

    def test_cleanup_batches_and_commits(self):
        fn = RETENTION_SRC[RETENTION_SRC.index("async def cleanup_expired_dlp_alerts"):]
        fn = fn[: fn.index("async def delete_expired_requests_no_archive")]
        assert "LIMIT :batch" in fn
        assert "await app_db.commit()" in fn

    def test_scanned_at_index_exists_for_the_sweep(self):
        models = (_DB_DIR / "models.py").read_text()
        assert 'Index("ix_dlp_alerts_scanned_at", "scanned_at")' in models

    def test_admin_ui_exposes_the_new_category(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "retention.html").read_text()
        assert 'name="retention.dlp_alerts_days"' in html
        assert '<option value="dlp_alerts">' in html

    def test_retention_post_allowlist_includes_the_key(self):
        routes_src = (_DASHBOARD_DIR / "routes.py").read_text()
        assert '"retention.dlp_alerts_days",' in routes_src


# ===================================================================
# Alert de-duplication (admin-toggleable) + scanner fail-open surfacing
# ===================================================================

class TestAlertDedup:
    """The same masked value from the same user collapses to one alert within
    the window, so a client's classifier/title calls and conversation-history
    re-scans don't mint an alert+email each."""

    def test_first_then_repeat_then_expiry(self, worker):
        worker._dedup_seen.clear()
        key = (7, ("12*******89",))
        # First sighting is not a duplicate; an immediate repeat is.
        assert worker._dedup_is_duplicate(key, 300.0) is False
        assert worker._dedup_is_duplicate(key, 300.0) is True
        # A different value for the same user is independent.
        assert worker._dedup_is_duplicate((7, ("ab****ef",)), 300.0) is False
        # Once the window elapses, the value alerts again (age the entry).
        worker._dedup_seen[key] = worker.time.monotonic() - 301.0
        assert worker._dedup_is_duplicate(key, 300.0) is False

    def test_dedup_keys_bounded(self, worker):
        worker._dedup_seen.clear()
        # Never grows without bound even under a flood of unique values.
        for i in range(worker._DEDUP_MAX_KEYS + 500):
            worker._dedup_is_duplicate((1, (f"v{i}",)), 300.0)
        assert len(worker._dedup_seen) <= worker._DEDUP_MAX_KEYS + 1

    def test_worker_gates_dedup_on_config(self):
        # _process_one honors the admin toggle + window and suppresses by return.
        assert 'config.get("dedup.enabled"' in WORKER_SRC
        assert "_dedup_is_duplicate((req.user_id, masked)" in WORKER_SRC
        assert 'get_config_json(db, "dlp.dedup.enabled"' in WORKER_SRC
        assert 'get_config_json(\n        db, "dlp.dedup.window_seconds"' in WORKER_SRC \
            or 'dlp.dedup.window_seconds' in WORKER_SRC

    def test_route_persists_dedup_toggle(self):
        # Admin -> DLP panel writes both keys and validates the window.
        assert '"dlp.dedup.enabled", dedup_enabled' in ROUTES_SRC
        assert '"dlp.dedup.window_seconds", dedup_window' in ROUTES_SRC
        assert 'form.get("dedup_enabled") == "on"' in ROUTES_SRC
        assert "0 <= dedup_window <= 86400" in ROUTES_SRC


class TestScannerFailOpenSurfaced:
    """A scanner that errors must be VISIBLE, never silently treated as clean."""

    def test_worker_surfaces_scanner_errors(self):
        assert "scan_result.scanner_errors" in WORKER_SRC
        assert "_surface_scanner_errors" in WORKER_SRC
        # An error with no findings still surfaces, then returns.
        assert "if not scan_result.findings:" in WORKER_SRC

    def test_error_alert_is_rate_limited(self):
        assert "SCANNER_ERROR_ALERT_WINDOW" in WORKER_SRC
        assert '"dlp_scanner_error"' in WORKER_SRC

    def test_scanner_raises_instead_of_returning_empty(self):
        # dlp_scanner surfaces failures as DlpScannerError (fail-closed).
        assert "class DlpScannerError" in SCANNER_SRC
        assert "raise DlpScannerError" in SCANNER_SRC
        # run_dlp_scan no longer treats an errored scan as clean (None).
        assert "if not all_findings and not scanner_errors:" in SCANNER_SRC


# ===================================================================
# Digest report: per-severity delivery mode + scheduled roll-up
# ===================================================================

class TestDigest:
    """Each severity routes to immediate / digest / off; digest-mode alerts are
    rolled into a scheduled report (hourly..daily), central recipients."""

    def test_frequencies_cover_hourly_to_daily(self, worker):
        f = worker.DIGEST_FREQUENCIES
        assert f["hourly"] == 3600
        assert f["6h"] == 21600
        assert f["12h"] == 43200
        assert f["daily"] == 86400

    def test_parse_iso_roundtrip_and_tolerance(self, worker):
        from datetime import datetime, timezone
        # tz-aware ISO round-trips
        dt = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
        assert worker._parse_iso(dt.isoformat()) == dt
        # naive input is coerced to UTC (not left tz-naive)
        parsed = worker._parse_iso("2026-08-14T12:00:00")
        assert parsed.tzinfo is not None
        # None / garbage never raise
        assert worker._parse_iso(None) is None
        assert worker._parse_iso("not-a-date") is None

    def test_immediate_delivery_gated_on_mode(self):
        # _maybe_send_email only sends when the severity is set to "immediate".
        assert 'get_config_json(db, f"dlp.email.{severity}.mode", "immediate")' in WORKER_SRC
        assert 'if mode != "immediate":' in WORKER_SRC

    def test_digest_loop_is_wired_and_scheduled(self):
        assert "async def dlp_digest_loop" in WORKER_SRC
        assert "async def _maybe_send_digest" in WORKER_SRC
        # First run establishes the watermark without emailing a backfill.
        assert "First run establishes the watermark" in WORKER_SRC
        # Only digest-mode severities are rolled up.
        assert '"immediate") == "digest"' in WORKER_SRC

    # --- 2.9.47: one digest per window across N worker processes ---------

    def test_window_is_claimed_before_sending(self):
        """Four identical digests went out after the 2.9.46 restart: every
        uvicorn worker's loop woke on the same tick, read the same stale
        watermark, and each sent before any had written it.  The claim must be
        a compare-and-set that happens BEFORE the send."""
        body = WORKER_SRC[WORKER_SRC.index("async def _maybe_send_digest"):WORKER_SRC.index("async def _claim_digest_window")]
        assert body.index("_claim_digest_window(") < body.index("_send_digest_email(")
        claim = WORKER_SRC[WORKER_SRC.index("async def _claim_digest_window"):]
        assert "update(AppConfig)" in claim
        assert 'AppConfig.value == _json.dumps(previous_raw)' in claim, "CAS must compare the stored JSON encoding"
        assert "rowcount" in claim

    @pytest.mark.asyncio
    async def test_claim_returns_true_for_exactly_the_matching_update(self, worker):
        """The statement compares key AND the previously-read value; rowcount 1
        means this worker owns the window, 0 means another one already moved it."""
        import json as _json
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
        from sqlalchemy import String, Text

        class _Base(DeclarativeBase):
            pass

        class AppConfig(_Base):
            __tablename__ = "app_config"
            key: Mapped[str] = mapped_column(String(100), primary_key=True)
            value: Mapped[str] = mapped_column(Text)

        captured = {}

        class _Result:
            def __init__(self, rc):
                self.rowcount = rc

        def _db(rowcount):
            db = MagicMock()

            async def execute(stmt):
                captured["stmt"] = stmt
                return _Result(rowcount)

            async def commit():
                captured["committed"] = True

            db.execute = execute
            db.commit = commit
            return db

        with patch.dict(sys.modules, {"backend.app.db.models": MagicMock(AppConfig=AppConfig)}):
            assert await worker._claim_digest_window(_db(1), "2026-08-20T14:02:00+00:00", "2026-08-21T14:06:00+00:00") is True
            assert captured["committed"] is True
            sql = str(captured["stmt"].compile(compile_kwargs={"literal_binds": True}))
            assert "app_config.key = 'dlp.digest.last_sent_at'" in sql
            assert "app_config.value = '" + _json.dumps("2026-08-20T14:02:00+00:00").replace("'", "''") + "'" in sql
            assert _json.dumps("2026-08-21T14:06:00+00:00").replace("'", "''") in sql
            assert await worker._claim_digest_window(_db(0), "x", "y") is False
            assert await worker._claim_digest_window(_db(2), "x", "y") is False

    def test_loop_jitters_so_workers_do_not_tick_together(self):
        loop = WORKER_SRC[WORKER_SRC.index("async def dlp_digest_loop"):WORKER_SRC.index("async def _maybe_send_digest")]
        assert "random.uniform(0, DIGEST_CHECK_JITTER)" in loop
        assert "DIGEST_CHECK_JITTER = 60.0" in WORKER_SRC

    def test_send_failure_restores_the_previous_watermark(self):
        """The claim already advanced the watermark, so a failed send must put
        the OLD value back (not merely skip the write) or the window is lost."""
        body = WORKER_SRC[WORKER_SRC.index("async def _maybe_send_digest"):WORKER_SRC.index("async def _claim_digest_window")]
        fail = body[body.index("dlp_digest_send_failed"):]
        assert 'set_config(db, "dlp.digest.last_sent_at", last_sent_raw)' in fail

    # --- 2.9.47: digest times in the configured display timezone ---------

    def test_digest_renders_in_configured_timezone_with_abbreviation(self, worker):
        from datetime import datetime, timezone
        tz = worker.zoneinfo.ZoneInfo("America/Los_Angeles")
        # naive DB value == UTC, never local wall clock
        naive = datetime(2026, 8, 20, 17, 19)
        assert worker._fmt_local(naive, tz) == "2026-08-20 10:19"
        aware = datetime(2026, 1, 15, 17, 19, tzinfo=timezone.utc)
        assert worker._fmt_local(aware, tz) == "2026-01-15 09:19"
        assert worker._tz_label(tz, datetime(2026, 8, 21, 14, 6, tzinfo=timezone.utc)) == "PDT"
        assert worker._tz_label(tz, datetime(2026, 1, 21, 14, 6, tzinfo=timezone.utc)) == "PST"
        assert worker._fmt_local(None, tz) == ""

    @pytest.mark.asyncio
    async def test_display_tz_reads_app_timezone_and_falls_back(self, worker):
        crud = MagicMock()

        async def cfg(db, key, default=None):
            return {"app.timezone": "America/New_York"}.get(key, default)

        crud.get_config_json = cfg
        # the worker does `from backend.app.db import crud` -> patch the parent
        with patch.dict(sys.modules, {"backend.app.db": MagicMock(crud=crud)}):
            assert (await worker._display_tz(None)).key == "America/New_York"

        async def bad(db, key, default=None):
            return "Mars/Olympus_Mons"

        crud.get_config_json = bad
        with patch.dict(sys.modules, {"backend.app.db": MagicMock(crud=crud)}):
            assert (await worker._display_tz(None)).key == "America/Los_Angeles"

    def test_digest_email_no_longer_labels_utc(self):
        body = WORKER_SRC[WORKER_SRC.index("async def _send_digest_email"):]
        assert "Time (UTC)" not in body
        assert "Time ({_esc(tz_label)})" in body
        assert "_fmt_local(since, tz)" in body and "_fmt_local(a.scanned_at, tz)" in body

    def test_digest_watermark_not_advanced_on_failure(self):
        # A send failure must not drop alerts: keep the window for the retry.
        assert "Do NOT keep the advanced watermark on send failure" in WORKER_SRC

    def test_digest_email_never_includes_matched_values(self):
        assert "Matched values are masked" in WORKER_SRC

    def test_main_starts_the_digest_loop(self):
        main_src = (_APP_DIR / "main.py").read_text()
        assert "dlp_digest_loop" in main_src
        assert "_dlp_digest_task = asyncio.create_task(dlp_digest_loop())" in main_src
        assert "_dlp_digest_task.cancel()" in main_src  # cancelled on shutdown

    def test_route_persists_modes_and_digest(self):
        assert '"dlp.email.minor.mode", email_modes["minor"]' in ROUTES_SRC
        assert '"dlp.email.major.mode", email_modes["major"]' in ROUTES_SRC
        assert '"dlp.digest.frequency", digest_frequency' in ROUTES_SRC
        assert '"dlp.digest.recipients", digest_recipients' in ROUTES_SRC
        assert 'if digest_frequency not in ("hourly", "6h", "12h", "daily")' in ROUTES_SRC
        # A digest-routed severity without recipients is rejected up front.
        assert '"digest" in email_modes.values() and not digest_recipients' in ROUTES_SRC

    def test_template_has_mode_selectors_and_digest_card(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert 'name="email_{{ row.key }}_mode"' in html
        assert 'name="digest_frequency"' in html
        assert 'name="digest_recipients"' in html


class TestGlinerScanCapWiring:
    """The GLiNER scan-length cap is admin-configurable end to end."""

    def test_worker_loads_and_passes_the_cap(self):
        assert 'get_config_json(\n        db, "dlp.gliner.max_scan_chars"' in WORKER_SRC \
            or 'dlp.gliner.max_scan_chars' in WORKER_SRC
        assert 'GLINER_DEFAULT_MAX_CHARS' in WORKER_SRC
        # run_dlp_scan hoists the cap into a local (shared by the on-host and
        # off-host GLiNER branches) and passes it through as max_chars.
        assert 'config.get("gliner.max_scan_chars")' in SCANNER_SRC
        assert 'max_chars=gliner_max_chars' in SCANNER_SRC

    def test_route_validates_and_persists_the_cap(self):
        assert '"dlp.gliner.max_scan_chars", gliner_max_chars' in ROUTES_SRC
        assert "500 <= gliner_max_chars <= 200000" in ROUTES_SRC

    def test_template_exposes_the_field(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert 'name="gliner_max_scan_chars"' in html


class TestWorkerConcurrency:
    """Supervisor + overflow-counter behavior added for burst support."""

    def test_enqueue_overflow_increments_dropped_counter(self, worker):
        async def run():
            # fill the queue to capacity, then overflow twice
            q = worker.get_dlp_queue()
            drained = 0
            while not q.empty():
                q.get_nowait(); drained += 1
            base = worker.get_queue_dropped_total()
            for i in range(q.maxsize):
                q.put_nowait(i)
            await worker.enqueue_for_dlp(999001)
            await worker.enqueue_for_dlp(999002)
            assert worker.get_queue_dropped_total() == base + 2
            while not q.empty():
                q.get_nowait()
        asyncio.run(run())

    def test_read_worker_concurrency_clamps_and_defaults(self, worker, monkeypatch):
        async def run():
            # DB unavailable -> default
            assert await worker._read_worker_concurrency() == worker.DLP_WORKER_DEFAULT_CONCURRENCY
        asyncio.run(run())

    def test_supervisor_spawns_and_resizes_consumers(self, worker, monkeypatch):
        async def run():
            targets = iter([3, 1, 1])

            async def fake_read(fallback=None):
                return next(targets)

            spawned = []

            async def fake_consume(idx):
                spawned.append(idx)
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    raise

            sleeps = {"n": 0}
            real_sleep = asyncio.sleep

            async def fast_sleep(_secs):
                sleeps["n"] += 1
                await real_sleep(0)   # let spawned consumer tasks start
                if sleeps["n"] >= 2:
                    raise asyncio.CancelledError  # simulate shutdown

            monkeypatch.setattr(worker, "_read_worker_concurrency", fake_read)
            monkeypatch.setattr(worker, "_consume_loop", fake_consume)
            monkeypatch.setattr(worker.asyncio, "sleep", fast_sleep, raising=False)

            await worker.dlp_worker_loop()   # swallows the CancelledError, cancels children
            assert spawned == [0, 1, 2]      # grew to 3, then shrank to 1 (no respawn)
        asyncio.run(run())


# ===================================================================
# 2.9.52 — DLP alert export (zip: json/jsonl/csv) + AJAX filter partial
# ===================================================================

class TestDlpAlertExport:
    """Filtered, zipped alert export for admins/auditors, and the pure
    render/parse helpers behind it. dlp_routes is spec-loaded via `routes`."""

    def _sample_alert(self):
        from datetime import datetime, timezone
        a = MagicMock()
        a.id = 42
        a.scanned_at = datetime(2026, 8, 20, 17, 19, tzinfo=timezone.utc)
        a.severity = "major"
        a.scanner = "gliner"
        a.categories = ["social security number", "person"]
        a.entities = [{"scanner": "regex", "category": "social security number", "text": "89*****88", "confidence": 1.0}]
        a.confidence = 1.0
        a.scan_latency_ms = 12
        a.detail = "5 finding(s)"
        a.acknowledged = False
        a.acknowledged_at = None
        a.user_id = 2
        a.user = MagicMock(email="u@x.edu")
        a.request_id = 999
        a.request = MagicMock(request_uuid="abc-uuid")
        return a

    def test_parse_export_date_handles_bounds(self, routes):
        from datetime import timezone
        assert routes._parse_export_date("") is None
        assert routes._parse_export_date("not-a-date") is None
        start = routes._parse_export_date("2026-08-20")
        assert start.year == 2026 and start.month == 8 and start.day == 20 and start.hour == 0
        assert start.tzinfo is not None
        # a date-only END bound rolls to the next midnight (inclusive whole day)
        end = routes._parse_export_date("2026-08-20", end=True)
        assert end.day == 21 and end.hour == 0
        # an explicit datetime end is used as-is
        end2 = routes._parse_export_date("2026-08-20T12:00", end=True)
        assert end2.day == 20 and end2.hour == 12

    def test_alert_to_record_is_json_safe_and_masked(self, routes):
        rec = routes._alert_to_record(self._sample_alert())
        assert rec["id"] == 42
        assert rec["scanned_at"] == "2026-08-20T17:19:00+00:00"
        assert rec["user_email"] == "u@x.edu"
        assert rec["request_uuid"] == "abc-uuid"
        assert rec["categories"] == ["social security number", "person"]
        # only masked snippet text is present
        import json as _json
        assert "89*****88" in _json.dumps(rec) and "89-" not in _json.dumps(rec)

    def test_render_json_jsonl_csv(self, routes):
        import json as _json
        recs = [routes._alert_to_record(self._sample_alert()),
                routes._alert_to_record(self._sample_alert())]
        # JSON: a single parseable array
        arr = _json.loads(routes._render_export(recs, "json").decode())
        assert isinstance(arr, list) and len(arr) == 2 and arr[0]["id"] == 42
        # JSONL: one object per line
        lines = routes._render_export(recs, "jsonl").decode().splitlines()
        assert len(lines) == 2 and all(_json.loads(l)["severity"] == "major" for l in lines)
        # CSV: header + N rows, nested cols JSON-encoded into one field
        import csv, io
        rows = list(csv.DictReader(io.StringIO(routes._render_export(recs, "csv").decode())))
        assert len(rows) == 2
        assert rows[0]["severity"] == "major"
        assert _json.loads(rows[0]["categories"]) == ["social security number", "person"]

    def test_render_empty_is_valid(self, routes):
        import json as _json
        assert _json.loads(routes._render_export([], "json").decode()) == []
        assert routes._render_export([], "jsonl").decode() == ""
        assert routes._render_export([], "csv").decode().strip().startswith("id,")

    def test_export_constants(self, routes):
        assert routes.EXPORT_FORMATS == ("json", "jsonl", "csv")
        assert routes.EXPORT_MAX_ROWS >= 1000

    def test_export_route_admin_read_gated_and_zips(self):
        # source-level guarantees on the export route
        assert '@dlp_router.get("/admin/dlp/alerts/export")' in ROUTES_SRC
        body = ROUTES_SRC[ROUTES_SRC.index("async def export_dlp_alerts"):ROUTES_SRC.index("async def", ROUTES_SRC.index("async def export_dlp_alerts") + 1)]
        assert "_require_admin_read(request, db)" in body
        assert "zipfile.ZipFile" in body
        assert 'z.writestr("manifest.json"' in body
        assert "application/zip" in body
        assert "start_at=start_at, end_at=end_at" in body

    def test_partial_route_shares_auth_and_template(self):
        assert '@dlp_router.get("/admin/dlp/alerts/partial"' in ROUTES_SRC
        body = ROUTES_SRC[ROUTES_SRC.index("async def dlp_alerts_partial"):]
        body = body[:body.index("@dlp_router.post")]
        assert "_require_admin_read(request, db)" in body
        assert '"admin/_dlp_alerts_table.html"' in body
        assert "start_at=start_at, end_at=end_at" in body


class TestDlpAlertsUX:
    def test_partial_template_exists_and_self_contained(self):
        partial = _DASHBOARD_DIR / "templates" / "admin" / "_dlp_alerts_table.html"
        assert partial.exists()
        html = partial.read_text()
        assert "data-dlp-total" in html          # feeds the header count on swap
        assert "page-link" in html                # pagination lives in the partial
        assert "{% for alert in alerts %}" in html

    def test_page_swaps_container_without_full_reload(self):
        html = (_DASHBOARD_DIR / "templates" / "admin" / "dlp.html").read_text()
        assert 'id="dlpAlertsContainer"' in html
        assert '{% include "admin/_dlp_alerts_table.html" %}' in html
        # the old jump-to-top behavior is gone
        assert 'onchange="this.form.submit()"' not in html
        # AJAX + history without reload, plus export wiring
        assert "history.replaceState" in html
        assert "/admin/dlp/alerts/partial" in html
        assert "/admin/dlp/alerts/export" in html
        assert 'id="dlpExportFmt"' in html and 'id="dlpExportBtn"' in html
        # date-range controls present
        assert 'name="start_date"' in html and 'name="end_date"' in html
