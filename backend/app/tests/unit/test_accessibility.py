"""
Accessibility tests for MindRouter2 HTML templates.

Validates WCAG 2.1 Level A and AA compliance by parsing Jinja2 templates
and checking for required ARIA attributes, semantic HTML, heading hierarchy,
form labels, and other accessibility requirements.
"""

import os
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "dashboard" / "templates"


def _read_template(relative_path: str) -> str:
    """Read a template file and return its contents."""
    path = TEMPLATE_DIR / relative_path
    assert path.exists(), f"Template not found: {path}"
    return path.read_text(encoding="utf-8")


class _TagCollector(HTMLParser):
    """Lightweight HTML parser that collects elements and their attributes."""

    def __init__(self):
        super().__init__()
        self.tags: list[dict] = []  # [{tag, attrs_dict, line}]
        self._current_line = 0

    def handle_starttag(self, tag, attrs):
        self.tags.append({
            "tag": tag,
            "attrs": dict(attrs),
            "line": self.getpos()[0],
        })

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def _parse_tags(html: str) -> list[dict]:
    """Parse HTML (may contain Jinja2 tags) and collect elements."""
    # Strip Jinja2 blocks/expressions so the HTML parser doesn't choke.
    cleaned = re.sub(r"\{%.*?%\}", "", html, flags=re.DOTALL)
    cleaned = re.sub(r"\{\{.*?\}\}", "", cleaned, flags=re.DOTALL)
    collector = _TagCollector()
    collector.feed(cleaned)
    return collector.tags


def _find_tags(tags: list[dict], tag_name: str) -> list[dict]:
    return [t for t in tags if t["tag"] == tag_name]


def _find_by_attr(tags: list[dict], attr: str, value: str) -> list[dict]:
    return [t for t in tags if t["attrs"].get(attr) == value]


def _find_by_class(tags: list[dict], cls: str) -> list[dict]:
    return [
        t for t in tags
        if cls in (t["attrs"].get("class", "").split())
    ]


def _classes(tag: dict) -> set[str]:
    return set(tag["attrs"].get("class", "").split())


# ── Fixtures ─────────────────────────────────────────────────────

ALL_TEMPLATES = sorted(
    str(p.relative_to(TEMPLATE_DIR))
    for p in TEMPLATE_DIR.rglob("*.html")
)


@pytest.fixture(params=ALL_TEMPLATES, ids=ALL_TEMPLATES)
def template_content(request):
    """Parameterized fixture: yields (relative_path, content) for every template."""
    return request.param, _read_template(request.param)


# ── 1. Base Template Tests ────────────────────────────────────────

class TestBaseTemplate:
    """Tests for base.html shared structure."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.html = _read_template("base.html")
        self.tags = _parse_tags(self.html)

    def test_html_lang_attribute(self):
        html_tags = _find_tags(self.tags, "html")
        assert html_tags, "No <html> tag found"
        assert html_tags[0]["attrs"].get("lang") == "en"

    def test_skip_navigation_link(self):
        assert "skip-link" in self.html, "Missing skip navigation link"
        assert 'href="#main-content"' in self.html, (
            "Skip link should target #main-content"
        )

    def test_main_content_id(self):
        main_tags = _find_tags(self.tags, "main")
        assert main_tags, "No <main> tag found"
        assert main_tags[0]["attrs"].get("id") == "main-content"

    def test_sr_only_class_defined(self):
        assert ".sr-only" in self.html, "Missing .sr-only CSS class definition"

    def test_navbar_aria_label(self):
        nav_tags = _find_tags(self.tags, "nav")
        main_nav = [n for n in nav_tags if "navbar" in _classes(n)]
        assert main_nav, "No navbar <nav> found"
        assert main_nav[0]["attrs"].get("aria-label"), (
            "Main <nav> should have aria-label"
        )

    def test_navbar_toggler_attributes(self):
        buttons = _find_by_class(self.tags, "navbar-toggler")
        assert buttons, "No navbar-toggler button found"
        btn = buttons[0]
        assert btn["attrs"].get("aria-controls"), "navbar-toggler missing aria-controls"
        assert btn["attrs"].get("aria-expanded") is not None, (
            "navbar-toggler missing aria-expanded"
        )
        assert btn["attrs"].get("aria-label"), "navbar-toggler missing aria-label"

    def test_main_landmark_present(self):
        main_tags = _find_tags(self.tags, "main")
        assert len(main_tags) >= 1, "Missing <main> landmark"

    def test_footer_present(self):
        footer_tags = _find_tags(self.tags, "footer")
        assert footer_tags, "Missing <footer> landmark"


# ── 2. Public Status Page Tests ───────────────────────────────────

class TestPublicIndex:
    """Tests for public/index.html."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.html = _read_template("public/index.html")
        self.tags = _parse_tags(self.html)

    def test_canvas_has_role_and_label(self):
        canvases = _find_by_attr(self.tags, "id", "tokenFlowCanvas")
        assert canvases, "Canvas element not found"
        c = canvases[0]
        assert c["attrs"].get("role") == "img", "Canvas missing role='img'"
        assert c["attrs"].get("aria-label"), "Canvas missing aria-label"

    def test_flow_stats_live_region(self):
        stats = _find_by_attr(self.tags, "id", "flowStats")
        assert stats, "flowStats element not found"
        assert stats[0]["attrs"].get("aria-live"), (
            "flowStats should have aria-live"
        )

    def test_sr_only_stats_region(self):
        sr = _find_by_attr(self.tags, "id", "flowStatsSR")
        assert sr, "Screen reader stat summary element not found"
        assert sr[0]["attrs"].get("aria-live"), (
            "SR summary should have aria-live"
        )
        assert sr[0]["attrs"].get("role") == "status"

    def test_idle_label_has_role(self):
        idle = _find_by_attr(self.tags, "id", "idleLabel")
        assert idle, "idleLabel element not found"
        assert idle[0]["attrs"].get("role") == "status"

    def test_stat_cards_use_proper_heading_hierarchy(self):
        """Stat card subtitles should be h2 (not h6) under the page h1."""
        h6_subtitles = [
            t for t in _find_tags(self.tags, "h6")
            if "card-subtitle" in _classes(t)
        ]
        assert not h6_subtitles, (
            f"Found {len(h6_subtitles)} card-subtitle <h6> elements; "
            "should be <h2 class='h6'> for proper heading hierarchy"
        )


# ── 3. Chat Page Tests ───────────────────────────────────────────

class TestChat:
    """Tests for chat.html."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.html = _read_template("chat.html")
        self.tags = _parse_tags(self.html)

    def test_model_select_has_label(self):
        selects = _find_by_attr(self.tags, "id", "modelSelect")
        assert selects, "modelSelect not found"
        s = selects[0]
        has_aria = s["attrs"].get("aria-label")
        # Also check for a <label for="modelSelect">
        labels = [
            t for t in _find_tags(self.tags, "label")
            if t["attrs"].get("for") == "modelSelect"
        ]
        assert has_aria or labels, (
            "modelSelect needs aria-label or an associated <label>"
        )

    def test_chat_input_has_label(self):
        inputs = _find_by_attr(self.tags, "id", "chatInput")
        assert inputs, "chatInput not found"
        inp = inputs[0]
        has_aria = inp["attrs"].get("aria-label")
        labels = [
            t for t in _find_tags(self.tags, "label")
            if t["attrs"].get("for") == "chatInput"
        ]
        assert has_aria or labels, (
            "chatInput needs aria-label or an associated <label>"
        )

    def test_send_button_accessible_name(self):
        btn = _find_by_attr(self.tags, "id", "sendBtn")
        assert btn, "sendBtn not found"
        assert btn[0]["attrs"].get("aria-label"), "sendBtn missing aria-label"

    def test_stop_button_accessible_name(self):
        btn = _find_by_attr(self.tags, "id", "stopBtn")
        assert btn, "stopBtn not found"
        assert btn[0]["attrs"].get("aria-label"), "stopBtn missing aria-label"

    def test_attach_button_accessible_name(self):
        btn = _find_by_attr(self.tags, "id", "attachBtn")
        assert btn, "attachBtn not found"
        assert btn[0]["attrs"].get("aria-label"), "attachBtn missing aria-label"

    def test_toggle_sidebar_accessible_name(self):
        btn = _find_by_attr(self.tags, "id", "toggleSidebar")
        assert btn, "toggleSidebar not found"
        assert btn[0]["attrs"].get("aria-label"), (
            "toggleSidebar missing aria-label"
        )
        assert btn[0]["attrs"].get("aria-expanded") is not None, (
            "toggleSidebar missing aria-expanded"
        )

    def test_image_modal_dialog_semantics(self):
        modal = _find_by_attr(self.tags, "id", "imageModal")
        assert modal, "imageModal not found"
        assert modal[0]["attrs"].get("role") == "dialog", (
            "imageModal missing role='dialog'"
        )
        assert modal[0]["attrs"].get("aria-modal") == "true", (
            "imageModal missing aria-modal='true'"
        )
        assert modal[0]["attrs"].get("aria-label"), (
            "imageModal missing aria-label"
        )

    def test_image_modal_close_button(self):
        close_btn = _find_by_attr(self.tags, "id", "imageModalClose")
        assert close_btn, "imageModalClose not found"
        assert close_btn[0]["attrs"].get("aria-label"), (
            "imageModalClose missing aria-label"
        )

    def test_chat_messages_live_region(self):
        msgs = _find_by_attr(self.tags, "id", "chatMessages")
        assert msgs, "chatMessages not found"
        m = msgs[0]
        assert m["attrs"].get("aria-live"), "chatMessages missing aria-live"
        assert m["attrs"].get("role") == "log", "chatMessages should have role='log'"

    def test_conversation_list_accessible(self):
        conv_list = _find_by_attr(self.tags, "id", "conversationList")
        assert conv_list, "conversationList not found"
        assert conv_list[0]["attrs"].get("role") == "list"

    def test_sidebar_is_nav(self):
        sidebar = _find_by_attr(self.tags, "id", "chatSidebar")
        assert sidebar, "chatSidebar not found"
        assert sidebar[0]["tag"] == "nav", "chatSidebar should be a <nav>"
        assert sidebar[0]["attrs"].get("aria-label"), (
            "chatSidebar nav missing aria-label"
        )

    def test_multimodal_warning_modal_labeled(self):
        modal = _find_by_attr(self.tags, "id", "multimodalWarningModal")
        assert modal, "multimodalWarningModal not found"
        assert modal[0]["attrs"].get("aria-labelledby") == "multimodalWarningLabel"


# ── 4. Login Page Tests ──────────────────────────────────────────

class TestLogin:
    """Tests for public/login.html."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.html = _read_template("public/login.html")
        self.tags = _parse_tags(self.html)

    def test_has_page_heading(self):
        h1s = _find_tags(self.tags, "h1")
        assert h1s, "Login page should have an h1"

    def test_username_has_label(self):
        labels = [
            t for t in _find_tags(self.tags, "label")
            if t["attrs"].get("for") == "username"
        ]
        assert labels, "Username input missing label"

    def test_password_has_label(self):
        labels = [
            t for t in _find_tags(self.tags, "label")
            if t["attrs"].get("for") == "password"
        ]
        assert labels, "Password input missing label"

    def test_username_autocomplete(self):
        inputs = _find_by_attr(self.tags, "id", "username")
        assert inputs, "Username input not found"
        assert inputs[0]["attrs"].get("autocomplete") == "username"

    def test_password_autocomplete(self):
        inputs = _find_by_attr(self.tags, "id", "password")
        assert inputs, "Password input not found"
        assert inputs[0]["attrs"].get("autocomplete") == "current-password"

    def test_error_alert_role(self):
        """Error alert should have role='alert'."""
        assert 'role="alert"' in self.html


# ── 6. Admin Template Tests ──────────────────────────────────────

ADMIN_TEMPLATES = [
    "admin/dashboard.html",
    "admin/backends.html",
    "admin/users.html",
    "admin/user_detail.html",
    "admin/groups.html",
    "admin/api_keys.html",
    "admin/requests.html",
    "admin/audit.html",
    "admin/nodes.html",
    "admin/metrics.html",
]


class TestAdminSidebarNav:
    """Tests for admin sidebar navigation (shared include)."""

    def test_sidebar_include_has_aria_label(self):
        """The shared _sidebar.html include has aria-label on the nav list."""
        html = _read_template("admin/_sidebar.html")
        tags = _parse_tags(html)
        nav_lists = [
            t for t in _find_tags(tags, "ul")
            if "nav" in _classes(t) and "flex-column" in _classes(t)
        ]
        assert nav_lists, "No sidebar nav <ul> found in admin/_sidebar.html"
        assert nav_lists[0]["attrs"].get("aria-label"), (
            "Sidebar nav missing aria-label in admin/_sidebar.html"
        )

    def test_sidebar_links_have_conditional_aria_current(self):
        """Each sidebar link has conditional aria-current='page'."""
        html = _read_template("admin/_sidebar.html")
        # Check that aria-current="page" appears in Jinja conditionals
        assert 'aria-current="page"' in html, (
            "Sidebar links missing aria-current='page' in admin/_sidebar.html"
        )

    @pytest.mark.parametrize("template_path", ADMIN_TEMPLATES)
    def test_admin_templates_include_sidebar(self, template_path):
        """Each admin template includes the shared sidebar."""
        html = _read_template(template_path)
        assert 'include "admin/_sidebar.html"' in html, (
            f"{template_path} does not include admin/_sidebar.html"
        )


# ── 7. Table Accessibility Tests ─────────────────────────────────

TEMPLATES_WITH_TABLES = [
    "admin/dashboard.html",
    "admin/users.html",
    "admin/user_detail.html",
    "admin/api_keys.html",
    "admin/audit.html",
    "user/dashboard.html",
]


@pytest.mark.parametrize("template_path", TEMPLATES_WITH_TABLES)
class TestTableAccessibility:
    """Tests for data table accessibility."""

    def test_th_elements_have_scope(self, template_path):
        html = _read_template(template_path)
        tags = _parse_tags(html)
        ths = _find_tags(tags, "th")
        for th in ths:
            assert th["attrs"].get("scope"), (
                f"<th> at line {th['line']} missing scope attribute "
                f"in {template_path}"
            )

    def test_tables_have_caption(self, template_path):
        html = _read_template(template_path)
        tags = _parse_tags(html)
        captions = _find_tags(tags, "caption")
        assert captions, (
            f"No <caption> found in {template_path}; "
            "tables should have a descriptive caption"
        )


# ── 8. Duplicate ID Tests ────────────────────────────────────────

@pytest.mark.parametrize("template_path", ALL_TEMPLATES)
def test_no_duplicate_ids(template_path):
    """Every template should have unique element IDs."""
    html = _read_template(template_path)
    relative = template_path
    tags = _parse_tags(html)
    ids = [t["attrs"]["id"] for t in tags if "id" in t["attrs"] and t["attrs"]["id"]]
    seen = set()
    dupes = set()
    for eid in ids:
        if eid in seen:
            dupes.add(eid)
        seen.add(eid)
    assert not dupes, (
        f"Duplicate IDs found in {relative}: {dupes}"
    )


# ── 9. User Dashboard Tests ──────────────────────────────────────

class TestUserDashboard:
    """Tests for user/dashboard.html."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.html = _read_template("user/dashboard.html")
        self.tags = _parse_tags(self.html)

    def test_progress_bar_aria_attributes(self):
        bars = _find_by_class(self.tags, "progress-bar")
        for bar in bars:
            attrs = bar["attrs"]
            assert "aria-valuenow" in attrs or "aria-valuenow" in self.html, (
                "Progress bar missing aria-valuenow"
            )
            assert "aria-valuemin" in attrs or "aria-valuemin" in self.html, (
                "Progress bar missing aria-valuemin"
            )
            assert "aria-valuemax" in attrs or "aria-valuemax" in self.html, (
                "Progress bar missing aria-valuemax"
            )
            assert "aria-label" in attrs or "aria-label" in self.html, (
                "Progress bar missing aria-label"
            )

    def test_create_key_modal_labeled(self):
        modal = _find_by_attr(self.tags, "id", "createKeyModal")
        assert modal, "createKeyModal not found"
        assert modal[0]["attrs"].get("aria-labelledby"), (
            "createKeyModal missing aria-labelledby"
        )

    def test_api_keys_table_has_caption(self):
        captions = _find_tags(self.tags, "caption")
        assert captions, "API keys table missing <caption>"


# ── 10. Metrics Page Tests ────────────────────────────────────────

class TestMetrics:
    """Tests for admin/metrics.html chart accessibility."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.html = _read_template("admin/metrics.html")
        self.tags = _parse_tags(self.html)

    @pytest.mark.parametrize("canvas_id", [
        "chart-gpu-util",
        "chart-gpu-mem",
        "chart-power",
        "chart-requests",
    ])
    def test_chart_canvas_accessible(self, canvas_id):
        canvases = _find_by_attr(self.tags, "id", canvas_id)
        assert canvases, f"Canvas #{canvas_id} not found"
        c = canvases[0]
        assert c["attrs"].get("role") == "img", (
            f"Canvas #{canvas_id} missing role='img'"
        )
        assert c["attrs"].get("aria-label"), (
            f"Canvas #{canvas_id} missing aria-label"
        )

    def test_time_range_button_group_labeled(self):
        groups = [
            t for t in _find_tags(self.tags, "div")
            if "btn-group" in _classes(t) and t["attrs"].get("role") == "group"
        ]
        for group in groups:
            assert group["attrs"].get("aria-label"), (
                f"Button group at line {group['line']} missing aria-label"
            )


# ── 11. Cross-Template Heading Hierarchy Tests ────────────────────

@pytest.mark.parametrize("template_path,expected_tag", [
    ("public/login.html", "h1"),
    ("user/key_created.html", "h1"),
    ("user/request_quota.html", "h1"),
])
def test_standalone_pages_have_h1(template_path, expected_tag):
    """Pages that are the primary content (not admin with sidebar) should have h1."""
    html = _read_template(template_path)
    tags = _parse_tags(html)
    h1s = _find_tags(tags, expected_tag)
    assert h1s, f"{template_path} should have an <{expected_tag}>"


# ── 12. Decorative Icon Tests ────────────────────────────────────

@pytest.mark.parametrize("template_path", [
    "public/login.html",
    "user/key_created.html",
    "user/request_quota.html",
    "chat.html",
])
def test_decorative_icons_hidden(template_path):
    """Icons that are purely decorative should have aria-hidden='true'."""
    html = _read_template(template_path)
    tags = _parse_tags(html)
    icons = [t for t in _find_tags(tags, "i") if "bi" in _classes(t)]
    hidden_count = sum(
        1 for i in icons if i["attrs"].get("aria-hidden") == "true"
    )
    # We expect at least SOME icons to be marked as decorative
    assert hidden_count > 0, (
        f"No decorative icons have aria-hidden='true' in {template_path}"
    )
