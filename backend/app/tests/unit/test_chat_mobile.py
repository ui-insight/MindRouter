"""
Mobile responsiveness tests for the chat interface (chat.html).

Validates that the chat template has correct CSS, HTML structure, and JS logic
for mobile viewports: sidebar collapse/backdrop, thinking block collapse,
and compact layout rules.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "dashboard" / "templates"


def _read_chat_html() -> str:
    path = TEMPLATE_DIR / "chat.html"
    assert path.exists(), f"Template not found: {path}"
    return path.read_text(encoding="utf-8")


def _extract_css(html: str) -> str:
    """Extract the contents of the first <style>...</style> block."""
    m = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
    assert m, "No <style> block found in chat.html"
    return m.group(1)


def _extract_js(html: str) -> str:
    """Extract the contents of the main <script>...</script> block (the IIFE)."""
    # The main JS block is the last <script> tag with inline code (not src=)
    blocks = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    assert blocks, "No inline <script> block found in chat.html"
    return blocks[-1]  # The IIFE is the last inline script


def _extract_media_block(css: str) -> str:
    """Extract the @media (max-width: 768px) block contents."""
    # Find the @media rule and extract its balanced braces
    start = css.find("@media (max-width: 768px)")
    assert start != -1, "@media (max-width: 768px) not found in CSS"
    brace_start = css.index("{", start)
    depth = 0
    for i in range(brace_start, len(css)):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[brace_start + 1 : i]
    raise AssertionError("Unbalanced braces in @media block")


class _TagCollector(HTMLParser):
    """Lightweight HTML parser that collects elements and their attributes."""

    def __init__(self):
        super().__init__()
        self.tags: list[dict] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append({"tag": tag, "attrs": dict(attrs)})

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def _parse_tags(html: str) -> list[dict]:
    cleaned = re.sub(r"\{%.*?%\}", "", html, flags=re.DOTALL)
    cleaned = re.sub(r"\{\{.*?\}\}", "", cleaned, flags=re.DOTALL)
    collector = _TagCollector()
    collector.feed(cleaned)
    return collector.tags


def _find_by_id(tags: list[dict], element_id: str) -> dict | None:
    for t in tags:
        if t["attrs"].get("id") == element_id:
            return t
    return None


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def chat_html():
    return _read_chat_html()


@pytest.fixture(scope="module")
def css(chat_html):
    return _extract_css(chat_html)


@pytest.fixture(scope="module")
def js(chat_html):
    return _extract_js(chat_html)


@pytest.fixture(scope="module")
def media_block(css):
    return _extract_media_block(css)


@pytest.fixture(scope="module")
def tags(chat_html):
    return _parse_tags(chat_html)


# ── 1. Sidebar backdrop: HTML structure ──────────────────────────


class TestSidebarBackdropHTML:
    """Verify the sidebar-backdrop element exists with correct attributes."""

    def test_backdrop_element_exists(self, tags):
        el = _find_by_id(tags, "sidebarBackdrop")
        assert el is not None, "sidebar-backdrop element not found"

    def test_backdrop_has_correct_class(self, tags):
        el = _find_by_id(tags, "sidebarBackdrop")
        assert "sidebar-backdrop" in el["attrs"].get("class", ""), \
            "sidebar-backdrop missing 'sidebar-backdrop' class"

    def test_backdrop_is_div(self, tags):
        el = _find_by_id(tags, "sidebarBackdrop")
        assert el["tag"] == "div"

    def test_backdrop_inside_chat_layout(self, chat_html):
        """Backdrop should appear after chat-layout opens and before it closes."""
        layout_start = chat_html.find('class="chat-layout"')
        backdrop_pos = chat_html.find('id="sidebarBackdrop"')
        assert layout_start < backdrop_pos, \
            "sidebar-backdrop should be inside .chat-layout"


# ── 2. Sidebar backdrop: CSS ────────────────────────────────────


class TestSidebarBackdropCSS:
    """Verify the .sidebar-backdrop CSS is correct."""

    def test_backdrop_base_styles(self, css):
        assert ".sidebar-backdrop" in css
        assert "position: fixed" in css or "position:fixed" in css.replace(" ", "")

    def test_backdrop_z_index(self, css):
        # z-index: 49 — just below sidebar's z-index: 50
        m = re.search(r"\.sidebar-backdrop\s*\{([^}]+)\}", css)
        assert m, ".sidebar-backdrop rule not found"
        assert "z-index" in m.group(1)
        assert "49" in m.group(1)

    def test_backdrop_hidden_by_default(self, css):
        m = re.search(r"\.sidebar-backdrop\s*\{([^}]+)\}", css)
        assert m
        assert "display" in m.group(1)
        assert "none" in m.group(1)

    def test_backdrop_active_visible(self, css):
        assert ".sidebar-backdrop.active" in css
        # Should set display: block
        m = re.search(r"\.sidebar-backdrop\.active\s*\{([^}]+)\}", css)
        assert m, ".sidebar-backdrop.active rule not found"
        assert "block" in m.group(1)

    def test_backdrop_outside_media_query(self, css):
        """Backdrop base styles should be outside @media block (always available)."""
        media_start = css.find("@media (max-width: 768px)")
        backdrop_rule = css.find(".sidebar-backdrop {")
        if backdrop_rule == -1:
            backdrop_rule = css.find(".sidebar-backdrop{")
        assert backdrop_rule < media_start, \
            ".sidebar-backdrop should be defined before @media block"


# ── 3. Mobile media query: layout rules ─────────────────────────


class TestMobileMediaQuery:
    """Verify the @media (max-width: 768px) block contains mobile polish rules."""

    def test_sidebar_absolute_position(self, media_block):
        assert "position: absolute" in media_block or "position:absolute" in media_block

    def test_sidebar_z_index_50(self, media_block):
        assert "z-index: 50" in media_block or "z-index:50" in media_block

    def test_message_compact_padding(self, media_block):
        # .message padding should be tighter than desktop default
        assert ".message" in media_block
        assert "0.5rem" in media_block

    def test_input_area_compact_padding(self, media_block):
        assert ".chat-input-area" in media_block

    def test_thumbnail_scaled_down(self, media_block):
        assert "140px" in media_block

    def test_image_modal_wider(self, media_block):
        assert "95vw" in media_block

    def test_header_compact(self, media_block):
        assert ".chat-header" in media_block

    def test_model_select_narrower(self, media_block):
        assert "max-width" in media_block
        assert "140px" in media_block

    def test_sidebar_header_compact(self, media_block):
        assert ".sidebar-header" in media_block


# ── 4. JS: Mobile sidebar init ──────────────────────────────────


class TestJSSidebarInit:
    """Verify JS initializes sidebar as collapsed on mobile."""

    def test_is_mobile_helper_defined(self, js):
        assert "isMobile" in js

    def test_is_mobile_checks_768(self, js):
        # Should check innerWidth <= 768
        m = re.search(r"isMobile\s*=\s*\(\)\s*=>\s*window\.innerWidth\s*<=\s*768", js)
        assert m, "isMobile should be an arrow fn checking window.innerWidth <= 768"

    def test_init_collapses_on_mobile(self, js):
        """Init block should add 'collapsed' class when isMobile()."""
        assert re.search(
            r"if\s*\(isMobile\(\)\)\s*\{[^}]*sidebar\.classList\.add\(['\"]collapsed['\"]\)",
            js,
            re.DOTALL,
        ), "Init block should collapse sidebar on mobile"

    def test_init_sets_aria_expanded_false(self, js):
        assert re.search(
            r"if\s*\(isMobile\(\)\)\s*\{[^}]*setAttribute\(['\"]aria-expanded['\"],\s*['\"]false['\"]\)",
            js,
            re.DOTALL,
        ), "Init block should set aria-expanded='false' on mobile"


# ── 5. JS: closeSidebarOnMobile helper ──────────────────────────


class TestJSCloseSidebarOnMobile:
    """Verify closeSidebarOnMobile() helper exists and has correct logic."""

    def test_function_defined(self, js):
        assert "function closeSidebarOnMobile()" in js

    def test_checks_is_mobile(self, js):
        # Extract the function body
        m = re.search(
            r"function closeSidebarOnMobile\(\)\s*\{(.*?)\n    \}",
            js,
            re.DOTALL,
        )
        assert m, "closeSidebarOnMobile function body not found"
        body = m.group(1)
        assert "isMobile()" in body, "Should check isMobile()"

    def test_adds_collapsed_class(self, js):
        m = re.search(
            r"function closeSidebarOnMobile\(\)\s*\{(.*?)\n    \}",
            js,
            re.DOTALL,
        )
        body = m.group(1)
        assert "classList.add" in body and "collapsed" in body

    def test_removes_backdrop_active(self, js):
        m = re.search(
            r"function closeSidebarOnMobile\(\)\s*\{(.*?)\n    \}",
            js,
            re.DOTALL,
        )
        body = m.group(1)
        assert "sidebarBackdrop" in body and "remove" in body and "active" in body


# ── 6. JS: Sidebar toggle backdrop integration ──────────────────


class TestJSToggleBackdrop:
    """Verify toggle button integrates with backdrop on mobile."""

    def test_toggle_handler_checks_mobile(self, js):
        # The toggleBtn click handler should contain isMobile check
        m = re.search(
            r"toggleBtn\.addEventListener\(['\"]click['\"],\s*\(\)\s*=>\s*\{(.*?)\}\);",
            js,
            re.DOTALL,
        )
        assert m, "toggleBtn click handler not found"
        body = m.group(1)
        assert "isMobile()" in body, "Toggle handler should check isMobile()"

    def test_toggle_handler_toggles_backdrop(self, js):
        m = re.search(
            r"toggleBtn\.addEventListener\(['\"]click['\"],\s*\(\)\s*=>\s*\{(.*?)\}\);",
            js,
            re.DOTALL,
        )
        body = m.group(1)
        assert "sidebarBackdrop" in body, "Toggle handler should reference backdrop"

    def test_backdrop_click_calls_close(self, js):
        assert "sidebarBackdrop.addEventListener" in js
        assert "closeSidebarOnMobile" in js


# ── 7. JS: Auto-close on navigation ─────────────────────────────


class TestJSAutoCloseOnNavigation:
    """Verify sidebar auto-closes on mobile when switching conversations or creating new chat."""

    def test_switch_conversation_closes(self, js):
        # switchConversation should call closeSidebarOnMobile
        m = re.search(
            r"async function switchConversation\(id\)\s*\{(.*?)\n    \}",
            js,
            re.DOTALL,
        )
        assert m, "switchConversation function not found"
        body = m.group(1)
        assert "closeSidebarOnMobile()" in body, \
            "switchConversation should call closeSidebarOnMobile()"

    def test_new_chat_closes(self, js):
        # newChatBtn click handler should call closeSidebarOnMobile
        m = re.search(
            r"newChatBtn\.addEventListener\(['\"]click['\"],\s*\(\)\s*=>\s*\{(.*?)\}\);",
            js,
            re.DOTALL,
        )
        assert m, "newChatBtn click handler not found"
        body = m.group(1)
        assert "closeSidebarOnMobile()" in body, \
            "newChatBtn handler should call closeSidebarOnMobile()"


# ── 8. JS: Thinking blocks collapsed on mobile ──────────────────


class TestJSThinkingBlockMobile:
    """Verify thinking blocks start collapsed on mobile during streaming."""

    def test_reasoning_checks_mobile(self, js):
        # appendStreamReasoning should reference isMobile
        m = re.search(
            r"function appendStreamReasoning\(delta\)\s*\{(.*?)\n    \}",
            js,
            re.DOTALL,
        )
        assert m, "appendStreamReasoning function not found"
        body = m.group(1)
        assert "isMobile()" in body, \
            "appendStreamReasoning should check isMobile()"

    def test_start_collapsed_variable(self, js):
        m = re.search(
            r"function appendStreamReasoning\(delta\)\s*\{(.*?)\n    \}",
            js,
            re.DOTALL,
        )
        body = m.group(1)
        assert "startCollapsed" in body, \
            "Should use startCollapsed variable"

    def test_collapsed_class_conditional(self, js):
        # The thinking-toggle should conditionally get 'collapsed' class
        assert "startCollapsed ? ' collapsed' : ''" in js or \
               'startCollapsed ? " collapsed" : ""' in js, \
            "thinking-toggle should conditionally add collapsed class"

    def test_display_none_conditional(self, js):
        # thinking-content should conditionally get display:none
        assert 'startCollapsed ? \' style="display:none"\' : \'\'' in js or \
               'startCollapsed ? " style=\\"display:none\\"" : ""' in js, \
            "thinking-content should conditionally set display:none"


# ── 9. CSS: Brace balancing ─────────────────────────────────────


class TestCSSIntegrity:
    """Basic structural integrity checks for the CSS."""

    def test_braces_balanced(self, css):
        opens = css.count("{")
        closes = css.count("}")
        assert opens == closes, \
            f"CSS brace mismatch: {opens} opens vs {closes} closes"

    def test_single_media_query(self, css):
        count = len(re.findall(r"@media\s*\(max-width:\s*768px\)", css))
        assert count == 1, f"Expected 1 @media block, found {count}"
