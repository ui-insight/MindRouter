############################################################
#
# mindrouter - unit tests for the admin-configurable chat notice
#
# Admin -> Chat can show users a message when they open the chat page (e.g.
# pointing them at VandalChat). Title, HTML body and link are configurable.
#
############################################################

"""Unit tests for the chat notice: rendering, gating, and admin handling."""

import pathlib
import re

import pytest
from jinja2 import Environment, FileSystemLoader

_DASH = pathlib.Path(__file__).resolve().parents[2] / "dashboard"
_TPL = _DASH / "templates"


def _fragment():
    """The notice block out of chat.html, renderable on its own."""
    env = Environment(loader=FileSystemLoader(str(_TPL)), autoescape=True)
    src = (_TPL / "chat.html").read_text()
    i = src.index("{# Admin-configurable chat notice")
    j = src.index("<!-- Multimodal warning modal -->")
    return env.from_string(src[i:j])


def _notice(**over):
    base = {
        "enabled": True,
        "title": "Looking for VandalChat?",
        "html": "<p>Try <strong>VandalChat</strong> instead.</p>",
        "link_url": "https://vandalchat.uidaho.edu",
        "link_text": "Open VandalChat",
        "show_once": True,
        "version": 3,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_modal_renders_when_enabled():
    out = _fragment().render(chat_notice=_notice())
    assert 'id="chatNoticeModal"' in out
    assert "Looking for VandalChat?" in out
    assert "Open VandalChat" in out
    assert 'href="https://vandalchat.uidaho.edu"' in out


def test_nothing_renders_when_disabled():
    """A disabled notice must emit no markup and no script at all."""
    out = _fragment().render(chat_notice=_notice(enabled=False)).strip()
    assert out == ""


def test_nothing_renders_when_context_missing():
    """Any page rendering chat.html without the key must not break."""
    assert _fragment().render().strip() == ""
    assert _fragment().render(chat_notice=None).strip() == ""


def test_admin_html_is_rendered_raw_not_escaped():
    """The body is admin-authored HTML — same trust model as the use agreement."""
    out = _fragment().render(chat_notice=_notice())
    assert "<strong>VandalChat</strong>" in out
    assert "&lt;strong&gt;" not in out


def test_link_omitted_when_no_url_configured():
    out = _fragment().render(chat_notice=_notice(link_url="", link_text=""))
    assert 'id="chatNoticeModal"' in out          # notice still shows
    assert "btn btn-primary" not in out           # but no dead button
    assert "Continue here" in out                 # dismiss always available


def test_defaults_when_title_or_link_text_blank():
    out = _fragment().render(chat_notice=_notice(title="", link_text=""))
    assert "A note before you start" in out
    assert "Learn more" in out


def test_body_omitted_when_no_message():
    out = _fragment().render(chat_notice=_notice(html=""))
    assert "modal-body" not in out
    assert 'id="chatNoticeModal"' in out


def test_link_opens_safely_in_a_new_tab():
    out = _fragment().render(chat_notice=_notice())
    assert 'target="_blank"' in out
    assert 'rel="noopener noreferrer"' in out


# --------------------------------------------------------------------------
# Show-once behaviour
# --------------------------------------------------------------------------


def test_version_is_emitted_so_edits_reshow_the_notice():
    out = _fragment().render(chat_notice=_notice(version=7))
    assert "var VERSION = 7;" in out


def test_show_once_flag_is_emitted_as_a_javascript_boolean():
    assert "var SHOW_ONCE = true;" in _fragment().render(chat_notice=_notice(show_once=True))
    assert "var SHOW_ONCE = false;" in _fragment().render(chat_notice=_notice(show_once=False))


def test_storage_access_is_guarded():
    """Private mode / blocked storage must not break the chat page."""
    out = _fragment().render(chat_notice=_notice())
    # every localStorage touch sits inside a try/catch
    assert out.count("try {") >= 2
    assert out.count("catch (e)") >= 2


def test_missing_bootstrap_does_not_throw():
    out = _fragment().render(chat_notice=_notice())
    assert "typeof bootstrap === 'undefined'" in out


# --------------------------------------------------------------------------
# Server-side gating and admin handling
# --------------------------------------------------------------------------


def _src(name):
    return (_DASH / name).read_text()


def test_chat_handler_passes_every_notice_field():
    src = _src("chat.py")
    for key in ("chat.notice_enabled", "chat.notice_title", "chat.notice_html",
                "chat.notice_link_url", "chat.notice_link_text",
                "chat.notice_show_once", "chat.notice_version"):
        assert key in src, key
    assert '"chat_notice": notice' in src


def test_empty_notice_is_treated_as_disabled_server_side():
    """Never pop an empty dialog because someone ticked the box."""
    src = _src("chat.py")
    assert 'if not (notice["title"] or notice["html"] or notice["link_url"]):' in src
    assert 'notice["enabled"] = False' in src


def test_admin_post_rejects_a_dangerous_link_scheme():
    """An admin typo like javascript: must not become a live script handle."""
    src = _src("routes.py")
    block = src[src.index('if action == "save_chat_notice":'):]
    block = block[: block.index('if action == "set_default":')]
    assert 'n_url.startswith("https://")' in block
    assert 'n_url.startswith("http://")' in block
    assert 'n_url.startswith("/")' in block
    assert "error=Link+URL+must+start+with" in block


def test_admin_post_bumps_the_version_and_commits():
    src = _src("routes.py")
    block = src[src.index('if action == "save_chat_notice":'):]
    block = block[: block.index('if action == "set_default":')]
    assert 'await crud.set_config(db, "chat.notice_version", _ver)' in block
    assert "_ver = int(_ver) + 1" in block
    assert "await db.commit()" in block
    assert "log_admin_action" in block


def test_admin_post_is_full_admin_only():
    """The GET page is admin-read; writing the notice needs full admin."""
    src = _src("routes.py")
    post = src[src.index("async def admin_chat_config_post("):]
    post = post[: post.index('if action == "save_chat_notice":')]
    assert "user.group.is_admin" in post


@pytest.mark.parametrize(
    "field",
    ["notice_enabled", "notice_title", "notice_html",
     "notice_link_url", "notice_link_text", "notice_show_once"],
)
def test_admin_form_exposes_every_field(field):
    tpl = (_TPL / "admin" / "chat_config.html").read_text()
    assert f'name="{field}"' in tpl


def test_admin_form_posts_the_right_action():
    tpl = (_TPL / "admin" / "chat_config.html").read_text()
    assert 'value="save_chat_notice"' in tpl
    assert "notice_updated" in tpl        # success banner wired
