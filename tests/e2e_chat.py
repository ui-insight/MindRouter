#!/usr/bin/env python3
"""
End-to-end tests for MindRouter2 chat subsystem.

Exercises: persistence, image preprocessing, storage, multi-turn context,
vision model Q&A, cross-user isolation, and CRUD operations against the
live Docker stack.

Usage:
    python tests/e2e_chat.py \
        --base-url http://localhost:8000 \
        --username admin --password <pw> \
        --text-model phi4:14b \
        --vision-model qwen2.5-VL-32k:7b \
        --username2 faculty1 --password2 <pw> \
        --cookie-file /tmp/mr_cookies.txt \
        --cookie-file2 /tmp/mr_cookies2.txt \
        --skip-vision \
        --docker-container mindrouter2-app-1
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import namedtuple

import requests

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

TestResult = namedtuple("TestResult", ["name", "passed", "skipped", "message", "duration"])


class SkipTest(Exception):
    """Raise to skip a test with a reason."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_cookie_file(path: str) -> str | None:
    """Extract mindrouter_session value from Netscape cookie file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # curl marks httponly cookies with #HttpOnly_ prefix — strip it
            if line.startswith("#HttpOnly_"):
                line = line[len("#HttpOnly_"):]
            elif line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7 and parts[5] == "mindrouter_session":
                return parts[6]
    return None


def _make_test_image_png() -> bytes:
    """Generate a 3000x2000 PNG with red circle, blue rectangle, green text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (3000, 2000), "white")
    draw = ImageDraw.Draw(img)
    # Red filled circle — upper-left quadrant
    draw.ellipse([200, 200, 1200, 1200], fill="red")
    # Blue filled rectangle — lower-right quadrant
    draw.rectangle([1800, 1000, 2800, 1800], fill="blue")
    # Green "TEST" text — center
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
        except (OSError, IOError):
            font = ImageFont.load_default()
    draw.text((1300, 900), "TEST", fill="green", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_test_text_file() -> bytes:
    """Generate a small text file for upload testing."""
    return b"The capital of France is Paris. MindRouter test file.\n"


# ---------------------------------------------------------------------------
# Chat E2E Test Suite
# ---------------------------------------------------------------------------

class ChatE2ETests:
    def __init__(self, args: argparse.Namespace):
        self.base = args.base_url.rstrip("/")
        self.args = args
        self.results: list[TestResult] = []

        # Sessions (populated during auth tests)
        self.session: requests.Session | None = None
        self.session2: requests.Session | None = None

        # State shared across tests
        self.conv_id: int | None = None
        self.conv_ids_to_cleanup: list[int] = []
        self.attachment_id_text: int | None = None
        self.attachment_id_image: int | None = None
        self.test_image_png: bytes | None = None

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _authenticate(self, username, password, cookie_file) -> requests.Session:
        """Return an authenticated requests.Session."""
        s = requests.Session()

        if cookie_file and os.path.isfile(cookie_file):
            val = _parse_cookie_file(cookie_file)
            if val:
                s.cookies.set("mindrouter_session", val)
                # Validate the session
                r = s.get(f"{self.base}/chat/api/conversations")
                if r.status_code == 200:
                    return s
                # Cookie didn't work — fall through to login

        if not username or not password:
            raise SkipTest("No valid credentials or cookie file provided")

        r = s.post(
            f"{self.base}/login",
            data={"username": username, "password": password},
            allow_redirects=False,
        )
        if r.status_code not in (302, 303):
            raise AssertionError(f"Login returned {r.status_code}, expected redirect")
        if "mindrouter_session" not in s.cookies:
            raise AssertionError("Login did not set mindrouter_session cookie")
        return s

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    def _stream_completion(self, session, conv_id, model, content,
                           attachment_ids=None) -> str:
        """Send a streaming completion and return accumulated content."""
        payload = {
            "conversation_id": conv_id,
            "model": model,
            "content": content,
            "stream": True,
        }
        if attachment_ids:
            payload["attachment_ids"] = attachment_ids

        r = session.post(
            f"{self.base}/chat/api/completions",
            json=payload,
            stream=True,
        )
        assert r.status_code == 200, f"Completions returned {r.status_code}: {r.text[:500]}"

        full = []
        saw_done = False
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                data = raw[6:]
                if data.strip() == "[DONE]":
                    saw_done = True
                    break
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        full.append(token)
                except json.JSONDecodeError:
                    pass
        assert saw_done, "Stream did not end with [DONE]"
        return "".join(full)

    def _nonstreaming_completion(self, session, conv_id, model, content,
                                  attachment_ids=None) -> str:
        """Send a non-streaming completion and return content."""
        payload = {
            "conversation_id": conv_id,
            "model": model,
            "content": content,
            "stream": False,
        }
        if attachment_ids:
            payload["attachment_ids"] = attachment_ids

        r = session.post(
            f"{self.base}/chat/api/completions",
            json=payload,
        )
        assert r.status_code == 200, f"Completions returned {r.status_code}: {r.text[:500]}"
        body = r.json()
        assert "choices" in body, f"No choices in response: {body}"
        return body["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def test_01_auth_login(self):
        """Login or load cookie, validate session."""
        self.session = self._authenticate(
            self.args.username, self.args.password, self.args.cookie_file
        )
        r = self.session.get(f"{self.base}/chat/api/conversations")
        assert r.status_code == 200, f"Session validation failed: {r.status_code}"
        return "Authenticated successfully"

    def test_02_conversation_create(self):
        """POST creates conversation with 'New Chat' title."""
        r = self.session.post(
            f"{self.base}/chat/api/conversations",
            json={"model": self.args.text_model},
        )
        assert r.status_code == 200, f"Create conv returned {r.status_code}: {r.text[:300]}"
        body = r.json()
        self.conv_id = body["id"]
        self.conv_ids_to_cleanup.append(self.conv_id)
        assert body["title"] == "New Chat", f"Expected 'New Chat', got '{body['title']}'"
        return f"Created conversation {self.conv_id}"

    def test_03_conversation_list(self):
        """GET lists the new conversation."""
        r = self.session.get(f"{self.base}/chat/api/conversations")
        assert r.status_code == 200
        body = r.json()
        convs = body.get("conversations", body) if isinstance(body, dict) else body
        ids = [c["id"] for c in convs]
        assert self.conv_id in ids, f"Conv {self.conv_id} not in list {ids}"
        return f"Conversation {self.conv_id} appears in list"

    def test_04_conversation_get(self):
        """GET returns conversation with empty messages."""
        r = self.session.get(f"{self.base}/chat/api/conversations/{self.conv_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == self.conv_id
        assert body["messages"] == [], f"Expected empty messages, got {len(body['messages'])}"
        return "Conversation has empty messages array"

    def test_05_conversation_patch(self):
        """PATCH updates title."""
        new_title = "E2E Test Conversation"
        r = self.session.patch(
            f"{self.base}/chat/api/conversations/{self.conv_id}",
            json={"title": new_title},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["title"] == new_title, f"Title mismatch: {body['title']}"
        return f"Title updated to '{new_title}'"

    def test_06_nonstreaming_completion(self):
        """Non-streaming math Q&A: model should answer '4' for 2+2."""
        answer = self._nonstreaming_completion(
            self.session, self.conv_id, self.args.text_model,
            "What is 2+2? Reply with ONLY the number, no other text."
        )
        assert "4" in answer, f"Expected '4' in answer, got: {answer[:200]}"
        return f"Model answered: {answer.strip()[:80]}"

    def test_07_auto_title(self):
        """First message on a new conversation auto-sets the title."""
        # Create a fresh conversation
        r = self.session.post(
            f"{self.base}/chat/api/conversations",
            json={"model": self.args.text_model},
        )
        assert r.status_code == 200
        new_id = r.json()["id"]
        self.conv_ids_to_cleanup.append(new_id)

        # Send a message (non-streaming to keep it simple)
        self._nonstreaming_completion(
            self.session, new_id, self.args.text_model,
            "Tell me a fun fact about penguins."
        )

        # Re-fetch and check title changed from "New Chat"
        r = self.session.get(f"{self.base}/chat/api/conversations/{new_id}")
        assert r.status_code == 200
        title = r.json()["title"]
        assert title != "New Chat", f"Title was not auto-updated, still '{title}'"
        return f"Auto-title set to: '{title[:60]}'"

    def test_08_streaming_completion(self):
        """SSE streaming: content extraction and DB persistence."""
        content = self._stream_completion(
            self.session, self.conv_id, self.args.text_model,
            "Name three primary colors. Be brief."
        )
        assert len(content) > 5, f"Stream content too short: {content}"
        # Wait for server to commit assistant message
        time.sleep(2)
        # Verify persistence
        r = self.session.get(f"{self.base}/chat/api/conversations/{self.conv_id}")
        assert r.status_code == 200
        messages = r.json()["messages"]
        roles = [m["role"] for m in messages]
        assert "assistant" in roles, f"No assistant message persisted. Roles: {roles}"
        return f"Streamed {len(content)} chars, persisted to DB"

    def test_09_context_maintenance(self):
        """Multi-turn: define a fact, then recall it."""
        # Create a fresh conv for isolation
        r = self.session.post(
            f"{self.base}/chat/api/conversations",
            json={"model": self.args.text_model},
        )
        ctx_conv = r.json()["id"]
        self.conv_ids_to_cleanup.append(ctx_conv)

        # Turn 1: define a fact
        self._nonstreaming_completion(
            self.session, ctx_conv, self.args.text_model,
            'Remember this: a "zibblefruit" is a purple vegetable that tastes like chocolate. Just say "OK".'
        )
        # Turn 2: recall it
        answer = self._nonstreaming_completion(
            self.session, ctx_conv, self.args.text_model,
            "What color is a zibblefruit?"
        )
        assert "purple" in answer.lower(), f"Context lost — expected 'purple' in: {answer[:200]}"
        return f"Context maintained: {answer.strip()[:80]}"

    def test_10_persistence_reload(self):
        """Reload conversation and verify all messages and role ordering."""
        # Give streaming persistence extra time to commit
        time.sleep(2)
        r = self.session.get(f"{self.base}/chat/api/conversations/{self.conv_id}")
        assert r.status_code == 200
        messages = r.json()["messages"]
        assert len(messages) >= 4, f"Expected >=4 messages, got {len(messages)}"
        # Verify alternating roles starting with user
        for i, m in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert m["role"] == expected_role, (
                f"Message {i}: expected '{expected_role}', got '{m['role']}'"
            )
        return f"Reloaded {len(messages)} messages with correct role ordering"

    def test_11_text_file_upload(self):
        """Upload .txt file, verify attachment_id and is_image=false."""
        data = _make_test_text_file()
        r = self.session.post(
            f"{self.base}/chat/api/upload",
            files={"file": ("test_document.txt", io.BytesIO(data), "text/plain")},
        )
        assert r.status_code == 200, f"Upload returned {r.status_code}: {r.text[:300]}"
        body = r.json()
        self.attachment_id_text = body["attachment_id"]
        assert body["is_image"] is False, f"Text file marked as image"
        assert body["filename"] == "test_document.txt"
        return f"Uploaded text file, attachment_id={self.attachment_id_text}"

    def test_12_text_file_completion(self):
        """Send message with text attachment; model reads file content."""
        # Create a fresh conv
        r = self.session.post(
            f"{self.base}/chat/api/conversations",
            json={"model": self.args.text_model},
        )
        file_conv = r.json()["id"]
        self.conv_ids_to_cleanup.append(file_conv)

        answer = self._nonstreaming_completion(
            self.session, file_conv, self.args.text_model,
            "What is the capital city mentioned in the attached file? Reply with only the city name.",
            attachment_ids=[self.attachment_id_text],
        )
        assert "paris" in answer.lower(), f"Expected 'Paris' in answer: {answer[:200]}"
        return f"Model read attachment: {answer.strip()[:80]}"

    def test_13_image_upload(self):
        """Upload 3000x2000 PNG, verify thumbnail and JPEG conversion."""
        self.test_image_png = _make_test_image_png()
        r = self.session.post(
            f"{self.base}/chat/api/upload",
            files={"file": ("test_image.png", io.BytesIO(self.test_image_png), "image/png")},
        )
        assert r.status_code == 200, f"Image upload returned {r.status_code}: {r.text[:300]}"
        body = r.json()
        self.attachment_id_image = body["attachment_id"]
        assert body["is_image"] is True, "Image not marked as is_image"
        assert body["thumbnail"] is not None, "No thumbnail returned"
        # Verify thumbnail is valid base64 PNG
        thumb_bytes = base64.b64decode(body["thumbnail"])
        assert thumb_bytes[:4] == b"\x89PNG", "Thumbnail is not a PNG"
        return f"Uploaded image, attachment_id={self.attachment_id_image}, thumbnail OK"

    def test_14_image_dimensions_check(self):
        """Docker exec: verify stored JPEG is 1536x1024 (max 1536 on longest side)."""
        container = self.args.docker_container
        if not container:
            raise SkipTest("No --docker-container specified")
        if not shutil.which("docker"):
            raise SkipTest("docker CLI not found")

        att_id = self.attachment_id_image
        if att_id is None:
            raise SkipTest("No image attachment from previous test")

        cmd = [
            "docker", "exec", container, "python3", "-c",
            f"from PIL import Image; img=Image.open('/data/chat_files/{att_id}.jpg'); print(img.size)",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise SkipTest(f"Docker exec failed: {e}")

        if result.returncode != 0:
            raise SkipTest(f"Docker exec error: {result.stderr.strip()}")

        size_str = result.stdout.strip()
        # Parse "(W, H)"
        match = re.match(r"\((\d+),\s*(\d+)\)", size_str)
        assert match, f"Could not parse size: {size_str}"
        w, h = int(match.group(1)), int(match.group(2))
        assert max(w, h) == 1536, f"Longest side should be 1536, got {max(w, h)} ({w}x{h})"
        return f"Stored JPEG dimensions: {w}x{h}"

    def test_15_vision_image_qa(self):
        """Vision model describes red circle and blue rectangle in test image."""
        if self.args.skip_vision:
            raise SkipTest("--skip-vision flag set")
        if not self.args.vision_model:
            raise SkipTest("No --vision-model specified")
        if self.attachment_id_image is None:
            raise SkipTest("No image attachment from previous test")

        # Create a fresh conv for vision
        r = self.session.post(
            f"{self.base}/chat/api/conversations",
            json={"model": self.args.vision_model},
        )
        vision_conv = r.json()["id"]
        self.conv_ids_to_cleanup.append(vision_conv)
        self._vision_conv_id = vision_conv

        answer = self._nonstreaming_completion(
            self.session, vision_conv, self.args.vision_model,
            "Describe the shapes and colors you see in this image. Be specific.",
            attachment_ids=[self.attachment_id_image],
        )
        answer_lower = answer.lower()
        has_red = "red" in answer_lower
        has_blue = "blue" in answer_lower
        assert has_red or has_blue, f"Vision model didn't mention red/blue: {answer[:300]}"
        return f"Vision detected colors: red={has_red}, blue={has_blue}"

    def test_16_vision_image_followup(self):
        """Follow-up about 'TEST' text without re-uploading image."""
        if self.args.skip_vision:
            raise SkipTest("--skip-vision flag set")
        if not self.args.vision_model:
            raise SkipTest("No --vision-model specified")
        if not hasattr(self, "_vision_conv_id"):
            raise SkipTest("No vision conversation from previous test")

        answer = self._nonstreaming_completion(
            self.session, self._vision_conv_id, self.args.vision_model,
            'Is there any text written in the image? If so, what does it say?'
        )
        assert "test" in answer.lower(), f"Vision didn't find 'TEST' text: {answer[:300]}"
        return f"Follow-up found text: {answer.strip()[:80]}"

    def test_17_cross_user_isolation_list(self):
        """User2 cannot see user1's conversations."""
        if not self._have_user2():
            raise SkipTest("No user2 credentials provided")

        self.session2 = self._authenticate(
            self.args.username2, self.args.password2, self.args.cookie_file2
        )
        r = self.session2.get(f"{self.base}/chat/api/conversations")
        assert r.status_code == 200
        body = r.json()
        convs = body.get("conversations", body) if isinstance(body, dict) else body
        ids2 = [c["id"] for c in convs]
        for cid in self.conv_ids_to_cleanup:
            assert cid not in ids2, f"User2 can see user1's conv {cid}"
        return "User2 cannot list user1's conversations"

    def test_18_cross_user_isolation_access(self):
        """User2 gets 404 on user1's conversation."""
        if not self._have_user2() or self.session2 is None:
            raise SkipTest("No user2 session")

        r = self.session2.get(
            f"{self.base}/chat/api/conversations/{self.conv_id}"
        )
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"
        return f"User2 correctly gets 404 on conv {self.conv_id}"

    def test_19_conversation_delete(self):
        """DELETE conversation, verify 404 on re-fetch."""
        # Create a throwaway conv to delete
        r = self.session.post(
            f"{self.base}/chat/api/conversations",
            json={"model": self.args.text_model},
        )
        del_id = r.json()["id"]

        r = self.session.delete(f"{self.base}/chat/api/conversations/{del_id}")
        assert r.status_code == 200, f"Delete returned {r.status_code}"
        body = r.json()
        assert body.get("ok") is True, f"Delete response: {body}"

        r = self.session.get(f"{self.base}/chat/api/conversations/{del_id}")
        assert r.status_code == 404, f"Re-fetch after delete: expected 404, got {r.status_code}"
        return f"Deleted conv {del_id}, confirmed 404"

    def test_20_cleanup(self):
        """Delete all auxiliary test conversations."""
        deleted = []
        failed = []
        for cid in self.conv_ids_to_cleanup:
            r = self.session.delete(f"{self.base}/chat/api/conversations/{cid}")
            if r.status_code == 200:
                deleted.append(cid)
            else:
                failed.append((cid, r.status_code))
        if failed:
            return f"Cleaned {len(deleted)}, failed: {failed}"
        return f"Cleaned up {len(deleted)} test conversations"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _have_user2(self) -> bool:
        return bool(
            (self.args.username2 and self.args.password2)
            or (self.args.cookie_file2 and os.path.isfile(self.args.cookie_file2))
        )

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def _run_test(self, method):
        """Execute a single test method, catching errors."""
        name = method.__name__
        t0 = time.time()
        try:
            msg = method()
            dur = time.time() - t0
            self.results.append(TestResult(name, True, False, msg or "OK", dur))
        except SkipTest as e:
            dur = time.time() - t0
            self.results.append(TestResult(name, False, True, str(e), dur))
        except AssertionError as e:
            dur = time.time() - t0
            self.results.append(TestResult(name, False, False, str(e), dur))
        except Exception as e:
            dur = time.time() - t0
            self.results.append(TestResult(name, False, False, f"{type(e).__name__}: {e}", dur))

    def run_all(self):
        """Run all tests in order and print summary."""
        tests = [
            self.test_01_auth_login,
            self.test_02_conversation_create,
            self.test_03_conversation_list,
            self.test_04_conversation_get,
            self.test_05_conversation_patch,
            self.test_06_nonstreaming_completion,
            self.test_07_auto_title,
            self.test_08_streaming_completion,
            self.test_09_context_maintenance,
            self.test_10_persistence_reload,
            self.test_11_text_file_upload,
            self.test_12_text_file_completion,
            self.test_13_image_upload,
            self.test_14_image_dimensions_check,
            self.test_15_vision_image_qa,
            self.test_16_vision_image_followup,
            self.test_17_cross_user_isolation_list,
            self.test_18_cross_user_isolation_access,
            self.test_19_conversation_delete,
            self.test_20_cleanup,
        ]

        print(f"\n{'='*70}")
        print("  MindRouter2 Chat E2E Test Suite")
        print(f"  Target: {self.base}")
        print(f"  Text model: {self.args.text_model}")
        print(f"  Vision model: {self.args.vision_model or '(none)'}")
        print(f"{'='*70}\n")

        for test in tests:
            name = test.__name__
            print(f"  {name} ... ", end="", flush=True)
            self._run_test(test)
            r = self.results[-1]
            if r.skipped:
                print(f"SKIP ({r.message})")
            elif r.passed:
                print(f"PASS ({r.duration:.1f}s) — {r.message}")
            else:
                print(f"FAIL ({r.duration:.1f}s) — {r.message}")

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        skipped = sum(1 for r in self.results if r.skipped)
        failed = sum(1 for r in self.results if not r.passed and not r.skipped)
        total = len(self.results)

        print(f"\n{'='*70}")
        print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped / {total} total")
        if failed:
            print("\n  Failed tests:")
            for r in self.results:
                if not r.passed and not r.skipped:
                    print(f"    - {r.name}: {r.message}")
        print(f"{'='*70}\n")

        return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MindRouter2 Chat E2E Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="MindRouter base URL (default: http://localhost:8000)")
    parser.add_argument("--username", default=None, help="Primary user username")
    parser.add_argument("--password", default=None, help="Primary user password")
    parser.add_argument("--cookie-file", default=None,
                        help="Netscape cookie file for primary user")
    parser.add_argument("--text-model", required=True, help="Text model ID")
    parser.add_argument("--vision-model", default=None, help="Vision model ID")
    parser.add_argument("--username2", default=None, help="Second user username")
    parser.add_argument("--password2", default=None, help="Second user password")
    parser.add_argument("--cookie-file2", default=None,
                        help="Netscape cookie file for second user")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip vision model tests")
    parser.add_argument("--docker-container", default=None,
                        help="Docker container name for dimension checks")
    args = parser.parse_args()

    suite = ChatE2ETests(args)
    sys.exit(suite.run_all())


if __name__ == "__main__":
    main()
