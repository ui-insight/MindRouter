############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_image_matting.py: `background: "transparent"` on the
# images API (issue #28). The diffusion backend returns
# opaque RGB, so the gateway cuts the finished, watermarked
# picture out with a matte from the matting server
# (matting_service/).
#
# Covers: the `background` value (accepted, refused, absent);
# attaching a matte with real pixels (the alpha that comes
# out, that every fully opaque and fully transparent pixel
# keeps its colour so the watermark survives in the file,
# that edge pixels lose the old background's colour, the two
# "not a cut-out" cases, a matte that does not fit or claims
# an enormous size); the call to the matting server (a reply
# that is too large or never finishes, and whose fault each
# failure is); failing open at every level; the response
# fields; the health lookup and circuit reports for a
# registered server (only sickness is reported); the request
# path (b64 forced, cut-out after completion, cancellation,
# nothing changes for callers who do not send the field);
# both endpoints and the playground; the admin settings form
# driven for real; the engine, the migration and the docs.
#
############################################################

"""Transparent backgrounds for generated images."""

import ast
import asyncio
import base64
import importlib.util
import io
import re
import struct
import time
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from PIL import Image

from backend.app.services import image_matting as im

_APP = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[4]
URL = "https://aspen4.example.edu:8005"
CONFIG = im.MattingConfig(enabled=True, url=URL, api_key="matting-key", timeout=30.0)
PNG = b"\x89PNG\r\n\x1a\n"


def _png(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _b64(image):
    return base64.b64encode(_png(image)).decode("ascii")


def _noise(size=(96, 80), seed=3):
    """A picture with no two neighbouring pixels alike, so any change shows."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")


def _half_matte(size=(96, 80), soft=0):
    """Left part subject, right part background, with ``soft`` columns of ramp between."""
    width, height = size
    row = np.zeros(width, dtype=np.float32)
    middle = width // 2
    row[: middle - soft // 2] = 255
    if soft:
        row[middle - soft // 2: middle + soft // 2] = np.linspace(255, 0, soft + 2)[1:-1][: 2 * (soft // 2)]
    return Image.fromarray(np.tile(row.round().astype(np.uint8), (height, 1)), "L")


def _open(png_bytes):
    return np.asarray(Image.open(io.BytesIO(png_bytes)))


# ---------------------------------------------------------------------------
# the `background` value
# ---------------------------------------------------------------------------

class TestParseBackground:
    @pytest.mark.parametrize("value,expected", [
        ("transparent", "transparent"), ("opaque", "opaque"), ("auto", "auto"),
        (" Transparent ", "transparent"), ("OPAQUE", "opaque"),
        (None, None), ("", None), ("   ", None),
    ])
    def test_accepted(self, value, expected):
        assert im.parse_background(value) == expected

    @pytest.mark.parametrize("value", ["clear", "white", "none", "true", "transparant", True, 1, ["transparent"],
                                       {"type": "transparent"}])
    def test_anything_else_is_read_as_auto_never_refused(self, value):
        # The field used to be ignored, so a client sending some other value
        # has working requests today. They must keep working.
        assert im.parse_background(value) == "auto"

    @pytest.mark.parametrize("value", [False, 0, [], {}, 0.0])
    def test_an_empty_value_of_any_kind_counts_as_absent(self, value):
        # `false` means "no", not "some background I did not spell right":
        # such a caller gets the response it always got, with no new fields.
        assert im.parse_background(value) is None

    def test_what_was_actually_sent_is_logged_because_the_audit_row_will_say_auto(self, monkeypatch):
        log = MagicMock()
        monkeypatch.setattr(im, "logger", log)
        im.parse_background("transparant")
        log.info.assert_called_once_with("image_background_value_not_recognised", sent="'transparant'")
        log.reset_mock()
        im.parse_background("x" * 500)
        assert len(log.info.call_args.kwargs["sent"]) == 60          # a caller cannot fill the log
        log.reset_mock()
        for known in ("transparent", "auto", None, "", False):
            im.parse_background(known)
        log.info.assert_not_called()

    async def test_an_unrecognised_value_gets_the_ordinary_picture_and_is_told_so(self, server):
        response = _response()
        original = response["data"][0]["b64_json"]
        await im.apply_background(im.parse_background("transparant"), response, CONFIG, _registry((7, None)))
        assert response["background"] == "opaque" and server["calls"] == 0
        assert response["data"][0]["b64_json"] == original and response["data"][0]["has_alpha"] is False


# ---------------------------------------------------------------------------
# attaching a matte (real pixels)
# ---------------------------------------------------------------------------

class TestBoxBlur:
    @pytest.mark.parametrize("size", [1, 2, 5, 6, 9, 90])
    def test_matches_a_plain_mean_over_the_window(self, size):
        rng = np.random.default_rng(1)
        values = rng.random((23, 17, 2), dtype=np.float32)
        before, after = size // 2, size - size // 2 - 1
        padded = np.pad(values, ((before, after), (before, after), (0, 0)), mode="edge")
        want = np.zeros_like(values)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                want[i, j] = padded[i:i + size, j:j + size].mean(axis=(0, 1))
        got = im._box_blur(values, size)
        assert got.shape == values.shape and got.dtype == np.float32
        assert np.abs(got - want).max() < 1e-4

    def test_a_flat_picture_stays_flat(self):
        flat = np.full((40, 30, 3), 0.25, dtype=np.float32)
        assert np.abs(im._box_blur(flat, 90) - 0.25).max() < 1e-5


class TestAttachMatte:
    def test_the_result_is_an_rgba_png_with_the_matte_as_alpha(self):
        picture, matte = _noise(), _half_matte()
        cut, outcome = im.attach_matte(_png(picture), _png(matte))
        assert outcome == im.OUTCOME_TRANSPARENT and cut.startswith(PNG)
        image = Image.open(io.BytesIO(cut))
        assert image.mode == "RGBA" and image.size == picture.size
        assert (_open(cut)[:, :, 3] == np.asarray(matte)).all()

    def test_pixels_that_are_fully_in_or_fully_out_keep_their_exact_colour(self):
        # This is what keeps the watermark in the delivered file: the cut-out
        # may only recolour the partly transparent edge band.
        picture, matte = _noise(), _half_matte(soft=10)
        out = _open(im.attach_matte(_png(picture), _png(matte))[0])
        alpha = out[:, :, 3]
        settled = (alpha == 0) | (alpha == 255)
        assert settled.any() and (~settled).any()
        assert (out[:, :, :3][settled] == np.asarray(picture)[settled]).all()
        assert (alpha == 0).sum() > 1000          # the hidden half is still there, colour intact

    def test_edge_pixels_lose_the_old_backgrounds_colour(self):
        # A red disc on white with a soft rim. On the rim the picture holds
        # pink (red mixed with white); the cut-out must hold red again, or the
        # disc keeps a white fringe on a dark slide.
        size = 160
        yy, xx = np.mgrid[:size, :size]
        distance = np.hypot(xx - size / 2, yy - size / 2)
        alpha = np.clip((50 - distance) / 12 + 0.5, 0, 1)
        red, white = np.array([220, 20, 20], dtype=np.float32), np.array([255, 255, 255], dtype=np.float32)
        mixed = alpha[:, :, None] * red + (1 - alpha[:, :, None]) * white
        picture = Image.fromarray(mixed.round().astype(np.uint8), "RGB")
        matte = Image.fromarray((alpha * 255).round().astype(np.uint8), "L")
        out = _open(im.attach_matte(_png(picture), _png(matte))[0]).astype(np.float32)
        rim = (out[:, :, 3] > 60) & (out[:, :, 3] < 200)
        assert rim.sum() > 200
        before = np.abs(np.asarray(picture, dtype=np.float32)[rim] - red).mean()
        after = np.abs(out[:, :, :3][rim] - red).mean()
        assert before > 40 and after < before / 4, (before, after)

    def test_almost_zero_and_almost_full_are_snapped(self):
        picture = _noise((8, 8))
        values = np.array([0, 1, im.ALPHA_FLOOR, im.ALPHA_FLOOR + 1, 128, im.ALPHA_CEILING - 1, im.ALPHA_CEILING, 255],
                          dtype=np.uint8)
        matte = Image.fromarray(np.tile(values, (8, 1)), "L")
        alpha = _open(im.attach_matte(_png(picture), _png(matte))[0])[0, :, 3]
        assert alpha.tolist() == [0, 0, 0, im.ALPHA_FLOOR + 1, 128, im.ALPHA_CEILING - 1, 255, 255]

    def test_without_a_solid_subject_it_is_not_a_cut_out(self):
        picture = _noise((100, 100))
        blank = Image.new("L", (100, 100), 0)
        assert im.attach_matte(_png(picture), _png(blank)) == (None, im.OUTCOME_NO_SUBJECT)
        assert im.attach_matte(_png(picture), _png(Image.new("L", (100, 100), im.ALPHA_FLOOR))) == \
            (None, im.OUTCOME_NO_SUBJECT)
        # A faint haze over everything, or a ghost of a blob: visible pixels, but nothing solid.
        haze = Image.fromarray(np.random.default_rng(5).integers(5, 31, (100, 100), dtype=np.uint8), "L")
        assert im.attach_matte(_png(picture), _png(haze))[1] == im.OUTCOME_NO_SUBJECT
        ghost = blank.copy()
        ghost.paste(20, (10, 10, 60, 60))
        assert im.attach_matte(_png(picture), _png(ghost))[1] == im.OUTCOME_NO_SUBJECT
        speck = blank.copy()
        speck.paste(255, (0, 0, 4, 4))            # 16 solid pixels of 10,000: under the 0.2 % floor
        assert im.attach_matte(_png(picture), _png(speck))[1] == im.OUTCOME_NO_SUBJECT
        speck.paste(255, (0, 0, 5, 5))            # 25 of 10,000: a small subject is still a subject
        assert im.attach_matte(_png(picture), _png(speck))[1] == im.OUTCOME_TRANSPARENT

    def test_without_a_removed_background_it_is_not_a_cut_out(self):
        picture = _noise((100, 100))
        full = Image.new("L", (100, 100), 255)
        assert im.attach_matte(_png(picture), _png(full)) == (None, im.OUTCOME_NOTHING_REMOVED)
        assert im.attach_matte(_png(picture), _png(Image.new("L", (100, 100), im.ALPHA_CEILING)))[1] == \
            im.OUTCOME_NOTHING_REMOVED
        # One softened pixel, or a sliver removed: the picture still looks opaque.
        dent = full.copy()
        dent.putpixel((3, 3), 200)
        assert im.attach_matte(_png(picture), _png(dent))[1] == im.OUTCOME_NOTHING_REMOVED
        sliver = full.copy()
        sliver.paste(0, (0, 0, 19, 1))            # 19 removed pixels of 10,000: under the 0.2 % floor
        assert im.attach_matte(_png(picture), _png(sliver))[1] == im.OUTCOME_NOTHING_REMOVED
        sliver.paste(0, (0, 0, 20, 1))
        assert im.attach_matte(_png(picture), _png(sliver))[1] == im.OUTCOME_TRANSPARENT

    def test_a_small_file_claiming_an_enormous_matte_is_refused_from_its_header(self, monkeypatch):
        # 22,000 x 22,000 declared in a few hundred bytes. Decoding it would
        # take about 480 MB, so the size is compared before any pixel is read.
        # Pillow's own bomb guard does not cover this: the OCR service raises
        # its limit process-wide (services/ocr.py), which is mimicked here.
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 256_000_000)
        bomb = PNG + chunk(b"IHDR", struct.pack(">IIBBBBB", 22000, 22000, 8, 0, 0, 0, 0)) \
            + chunk(b"IDAT", zlib.compress(b"\x00" * 4096)) + chunk(b"IEND", b"")
        started = time.monotonic()
        with pytest.warns(Image.DecompressionBombWarning), pytest.raises(im.MattingError) as error:
            im.attach_matte(_png(_noise((64, 64))), bomb)
        assert str(error.value) == "matte and picture are not the same size"
        assert time.monotonic() - started < 1.0

    def test_a_hard_matte_changes_no_colour_at_all(self):
        picture, matte = _noise(), _half_matte()
        out = _open(im.attach_matte(_png(picture), _png(matte))[0])
        assert (out[:, :, :3] == np.asarray(picture)).all()

    @pytest.mark.parametrize("picture,matte", [
        (_png(_noise((10, 10))), _png(Image.new("L", (11, 10), 128))),          # not the same size
        (_png(_noise((10, 10))), b"not a png"),
        (b"not a picture", _png(Image.new("L", (10, 10), 128))),
    ])
    def test_a_matte_that_does_not_fit_is_an_error_not_a_guess(self, picture, matte):
        with pytest.raises(im.MattingError):
            im.attach_matte(picture, matte)

    def test_the_matte_must_be_a_png(self):
        jpeg = io.BytesIO()
        Image.new("L", (10, 10), 128).save(jpeg, format="JPEG")
        with pytest.raises(im.MattingError):
            im.attach_matte(_png(_noise((10, 10))), jpeg.getvalue())

    def test_an_edge_band_at_the_picture_border_is_handled(self):
        picture = _noise((40, 30))
        ramp = np.tile(np.linspace(0, 255, 40).round().astype(np.uint8), (30, 1))     # soft from edge to edge
        out = _open(im.attach_matte(_png(picture), _png(Image.fromarray(ramp, "L")))[0])
        assert out.shape == (30, 40, 4)


# ---------------------------------------------------------------------------
# the matting server
# ---------------------------------------------------------------------------

def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestFetchMatte:
    async def test_sends_the_picture_and_the_key_and_returns_the_matte(self):
        seen = {}
        matte = _png(Image.new("L", (4, 4), 9))

        async def handler(request):
            seen.update(url=str(request.url), auth=request.headers.get("authorization"),
                        type=request.headers.get("content-type"), body=request.content)
            return httpx.Response(200, content=matte, headers={"content-type": "image/png"})

        async with _client(handler) as client:
            assert await im.fetch_matte(b"PICTURE", CONFIG, client=client) == matte
        assert seen == {"url": f"{URL}/v1/matte", "auth": "Bearer matting-key", "type": "image/png",
                        "body": b"PICTURE"}

    async def test_no_key_configured_sends_no_authorization(self):
        seen = {}

        async def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, content=_png(Image.new("L", (2, 2))))

        async with _client(handler) as client:
            await im.fetch_matte(b"x", im.MattingConfig(enabled=True, url=URL), client=client)
        assert seen["auth"] is None

    @pytest.mark.parametrize("status,outcome", [
        (503, im.OUTCOME_BUSY),                                   # queue full: load
        (500, im.OUTCOME_FAILED), (502, im.OUTCOME_FAILED), (504, im.OUTCOME_FAILED),   # sickness
        (401, im.OUTCOME_REJECTED), (413, im.OUTCOME_REJECTED), (422, im.OUTCOME_REJECTED),   # this request
    ])
    async def test_a_refusal_is_an_error_that_says_whose_fault_it_was(self, status, outcome):
        async def handler(request):
            return httpx.Response(status, json={"detail": "SECRET-BODY"})

        async with _client(handler) as client:
            with pytest.raises(im.MattingError) as error:
                await im.fetch_matte(b"x", CONFIG, client=client)
        assert error.value.outcome == outcome and str(error.value) == f"matting server returned HTTP {status}"
        assert "SECRET" not in str(error.value)

    async def test_a_200_that_is_not_a_png_is_an_error(self):
        async def handler(request):
            return httpx.Response(200, content=b"<html>proxy error</html>")

        async with _client(handler) as client:
            with pytest.raises(im.MattingError) as error:
                await im.fetch_matte(b"x", CONFIG, client=client)
        assert error.value.outcome == im.OUTCOME_FAILED

    @pytest.mark.parametrize("exc,outcome", [
        (httpx.ConnectError("refused"), im.OUTCOME_FAILED),
        (httpx.ConnectTimeout("no route"), im.OUTCOME_FAILED),
        (httpx.RemoteProtocolError("closed"), im.OUTCOME_FAILED),
        # Connected, then slow: the picture outran the wait. Not sickness.
        (httpx.ReadTimeout("slow"), im.OUTCOME_BUSY), (httpx.WriteTimeout("slow"), im.OUTCOME_BUSY),
        (httpx.PoolTimeout("slow"), im.OUTCOME_BUSY),
    ])
    async def test_unreachable_is_sickness_and_slow_is_not(self, exc, outcome):
        async def handler(request):
            raise exc

        async with _client(handler) as client:
            with pytest.raises(im.MattingError) as error:
                await im.fetch_matte(b"x", CONFIG, client=client)
        assert type(exc).__name__ in str(error.value) and error.value.outcome == outcome

    @pytest.mark.parametrize("declared", [True, False])
    async def test_a_reply_over_the_size_limit_is_dropped_not_buffered(self, monkeypatch, declared):
        monkeypatch.setattr(im, "MAX_MATTE_BYTES", 1000)
        sent = {"chunks": 0}

        async def body():
            for _ in range(50):
                sent["chunks"] += 1
                yield PNG + b"x" * 92                    # 100 bytes a time, 5,000 in all

        async def handler(request):
            headers = {"content-length": "5000"} if declared else {}
            return httpx.Response(200, content=body(), headers=headers)

        async with _client(handler) as client:
            with pytest.raises(im.MattingError) as error:
                await im.fetch_matte(b"x", CONFIG, client=client)
        assert str(error.value) == "matting server reply is too large"
        assert sent["chunks"] <= (1 if declared else 11)    # stopped at the limit, not read to the end

    async def test_a_reply_at_the_size_limit_is_accepted(self, monkeypatch):
        matte = _png(Image.new("L", (4, 4), 9))
        monkeypatch.setattr(im, "MAX_MATTE_BYTES", len(matte))

        async def handler(request):
            return httpx.Response(200, content=matte)

        async with _client(handler) as client:
            assert await im.fetch_matte(b"x", CONFIG, client=client) == matte

    async def test_a_reply_that_drips_forever_is_cut_off_at_the_timeout(self):
        # One byte every 50 ms never trips a per-read timeout; the whole
        # exchange has a deadline of its own.
        async def body():
            yield PNG
            while True:
                await asyncio.sleep(0.05)
                yield b"x"

        async def handler(request):
            return httpx.Response(200, content=body())

        config = im.MattingConfig(enabled=True, url=URL, timeout=0.3)
        started = time.monotonic()
        async with _client(handler) as client:
            with pytest.raises(im.MattingError) as error:
                await im.fetch_matte(b"x", config, client=client)
        assert 0.25 < time.monotonic() - started < 2.0
        assert error.value.outcome == im.OUTCOME_BUSY and str(error.value) == "no matte within 0.3 s"

    async def test_a_later_image_is_given_what_is_left_and_the_message_says_so(self):
        async def handler(request):
            await asyncio.sleep(30)

        started = time.monotonic()
        async with _client(handler) as client:
            with pytest.raises(im.MattingError) as error:
                await im.fetch_matte(b"x", CONFIG, client=client, timeout=0.2)      # CONFIG allows 30 s
        assert time.monotonic() - started < 2.0
        assert str(error.value) == "no matte within 0.2 s (what was left of the 30 s limit)"

    async def test_a_server_that_never_answers_is_cut_off_at_the_timeout(self):
        async def handler(request):
            await asyncio.sleep(30)
            return httpx.Response(200, content=_png(Image.new("L", (2, 2))))

        started = time.monotonic()
        async with _client(handler) as client:
            with pytest.raises(im.MattingError) as error:
                await im.fetch_matte(b"x", im.MattingConfig(enabled=True, url=URL, timeout=0.2), client=client)
        assert time.monotonic() - started < 2.0 and error.value.outcome == im.OUTCOME_BUSY

    def test_the_real_client_follows_the_timeout_and_tls_settings(self, monkeypatch):
        import backend.app.settings as settings_mod

        for setting in (True, False):
            made = {}
            monkeypatch.setattr(settings_mod, "get_settings", lambda: MagicMock(internal_tls_verify=setting))
            monkeypatch.setattr(im.httpx, "AsyncClient", lambda **kwargs: made.update(kwargs) or "client")
            assert im.new_client(im.MattingConfig(enabled=True, url=URL, timeout=7.0)) == "client"
            assert made["verify"] is setting
            assert (made["timeout"].read, made["timeout"].write, made["timeout"].connect) == (7.0, 7.0, 10.0)

    async def test_without_a_client_it_makes_one_and_closes_it(self, monkeypatch):
        made = []

        def factory(config):
            made.append(_client(lambda request: httpx.Response(200, content=_png(Image.new("L", (2, 2))))))
            return made[-1]

        monkeypatch.setattr(im, "new_client", factory)
        assert (await im.fetch_matte(b"x", CONFIG)).startswith(PNG)
        assert len(made) == 1 and made[0].is_closed


class TestMakeTransparent:
    def _server(self, matte=None, status=200):
        async def handler(request):
            if status != 200:
                return httpx.Response(status)
            return httpx.Response(200, content=_png(matte))
        return _client(handler)

    async def test_a_picture_comes_back_cut_out(self):
        picture = _noise()
        async with self._server(_half_matte()) as client:
            cut, outcome = await im.make_transparent(_b64(picture), CONFIG, client=client)
        assert outcome == im.OUTCOME_TRANSPARENT
        out = _open(base64.b64decode(cut))
        assert out.shape == (80, 96, 4) and out[0, 0, 3] == 255 and out[0, 95, 3] == 0

    @pytest.mark.parametrize("status,outcome", [(503, im.OUTCOME_BUSY), (500, im.OUTCOME_FAILED),
                                                (401, im.OUTCOME_REJECTED), (422, im.OUTCOME_REJECTED)])
    async def test_a_server_failure_returns_the_original_picture(self, status, outcome):
        original = _b64(_noise())
        async with self._server(status=status) as client:
            assert await im.make_transparent(original, CONFIG, client=client) == (original, outcome)

    @pytest.mark.parametrize("matte,outcome", [(Image.new("L", (96, 80), 0), im.OUTCOME_NO_SUBJECT),
                                               (Image.new("L", (96, 80), 255), im.OUTCOME_NOTHING_REMOVED),
                                               (Image.new("L", (5, 5), 128), im.OUTCOME_FAILED)])
    async def test_a_useless_matte_returns_the_original_picture(self, matte, outcome):
        original = _b64(_noise())
        async with self._server(matte) as client:
            assert await im.make_transparent(original, CONFIG, client=client) == (original, outcome)

    async def test_it_never_raises(self):
        async def handler(request):
            raise RuntimeError("anything at all")

        async with _client(handler) as client:
            # The gateway's own trouble is "error": it says nothing about the server.
            assert await im.make_transparent("not-base64!!!", CONFIG, client=client) == \
                ("not-base64!!!", im.OUTCOME_ERROR)
            assert await im.make_transparent(None, CONFIG, client=client) == (None, im.OUTCOME_ERROR)
            original = _b64(_noise())
            assert await im.make_transparent(original, CONFIG, client=client) == (original, im.OUTCOME_ERROR)

    async def test_a_matte_that_does_not_fit_is_the_servers_failure(self):
        original = _b64(_noise())
        async with self._server(Image.new("L", (5, 5), 128)) as client:
            assert await im.make_transparent(original, CONFIG, client=client) == (original, im.OUTCOME_FAILED)

    async def test_cancellation_is_not_swallowed(self):
        async def handler(request):
            await asyncio.sleep(30)

        async with _client(handler) as client:
            task = asyncio.ensure_future(im.make_transparent(_b64(_noise()), CONFIG, client=client))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


# ---------------------------------------------------------------------------
# answering `background` on a finished response
# ---------------------------------------------------------------------------

def _registry(state=(None, None)):
    registry = MagicMock()
    registry.matting_server_state = AsyncMock(return_value=state)
    registry.report_live_failure = AsyncMock()
    registry.report_live_success = AsyncMock()
    return registry


@pytest.fixture
def server(monkeypatch):
    """A stand-in matting server for apply_background. ``box["replies"]`` is
    what each call gets, in order (a matte image, or an HTTP status)."""
    box = {"replies": [], "calls": 0, "configs": [], "delay": 0.0}

    async def handler(request):
        box["calls"] += 1                      # counted first: a call is a call, whatever it gets back
        if box["delay"]:
            await asyncio.sleep(box["delay"])
        replies = box["replies"] or [500]
        reply = replies[min(box["calls"], len(replies)) - 1]
        if isinstance(reply, int):
            return httpx.Response(reply)
        if isinstance(reply, BaseException):
            raise reply
        return httpx.Response(200, content=_png(reply))

    def factory(config):
        box["configs"].append(config)
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(im, "new_client", factory)
    return box


def _response(n=1, **extra):
    return {"created": 1, "data": [{"b64_json": _b64(_noise(seed=i)), "revised_prompt": "p"} for i in range(n)], **extra}


def _count(outcome):
    return im.IMAGE_BACKGROUNDS.labels(outcome=outcome)._value.get()


class TestApplyBackground:
    async def test_a_caller_who_did_not_send_the_field_gets_the_response_untouched(self, server):
        response = _response()
        before = {"created": 1, "data": [dict(response["data"][0])]}
        await im.apply_background(None, response, CONFIG, _registry())
        assert response == before and server["calls"] == 0

    @pytest.mark.parametrize("background", ["opaque", "auto"])
    async def test_opaque_and_auto_say_so_and_dial_nothing(self, server, background):
        response = _response(2)
        pictures = [item["b64_json"] for item in response["data"]]
        registry = _registry((7, None))
        await im.apply_background(background, response, CONFIG, registry)
        assert response["background"] == "opaque" and server["calls"] == 0
        registry.matting_server_state.assert_not_awaited()
        registry.report_live_failure.assert_not_awaited()
        assert [item["has_alpha"] for item in response["data"]] == [False, False]
        assert [item["b64_json"] for item in response["data"]] == pictures

    async def test_transparent(self, server):
        server["replies"] = [_half_matte()]
        registry, response = _registry((7, None)), _response()
        before = _count(im.OUTCOME_TRANSPARENT)
        await im.apply_background("transparent", response, CONFIG, registry)
        assert response["background"] == "transparent" and response["data"][0]["has_alpha"] is True
        assert Image.open(io.BytesIO(base64.b64decode(response["data"][0]["b64_json"]))).mode == "RGBA"
        assert response["data"][0]["revised_prompt"] == "p"          # nothing else about the entry changes
        registry.matting_server_state.assert_awaited_once_with(URL)
        registry.report_live_success.assert_awaited_once_with(7)
        registry.report_live_failure.assert_not_awaited()
        assert _count(im.OUTCOME_TRANSPARENT) == before + 1
        assert server["configs"] == [CONFIG]            # one client for the response, built from the settings

    @pytest.mark.parametrize("config", [None, im.MattingConfig(), im.MattingConfig(enabled=False, url=URL),
                                        im.MattingConfig(enabled=True, url="")])
    async def test_off_or_without_a_server_returns_the_opaque_picture(self, server, config):
        registry, response = _registry(), _response()
        original = response["data"][0]["b64_json"]
        before = _count(im.OUTCOME_DISABLED)
        await im.apply_background("transparent", response, config, registry)
        assert response["background"] == "opaque" and response["data"][0] == \
            {"b64_json": original, "revised_prompt": "p", "has_alpha": False}
        assert server["calls"] == 0 and _count(im.OUTCOME_DISABLED) == before + 1
        registry.matting_server_state.assert_not_awaited()

    @pytest.mark.parametrize("reason", ["unhealthy", "disabled", "draining", "circuit open"])
    async def test_a_registered_server_known_to_be_down_is_not_dialed(self, server, reason):
        registry, response = _registry((7, reason)), _response(2)
        before = _count(im.OUTCOME_UNAVAILABLE)
        await im.apply_background("transparent", response, CONFIG, registry)
        assert server["calls"] == 0 and response["background"] == "opaque"
        assert [item["has_alpha"] for item in response["data"]] == [False, False]
        assert _count(im.OUTCOME_UNAVAILABLE) == before + 2
        registry.report_live_failure.assert_not_awaited()             # not dialed: nothing to report

    async def test_a_failed_health_lookup_dials_the_server_anyway(self, server):
        server["replies"] = [_half_matte()]
        registry = _registry()
        registry.matting_server_state = AsyncMock(side_effect=TimeoutError("pool exhausted"))
        response = _response()
        await im.apply_background("transparent", response, CONFIG, registry)
        assert response["background"] == "transparent" and server["calls"] == 1
        registry.report_live_success.assert_not_awaited()             # no backend id to report against

    async def test_an_unregistered_server_is_dialed_and_nothing_is_reported(self, server):
        server["replies"] = [500]
        registry, response = _registry((None, None)), _response()
        await im.apply_background("transparent", response, CONFIG, registry)
        assert server["calls"] == 1 and response["background"] == "opaque"
        registry.report_live_failure.assert_not_awaited()
        registry.report_live_success.assert_not_awaited()

    async def test_without_a_registry_the_server_is_simply_dialed(self, server):
        server["replies"] = [_half_matte()]
        response = _response()
        await im.apply_background("transparent", response, CONFIG, None)
        assert response["background"] == "transparent"

    async def test_a_server_failure_is_reported_once_and_the_rest_are_not_dialed(self, server):
        server["replies"] = [500]
        registry, response = _registry((7, None)), _response(4)
        originals = [item["b64_json"] for item in response["data"]]
        before = _count(im.OUTCOME_FAILED)
        await im.apply_background("transparent", response, CONFIG, registry)
        assert server["calls"] == 1                                   # not four timeouts in a row
        assert [item["b64_json"] for item in response["data"]] == originals
        assert [item["has_alpha"] for item in response["data"]] == [False] * 4
        assert response["background"] == "opaque" and _count(im.OUTCOME_FAILED) == before + 4
        registry.report_live_failure.assert_awaited_once_with(7)
        registry.report_live_success.assert_not_awaited()

    @pytest.mark.parametrize("reply,outcome", [
        (503, im.OUTCOME_BUSY),                             # queue full
        (httpx.ReadTimeout("slow"), im.OUTCOME_BUSY),       # the picture outran the wait behind others
        (401, im.OUTCOME_REJECTED),                         # wrong key: a settings problem
        (422, im.OUTCOME_REJECTED), (413, im.OUTCOME_REJECTED),   # picture too large for the server
    ])
    async def test_load_and_refusals_never_open_the_circuit(self, server, reply, outcome):
        # Three "failures" mark a backend unhealthy for everyone. A queue, a
        # slow answer or a refused request must not count as one.
        server["replies"] = [reply]
        registry, response = _registry((7, None)), _response(2)
        before = _count(outcome)
        await im.apply_background("transparent", response, CONFIG, registry)
        assert server["calls"] == 1 and _count(outcome) == before + 2     # and the rest are not dialed
        assert response["background"] == "opaque"
        registry.report_live_failure.assert_not_awaited()
        registry.report_live_success.assert_not_awaited()

    @pytest.mark.parametrize("reply", [500, 502, 504, httpx.ConnectError("refused"),
                                       Image.new("L", (5, 5), 128)])     # last: a matte that does not fit
    async def test_sickness_is_reported(self, server, reply):
        server["replies"] = [reply]
        registry = _registry((7, None))
        await im.apply_background("transparent", _response(), CONFIG, registry)
        registry.report_live_failure.assert_awaited_once_with(7)

    @pytest.mark.parametrize("matte", [Image.new("L", (96, 80), 0), Image.new("L", (96, 80), 255)])
    async def test_a_matte_that_is_not_a_cut_out_still_shows_the_server_is_well(self, server, matte):
        server["replies"] = [matte]
        registry, response = _registry((7, None)), _response()
        await im.apply_background("transparent", response, CONFIG, registry)
        assert response["background"] == "opaque"
        registry.report_live_success.assert_awaited_once_with(7)
        registry.report_live_failure.assert_not_awaited()

    async def test_a_failure_after_a_success_is_still_a_failure(self, server):
        server["replies"] = [_half_matte(), 500]
        registry, response = _registry((7, None)), _response(3)
        await im.apply_background("transparent", response, CONFIG, registry)
        assert [item["has_alpha"] for item in response["data"]] == [True, False, False]
        assert response["background"] == "opaque"                     # "transparent" only when every image is
        assert server["calls"] == 2
        registry.report_live_failure.assert_awaited_once_with(7)
        registry.report_live_success.assert_not_awaited()

    async def test_the_time_limit_is_for_the_whole_response_not_for_each_image(self, server):
        # Four images, 0.25 s each, 0.6 s allowed in all: two are cut out, the
        # third runs out of time, the fourth is not even sent.
        server["replies"], server["delay"] = [_half_matte()], 0.25
        config = im.MattingConfig(enabled=True, url=URL, timeout=0.6)
        registry, response = _registry((7, None)), _response(4)
        before = _count(im.OUTCOME_BUSY)
        started = time.monotonic()
        await im.apply_background("transparent", response, config, registry)
        elapsed = time.monotonic() - started
        assert [item["has_alpha"] for item in response["data"]] == [True, True, False, False]
        assert 0.55 < elapsed < 1.2 and server["calls"] == 3
        assert _count(im.OUTCOME_BUSY) == before + 2
        registry.report_live_failure.assert_not_awaited()             # slow is not sick
        registry.report_live_success.assert_awaited_once_with(7)

    async def test_time_used_up_between_two_images_is_load_not_sickness(self, server, monkeypatch):
        # The first image is answered in time, but attaching its matte runs
        # past the limit. The second is then not sent at all, and nobody is blamed.
        server["replies"] = [_half_matte()]
        real = im.attach_matte

        def slow_attach(picture, matte):
            time.sleep(0.3)
            return real(picture, matte)

        monkeypatch.setattr(im, "attach_matte", slow_attach)
        config = im.MattingConfig(enabled=True, url=URL, timeout=0.2)
        registry, response = _registry((7, None)), _response(2)
        before = (_count(im.OUTCOME_OUT_OF_TIME), _count(im.OUTCOME_BUSY), _count(im.OUTCOME_FAILED))
        await im.apply_background("transparent", response, config, registry)
        assert [item["has_alpha"] for item in response["data"]] == [True, False]
        assert server["calls"] == 1
        # Not "busy": the server answered promptly; the time went elsewhere.
        assert (_count(im.OUTCOME_OUT_OF_TIME), _count(im.OUTCOME_BUSY), _count(im.OUTCOME_FAILED)) == \
            (before[0] + 1, before[1], before[2])
        registry.report_live_failure.assert_not_awaited()
        registry.report_live_success.assert_awaited_once_with(7)

    @pytest.mark.parametrize("budget", [0, -5.0])
    async def test_a_request_that_has_already_used_its_time_gets_no_cut_out_at_all(self, server, budget):
        # It waited for a worker, or was retried: starting a cut-out now could
        # get a finished, billed picture cut off by the front proxy.
        server["replies"] = [_half_matte()]
        registry, response = _registry((7, None)), _response(2)
        before = _count(im.OUTCOME_OUT_OF_TIME)
        await im.apply_background("transparent", response, CONFIG, registry, budget)
        assert server["calls"] == 0 and response["background"] == "opaque"
        assert [item["has_alpha"] for item in response["data"]] == [False, False]
        assert _count(im.OUTCOME_OUT_OF_TIME) == before + 2
        registry.matting_server_state.assert_not_awaited()
        registry.report_live_failure.assert_not_awaited()

    async def test_a_request_with_little_time_left_gets_only_that(self, server):
        server["replies"], server["delay"] = [_half_matte()], 0.6
        registry, response = _registry((7, None)), _response()
        started = time.monotonic()
        await im.apply_background("transparent", response, CONFIG, registry, 0.2)      # CONFIG allows 30 s
        assert time.monotonic() - started < 0.55 and response["background"] == "opaque"
        registry.report_live_failure.assert_not_awaited()

    async def test_plenty_of_time_left_changes_nothing(self, server):
        server["replies"] = [_half_matte()]
        response = _response()
        await im.apply_background("transparent", response, CONFIG, _registry((7, None)), 600.0)
        assert response["background"] == "transparent"
        # ...and it does not lengthen the configured limit either.
        server["delay"] = 0.6
        response, started = _response(), time.monotonic()
        await im.apply_background("transparent", response, im.MattingConfig(enabled=True, url=URL, timeout=0.2),
                                  _registry((7, None)), 600.0)
        assert time.monotonic() - started < 0.55 and response["background"] == "opaque"

    async def test_each_response_leaves_a_log_line_with_its_outcomes(self, server, monkeypatch):
        server["replies"] = [_half_matte(), 500]
        log = MagicMock()
        monkeypatch.setattr(im, "logger", log)
        await im.apply_background("transparent", _response(3), CONFIG, _registry((7, None)))
        log.info.assert_any_call("image_background_outcome",
                                 outcomes=[im.OUTCOME_TRANSPARENT, im.OUTCOME_FAILED, im.OUTCOME_FAILED])
        log.reset_mock()
        await im.apply_background("opaque", _response(), CONFIG, _registry())
        assert not any(call.args[:1] == ("image_background_outcome",) for call in log.info.call_args_list)

    async def test_has_alpha_is_per_image(self, server):
        server["replies"] = [_half_matte(), Image.new("L", (96, 80), 255), _half_matte()]
        registry, response = _registry((7, None)), _response(3)
        originals = [item["b64_json"] for item in response["data"]]
        await im.apply_background("transparent", response, CONFIG, registry)
        assert [item["has_alpha"] for item in response["data"]] == [True, False, True]
        assert response["data"][1]["b64_json"] == originals[1]        # the model kept everything: original back
        assert response["background"] == "opaque" and server["calls"] == 3
        registry.report_live_success.assert_awaited_once_with(7)      # the server answered every time

    async def test_an_entry_without_bytes_cannot_be_cut_out(self, server):
        server["replies"] = [_half_matte()]
        response = {"created": 1, "data": [{"url": "/images/x.png"}]}
        registry = _registry((7, None))
        before = _count(im.OUTCOME_ERROR)
        await im.apply_background("transparent", response, CONFIG, registry)
        assert response["data"][0] == {"url": "/images/x.png", "has_alpha": False}
        assert response["background"] == "opaque" and server["calls"] == 0
        assert _count(im.OUTCOME_ERROR) == before + 1
        # The server was never asked, so this says nothing about its health.
        registry.report_live_failure.assert_not_awaited()
        registry.report_live_success.assert_not_awaited()

    async def test_no_images_is_opaque_not_a_crash(self, server):
        for response in ({"created": 1, "data": []}, {"created": 1}):
            await im.apply_background("transparent", response, CONFIG, _registry())
            assert response["background"] == "opaque"

    async def test_bookkeeping_that_fails_does_not_fail_the_image(self, server):
        server["replies"] = [_half_matte()]
        registry = _registry((7, None))
        registry.report_live_success = AsyncMock(side_effect=RuntimeError("db down"))
        response = _response()
        await im.apply_background("transparent", response, CONFIG, registry)
        assert response["background"] == "transparent"

    async def test_whatever_breaks_the_caller_still_gets_consistent_opaque_images(self, monkeypatch):
        async def explode(*args, **kwargs):
            raise RuntimeError("bug")

        monkeypatch.setattr(im, "_cut_out_all", explode)
        response = _response(2)
        originals = [item["b64_json"] for item in response["data"]]
        before = _count(im.OUTCOME_ERROR)
        await im.apply_background("transparent", response, CONFIG, _registry())
        assert _count(im.OUTCOME_ERROR) == before + 2
        assert [item["b64_json"] for item in response["data"]] == originals
        assert [item["has_alpha"] for item in response["data"]] == [False, False]
        assert response["background"] == "opaque"


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

class TestConfig:
    async def _load(self, monkeypatch, stored):
        from backend.app.db import crud

        async def fake(db, key, default=None):
            return stored.get(key, default)

        monkeypatch.setattr(crud, "get_config_json", fake)
        return await im.load_config(object())

    async def test_off_by_default(self, monkeypatch):
        config = await self._load(monkeypatch, {})
        assert config == im.MattingConfig(enabled=False, url="", api_key=None, timeout=30.0)
        assert config.usable is False

    async def test_stored_values(self, monkeypatch):
        config = await self._load(monkeypatch, {
            "img.transparent_enabled": True, "img.matting_url": " https://h:8005/ ",
            "img.matting_api_key": " k ", "img.matting_timeout": 12})
        assert config == im.MattingConfig(enabled=True, url="https://h:8005", api_key="k", timeout=12.0)
        assert config.usable is True

    async def test_on_without_a_url_is_not_usable(self, monkeypatch):
        assert (await self._load(monkeypatch, {"img.transparent_enabled": True})).usable is False

    async def test_junk_in_the_settings_is_tolerated(self, monkeypatch):
        config = await self._load(monkeypatch, {"img.transparent_enabled": 1, "img.matting_url": 42,
                                                "img.matting_api_key": None, "img.matting_timeout": "soon"})
        assert config.url == "" and config.api_key is None and config.timeout == 30.0

    @pytest.mark.parametrize("value,expected", [(5, 5.0), ("12", 12.0), (100, 100.0), (0, 30.0), (-1, 30.0),
                                                (None, 30.0), ("x", 30.0), (True, 30.0),
                                                # above the ceiling: someone wanted a long limit, so the ceiling
                                                (101, 100.0), (300, 100.0), ("250", 100.0), (float("inf"), 100.0),
                                                # not a number at all, or one no float can hold
                                                (float("nan"), 30.0), (10 ** 400, 30.0)])
    def test_clean_timeout(self, value, expected):
        assert im.clean_timeout(value) == expected

    def test_the_longest_cut_out_still_fits_inside_the_front_proxys_limit(self):
        # Generation may use its whole budget before the cut-out starts; the
        # two together must stay under the gateway nginx's proxy_read_timeout.
        from backend.app.settings import Settings

        generation = Settings.model_fields["backend_image_request_timeout"].default
        assert generation + im.MAX_TIMEOUT < 720
        # ...and since a request can also wait for a worker and retry, the
        # cut-out is held to a budget counted from the request's first moment.
        assert im.DEFAULT_TIMEOUT < im.MAX_TIMEOUT < im.REQUEST_BUDGET < 720

    @pytest.mark.parametrize("key", ["abc", "k" * 512, "a-b_c.d~e!f", "0123456789abcdef" * 4,
                                     "pass phrase 1"])        # a space inside is a legal header value
    def test_a_key_that_can_be_sent(self, key):
        assert im.validate_api_key(key) is None

    @pytest.mark.parametrize("key", ["", "k" * 513, " leading", "trailing ", "tab\tkey", "smart\u201cquote",
                                     "nbsp\u00a0key", "caf\u00e9", "line\nbreak", None, 5])
    def test_a_key_that_cannot(self, key):
        assert im.validate_api_key(key)

    async def test_a_stored_key_that_cannot_be_sent_is_treated_as_no_key_and_says_so(self, monkeypatch):
        # No key means a 401 from the server, which is counted as "rejected"
        # and stops the dialing: a clearer signal than an encoding error on every image.
        log = MagicMock()
        monkeypatch.setattr(im, "logger", log)
        config = await self._load(monkeypatch, {"img.transparent_enabled": True, "img.matting_url": URL,
                                                "img.matting_api_key": "smart\u201cquote"})
        assert config.api_key is None and config.usable is True
        log.warning.assert_called_once_with("image_matting_stored_key_unusable_sending_none")
        assert "smart" not in repr(log.mock_calls)                   # the key itself is never logged

    async def test_a_usable_stored_key_is_kept_without_comment(self, monkeypatch):
        log = MagicMock()
        monkeypatch.setattr(im, "logger", log)
        config = await self._load(monkeypatch, {"img.matting_url": URL, "img.matting_api_key": " pass phrase 1 "})
        assert config.api_key == "pass phrase 1"
        log.warning.assert_not_called()

    @pytest.mark.parametrize("url", ["https://aspen4.hpc.uidaho.edu:8005", "http://127.0.0.1:18006"])
    def test_a_base_address_is_a_valid_server_url(self, url):
        assert im.validate_server_url(url) is None

    @pytest.mark.parametrize("url", [
        "aspen4:8005", "ftp://h/x", "https://", "https://user:pw@h:1", "https://h/?a=1", "https://h/#x",
        "https://[bad", "javascript:alert(1)",
        "https://h:notaport", "https://h:99999",                      # a port that is not one
        "https://ho st", "https://h\r\nX: y", "https://h\t", " https://h",   # spaces and control characters
        "https://m:18006/v1/matte", "https://m:18006/x",              # the endpoint pasted instead of the server
    ])
    def test_anything_else_is_refused(self, url):
        assert im.validate_server_url(url)

    def test_a_trailing_slash_is_still_the_base_address(self):
        assert im.validate_server_url("https://h:8005/") is None

    def test_the_key_is_not_in_the_settings_repr(self):
        assert "matting-key" not in repr(CONFIG) and "matting-key" not in str(CONFIG)
        assert CONFIG.api_key == "matting-key"


# ---------------------------------------------------------------------------
# the request path
# ---------------------------------------------------------------------------

def _image_request(**over):
    from backend.app.core.canonical_schemas import CanonicalImageRequest

    return CanonicalImageRequest(model="flux", prompt="a hedgehog", **over)


def _service(monkeypatch, stored, backend_reply=None):
    """An InferenceService with only what the image path touches, a fake
    diffusion backend, and app_config answered from ``stored``."""
    import backend.app.services.inference as inference_mod
    from backend.app.db import session as session_mod
    from backend.app.services import image_watermark
    from backend.app.services.inference import InferenceService

    seen = {}

    async def handler(request):
        import json

        seen["url"], seen["payload"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json=backend_reply or {"created": 1, "data": [{"b64_json": _b64(_noise())}]})

    async def config(db, key, default=None):
        seen.setdefault("config_reads", []).append(key)
        if isinstance(stored, Exception):
            raise stored
        return stored.get(key, default)

    class _Db:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *exc):
            return False

    async def marked(picture, text):
        seen.setdefault("watermarked", []).append(picture)
        return picture

    monkeypatch.setattr(inference_mod.crud, "get_config_json", config)
    monkeypatch.setattr(session_mod, "get_async_db_context", lambda: _Db())
    monkeypatch.setattr(image_watermark, "apply_watermark_b64", marked)
    svc = InferenceService.__new__(InferenceService)
    svc._settings = MagicMock(backend_image_request_timeout=600)
    svc._make_inference_client = lambda read_timeout=None: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    svc._registry = _registry()
    svc._matting_config = None
    return svc, seen


_ON = {"img.transparent_enabled": True, "img.matting_url": URL, "img.matting_api_key": "k"}


class TestProxyImageRequest:
    async def test_without_the_field_nothing_about_the_request_changes(self, monkeypatch):
        svc, seen = _service(monkeypatch, {"img.watermark_enabled": False, **_ON})
        await svc._proxy_image_request(_image_request(response_format="url"), MagicMock(url="http://flux"))
        assert seen["payload"]["response_format"] == "url" and "background" not in seen["payload"]
        assert not any(key.startswith("img.matting") or key == "img.transparent_enabled"
                       for key in seen["config_reads"])
        assert svc._matting_config is None

    async def test_transparent_needs_the_bytes_even_with_the_watermark_off(self, monkeypatch):
        svc, seen = _service(monkeypatch, {"img.watermark_enabled": False, **_ON})
        await svc._proxy_image_request(_image_request(response_format="url", background="transparent"),
                                       MagicMock(url="http://flux"))
        assert seen["payload"]["response_format"] == "b64_json"
        assert "background" not in seen["payload"]                    # never forwarded to the image model
        assert svc._matting_config == im.MattingConfig(enabled=True, url=URL, api_key="k", timeout=30.0)

    async def test_transparent_with_the_feature_off_leaves_the_format_alone(self, monkeypatch):
        svc, seen = _service(monkeypatch, {"img.watermark_enabled": False, "img.matting_url": URL})
        await svc._proxy_image_request(_image_request(response_format="url", background="transparent"),
                                       MagicMock(url="http://flux"))
        assert seen["payload"]["response_format"] == "url" and svc._matting_config.usable is False

    @pytest.mark.parametrize("background", ["opaque", "auto"])
    async def test_opaque_and_auto_read_no_matting_settings(self, monkeypatch, background):
        svc, seen = _service(monkeypatch, {"img.watermark_enabled": False, **_ON})
        await svc._proxy_image_request(_image_request(response_format="url", background=background),
                                       MagicMock(url="http://flux"))
        assert seen["payload"]["response_format"] == "url" and svc._matting_config is None

    async def test_a_settings_read_that_fails_means_an_opaque_picture_not_a_failed_one(self, monkeypatch):
        svc, seen = _service(monkeypatch, RuntimeError("db down"))
        out = await svc._proxy_image_request(_image_request(background="transparent"), MagicMock(url="http://flux"))
        assert out["data"][0]["b64_json"] and svc._matting_config is None

    async def test_the_picture_is_watermarked_before_any_cut_out(self, monkeypatch):
        # The proxy step watermarks; the cut-out happens later, on its output.
        svc, seen = _service(monkeypatch, {"img.watermark_enabled": True, **_ON})
        out = await svc._proxy_image_request(_image_request(background="transparent"), MagicMock(url="http://flux"))
        assert seen["watermarked"] == [out["data"][0]["b64_json"]]
        assert "has_alpha" not in out["data"][0] and "background" not in out


class TestImageGeneration:
    def _wire(self, svc, response, order):
        async def complete(*args, **kwargs):
            order.append("completed")

        async def proxied(request, job, user, **kwargs):
            order.append("generated")
            return response, MagicMock(id=45)

        svc._check_quota = AsyncMock()
        svc._create_request_record = AsyncMock(return_value=MagicMock(request_uuid="u"))
        svc._scheduler = MagicMock()
        svc._proxy_with_retry = proxied
        svc._complete_request = complete
        svc._fail_request = AsyncMock()

    async def test_the_cut_out_happens_after_the_worker_slot_is_released(self, monkeypatch, server):
        server["replies"] = [_half_matte()]
        svc, _ = _service(monkeypatch, {})
        svc._matting_config = CONFIG
        svc._registry = _registry((7, None))
        order, response = [], _response()
        self._wire(svc, response, order)
        real = im.apply_background

        async def applied(*args, **kwargs):
            order.append("cut out")
            return await real(*args, **kwargs)

        monkeypatch.setattr(im, "apply_background", applied)
        out = await svc.image_generation(_image_request(background="transparent"), MagicMock(id=1), MagicMock(id=2),
                                         MagicMock())
        assert order == ["generated", "completed", "cut out"]
        assert out["background"] == "transparent" and out["data"][0]["has_alpha"] is True
        svc._fail_request.assert_not_awaited()

    async def test_a_request_without_the_field_gets_the_old_response_shape(self, monkeypatch, server):
        svc, _ = _service(monkeypatch, {})
        order, response = [], _response()
        self._wire(svc, response, order)
        out = await svc.image_generation(_image_request(), MagicMock(id=1), MagicMock(id=2), MagicMock())
        assert set(out) == {"created", "data"} and set(out["data"][0]) == {"b64_json", "revised_prompt"}
        assert server["calls"] == 0

    async def test_a_broken_cut_out_step_still_returns_the_picture(self, monkeypatch, server):
        svc, _ = _service(monkeypatch, {})
        order, response = [], _response()
        self._wire(svc, response, order)

        async def explode(*args, **kwargs):
            raise RuntimeError("bug")

        monkeypatch.setattr(im, "apply_background", explode)
        out = await svc.image_generation(_image_request(background="transparent"), MagicMock(id=1), MagicMock(id=2),
                                         MagicMock())
        assert out["background"] == "opaque" and out["data"][0]["has_alpha"] is False
        assert out["data"][0]["b64_json"] and order == ["generated", "completed"]
        svc._fail_request.assert_not_awaited()                         # completed stays completed

    async def test_the_cut_out_is_given_what_the_request_has_left(self, monkeypatch, server):
        svc, _ = _service(monkeypatch, {})
        svc._matting_config = CONFIG
        seen = {}

        async def applied(background, response, config, registry, budget):
            seen["budget"] = budget

        monkeypatch.setattr(im, "apply_background", applied)
        request = _image_request(background="transparent")
        await svc._apply_image_background(request, _response(), time.monotonic() - 100.0)
        assert im.REQUEST_BUDGET - 101.0 < seen["budget"] <= im.REQUEST_BUDGET - 100.0
        await svc._apply_image_background(request, _response(), time.monotonic() - 800.0)
        assert seen["budget"] < 0                                     # nothing left: no cut-out is attempted

    async def test_a_request_that_ran_long_returns_its_picture_without_a_cut_out(self, monkeypatch, server):
        server["replies"] = [_half_matte()]
        svc, _ = _service(monkeypatch, {})
        svc._matting_config = CONFIG
        svc._registry = _registry((7, None))
        response = _response()
        await svc._apply_image_background(_image_request(background="transparent"), response,
                                          time.monotonic() - im.REQUEST_BUDGET - 1)
        assert server["calls"] == 0 and response["background"] == "opaque"
        assert response["data"][0]["has_alpha"] is False

    async def test_the_clock_starts_before_quota_and_queueing(self, monkeypatch, server):
        svc, _ = _service(monkeypatch, {})
        order, response = [], _response()
        self._wire(svc, response, order)
        seen = {}

        async def slow_quota(*args, **kwargs):
            await asyncio.sleep(0.2)

        async def applied(request, response, started=None):
            seen["waited"] = time.monotonic() - started

        svc._check_quota = slow_quota
        svc._apply_image_background = applied
        await svc.image_generation(_image_request(background="transparent"), MagicMock(id=1), MagicMock(id=2),
                                   MagicMock())
        assert seen["waited"] >= 0.2

    async def test_a_caller_who_leaves_during_the_cut_out_leaves_a_completed_request(self, monkeypatch, server):
        # The picture was generated and the row completed; a disconnect now
        # must not flip it to failed.
        svc, _ = _service(monkeypatch, {})
        svc._matting_config = CONFIG
        order, response = [], _response()
        self._wire(svc, response, order)
        started = asyncio.Event()

        async def slow(*args, **kwargs):
            started.set()
            await asyncio.sleep(30)

        monkeypatch.setattr(im, "apply_background", slow)
        task = asyncio.ensure_future(svc.image_generation(
            _image_request(background="transparent"), MagicMock(id=1), MagicMock(id=2), MagicMock()))
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert order == ["generated", "completed"]
        svc._fail_request.assert_not_awaited()

    async def test_the_requested_background_is_kept_in_the_audit_row(self):
        source = (_APP / "services" / "inference.py").read_text()
        line = next(l for l in source.splitlines() if '"num_inference_steps", "guidance_scale", "seed"' in l)
        assert '"background"' in line


class TestSchemas:
    def test_the_new_fields_are_absent_unless_set(self):
        from backend.app.core.canonical_schemas import CanonicalImageData, CanonicalImageResponse

        plain = CanonicalImageResponse(created=1, data=[CanonicalImageData(b64_json="x")])
        assert plain.model_dump(exclude_none=True) == {"created": 1, "data": [{"b64_json": "x"}]}

    def test_the_request_carries_background_but_the_backend_payload_does_not(self):
        from backend.app.core.translators.diffusion_out import DiffusionOutTranslator

        request = _image_request(background="transparent")
        assert request.background == "transparent" and _image_request().background is None
        assert "background" not in DiffusionOutTranslator.translate_image_request(request)


# ---------------------------------------------------------------------------
# the endpoints
# ---------------------------------------------------------------------------

class TestEndpoints:
    async def _prepare(self, monkeypatch, params):
        import backend.app.api.v1_openai as api
        from backend.app.services import feature_access

        judged = []

        async def config(db, key, default=None):
            return {"img.policy": "no violence"}.get(key, default)

        async def judge(**kwargs):
            judged.append(kwargs["prompt"])
            return MagicMock(passed=True, to_dict=lambda: {"passed": True})

        registry = MagicMock()
        registry.resolve_alias = lambda name: (name, None)
        monkeypatch.setattr(api.crud, "get_config_json", config)
        monkeypatch.setattr(feature_access, "image_generation_allowed", AsyncMock(return_value=True))
        monkeypatch.setattr("backend.app.services.image_policy.evaluate_prompt", judge)
        monkeypatch.setattr(api, "get_registry", lambda: registry)
        monkeypatch.setattr(api, "model_availability", AsyncMock(return_value=api.AVAILABLE))
        canonical = await api._prepare_image_canonical(
            db=object(), request=MagicMock(), user=MagicMock(id=1), api_key=MagicMock(id=2),
            request_id="img-1", endpoint="/v1/images/generations", params=params)
        return canonical, judged

    @pytest.mark.parametrize("sent,carried", [("transparent", "transparent"), ("Opaque", "opaque"),
                                              ("auto", "auto"), (None, None), ("", None)])
    async def test_the_value_reaches_the_request(self, monkeypatch, sent, carried):
        params = {"prompt": "a hedgehog", "size": "1024x1024"}
        if sent is not None:
            params["background"] = sent
        canonical, _ = await self._prepare(monkeypatch, params)
        assert canonical.background == carried

    @pytest.mark.parametrize("value", ["clear", "none", "transparant", True, 5, ["transparent"]])
    async def test_an_unknown_value_does_not_fail_the_request(self, monkeypatch, value):
        # 2.9.89 ignored the field entirely; a request that worked then must not become a 400.
        canonical, judged = await self._prepare(monkeypatch, {"prompt": "a hedgehog", "background": value})
        assert canonical.background == "auto" and judged == ["a hedgehog"]

    def test_the_edits_form_takes_the_field_and_passes_it_on(self):
        import inspect

        import backend.app.api.v1_openai as api

        parameter = inspect.signature(api.image_edits).parameters["background"]
        assert parameter.default.default is None                      # an optional form field
        source = inspect.getsource(api.image_edits)
        assert '"background": background,' in source

    def test_the_playground_takes_the_field_and_never_refuses_a_value(self):
        source = (_APP / "dashboard" / "images.py").read_text()
        tree = ast.parse(source)
        generate = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                        and n.name == "images_api_generate")
        body = ast.get_source_segment(source, generate)
        assert 'background = parse_background(body.get("background"))' in body
        assert "background=background," in body
        # Nothing about the value is an error: no branch between reading it and loading the settings.
        between = body[body.index("background = parse_background("):body.index("# Load config for defaults/guardrails")]
        assert "return" not in between and "except" not in between


class TestPlayground:
    def test_the_option_is_offered_only_when_the_feature_is_on(self):
        html = (_APP / "dashboard" / "templates" / "user" / "images.html").read_text()
        box = html.index('id="transparentBg"')
        assert html.rindex("{% if transparent_enabled %}", 0, box) > html.rindex("{% endif %}", 0, box)
        source = (_APP / "dashboard" / "images.py").read_text()
        assert '"transparent_enabled": bool(await crud.get_config_json(db, "img.transparent_enabled", False))' in source

    def test_it_sends_the_field_and_says_when_the_background_was_not_removed(self):
        html = (_APP / "dashboard" / "templates" / "user" / "images.html").read_text()
        assert "if (wantTransparent) body.background = 'transparent';" in html
        assert "imgData.has_alpha === true" in html and "background could not be removed" in html
        # The checkerboard backdrop only for a picture that really has transparency.
        assert "resultImg.classList.toggle('has-alpha', imgData.has_alpha === true);" in html
        # Without the checkbox on the page (feature off) nothing is sent.
        assert "!!(transparentBox && transparentBox.checked)" in html


# ---------------------------------------------------------------------------
# admin settings, the engine, the migration, the docs
# ---------------------------------------------------------------------------

_FORM = {"action": "save_config", "quota_tokens_per_image": "1000", "watermark_enabled": "on",
         "watermark_text": "UIMR-AI", "enabled": "on", "default_model": "m", "default_size": "1024x1024",
         "allowed_sizes": "", "policy": "", "judge_model": "", "judge_model_secondary": ""}
_CARD = {"matting_settings": "1", "matting_url": "https://m:8010", "matting_api_key": "", "matting_timeout": "30"}


async def _save(fields):
    """POST the admin image settings form for real; returns (redirect, {setting: written value}, audit)."""
    from unittest.mock import patch

    from starlette.datastructures import FormData

    import backend.app.dashboard.routes as routes
    from backend.app.services import feature_access

    writes = {}

    async def set_config(db, key, value, description=None):
        writes[key] = value

    request = MagicMock()
    request.form = AsyncMock(return_value=FormData({**_FORM, **fields}))
    db = MagicMock()
    db.commit = AsyncMock()
    admin = MagicMock()
    admin.group.is_admin = True
    with patch.object(routes, "get_session_user_id", return_value=1), \
            patch.object(routes.crud, "get_user_by_id", AsyncMock(return_value=admin)), \
            patch.object(routes.crud, "set_config", set_config), \
            patch.object(routes.crud, "log_admin_action", AsyncMock()) as audit, \
            patch.object(routes, "get_client_ip", return_value="10.0.0.1"), \
            patch.object(feature_access, "refresh_feature_access_cache", AsyncMock()):
        response = await routes.admin_images_config_post(request, db)
    return response.headers["location"], writes, (audit.await_args.kwargs if audit.await_args else None)


def _matting(writes):
    return {key: value for key, value in writes.items() if "matting" in key or "transparent" in key}


class TestAdminForm:
    async def test_switching_it_on_stores_the_switch_the_url_and_the_timeout(self):
        location, writes, audit = await _save({**_CARD, "transparent_enabled": "on", "matting_url": " https://m:8010/ ",
                                               "matting_timeout": "45"})
        assert "success" in location
        assert _matting(writes) == {"img.transparent_enabled": True, "img.matting_url": "https://m:8010",
                                    "img.matting_timeout": 45}
        assert audit["after_value"]["transparent_backgrounds"] is True
        assert audit["after_value"]["matting_url"] == "https://m:8010"

    async def test_switching_it_off_keeps_the_url(self):
        _, writes, _ = await _save(_CARD)
        assert _matting(writes) == {"img.transparent_enabled": False, "img.matting_url": "https://m:8010",
                                    "img.matting_timeout": 30}

    async def test_a_blank_key_field_keeps_the_stored_key(self):
        _, writes, _ = await _save({**_CARD, "transparent_enabled": "on"})
        assert "img.matting_api_key" not in writes

    async def test_a_new_key_is_stored_trimmed_and_not_put_in_the_audit_log(self):
        _, writes, audit = await _save({**_CARD, "matting_api_key": "  s3cret-key  "})
        assert writes["img.matting_api_key"] == "s3cret-key"
        assert "s3cret" not in repr(audit)

    async def test_remove_the_stored_key(self):
        _, writes, _ = await _save({**_CARD, "matting_api_key_clear": "on"})
        assert writes["img.matting_api_key"] == ""
        _, writes, _ = await _save({**_CARD, "matting_api_key_clear": "on", "matting_api_key": "typed-too"})
        assert writes["img.matting_api_key"] == ""            # "remove" wins over a typed value

    @pytest.mark.parametrize("fields,needle", [
        ({"transparent_enabled": "on", "matting_url": ""}, "need+a+matting+server+URL"),
        ({"matting_url": "m:8010"}, "http"),
        ({"matting_url": "https://m:notaport"}, "port"),
        ({"matting_url": "https://m:8010/v1/matte"}, "base+address"),
        ({"matting_url": "https://a:b@m"}, "base+address"),
        ({"matting_url": "https://ho st"}, "spaces"),
        ({"matting_timeout": "0"}, "timeout"), ({"matting_timeout": "101"}, "timeout"),
        ({"matting_timeout": "30.5"}, "timeout"), ({"matting_timeout": "soon"}, "timeout"),
        ({"matting_api_key": "smart\u201cquote"}, "printable+ASCII"),
        ({"matting_api_key": "tab\tinside"}, "printable+ASCII"),
    ])
    async def test_a_bad_value_is_refused_and_nothing_at_all_is_saved(self, fields, needle):
        location, writes, audit = await _save({**_CARD, **fields})
        assert "error=" in location and needle in location
        assert writes == {} and audit is None

    async def test_the_longest_allowed_timeout_is_accepted(self):
        _, writes, _ = await _save({**_CARD, "matting_timeout": str(int(im.MAX_TIMEOUT))})
        assert writes["img.matting_timeout"] == int(im.MAX_TIMEOUT)

    async def test_a_bad_typed_key_does_not_matter_when_the_key_is_being_removed(self):
        location, writes, _ = await _save({**_CARD, "matting_api_key_clear": "on",
                                           "matting_api_key": "smart\u201cquote"})
        assert "success" in location and writes["img.matting_api_key"] == ""

    async def test_a_key_with_a_space_inside_is_saved(self):
        _, writes, _ = await _save({**_CARD, "matting_api_key": " pass phrase 1 "})
        assert writes["img.matting_api_key"] == "pass phrase 1"

    async def test_a_blank_timeout_means_the_default(self):
        _, writes, _ = await _save({**_CARD, "matting_timeout": ""})
        assert writes["img.matting_timeout"] == 30

    async def test_a_form_without_the_card_leaves_these_settings_alone(self):
        # A tab opened before this card existed: "absent" must not be read as "off".
        location, writes, audit = await _save({})
        assert "success" in location and _matting(writes) == {}
        assert writes["img.enabled"] is True                 # the rest of the form still saves
        assert "transparent_backgrounds" not in audit["after_value"]

    async def test_a_bad_watermark_text_still_saves_nothing_of_this_card(self):
        location, writes, _ = await _save({**_CARD, "transparent_enabled": "on", "watermark_text": "waytoolongtext"})
        assert "error=" in location and writes == {}

    async def test_a_hostile_url_is_not_reflected_into_the_redirect(self):
        location, writes, _ = await _save({**_CARD, "matting_url": 'https://h/"><script>alert(1)</script>'})
        assert "script" not in location and writes == {}


class TestAdminSettings:
    def _segments(self):
        source = (_APP / "dashboard" / "routes.py").read_text()
        get = source[source.index("async def admin_images_config("):source.index("async def admin_images_config_post(")]
        post = source[source.index("async def admin_images_config_post("):]
        post = post[:post.index('elif action == "set_user_override"')]
        return get, post

    def test_the_page_is_told_whether_a_key_is_stored_never_the_key(self):
        get, _ = self._segments()
        assert '"matting_key_set": bool(await crud.get_config_json(db, "img.matting_api_key", ""))' in get
        assert get.count('"img.matting_api_key"') == 1
        html = (_APP / "dashboard" / "templates" / "admin" / "images_config.html").read_text()
        field = re.search(r'<input type="password"[^>]*name="matting_api_key"[^>]*>', html).group(0)
        assert 'value=""' in field
        assert "matting_api_key }}" not in html and "{{ matting_api_key" not in html

    def test_the_form_has_every_field(self):
        html = (_APP / "dashboard" / "templates" / "admin" / "images_config.html").read_text()
        for name in ("matting_settings", "transparent_enabled", "matting_url", "matting_api_key",
                     "matting_api_key_clear", "matting_timeout"):
            assert f'name="{name}"' in html, name

    def test_bad_values_are_refused_before_anything_is_written(self):
        _, post = self._segments()
        first_write = post.index("await crud.set_config(")
        for needle in ("validate_server_url(matting_url)", "Transparent backgrounds need a matting server URL.",
                       "Matting timeout must be 1 to"):
            assert post.index(needle) < first_write, needle
        refusal = post.index("if matting_error:")
        assert "return RedirectResponse(" in post[refusal:first_write]

    def test_a_blank_key_field_keeps_the_stored_key(self):
        _, post = self._segments()
        block = post[post.index('matting_key = (form.get("matting_api_key")'):]
        block = block[:block.index("await crud.set_config(db, \"img.quota_tokens_per_image\"")]
        assert 'if "matting_api_key_clear" in form:' in block and "elif matting_key:" in block
        assert block.count('set_config(db, "img.matting_api_key"') == 2      # clear, or replace; never blank by default


class TestEngine:
    def test_matting_servers_are_model_less(self):
        from backend.app.db.models import MODELLESS_ENGINES, BackendEngine

        assert BackendEngine.MATTING.value == "matting" and BackendEngine.MATTING in MODELLESS_ENGINES

    def test_the_registry_health_checks_one_like_a_decision_server(self):
        from backend.app.core.telemetry.adapters.decision import DecisionAdapter
        from backend.app.core.telemetry.registry import BackendRegistry
        from backend.app.db.models import BackendEngine

        registry = MagicMock()
        registry._settings.backend_health_timeout = 5
        backend = MagicMock(url="https://h:8005", engine=BackendEngine.MATTING)
        adapter = BackendRegistry._create_adapter(registry, backend)
        assert isinstance(adapter, DecisionAdapter) and adapter.base_url == "https://h:8005"

    async def _state(self, rows, url, open_circuits=(), db="request-session"):
        from backend.app.core.telemetry import registry as registry_mod
        from backend.app.core.telemetry.registry import BackendRegistry
        from backend.app.db.models import BackendStatus
        from unittest.mock import patch

        servers = [(i, u, BackendStatus(s)) for i, u, s in rows]
        registry = MagicMock()
        registry.is_backend_available = AsyncMock(side_effect=lambda backend_id: backend_id not in open_circuits)
        lookup = AsyncMock(return_value=servers)
        decisions = AsyncMock(return_value=[(99, url, BackendStatus("unhealthy"))])
        with patch.object(registry_mod.crud, "get_matting_servers", lookup), \
                patch.object(registry_mod.crud, "get_decision_servers", decisions):
            state = await BackendRegistry.matting_server_state(registry, url, db)
        lookup.assert_awaited_once_with(db)
        decisions.assert_not_awaited()                 # a decision server at the same URL is another matter
        return state

    async def test_server_states(self):
        assert await self._state([], URL) == (None, None)
        assert await self._state([(9, "https://other:8005", "healthy")], URL) == (None, None)
        assert await self._state([(9, URL + "/", "healthy")], URL) == (9, None)
        assert await self._state([(9, URL.upper().replace("HTTPS", "https"), "unknown")], URL) == (9, None)
        for status in ("unhealthy", "disabled", "draining"):
            assert await self._state([(9, URL, status)], URL) == (9, status)
        assert await self._state([(9, URL, "healthy")], URL, open_circuits={9}) == (9, "circuit open")

    def test_the_lookup_query_is_for_the_matting_engine(self):
        source = (_APP / "db" / "crud.py").read_text()
        block = source.split("async def get_matting_servers", 1)[1].split("\nasync def ", 1)[0]
        assert "Backend.engine == BackendEngine.MATTING" in block

    def test_the_orm_enum_and_the_migration_agree(self):
        from backend.app.db.models import BackendEngine

        versions = _APP / "db" / "migrations" / "versions"
        path = versions / "20261006_000000_089_add_matting_backend_engine.py"
        spec = importlib.util.spec_from_file_location("migration_089", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.revision == "089" and module.down_revision == "088"
        assert callable(module.upgrade) and callable(module.downgrade)
        # Same values, same order: a mismatch would make the ORM write a value the column rejects.
        assert module.NEW_ENGINE == ",".join(f"'{e.value}'" for e in BackendEngine)
        assert module.NEW_ENGINE == module.OLD_ENGINE + ",'matting'"
        previous = (versions / "20261004_000000_087_add_decision_backend_engine.py").read_text()
        assert f'NEW_ENGINE = "{module.OLD_ENGINE}"' in previous       # 089 starts where 087 ended
        assert sorted(p.name for p in versions.glob("*_089_*.py")) == [path.name]

    def test_admin_can_register_and_edit_one(self):
        html = (_APP / "dashboard" / "templates" / "admin" / "backends.html").read_text()
        assert html.count('<option value="matting"') == 2


class TestDocs:
    def test_the_docs_do_not_promise_an_error_for_an_unknown_value(self):
        text = (_REPO / "docs" / "images-api.md").read_text()
        assert "unknown `background` value" not in text and 'treated as `"auto"` (never an error)' in text
        html = (_APP / "dashboard" / "templates" / "public" / "documentation.html").read_text()
        assert "any other value is a 400" not in html

    def test_the_api_reference_documents_the_field_and_the_response(self):
        text = (_REPO / "docs" / "images-api.md").read_text()
        for needle in ("`background`", '"transparent"', "`has_alpha`", "img.transparent_enabled",
                       "img.matting_url"):
            assert needle in text, needle
        assert text.count("| `background` | string | no |") == 2       # both endpoints' request tables
        assert "| `data[].has_alpha` | boolean |" in text and "| `background` | string | Only when" in text

    def test_the_in_app_documentation_mentions_it_for_both_endpoints(self):
        html = (_APP / "dashboard" / "templates" / "public" / "documentation.html").read_text()
        assert html.count("<code>background</code>") >= 2 and "has_alpha" in html

    def test_both_engine_references_list_every_engine(self):
        from backend.app.db.models import BackendEngine

        html = (_APP / "dashboard" / "templates" / "public" / "documentation.html").read_text()
        markdown = (_REPO / "docs" / "index.md").read_text()
        for engine in BackendEngine:
            assert f"(<code>{engine.value}</code>)</td>" in html, engine.value
            assert f"(`{engine.value}`) |" in markdown, engine.value
        words = {8: "eight", 9: "nine", 10: "ten"}[len(BackendEngine)]
        assert f"exactly these {words} values" in html and f"exactly these {words} values" in markdown

    def test_the_test_manifest_lists_both_new_files(self):
        manifest = (_REPO / "TESTING.md").read_text()
        assert "test_image_matting.py" in manifest and "test_matting_service.py" in manifest
