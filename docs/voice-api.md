# MindRouter Voice API (Audio)

Last updated: 2026-07-23

Base URL: `https://mindrouter.uidaho.edu`

This is the public reference for MindRouter's **voice** primitives: text-to-speech
(TTS) and speech-to-text (STT). Both endpoints are OpenAI-compatible.

## Where this fits

MindRouter is a **provider of single-shot generative primitives**. The voice API
turns **one block of text into one audio utterance** (TTS), or **one uploaded audio
file into one transcript** (STT). It does **not** do multi-clip audio assembly,
mixing, timeline editing, subtitle burn-in, dubbing sync, or cross-shot coherence.
That orchestration belongs to the *separate* studio/storyboarding app that consumes
these APIs and does its own ffmpeg assembly. Treat MindRouter as the raw
speech-in / speech-out engine and keep the composition logic in your app.

Internally, both endpoints are thin authenticated proxies: `/v1/audio/speech`
forwards to a Kokoro-family TTS service, and `/v1/audio/transcriptions` forwards to
a Whisper-family STT service. The shapes below are what MindRouter accepts and
returns to you.

## Authentication

Every request requires an API key, sent as **either** header:

- `Authorization: Bearer <key>`
- `X-API-Key: <key>`

If the key is missing you get `401` with detail
`"Missing API key. Provide via 'Authorization: Bearer <key>' or 'X-API-Key: <key>'"`.
An invalid, inactive, or expired key also returns `401`.

Requests are subject to per-user token quota and per-key/per-user RPM rate limits;
exceeding either returns `429 Too Many Requests`.

---

## POST /v1/audio/speech — Text-to-Speech (TTS)

Converts text into spoken audio. The response is the **raw audio byte stream** (not
JSON), streamed back with an audio `Content-Type`.

### Request

- Method: `POST`
- Path: `/v1/audio/speech`
- Content-Type: `application/json`

Body fields:

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `input` | string | **yes** | — | The text to synthesize. Must be non-empty after trimming whitespace, or you get `400 "No text provided"`. |
| `model` | string | no | `"kokoro"` | TTS model/voice engine name. |
| `voice` | string | no | `"af_heart"` | Voice identifier passed through to the TTS engine. |
| `response_format` | string | no | `"mp3"` | Output audio container/codec. Passed through to the engine; `mp3` maps to `Content-Type: audio/mpeg`, any other value maps to `audio/<response_format>`. |
| `speed` | number | no | `1.0` | Playback speed multiplier. Constrained to the range `0.25`–`4.0` (inclusive); out-of-range values are rejected with `422`. |

Notes on voices/formats: MindRouter does not validate or enumerate `voice` or
`response_format` values itself — it forwards them to the backing Kokoro TTS
service, which is the source of truth for the supported voice catalog and output
formats. `af_heart` is the default Kokoro voice and `mp3` the default format.
There is no `/v1/audio/voices` discovery endpoint; consult your Kokoro deployment
for the full voice list.

### Response

- `200 OK` with a streamed audio body.
- `Content-Type`: `audio/mpeg` when `response_format` is `mp3`, otherwise
  `audio/<response_format>` (e.g. `audio/wav`, `audio/opus`).
- The body is the audio bytes — write them straight to a file.

### Error cases

| Status | When |
|--------|------|
| `400` | `input` is empty/whitespace-only (`"No text provided"`). |
| `401` | Missing/invalid/inactive/expired API key. |
| `404` | TTS is not enabled on this MindRouter instance (`"TTS is not enabled"`). |
| `422` | `speed` outside `0.25`–`4.0`, or a malformed body. |
| `429` | Token quota or RPM rate limit exceeded. |
| `500` | TTS service URL not configured server-side. |

Note: if the upstream TTS service errors mid-stream, MindRouter logs the failure and
the stream simply ends; you may receive a `200` with a truncated/empty body rather
than an error status. Check the byte length of what you receive.

### curl example

```bash
curl -sS -X POST "https://mindrouter.uidaho.edu/v1/audio/speech" \
  -H "Authorization: Bearer $MINDROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "kokoro",
        "input": "Storyboard shot 3: the lighthouse beam sweeps across the harbor.",
        "voice": "af_heart",
        "response_format": "mp3",
        "speed": 1.0
      }' \
  --output shot3_vo.mp3
```

### Python example

```python
import os
import requests

BASE_URL = "https://mindrouter.uidaho.edu"
API_KEY = os.environ["MINDROUTER_API_KEY"]

resp = requests.post(
    f"{BASE_URL}/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "kokoro",
        "input": "Storyboard shot 3: the lighthouse beam sweeps across the harbor.",
        "voice": "af_heart",
        "response_format": "mp3",
        "speed": 1.0,
    },
    stream=True,
)
resp.raise_for_status()

with open("shot3_vo.mp3", "wb") as f:
    for chunk in resp.iter_content(chunk_size=4096):
        f.write(chunk)

print("wrote shot3_vo.mp3")
```

---

## POST /v1/audio/transcriptions — Speech-to-Text (STT)

Transcribes an uploaded audio file to text. This endpoint takes
**`multipart/form-data`**, not JSON.

### Request

- Method: `POST`
- Path: `/v1/audio/transcriptions`
- Content-Type: `multipart/form-data`

Form fields:

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `file` | file upload | **yes** | — | The audio file to transcribe. Sent as a multipart file part. The original filename and content-type are forwarded to the STT engine; a missing filename defaults to `audio.webm`. |
| `model` | string | no | `whisper-large-v3-turbo` | STT model name. When omitted, the server default (config key `voice.stt_model`, default `whisper-large-v3-turbo`) is used. |
| `language` | string | no | (auto-detect) | Optional ISO language hint (e.g. `en`). Only forwarded to the engine when provided; otherwise the engine auto-detects. |
| `response_format` | string | no | `json` | One of `json`, `text`, `verbose_json`, `srt`, `vtt`. Case-insensitive. An unsupported value returns `400`. |

### Response

The response shape depends on `response_format`:

- `json` (default) and `verbose_json` → `200 OK`, `Content-Type: application/json`.
  The JSON body is passed through from the STT engine. For `json` this is typically
  `{"text": "..."}`; `verbose_json` additionally includes timing/segment metadata
  as produced by the Whisper engine.
- `text` → `200 OK`, `Content-Type: text/plain` — the raw transcript text.
- `srt` → `200 OK`, `Content-Type: text/plain` — SubRip subtitle text.
- `vtt` → `200 OK`, `Content-Type: text/vtt` — WebVTT subtitle text.

### Error cases

| Status | When |
|--------|------|
| `400` | `response_format` is not one of `json`, `text`, `verbose_json`, `srt`, `vtt`. |
| `401` | Missing/invalid/inactive/expired API key. |
| `404` | STT is not enabled on this MindRouter instance (`"STT is not enabled"`). |
| `422` | Missing required `file` part. |
| `429` | Token quota or RPM rate limit exceeded. |
| `500` | STT service URL not configured server-side. |
| `502` | Upstream STT service returned an error, was unavailable, or timed out. The timeout message notes the model may still be loading — retry after a short delay. |

The upstream request has a generous 600-second timeout to accommodate long audio and
cold model loads, so large files may take a while.

### curl example

```bash
curl -sS -X POST "https://mindrouter.uidaho.edu/v1/audio/transcriptions" \
  -H "X-API-Key: $MINDROUTER_API_KEY" \
  -F "file=@narration_take2.wav" \
  -F "model=whisper-large-v3-turbo" \
  -F "language=en" \
  -F "response_format=json"
```

Get subtitles directly for a storyboard clip:

```bash
curl -sS -X POST "https://mindrouter.uidaho.edu/v1/audio/transcriptions" \
  -H "X-API-Key: $MINDROUTER_API_KEY" \
  -F "file=@shot3_dialogue.mp3" \
  -F "response_format=srt" \
  --output shot3_dialogue.srt
```

### Python example

```python
import os
import requests

BASE_URL = "https://mindrouter.uidaho.edu"
API_KEY = os.environ["MINDROUTER_API_KEY"]

with open("narration_take2.wav", "rb") as audio:
    resp = requests.post(
        f"{BASE_URL}/v1/audio/transcriptions",
        headers={"X-API-Key": API_KEY},
        files={"file": ("narration_take2.wav", audio, "audio/wav")},
        data={
            "model": "whisper-large-v3-turbo",
            "language": "en",
            "response_format": "json",
        },
    )
resp.raise_for_status()

# response_format=json -> parse JSON; for text/srt/vtt use resp.text
print(resp.json()["text"])
```

---

## Quick reference

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/v1/audio/speech` | POST | JSON (`input`, `voice`, `model`, `response_format`, `speed`) | Streamed audio bytes (`audio/mpeg` for mp3) |
| `/v1/audio/transcriptions` | POST | multipart (`file`, `model`, `language`, `response_format`) | JSON or plain text/SRT/VTT transcript |

Both endpoints must be **enabled** on the target MindRouter instance (server config
keys `voice.tts_enabled` / `voice.stt_enabled`); if disabled you receive `404`.
