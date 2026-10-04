# Decisions API (System One) — `POST /v1/systemone`

> **STATUS: EXPERIMENTAL / TRANSITIONAL.** Off by default
> (`decisions.enabled`). The wire format is TypeSafe's System One API, the one
> their Jev model is served through, so it is stable in the sense that it is
> someone else's published contract; which models answer, how well, and the
> MindRouter-specific `metadata` block may change. Measured on our fleet in
> [Benchmarks](#benchmarks).

Last updated: 2026-10-04 (release 2.9.85)

## What it is

Ordinary chat completions generate text. Many application decisions do not
need text: *should this ticket be escalated?*, *which of these four queues?*,
*how severe?* A decision API takes one shared **state** and one or more
**typed questions**, and returns a typed answer with probabilities for each.
No prose is generated and nothing has to be parsed.

MindRouter speaks **TypeSafe's System One wire format** on
`POST /v1/systemone` (`POST /v1/decisions` is an alias with the identical
shape). A client written for TypeSafe's hosted Jev, including their SDK, works
by changing the base URL and using a MindRouter API key:

```bash
export TYPESAFE_BASE_URL=https://mindrouter.uidaho.edu
export TYPESAFE_API_KEY=mr2_...
```

```python
from typesafe_sdk import TypeSafeClient

with TypeSafeClient() as client:                       # model defaults to "jev-latest"
    res = client.system_one(
        "Help! My payouts have been failing for 3 days.",
        {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    )
    print(res.model, res.nouls["is_urgent"].noul)
```

**The wire is Jev's; the model is not.** Answers come from whichever model the
request's `model` field selects (below). Thresholds tuned against TypeSafe's
Jev do not carry over. The response's `model` field always names the model
that actually answered.

## Choosing the model

The `model` field decides how an answer is produced.

| `model` | Answered by | Configured in |
|---|---|---|
| a vLLM model's catalog name, e.g. `qwen/qwen3.8-27b` | One-token letter scoring on a chat model MindRouter already serves. No dedicated model, no extra GPU. | `decisions.allowed_models` |
| an upstream name, e.g. `clef` or `laya` | A purpose-built decision model running as its own System One server (Cloudflare's Clef behind `clef_service`, Laya's `laya-serve`, Open-Jev). MindRouter forwards the state and each question's `type`, `instructions` and `criteria` (nothing else), and checks the reply against what was asked. | `decisions.upstreams` |
| `jev-latest`, `jev-preview`, or omitted | Whichever of the above the admin set as the default. `jev-latest` is what TypeSafe's SDK sends unless told otherwise. | `decisions.default_model` |

Anything else, including a pinned TypeSafe version such as `jev-1.13.0`, is
refused with 422 on `model` rather than silently answered by a different
model. `GET /v1/models` lists the accepted names in TypeSafe's shape for
callers that send the SDK's `X-TypeSafe-SDK` header (everyone else gets the
normal OpenAI model list).

How the two kinds differ:

| | vLLM letter scoring (`qwen/qwen3.8-27b`) | Upstream server (`clef`, `laya`) |
|---|---|---|
| What the number is | The chat model's next-token likelihood over option letters, renormalized, then softened by a per-type temperature fitted on one public benchmark (`metadata.score_semantics = normalized_label_likelihood`, `metadata.temperature`). Calibrated on that benchmark, not on your data. | Whatever that model reports (`upstream_model_probability`); MindRouter does not adjust it. |
| Choice options | At most **20** (one single-token letter each); more is a 422 | The upstream's own limit (Clef: 255; Laya: 100) |
| Images | Yes, when the model's replicas serve it with vision (Qwen3.8-27B does) | Only upstreams marked `"images": true` (Clef); others refuse a request with images |
| Model passes per request | One per question (two for a `choice`) | Clef: one for the whole request |
| State size | `decisions.max_state_chars`, then the model's context (large) | `decisions.max_state_chars`, then the upstream's (Clef: 16,384 tokens as deployed; Laya: 512–1,024). A cut state is reported in `metadata.truncated`. |
| Cost | One 1-token forward pass per question on a 27B model; state prefix-cached after the first | One small encoder pass |
| Routing | A healthy, circuit-closed replica chosen per request | A fixed URL; not health-checked |

## Request

```json
{
  "model": "jev-latest",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent":  {"type": "noul",   "instructions": "Does this convey urgency?",
                   "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds",
                                "technical": "Bugs, outages, integrations",
                                "sales": null}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
```

* `state` — string, object or array. Structure is shown to the model as JSON.
* `model` — see above. Required by TypeSafe; MindRouter treats a missing
  `model` as `jev-latest`.
* `questions` — a map of ids you choose to questions; at most 32. Answers come
  back under the same ids. The ids are never shown to the model.
  * `noul` — a yes/no question. Optional `criteria.true` / `criteria.false`
    describe what yes and no mean.
  * `choice` — `criteria` maps option name to a description (or `null`).
  * `score` — `criteria` is an ordered array of 1 to 10 level descriptions;
    a level's position is its value, starting at 0.
  * `instructions`, option descriptions and score levels may each be a string,
    an object or an array.
* `images` *(Cloudflare Clef's extension)* — up to 4 images the model looks at
  before the state. Each is a base64 data URL (`"data:image/png;base64,..."`)
  or an object `{"content_type": "image/png", "base64": "..."}`. PNG, JPEG or
  WebP; 4 MiB and 16 megapixels each, 8 MiB in total. Remote URLs are not
  accepted: MindRouter does not fetch on a caller's behalf. Every image is
  checked before any model sees it (it must be a readable file of the type it
  claims, within the limits; an image whose pixel data turns out to be damaged
  is a 422 from the model). A model that cannot see refuses the request with
  422 rather than answering from the text alone; for a vLLM model that is the
  model's multimodal capability flag, the one the admin override sets. This is the same field and format
  Cloudflare publishes for Clef on Workers AI.
* `permutations` *(MindRouter extension, vLLM models only)* — how many option
  orders are scored. With `2` a question is also scored with its options
  reversed and the two distributions averaged, which cancels most position
  bias at twice the calls. **Omit it** to get the server's per-type defaults
  (`decisions.permutations`: `choice` 2, `noul` 1, `score` 1, which is where
  the measured gain is). Send `1` or `2` to force that for every question in
  the request.

## Response

```json
{
  "model": "qwen/qwen3.8-27b",
  "answers": {
    "is_urgent":  {"type": "noul", "noul": 0.95},
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                   "confidence": 0.82},
    "frustration": {"type": "score", "score": 1.05,
                    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                    "confidence": 0.92}
  },
  "usage": {"input_tokens": 296, "output_tokens": 3},
  "id": "dec-4f1c…",
  "metadata": {"score_semantics": "normalized_label_likelihood", "backend": "vllm_logprobs",
               "backend_calls": 3, "prompt_tokens": 890, "cached_tokens": 594,
               "temperature": {"noul": 1.35, "choice": 1.05, "score": 1.45},
               "questions": {"department": {"complete": true, "label_mass": 0.97, "permutations": 2}}}
}
```

`model`, `answers` and `usage` are TypeSafe's fields, with their meanings:

* `noul` — probability the answer is yes.
* `choice` — the highest-probability option, every option's probability
  (summing to 1), and `confidence`.
* `score` — the probability-weighted level (can fall between levels), the
  `legend` mapping level numbers back to what you sent, per-level
  probabilities, and `confidence`.
* `confidence` uses TypeSafe's published formulas: for a choice,
  `(n·max − 1) / (n − 1)`; for a score, `1 − E|level − mode| / (the same for a
  uniform spread)`, both clamped to [0, 1].
* `usage.input_tokens` is what is charged against your quota (see
  [Metering](#metering)).

`id` and `metadata` are MindRouter additions; TypeSafe's SDK ignores unknown
fields. The response header `x-typesafe-request-id` carries the same `id`.
For vLLM models, `metadata.questions[id]` reports `label_mass` (how much of
the model's next-token probability fell on any option letter at all; low
means the model did not want to answer with a letter), `complete` (false
when a label's probability had to be estimated) and `permutations` (how many
option orders were scored), and `metadata.temperature` lists the temperature
applied to each question type's probabilities. Temperature changes how
confident the numbers are, never which answer is chosen. For an upstream,
`metadata.upstream_model` and, for Laya, `metadata.routing` say which
checkpoint answered, and `metadata.truncated` / `metadata.state_tokens_dropped`
appear when the upstream says it cut the state to fit its context.

`confidence` means the same thing whichever model answered: for an upstream
it is recomputed from the returned probabilities with TypeSafe's formulas
(Clef, for one, reports its top probability in that field).

## Errors

Bodies are FastAPI-shaped, which is also what TypeSafe's API returns and what
its SDK parses.

| Status | Meaning |
|---|---|
| `401` | Missing or invalid MindRouter API key |
| `404` | The decisions API is disabled on this server |
| `422` | Invalid request. `detail` is a list of `{loc, msg, type}` with `loc` starting at `body`, e.g. `["body", "questions", "urgency", "score", "criteria"]`. Also: `model` not offered here, state over the server's limit, an invalid image or images sent to a model that cannot see, more than 20 choice options on a vLLM model, text that is not valid Unicode (an unpaired surrogate) or a non-finite number anywhere in the body, or the upstream model rejecting the content. |
| `429` | Token quota or requests-per-minute limit |
| `500` | Unexpected failure (the request is still recorded as failed) |
| `502` | The model's server failed or returned an invalid reply. Nothing is charged. |
| `503` | Model unavailable, no healthy replica, or the upstream is busy. `Retry-After` is set where known. |

TypeSafe's SDK retries 429 and 5xx with backoff by default. A disabled API is
404 rather than 503 so that it is not retried.

## How a vLLM model answers

For each question MindRouter sends one chat completion to a replica of the
model: a single user turn containing the state, the question and the options
labelled `A.`, `B.`, …, with thinking off, `max_tokens=1`,
`allowed_token_ids` restricted to the option letters and `logprob_token_ids`
requesting each letter's log-probability. The reply's distribution over the
letters, renormalized, is the answer. A `noul` is a two-option question
(yes/no); a `score` lists its levels as options.

This is the `separate` mode of
[open-alternative-jev](https://github.com/ikermoel/open-alternative-jev)
transcribed onto vLLM's HTTP API (that library needs an in-process engine;
MindRouter does not depend on it). The state comes first in every prompt, so
the first question fills vLLM's prefix cache and the rest reuse it:
MindRouter scores the first question alone, then the others concurrently.

Requirements on the model: a chat template that honours
`enable_thinking=false`, single-token capital letters, and vLLM ≥ 0.29 for
`logprob_token_ids`. That is Qwen3.x on our fleet. It is not gpt-oss (Harmony
puts analysis text first) and not Ollama backends. Hence the allow-list.

## Operations

Admin → Settings → "Decisions API (System One)", or `app_config`:

| Key | Default | Meaning |
|---|---|---|
| `decisions.enabled` | `false` | Master switch (404 when off) |
| `decisions.default_model` | `qwen/qwen3.8-27b` | What `jev-latest`, `jev-preview` and a missing `model` resolve to. A vLLM model or an upstream name. |
| `decisions.allowed_models` | `["qwen/qwen3.8-27b"]` | vLLM chat models that may be letter-scored. Use the catalog name exactly as `/v1/models` lists it. |
| `decisions.permutations` | `{"noul": 1, "choice": 2, "score": 1}` | Option orders scored and averaged per question type on vLLM models (1 or 2). A request's own `permutations` overrides it. |
| `decisions.temperature` | `{"noul": 1.35, "choice": 1.05, "score": 1.45}` | Temperature applied to each question type's probabilities on vLLM models (0.2–5; `1` is off; above 1 softens). Fitted on Qwen3.8-27B; refit if you change the model. |
| `decisions.upstreams` | `{}` | Upstream System One servers: `{"clef": {"url": "https://host:8004", "api_key": "…", "model": "clef", "images": true, "timeout": 30}}`. `model` is the name sent upstream (`null` omits it; Laya then picks a checkpoint by language). `images: true` marks an upstream that accepts the `images` extension. |
| `decisions.max_state_chars` | `32000` | Ceiling on the rendered state (hard cap 64,000) |
| `decisions.fanout` | `8` | Concurrent scoring calls per request (vLLM models) |
| `decisions.backend_concurrency` | `4` | Concurrent calls to one backend or upstream across all requests, per app worker process. With 8 workers the worst case on one replica is 32. |

### Metering

Every request writes a `requests` row (`endpoint`, model, backend, token
counts, timings, status). Its `parameters` column holds only the request's
shape (question count and types, state length, the requested model name):
**the state and question text are never stored or logged.** That includes
failures: error bodies from an engine or an upstream are not logged, an
upstream's rejection text is returned to the caller but the row records only
a fixed summary, and an unexpected error records its type, not its message.
A request cancelled mid-flight is closed as failed (`error_code` 499) rather
than left in `processing`.

Quota is charged for what had to be computed: for a vLLM model, prompt tokens
that were *not* served from the prefix cache, plus one scoring token per
question; for an upstream, the token counts it reports. A failed request is
charged nothing.

Prometheus: `mindrouter_decisions_requests_total{model,backend,status}`,
`mindrouter_decisions_questions_total{model,type}`,
`mindrouter_decisions_latency_seconds{model}`,
`mindrouter_decisions_tokens_total{model,type=prompt|scoring|cached}`. Log
event `decision_request`.

### Adding Clef as an upstream

[Clef](https://huggingface.co/Cloudflare/clef) (Cloudflare, Apache-2.0) is
Qwen3.8-27B post-trained for decisions with a small joint head. It answers a
whole request, every question at once, in one forward pass, and it can look at
images. Cloudflare publishes weights and a Python function but no server;
`clef_service/` in this repository is that server (bearer key, dynamic
batching, bounded queue; see its README). It needs one GPU with about 60 GB
free. Then:

```json
{"clef": {"url": "https://<host>:<tls-port>", "api_key": "<CLEF_API_KEY>", "model": "clef", "images": true}}
```

Callers select it with `"model": "clef"`. It cannot share the Qwen3.8-27B
vLLM replicas: its backbone weights are modified and its head reads the
model's internal states.

### Adding Laya as an upstream

[Laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0) is a small
encoder decision model whose `laya-serve` speaks this same wire format:

```bash
pip install "laya[serve]"
LAYA_DEVICE=cuda LAYA_API_KEY=<secret> LAYA_PORT=8010 laya-serve
```

Then in Admin → Settings add
`{"laya": {"url": "https://<host>:8010", "api_key": "<secret>", "model": null}}`
under upstream servers. Callers select it with `"model": "laya"`. Its own
model card reports that the base checkpoints are near chance zero-shot on the
`typed-decisions` benchmark and that its headline accuracy is a checkpoint
fine-tuned on that benchmark; measure before relying on it
(`tests/decisions_bench.py --model laya --only calibration`).

## Limitations and known risks

* **Not Jev.** Same request and response shape; different model, different
  numbers. Re-tune thresholds.
* **vLLM probabilities are calibrated on one benchmark, not yours.** They are
  the model's next-token distribution, softened by a temperature that made
  them well calibrated on `typed-decisions` (below). On a different workload
  they may be over- or under-confident; measure before setting thresholds.
* **The model leans cautious.** On the benchmark Qwen3.8-27B over-picks the
  conservative option (`human_review`, `escalate_to_human`, `hold`) relative
  to the reference labels; in 70% of its errors the reference answer was its
  second choice.
* **20 choice options** on vLLM models. TypeSafe allows 255.
* **No scheduler slot.** Scoring calls go straight to a replica and do not
  count against `max_concurrent`; `decisions.backend_concurrency` and the
  per-user RPM limit bound the load instead.
* **Upstreams are not registered backends.** No health polling, circuit
  breaker or GPU telemetry; a broken upstream returns 502 until fixed, and it
  does not appear in the backend list. Making them a backend engine is the
  natural next step and needs a migration.
* **No retry within a request.** One failing call fails the request; clients
  should retry (TypeSafe's SDK does).
* **Position bias** on vLLM models: letter readouts favour positions.
  Averaging over reversed options cancels most of it and is on by default for
  `choice` questions only, where it raised accuracy from 0.643 to 0.718; it
  made no measurable difference for `score` and slightly hurt `noul`.
* **Template drift.** A future chat template that changes the thinking-off
  suffix would show up as `label_mass` collapsing; watch it after model
  upgrades.
* **Upstream credentials** are stored in `app_config` like other service
  keys. The settings page never renders them back: a stored key shows as
  `(stored)`, and saving the form with that placeholder keeps it.
* **An upstream's token counts are charged as reported.** A reply claiming
  more than 2,000,000 tokens, or a non-numeric count, is rejected as invalid
  (502) rather than billed.

## Benchmarks

`tests/decisions_bench.py` runs against a live deployment and takes `--model`,
so the same script compares a vLLM model with an upstream:

```bash
MINDROUTER_API_KEY=mr2_… python tests/decisions_bench.py \
    --base-url https://mindrouter.uidaho.edu --model qwen/qwen3.8-27b \
    --only latency,cache,concurrency,stability,order,ambiguous

# accuracy + reliability on LocalLLaMA/typed-decisions (2,000 decisions; pip install datasets)
MINDROUTER_API_KEY=mr2_… python tests/decisions_bench.py --model qwen/qwen3.8-27b --only calibration --calib-cases 400
```

### Results (2026-10-03, through production MindRouter)

`LocalLLaMA/typed-decisions`, config `all`, test split: 400 cases, 2,000
decisions, four workflows. Accuracy is the share of decisions whose top answer
matches the reference label. No request failed in any run.

| | Accuracy | noul | choice | score | Calibration error (ECE) | Score MAE |
|---|---|---|---|---|---|---|
| Qwen3.8-27B, 2.9.83 (one option order, no temperature) | 0.687 | 0.770 | 0.643 | 0.657 | 0.090 | 0.403 |
| Qwen3.8-27B, 2.9.84 defaults, predicted offline | 0.710 | 0.770 | 0.718 | 0.657 | 0.022 | 0.377 |
| **Qwen3.8-27B, 2.9.84 defaults, measured on the deployed code** | **0.713** | 0.772 | 0.717 | 0.665 | **0.019** | — |
| Qwen3.8-27B, `permutations: 2` for every type | 0.707 | 0.752 | 0.718 | 0.664 | 0.063 | 0.382 |
| **Clef 27B (Cloudflare), upstream on one H200, measured 2026-10-04** | **0.726** | 0.847 | 0.647 | 0.694 | 0.023 | — |
| Laya (base checkpoints, zero-shot) | 0.361 | 0.487 | 0.287 | 0.323 | 0.175 | — |
| *TypeSafe Jev 1.13.0 (published, not measured here)* | *0.727* | | | | *0.144* | *0.391* |
| *random / majority class / teacher self-agreement ceiling* | *0.318 / 0.461 / 0.735* | | | | | |

Clef and the native path trade places by question type: Clef is clearly
better on yes/no questions (0.847 vs 0.772), the native path on `choice`
(0.717 vs 0.647). On speed, the same 400 cases at six concurrent callers took
44 s on Clef (one GPU, p50 612 ms) and 36 s on the native path (five replicas
shared with chat, p50 468 ms). Clef reads the state once per request, so it
bills far fewer tokens when a request asks many questions about a long state.
Clef's numbers need its fast path, see `clef_service/README.md` (Performance).

How the 2.9.84 defaults were chosen: the per-type option-order setting and the
temperatures were fitted on a 400-case sample of the benchmark's **training**
split and then scored once on the test split; the "predicted offline" row is
that result (computed from per-question answers collected through production
at one and two option orders), and the "measured" row is the same benchmark
re-run end to end after 2.9.84 was deployed. With those defaults, answers
reported at 0.9 confidence or higher were correct 94.5% of the time (89.8%
before).

Things the numbers say:

* **Much of the remaining gap is label noise.** On the 1,188 decisions where
  the benchmark's teacher labels agree with themselves, Qwen3.8-27B scores
  0.80; where the teacher is split, 0.52. That is why the ceiling is 0.735.
* **Laya's base checkpoints are near chance here**, which matches its own
  model card (0.362). Its published 0.766 is a separate checkpoint fine-tuned
  on this benchmark's workflows.
* **More is available with per-question statistics.** Dividing a yes/no
  answer by the model's own average answer to that same question (no labels
  needed) raised `noul` from 0.770 to about 0.81, and a recipe chosen on the
  training split reached 0.723 overall. With labeled examples of a workflow, a
  per-question bias fit reached 0.75–0.76. Neither is built in: both need a
  history of the same question.

Latency from a client outside the cluster, one request at a time (p50):

| Questions sharing one state | Qwen3.8-27B (one option order) | Laya |
|---|---|---|
| 1 | 183 ms | 128 ms |
| 4 | 424 ms | 135 ms |
| 16 | 715 ms | 162 ms |
| 16, 24,000-character state | 3.2 s | 0.43 s (state cut to its 512-token context) |

A second option order doubles the calls for that question: a five-question
request took 394 ms at one order and 606 ms at two for every type. Jev's
published single-question p50 is 236–276 ms (third-party measurements).
Under parallel load: Qwen3.8-27B 32 decisions/s from 6 clients; Laya
49 decisions/s from 8 clients (its single inference thread is the limit).

Known gap: vLLM does not report prefix-cache hits unless started with
`--enable-prompt-tokens-details`, which our units do not set, so
`metadata.cached_tokens` is null and a multi-question request is charged the
state once per scoring call rather than once.

## Compatibility evidence

Checked against TypeSafe's published contract
([API reference](https://docs.typesafe.ai/api.md), their OpenAPI-generated SDK
models) and against `typesafe-sdk` 0.7.2 itself: the SDK's client was run with
its HTTP transport captured; MindRouter's parser accepted the requests it
built, and the SDK's strict response validation accepted MindRouter's
responses. The captured request body is a fixture in
`backend/app/tests/unit/test_decisions.py`.

## Replacing or removing this layer

Everything lives in `backend/app/services/decisions/`
(`systemone.py` wire format, `images.py` image checks, `vllm_logprobs.py`
letter scoring, `upstream.py` forwarding) and `backend/app/api/decisions_api.py`, plus one router line in
`backend/app/api/__init__.py`, a branch in `GET /v1/models` for TypeSafe's
SDK, one admin card, this document and
`backend/app/tests/unit/test_decisions.py`. No migration. To remove: delete
those and the `decisions.*` config rows.
