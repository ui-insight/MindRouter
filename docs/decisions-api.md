# Decisions API (System One) — `POST /v1/systemone`

> **STATUS: EXPERIMENTAL / TRANSITIONAL.** Off by default
> (`decisions.enabled`). The wire format is TypeSafe's System One API, the one
> their Jev model is served through, so it is stable in the sense that it is
> someone else's published contract; which models answer, how well, and the
> MindRouter-specific `metadata` block may change. Not yet benchmarked on our
> hardware (see [Benchmarks](#benchmarks)).

Last updated: 2026-10-03 (release 2.9.83)

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
| a vLLM model name, e.g. `qwen3.8-27b` | One-token letter scoring on a chat model MindRouter already serves. No dedicated model, no extra GPU. | `decisions.allowed_models` |
| an upstream name, e.g. `laya` | A purpose-built decision model running as its own System One server (Laya's `laya-serve`, Open-Jev). MindRouter forwards the state and each question's `type`, `instructions` and `criteria` (nothing else), and checks the reply against what was asked. | `decisions.upstreams` |
| `jev-latest`, `jev-preview`, or omitted | Whichever of the above the admin set as the default. `jev-latest` is what TypeSafe's SDK sends unless told otherwise. | `decisions.default_model` |

Anything else, including a pinned TypeSafe version such as `jev-1.13.0`, is
refused with 422 on `model` rather than silently answered by a different
model. `GET /v1/models` lists the accepted names in TypeSafe's shape for
callers that send the SDK's `X-TypeSafe-SDK` header (everyone else gets the
normal OpenAI model list).

How the two kinds differ:

| | vLLM letter scoring (`qwen3.8-27b`) | Upstream server (`laya`) |
|---|---|---|
| What the number is | The chat model's next-token likelihood over option letters, renormalized (`metadata.score_semantics = normalized_label_likelihood`). Not calibrated. | Whatever that model reports (`upstream_model_probability`). Laya's are temperature-calibrated on its own data. |
| Choice options | At most **20** (one single-token letter each); more is a 422 | The upstream's own limit (Laya: 100) |
| State size | `decisions.max_state_chars`, then the model's context (large) | `decisions.max_state_chars`, then the upstream's (Laya: 512–1,024 tokens by default) |
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
* `permutations` *(MindRouter extension, vLLM models only)* — `1` (default) or
  `2`. With `2` each question is also scored with its options reversed and the
  two distributions averaged, which cancels most position bias at twice the
  cost.

## Response

```json
{
  "model": "qwen3.8-27b",
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
               "questions": {"department": {"complete": true, "label_mass": 0.97}}}
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
means the model did not want to answer with a letter) and `complete` (false
when a label's probability had to be estimated). For an upstream,
`metadata.upstream_model` and, for Laya, `metadata.routing` say which
checkpoint answered.

## Errors

Bodies are FastAPI-shaped, which is also what TypeSafe's API returns and what
its SDK parses.

| Status | Meaning |
|---|---|
| `401` | Missing or invalid MindRouter API key |
| `404` | The decisions API is disabled on this server |
| `422` | Invalid request. `detail` is a list of `{loc, msg, type}` with `loc` starting at `body`, e.g. `["body", "questions", "urgency", "score", "criteria"]`. Also: `model` not offered here, state over the server's limit, more than 20 choice options on a vLLM model, text that is not valid Unicode (an unpaired surrogate) or a non-finite number anywhere in the body, or the upstream model rejecting the content. |
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
| `decisions.default_model` | `qwen3.8-27b` | What `jev-latest`, `jev-preview` and a missing `model` resolve to. A vLLM model or an upstream name. |
| `decisions.allowed_models` | `["qwen3.8-27b"]` | vLLM chat models that may be letter-scored |
| `decisions.upstreams` | `{}` | Upstream System One servers: `{"laya": {"url": "https://host:8010", "api_key": "…", "model": null, "timeout": 30}}`. `model` is the name sent upstream; `null` omits it (Laya then picks a checkpoint by language). |
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
* **vLLM likelihoods are not calibrated.** They are the model's own
  next-token distribution. Do not present them as probabilities of being
  correct without measuring on your data.
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
  `permutations: 2` cancels most of it.
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
    --base-url https://mindrouter.uidaho.edu --model qwen3.8-27b \
    --only latency,cache,concurrency,stability,order,ambiguous

# accuracy + reliability on LocalLLaMA/typed-decisions (2,000 decisions; pip install datasets)
MINDROUTER_API_KEY=mr2_… python tests/decisions_bench.py --model qwen3.8-27b --only calibration --calib-cases 400
```

**Results on our fleet: not yet collected.** Published reference points on
`typed-decisions`: TypeSafe Jev 1.13.0 0.727 accuracy; the benchmark's teacher
self-agreement ceiling 0.735; open-alternative-jev reports 0.737 for
Qwen3.6-27B with this same letter-scoring method. Qwen3.8-27B through
MindRouter has not been measured.

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
(`systemone.py` wire format, `vllm_logprobs.py` letter scoring, `upstream.py`
forwarding) and `backend/app/api/decisions_api.py`, plus one router line in
`backend/app/api/__init__.py`, a branch in `GET /v1/models` for TypeSafe's
SDK, one admin card, this document and
`backend/app/tests/unit/test_decisions.py`. No migration. To remove: delete
those and the `decisions.*` config rows.
