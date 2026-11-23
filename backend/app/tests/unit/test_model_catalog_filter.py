############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_model_catalog_filter.py: Non-text models stay out of
#     the LLM catalogs (2.9.10)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The general model catalogs must publish text models only.

Until 2.9.10 `/v1/models` iterated every healthy backend's models with no
modality filter, so `black-forest-labs/FLUX.2-klein-9B` and
`lightricks/ltx-2.3-distilled` were advertised alongside chat models (verified
against production: both appeared among 73 entries). A client selecting one for
`/v1/chat/completions` gets a confusing failure, and registering voice backends
would have added Kokoro and whisper to the same list.

2.9.10 covered only the API catalogs. The human-facing pages kept leaking:
the landing-page model grid, the public /models catalog (which labelled FLUX
and LTX "LLM"), and the chat model picker — users picked voice models and
tried to talk to them. Those surfaces now filter too (`is_catalog_model` for
browse pages, the narrower `is_chat_model` for pickers whose purpose is
starting a conversation), guarded by TestDashboardSurfacesApplyTheFilter.

Image, video and speech each have their own discovery endpoint instead:
`/v1/images/models`, `/videos/models`, `/v1/audio/voices`.

Source/behaviour split: `is_catalog_model`/`is_chat_model` are pure and
imported directly; the endpoints are checked structurally because importing
them pulls the DB chain (see MEMORY.md "Import Chain Gotcha").
"""

import ast
import importlib.util
import io
import re
import sys
import tokenize
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP = Path(__file__).resolve().parents[2]
_DASHBOARD = _APP / "dashboard"


def _load_models_api():
    """Direct-load models_api for its pure helper, stubbing the heavy imports."""
    saved = {k: sys.modules.get(k) for k in (
        "backend", "backend.app", "backend.app.api", "backend.app.api.auth",
        "backend.app.core", "backend.app.core.canonical_schemas",
        "backend.app.core.telemetry", "backend.app.core.telemetry.registry",
        "backend.app.db", "backend.app.db.models", "backend.app.db.session",
    )}

    class _Modality:
        CHAT = "chat"
        COMPLETION = "completion"
        MULTIMODAL = "multimodal"
        EMBEDDING = "embedding"
        RERANKING = "reranking"
        IMAGE_GENERATION = "image_generation"
        VIDEO_GENERATION = "video_generation"
        TTS = "tts"
        STT = "stt"

    for name in ("backend", "backend.app", "backend.app.api", "backend.app.core",
                 "backend.app.core.telemetry", "backend.app.db"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["backend.app.api.auth"] = MagicMock()
    # FastAPI validates the response_model annotation when @router.get runs,
    # so CanonicalModelList must be the real pydantic class, not a MagicMock.
    _cs_spec = importlib.util.spec_from_file_location(
        "backend.app.core.canonical_schemas",
        _APP / "core" / "canonical_schemas.py", submodule_search_locations=[],
    )
    _cs = importlib.util.module_from_spec(_cs_spec)
    _cs_spec.loader.exec_module(_cs)
    sys.modules["backend.app.core.canonical_schemas"] = _cs
    sys.modules["backend.app.core.telemetry.registry"] = MagicMock()
    sys.modules["backend.app.db.session"] = MagicMock()
    sys.modules["backend.app.db.models"] = MagicMock(
        Modality=_Modality, ApiKey=MagicMock(), User=MagicMock()
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "models_api_under_test", _APP / "api" / "models_api.py",
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, _Modality
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture(scope="module")
def api():
    return _load_models_api()


def _model(modality):
    m = MagicMock()
    m.modality = modality
    return m


class TestIsCatalogModel:
    @pytest.mark.parametrize("attr", ["CHAT", "COMPLETION", "MULTIMODAL",
                                      "EMBEDDING", "RERANKING"])
    def test_text_modalities_are_published(self, api, attr):
        mod, Modality = api
        assert mod.is_catalog_model(_model(getattr(Modality, attr))) is True

    @pytest.mark.parametrize("attr", ["IMAGE_GENERATION", "VIDEO_GENERATION",
                                      "TTS", "STT"])
    def test_non_text_modalities_are_excluded(self, api, attr):
        mod, Modality = api
        assert mod.is_catalog_model(_model(getattr(Modality, attr))) is False

    def test_unknown_modality_is_published(self, api):
        """Fail open: a discovery gap must never hide a working chat model."""
        mod, _ = api
        assert mod.is_catalog_model(_model(None)) is True

    def test_embeddings_and_rerankers_stay(self, api):
        """These are text models with OpenAI precedent for being listed; the
        request was to hide image/video/voice, not to empty the catalog."""
        mod, Modality = api
        assert mod.is_catalog_model(_model(Modality.EMBEDDING)) is True
        assert mod.is_catalog_model(_model(Modality.RERANKING)) is True


class TestIsChatModel:
    @pytest.mark.parametrize("attr", ["CHAT", "COMPLETION", "MULTIMODAL"])
    def test_conversational_modalities_are_offered(self, api, attr):
        mod, Modality = api
        assert mod.is_chat_model(_model(getattr(Modality, attr))) is True

    @pytest.mark.parametrize("attr", ["EMBEDDING", "RERANKING",
                                      "IMAGE_GENERATION", "VIDEO_GENERATION",
                                      "TTS", "STT"])
    def test_non_conversational_modalities_are_excluded(self, api, attr):
        """Embedding/reranking belong in the catalog but not in a picker
        whose purpose is starting a conversation."""
        mod, Modality = api
        assert mod.is_chat_model(_model(getattr(Modality, attr))) is False

    def test_unknown_modality_fails_open(self, api):
        """Same fail-open rule as the catalog: a discovery gap must never
        hide a working chat model from the picker."""
        mod, _ = api
        assert mod.is_chat_model(_model(None)) is True

    def test_chat_set_is_a_subset_of_the_catalog_set(self, api):
        """A model offered for chat must also be browsable in the catalog."""
        mod, _ = api
        assert mod.CHAT_MODALITIES <= mod.CATALOG_MODALITIES


def _strip_comments(source: str) -> str:
    """Blank out comment tokens so a commented-out filter line can't satisfy
    an exact-statement guard (or the sweep's call regex)."""
    lines = source.splitlines(keepends=True)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                row, col = tok.start
                line = lines[row - 1]
                keep = line[:col]
                lines[row - 1] = keep + "\n" if line.endswith("\n") else keep
    except (tokenize.TokenError, IndentationError):
        pass
    return "".join(lines)


def _function_source(path: Path, name: str) -> str:
    """Comment-stripped source of a module-level function, nested defs
    included."""
    src = path.read_text()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _strip_comments(ast.get_source_segment(src, node))
    raise AssertionError(f"{path.name} has no module-level function {name!r}")


# Matches an actual CALL to a filter helper. A bare name is not enough: the
# imports are function-local, so `"is_chat_model" in seg` would be satisfied
# by the import line alone even after the filter itself was deleted (proven
# by mutation during review).
_FILTER_CALL = re.compile(r"is_(catalog|chat)_model\(")


class TestDashboardSurfacesApplyTheFilter:
    """The human-facing pages leaked after 2.9.10 filtered the API catalogs:
    users browsed FLUX/LTX/Kokoro on /models (as category "LLM") and picked
    them in the chat UI. Guard each fixed surface, keep the admin surfaces
    deliberately unfiltered, and sweep for new surfaces appearing bare.

    Each per-surface guard asserts the exact filter statement, not the
    helper's name — the function-local import line contains the name."""

    def test_landing_page_grid_filters(self):
        seg = _function_source(_DASHBOARD / "routes.py", "public_dashboard")
        assert "if is_catalog_model(m):" in seg

    def test_public_models_catalog_filters(self):
        seg = _function_source(_DASHBOARD / "routes.py", "models_catalog")
        assert "if not is_catalog_model(model):" in seg

    def test_public_models_chart_matches_the_grid(self):
        """The popularity chart comes from an unfiltered usage cache; it must
        be narrowed to the models the grid shows. Casefolded: Request.model
        keeps client casing while MariaDB groups case-insensitively, so an
        exact match would drop a visible model whose usage row is miscased."""
        seg = _function_source(_DASHBOARD / "routes.py", "models_catalog")
        assert re.search(r"\.casefold\(\) in visible_names", seg)

    def test_chart_cache_overfetches_so_filtering_can_backfill(self):
        """The warm cache is truncated BEFORE the visibility filter runs; it
        must hold comfortably more rows than the chart shows, or every hidden
        voice/image/retired name would eat a chart slot instead of a visible
        model backfilling it."""
        main_src = (_APP / "main.py").read_text()
        m = re.search(r"get_model_token_totals\(db, limit=(\d+)\)", main_src)
        assert m, "cache warm no longer sets an explicit limit"
        seg = _function_source(_DASHBOARD / "routes.py", "models_catalog")
        n = re.search(r"\]\[:(\d+)\]", seg)
        assert n, "models_catalog no longer slices the chart to a top-N"
        assert int(m.group(1)) >= 3 * int(n.group(1))

    def test_chat_picker_offers_only_chat_models(self):
        seg = _function_source(_DASHBOARD / "chat.py", "chat_list_models")
        assert "if not is_chat_model(model):" in seg

    def test_chat_picker_selection_cascades_past_missing_models(self):
        """A stale localStorage model (e.g. one the filter now hides) must
        fall through to the server default, not shadow it: the old
        first-truthy `a || b || c` chain landed such users on the first
        alphabetical model forever."""
        html = (_DASHBOARD / "templates" / "chat.html").read_text()
        assert "currentValue || lastModel || serverDefaultModel" not in html
        assert "[currentValue, lastModel, serverDefaultModel]" in html
        # The cascade only works if each candidate is checked for existence
        # before being applied — `if (candidate)` alone re-shadows the default.
        assert "if (candidate && modelSelect.querySelector" in html

    def test_chat_context_suggestions_offer_only_chat_models(self):
        """The 'file too large, try model X' hint enumerates models by
        context window; it must not suggest talking to a voice model."""
        seg = _function_source(_DASHBOARD / "chat.py", "chat_completions")
        assert "if not is_chat_model(m):" in seg

    def test_admin_chat_config_offers_only_chat_models(self):
        """Admin picks core/default CHAT models here; offering kokoro or
        FLUX invites configuring an unusable default."""
        seg = _function_source(_DASHBOARD / "routes.py", "admin_chat_config")
        assert "if is_chat_model(m) and" in seg

    def test_admin_backends_page_stays_unfiltered(self):
        """Admins manage every backend, image/video/voice included — the
        operational view must NOT gain the catalog filter by accident."""
        seg = _function_source(_DASHBOARD / "routes.py", "admin_backends")
        assert "is_catalog_model" not in seg
        assert "is_chat_model" not in seg

    def test_no_other_dashboard_surface_lists_all_backend_models(self):
        """Sweep at function granularity (one file mixes public and admin
        surfaces): every function that enumerates models — from the registry
        or from the DB — must contain a filter CALL or carry an explicit
        exemption here. Covers subpackages and methods of module-level
        classes; nested defs are covered via their parent's source."""
        LISTING_CALLS = (
            "get_backend_models",            # registry
            "get_models_grouped_by_name",    # DB
            "get_all_models_with_backends",  # DB
        )
        EXEMPT = {
            # admin operational/config views: seeing every model is the point
            ("routes.py", "admin_backends"),
            ("routes.py", "admin_models"),
            ("routes.py", "admin_ocr_config"),
            ("routes.py", "admin_images_config"),
            ("routes.py", "admin_video_config"),
            ("routes.py", "admin_mesh_config"),
        }
        offenders = []
        for path in sorted(_DASHBOARD.rglob("*.py")):
            src = path.read_text()
            funcs = []
            for node in ast.parse(src).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs.append(node)
                elif isinstance(node, ast.ClassDef):
                    funcs.extend(
                        n for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    )
            for node in funcs:
                seg = _strip_comments(ast.get_source_segment(src, node) or "")
                if not any(call in seg for call in LISTING_CALLS):
                    continue
                if (path.name, node.name) in EXEMPT:
                    continue
                if _FILTER_CALL.search(seg):
                    continue
                offenders.append(f"{path.name}:{node.name}")
        assert not offenders, (
            "these dashboard functions list models without a modality "
            f"filter call or an explicit exemption: {offenders}"
        )


class TestEveryCatalogAppliesTheFilter:
    """Three endpoints publish the catalog. If one forgets the filter, the
    models reappear on that surface only — a drift that is easy to miss."""

    def test_v1_models_filters(self):
        src = (_APP / "api" / "models_api.py").read_text()
        assert "if not is_catalog_model(model):" in src

    def test_api_tags_filters(self):
        src = (_APP / "api" / "ollama_api.py").read_text()
        assert "is_catalog_model" in src, "/api/tags must apply the same filter"

    def test_anthropic_models_delegates(self):
        """It reuses list_models, so it inherits the filter — assert the
        delegation rather than a duplicated filter."""
        src = (_APP / "api" / "anthropic_api.py").read_text()
        i = src.index("async def anthropic_models")
        assert "list_models" in src[i:i + 600]

    def test_no_other_endpoint_lists_all_backend_models(self):
        """Catch a fourth catalog appearing without the filter."""
        offenders = []
        for path in (_APP / "api").glob("*.py"):
            src = path.read_text()
            if "get_backend_models" not in src:
                continue
            if path.name in ("models_api.py", "ollama_api.py", "health.py"):
                continue  # these apply is_catalog_model
            if path.name == "telemetry_api.py":
                # admin operational telemetry: seeing every backend's models,
                # including image/video/voice, is the point
                continue
            # video/image/mesh discovery filter by engine instead, which is fine
            if re.search(r"engine.*!=.*BackendEngine\.(VIDEO|DIFFUSION|MESH)", src):
                continue
            offenders.append(path.name)
        assert not offenders, (
            f"these list backend models without a modality or engine filter: {offenders}"
        )


class TestPerModalityDiscoveryExists:
    """Filtering is only acceptable because each excluded family has its own
    endpoint — otherwise clients have no way to discover those models."""

    def test_images_have_a_discovery_endpoint(self):
        src = (_APP / "api" / "v1_openai.py").read_text()
        assert '@router.get("/images/models")' in src

    def test_videos_have_a_discovery_endpoint(self):
        src = (_APP / "api" / "video_api.py").read_text()
        assert '@router.get("/videos/models")' in src

    def test_meshes_have_a_discovery_endpoint(self):
        src = (_APP / "api" / "mesh_api.py").read_text()
        assert '@router.get("/meshes/models")' in src
        assert "!= BackendEngine.MESH" in src

    def test_image_endpoint_selects_by_diffusion_engine(self):
        src = (_APP / "api" / "v1_openai.py").read_text()
        i = src.index('@router.get("/images/models")')
        block = src[i:i + 1400]
        assert "BackendEngine.DIFFUSION" in block

    def test_docs_no_longer_point_images_at_v1_models(self):
        html = (_APP / "dashboard" / "templates" / "public" / "documentation.html").read_text()
        assert "Image models appear in <code>GET /v1/models</code>" not in html
        assert "/v1/images/models" in html
