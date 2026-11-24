############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mesh_models.py: What the gateway knows about each image-to-mesh
#     model family: its per-job settings (mirrors ENGINE_OPTIONS in
#     mesh-worker/config.py; a contract test keeps them equal), the
#     recommended per-model caps, and the license terms the admin page,
#     the 3D tab and /v1/meshes show. Pure data + helpers, no I/O.
#
############################################################

"""Image-to-mesh model families: settings, caps and licenses."""

from typing import Any, Dict, Optional

# Served model name per family (the worker's DEFAULT_MODEL).
FAMILY_MODEL_IDS: Dict[str, str] = {
    "hunyuan3d21": "tencent/hunyuan3d-2.1",
    "trellis2": "microsoft/trellis.2-4b",
    "pixal3d": "tencentarc/pixal3d",
}

FAMILY_LABELS: Dict[str, str] = {
    "hunyuan3d21": "Hunyuan3D 2.1 (Tencent)",
    "trellis2": "TRELLIS.2 (Microsoft)",
    "pixal3d": "Pixal3D (Tencent ARC)",
}

# Per-job settings an admin can set per model, sent to the worker as
# "options". MUST equal ENGINE_OPTIONS in mesh-worker/config.py.
FAMILY_OPTIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "trellis2": {
        "pipeline_type": {
            "type": "choice", "choices": ["512", "1024", "1024_cascade", "1536_cascade"],
            "description": "Generation resolution; the cascades refine a coarse shape (1536 is the most detailed and slowest)",
        },
    },
    "pixal3d": {
        "pipeline_type": {
            "type": "choice", "choices": ["1024_cascade", "1536_cascade"],
            "description": "Generation resolution (1536 is the most detailed and slowest)",
        },
    },
    "hunyuan3d21": {
        "octree_resolution": {
            "type": "choice", "choices": [256, 384, 512],
            "description": "Surface-extraction grid; higher keeps finer shape detail but meshes and paints much more slowly",
        },
        "steps": {
            "type": "int", "min": 5, "max": 50,
            "description": "Shape diffusion steps",
        },
        "guidance_scale": {
            "type": "float", "min": 1.0, "max": 10.0,
            "description": "How closely the shape follows the picture",
        },
    },
}

# The worker's defaults when an option is left unset (its env defaults).
FAMILY_OPTION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "trellis2": {"pipeline_type": "1024_cascade"},
    "pixal3d": {"pipeline_type": "1536_cascade"},
    "hunyuan3d21": {"octree_resolution": 256, "steps": 30, "guidance_scale": 5.0},
}

# Recommended per-model caps (shown on the admin page; the admin decides).
# Hunyuan3D's mesh cleanup and texture paint run on the CPU and grow with the
# face count: a 1M-face, 4096 px job held its GPU for about 11.5 minutes.
RECOMMENDED_CAPS: Dict[str, Dict[str, int]] = {
    "hunyuan3d21": {"max_faces": 200_000, "max_texture_size": 4096},
    "trellis2": {"max_faces": 500_000, "max_texture_size": 4096},
    "pixal3d": {"max_faces": 500_000, "max_texture_size": 4096},
}

HUNYUAN_LICENSE_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE"

LICENSES: Dict[str, Dict[str, Any]] = {
    "hunyuan3d21": {
        "name": "Tencent Hunyuan 3D 2.1 Community License Agreement",
        "url": HUNYUAN_LICENSE_URL,
        "requires_acceptance": True,
        # What the license asks of an operator offering it as a hosted service.
        "obligations": [
            "Territory: the license does not apply in the European Union, the United Kingdom or "
            "South Korea. The model and its outputs may not be used, reproduced or displayed there.",
            "Provider disclosure: users must be told clearly who provides the service (the "
            "University of Idaho), and that Tencent is not affiliated with, associated with, "
            "sponsoring or endorsing it. Tencent's names and marks may not be used in a way that "
            "suggests otherwise.",
            "A copy of the license must be made available to the people who use the service.",
            "Outputs may not be used to improve any other AI model (only Hunyuan3D 2.1 or its "
            "derivatives).",
            "Use must follow Tencent's Acceptable Use Policy (part of the license): among other "
            "things, no impersonation without consent, no high-stakes automated decisions, no "
            "military use, and machine-generated content placed in public must be identified as such.",
            "If the operator's products had more than 1 million monthly active users when 2.1 was "
            "released, a separate license from Tencent is required.",
            "Tencent may terminate the license if its terms are breached.",
        ],
        # Shown to users wherever the model is offered (3D tab, API).
        "user_notice": (
            "Hunyuan3D 2.1 is provided here by the University of Idaho (MindRouter). Tencent is not "
            "affiliated with, associated with, sponsoring or endorsing this service. Its use is "
            "governed by the Tencent Hunyuan 3D 2.1 Community License Agreement, including its "
            "Acceptable Use Policy: it is not licensed for use in the European Union, the United "
            "Kingdom or South Korea, and its outputs may not be used to improve other AI models."
        ),
    },
    "trellis2": {
        "name": "MIT License",
        "url": "https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE",
        "requires_acceptance": False,
        "obligations": [
            "MIT: keep the copyright and license notice with copies of the code.",
            "Its image encoder is Meta's gated DINOv3: an administrator must accept the DINOv3 "
            "license on Hugging Face before staging its weights.",
            "Upstream's background remover (RMBG-2.0, non-commercial) is never used: the worker "
            "swaps in BiRefNet (MIT).",
        ],
        "user_notice": "",
    },
    "pixal3d": {
        "name": "MIT License",
        "url": "https://github.com/TencentARC/Pixal3D/blob/main/LICENSE",
        "requires_acceptance": False,
        "obligations": [
            "MIT: keep the copyright and license notice with copies of the code.",
            "Built on TRELLIS.2, so it also uses Meta's gated DINOv3 (accept its license before "
            "staging the weights); RMBG-2.0 is replaced by BiRefNet (MIT).",
        ],
        "user_notice": "",
    },
}


def family_for_model(name: Optional[str]) -> Optional[str]:
    """The family a served model name belongs to (exact id first, then a
    name hint for renamed deployments); None for unknown models."""
    n = (name or "").strip().lower()
    for family, model_id in FAMILY_MODEL_IDS.items():
        if n == model_id:
            return family
    if "hunyuan3d" in n:
        return "hunyuan3d21"
    if "pixal3d" in n:
        return "pixal3d"
    if "trellis" in n:
        return "trellis2"
    return None


def license_for_model(name: Optional[str]) -> Optional[Dict[str, Any]]:
    family = family_for_model(name)
    return LICENSES.get(family) if family else None


def clean_options(family: Optional[str], raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate admin-entered settings for a family (form strings allowed).
    Returns only the settings that were given; raises ValueError naming the
    first bad one. Unknown families take no settings."""
    schema = FAMILY_OPTIONS.get(family or "", {})
    out: Dict[str, Any] = {}
    for name, value in (raw or {}).items():
        if name not in schema:
            raise ValueError(f"unknown setting '{name}'")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        spec = schema[name]
        kind = spec["type"]
        try:
            if kind == "choice":
                choices = spec["choices"]
                cast = type(choices[0])
                value = cast(value)
                if value not in choices:
                    raise ValueError
            elif kind == "int":
                value = int(str(value).strip())
                if not spec["min"] <= value <= spec["max"]:
                    raise ValueError
            elif kind == "float":
                value = float(str(value).strip())
                if not spec["min"] <= value <= spec["max"]:
                    raise ValueError
        except (TypeError, ValueError):
            allowed = (
                ", ".join(str(c) for c in spec["choices"]) if kind == "choice"
                else f"{spec['min']} to {spec['max']}"
            )
            raise ValueError(f"'{name}' must be {allowed}") from None
        out[name] = value
    return out


def model_limits(base: Dict[str, Any], settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-model limits: the service-wide ones narrowed by the model's own caps
    (Admin -> Meshes). The default face count is clamped into the range."""
    settings = settings or {}
    out = dict(base)
    cap_faces = settings.get("max_faces")
    if cap_faces:
        out["max_faces"] = max(out["min_faces"], min(out["max_faces"], int(cap_faces)))
    cap_tex = settings.get("max_texture_size")
    if cap_tex:
        sizes = [s for s in out["texture_sizes"] if s <= int(cap_tex)] or [min(out["texture_sizes"])]
        out["texture_sizes"] = sizes
        if out["default_texture_size"] not in sizes:
            out["default_texture_size"] = max(sizes)
    out["default_faces"] = max(out["min_faces"], min(out["max_faces"], out["default_faces"]))
    return out
