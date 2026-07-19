"""Contract test for idempotent backend registration (#8, source-check).

The register endpoint is DB/registry-heavy, so verify the upsert wiring by
source inspection (mirrors the install-friction tests' approach).
"""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ADMIN = os.path.join(ROOT, "backend/app/api/admin_api.py")


def _read(p):
    with open(p) as f:
        return f.read()


def test_backend_register_supports_upsert():
    src = _read(ADMIN)
    # schema flag
    assert "class BackendRegisterRequest" in src
    reg = src[src.index("class BackendRegisterRequest"):]
    reg = reg[: reg.index("class BackendResponse")]
    assert "upsert: bool = Field(" in reg

    # handler: upsert updates in place instead of 409; still 409 without upsert
    fn = src[src.index("async def register_backend"):]
    fn = fn[: fn.index("\n@router")]
    assert "upsert_target = (existing or existing_url) if request.upsert else None" in fn
    assert "await registry.update_backend(" in fn
    assert 'audit_action = "backend.upsert"' in fn
    # the 409 path is still there for non-upsert duplicates
    assert "HTTP_409_CONFLICT" in fn
    assert 'audit_action = "backend.register"' in fn
