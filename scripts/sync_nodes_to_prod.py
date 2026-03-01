#!/usr/bin/env python3
############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# sync_nodes_to_prod.py: Export nodes & backends from one
# database and import them into another.
#
# Usage:
#   # Export from dev to JSON:
#   python scripts/sync_nodes_to_prod.py export > nodes_export.json
#
#   # Import from JSON into current database:
#   python scripts/sync_nodes_to_prod.py import nodes_export.json
#
# The DATABASE_URL env var (or .env file) controls which
# database is targeted.
#
############################################################

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.db.session import get_async_db_context
from backend.app.db import crud
from backend.app.db.models import BackendEngine, BackendStatus, NodeStatus


async def export_data():
    """Export nodes and backends to JSON on stdout."""
    async with get_async_db_context() as db:
        nodes = await crud.get_all_nodes(db)
        backends = await crud.get_all_backends(db)

        data = {
            "nodes": [],
            "backends": [],
        }

        for n in nodes:
            data["nodes"].append({
                "name": n.name,
                "hostname": n.hostname,
                "sidecar_url": n.sidecar_url,
                "sidecar_key": n.sidecar_key,
                "gpu_count": n.gpu_count,
                "driver_version": n.driver_version,
                "cuda_version": n.cuda_version,
                "sidecar_version": n.sidecar_version,
            })

        for b in backends:
            node_name = None
            if b.node_id:
                # Find the node name for this backend
                for n in nodes:
                    if n.id == b.node_id:
                        node_name = n.name
                        break

            data["backends"].append({
                "name": b.name,
                "url": b.url,
                "engine": b.engine.value,
                "max_concurrent": b.max_concurrent,
                "gpu_memory_gb": b.gpu_memory_gb,
                "gpu_type": b.gpu_type,
                "priority": b.priority,
                "node_name": node_name,
                "gpu_indices": b.gpu_indices,
                "supports_multimodal": b.supports_multimodal,
                "supports_embeddings": b.supports_embeddings,
                "supports_structured_output": b.supports_structured_output,
            })

    print(json.dumps(data, indent=2))


async def import_data(filepath: str):
    """Import nodes and backends from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    async with get_async_db_context() as db:
        # Import nodes first (backends reference them)
        node_map = {}  # name -> db node object
        for ndata in data["nodes"]:
            existing = await crud.get_node_by_name(db, ndata["name"])
            if existing:
                print(f"  Node '{ndata['name']}' already exists (id={existing.id}), skipping...")
                node_map[ndata["name"]] = existing
            else:
                node = await crud.create_node(
                    db=db,
                    name=ndata["name"],
                    hostname=ndata.get("hostname"),
                    sidecar_url=ndata.get("sidecar_url"),
                    sidecar_key=ndata.get("sidecar_key"),
                )
                if ndata.get("gpu_count"):
                    await crud.update_node_hardware(
                        db=db,
                        node_id=node.id,
                        gpu_count=ndata["gpu_count"],
                        driver_version=ndata.get("driver_version"),
                        cuda_version=ndata.get("cuda_version"),
                        sidecar_version=ndata.get("sidecar_version"),
                    )
                node_map[ndata["name"]] = node
                print(f"  Created node: {ndata['name']}")

        await db.commit()

        # Import backends
        for bdata in data["backends"]:
            existing = await crud.get_backend_by_name(db, bdata["name"])
            if existing:
                print(f"  Backend '{bdata['name']}' already exists (id={existing.id}), skipping...")
                continue

            node_id = None
            if bdata.get("node_name") and bdata["node_name"] in node_map:
                node_id = node_map[bdata["node_name"]].id

            backend = await crud.create_backend(
                db=db,
                name=bdata["name"],
                url=bdata["url"],
                engine=BackendEngine(bdata["engine"]),
                max_concurrent=bdata.get("max_concurrent", 4),
                gpu_memory_gb=bdata.get("gpu_memory_gb"),
                gpu_type=bdata.get("gpu_type"),
                node_id=node_id,
                gpu_indices=bdata.get("gpu_indices"),
            )
            # Update fields not covered by create_backend
            if bdata.get("priority"):
                backend.priority = bdata["priority"]
            backend.supports_multimodal = bdata.get("supports_multimodal", False)
            backend.supports_embeddings = bdata.get("supports_embeddings", False)
            backend.supports_structured_output = bdata.get("supports_structured_output", True)
            db.add(backend)
            print(f"  Created backend: {bdata['name']} ({bdata['url']})")

        await db.commit()

    print("\nImport complete!")


async def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/sync_nodes_to_prod.py export > nodes_export.json")
        print("  python scripts/sync_nodes_to_prod.py import nodes_export.json")
        sys.exit(1)

    command = sys.argv[1]

    if command == "export":
        await export_data()
    elif command == "import":
        if len(sys.argv) < 3:
            print("Error: import requires a JSON file path")
            sys.exit(1)
        await import_data(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
