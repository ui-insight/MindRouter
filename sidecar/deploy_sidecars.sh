#!/usr/bin/env bash
############################################################
#
# deploy_sidecars.sh — Deploy sidecar v2 to all GPU nodes
#
# Standardized deployment:
#   - Source at /opt/mindrouter/sidecar/ on every node
#   - Docker on 10.200.0.0/24 network
#   - Reverse proxy on :8007 → container :18007 (nginx or Apache)
#   - Per-node sidecar keys (from MindRouter DB)
#   - /dev/ipmi0 for server power monitoring
#
# Usage:
#   ./sidecar/deploy_sidecars.sh              # full deploy
#   ./sidecar/deploy_sidecars.sh --verify     # health check only
#   ./sidecar/deploy_sidecars.sh <node>       # deploy single node
#
############################################################
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/opt/mindrouter/sidecar"
DOCKER_NETWORK="mindrouter-sidecar"
DOCKER_SUBNET="10.200.0.0/24"
DOCKER_GATEWAY="10.200.0.1"
SSH_USER="sheneman"

ALL_NODES="aspen1 aspen2 aspen3 aspen4 aspen5 marten lynx calvin eunice webbyg1 webbyg2 neuromancer"

# Per-node sidecar keys (from MindRouter DB)
node_key() {
    case "$1" in
        marten)      echo "b0447d8d7efefac27a761783447e37dce366395b802f69c5e578014966507953" ;;
        aspen1)      echo "dd4bb8de54cb3ee1f732b1d21ac428180b0a76c6e5b740dcde5daf8e33d5fcbc" ;;
        aspen2)      echo "udiydhdy7d7d7hdjhxhxhxhxh67sdgo" ;;
        lynx)        echo "ce6187b6c36098a4a23f34c62b5112d2304000b4a9dd616fefec502e3a588428" ;;
        webbyg2)     echo "a2663b63a2b036a88f9bb56a332dfd019f34c6398b9825ea0ec5aa940adf4830" ;;
        webbyg1)     echo "67be9f8d222c05656af048d6dd81368237890ce43aab039a66736bd9429ca4b6" ;;
        aspen3)      echo "Y8wzaoMh1aqYfOrSqJtTpyzvCauu9gyEEFDuoMh_tcc" ;;
        aspen4)      echo "Y8wza1oMh1aqYfOrSqJtTpyzvCauu9gyEEFDuoMh_xxxt" ;;
        calvin)      echo "491023f96203f67cf3d86bf81aacb98604b657f78b53b955407d35f51f3006ef" ;;
        eunice)      echo "71f3997b67b423a52e243b081413f64591f1bf64272bf0e58f4cd03bc7506ee3" ;;
        aspen5)      echo "f0426bf2f23aa5d6810ebc233b4944bee462a83bd4eb7ebfffe6581cacbbd431" ;;
        neuromancer) echo "5871a710564c92d079a42ecdbbb3b0183f5a7accbf2ffc368f90421bf082896d" ;;
        *) echo "UNKNOWN_NODE"; return 1 ;;
    esac
}

# Old sidecar paths to clean up
old_paths() {
    case "$1" in
        aspen[1-5]) echo "/scratch/mindrouter2/sidecar" ;;
        *)          echo "/space/mindrouter/sidecar" ;;
    esac
}

verify_node() {
    local node="$1"
    local key
    key="$(node_key "$node")"

    # Internal (container direct)
    local internal
    internal=$(ssh $SSH_USER@"$node" \
        "curl -sf -m5 -H 'X-Sidecar-Key: $key' http://127.0.0.1:18007/health 2>&1" || echo "UNREACHABLE")

    # Via nginx proxy
    local external
    external=$(ssh $SSH_USER@"$node" \
        "curl -sf -m5 -H 'X-Sidecar-Key: $key' http://127.0.0.1:8007/health 2>&1" || echo "UNREACHABLE")

    # Server power
    local power
    power=$(ssh $SSH_USER@"$node" \
        "curl -sf -m10 -H 'X-Sidecar-Key: $key' http://127.0.0.1:8007/gpu-info 2>/dev/null" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sp=d.get('server_power',{}); print(sp.get('instantaneous_watts', sp.get('error','N/A')))" 2>/dev/null \
        || echo "N/A")

    printf "  %-14s internal=%-40s nginx=%-40s power=%s W\n" "$node" "$internal" "$external" "$power"
}

deploy_node() {
    local node="$1"
    local key
    key="$(node_key "$node")"

    echo ""
    echo "=========================================="
    echo "  Deploying sidecar to $node"
    echo "=========================================="

    # 1. Copy sidecar files to /opt/mindrouter/sidecar/
    echo "  [$node] Copying files..."
    scp -q "$SCRIPT_DIR/gpu_agent.py" \
           "$SCRIPT_DIR/Dockerfile.sidecar" \
           "$SCRIPT_DIR/requirements.txt" \
           "$SCRIPT_DIR/VERSION" \
           $SSH_USER@"$node":~/
    ssh $SSH_USER@"$node" "sudo mkdir -p $REMOTE_DIR && sudo cp ~/gpu_agent.py ~/Dockerfile.sidecar ~/requirements.txt ~/VERSION $REMOTE_DIR/"

    # 2. Create Docker network (10.200.0.0/24)
    #    neuromancer already has a 'mindrouter' network on this subnet (shared with voice containers)
    local net_name="$DOCKER_NETWORK"
    if ssh $SSH_USER@"$node" "sudo docker network inspect mindrouter >/dev/null 2>&1" && \
       [ "$node" != "" ] && \
       ! ssh $SSH_USER@"$node" "sudo docker network inspect $DOCKER_NETWORK >/dev/null 2>&1"; then
        # An existing 'mindrouter' network on the same subnet — reuse it
        net_name="mindrouter"
        echo "  [$node] Reusing existing 'mindrouter' Docker network..."
    else
        echo "  [$node] Ensuring Docker network ($DOCKER_SUBNET)..."
        ssh $SSH_USER@"$node" "sudo docker network inspect $DOCKER_NETWORK >/dev/null 2>&1 || \
            sudo docker network create --driver bridge \
            --subnet $DOCKER_SUBNET --gateway $DOCKER_GATEWAY \
            $DOCKER_NETWORK"
    fi

    # 3. Build Docker image
    echo "  [$node] Building Docker image..."
    ssh $SSH_USER@"$node" "cd $REMOTE_DIR && sudo docker build --network host --no-cache \
        -t mindrouter-sidecar:latest -f Dockerfile.sidecar . >/dev/null 2>&1"

    # 4. Stop and remove old container
    echo "  [$node] Removing old container..."
    ssh $SSH_USER@"$node" "sudo docker rm -f gpu-sidecar 2>/dev/null || true"

    # 5. Start new container
    local ipmi_flag=""
    if ssh $SSH_USER@"$node" "test -c /dev/ipmi0" 2>/dev/null; then
        ipmi_flag="--device /dev/ipmi0:/dev/ipmi0"
    else
        echo "  [$node] WARNING: /dev/ipmi0 not found, skipping IPMI"
    fi

    echo "  [$node] Starting container (key=${key:0:8}...)..."
    ssh $SSH_USER@"$node" "sudo docker run -d \
        --name gpu-sidecar \
        --restart unless-stopped \
        --gpus all \
        $ipmi_flag \
        --network $net_name \
        -p 127.0.0.1:18007:8007 \
        -e SIDECAR_SECRET_KEY='$key' \
        mindrouter-sidecar:latest"

    # 6. Install/configure reverse proxy (:8007 → :18007)
    #    webbyg2 uses Apache (httpd) for multiple services — skip nginx there
    if ssh $SSH_USER@"$node" "test -f /etc/httpd/conf.d/sidecar-proxy.conf" 2>/dev/null; then
        echo "  [$node] Apache sidecar proxy already configured, skipping nginx..."
        ssh $SSH_USER@"$node" "sudo systemctl reload httpd 2>/dev/null || true"
    else
        echo "  [$node] Configuring nginx..."
        ssh $SSH_USER@"$node" "command -v nginx >/dev/null 2>&1 || sudo dnf install -y nginx >/dev/null 2>&1"
        scp -q "$SCRIPT_DIR/mindrouter-sidecar-nginx.conf" $SSH_USER@"$node":/tmp/mindrouter-sidecar.conf
        ssh $SSH_USER@"$node" "sudo cp /tmp/mindrouter-sidecar.conf /etc/nginx/conf.d/sidecar-proxy.conf && \
            sudo nginx -t 2>/dev/null && sudo systemctl enable nginx >/dev/null 2>&1 && \
            (sudo systemctl is-active nginx >/dev/null 2>&1 && sudo systemctl reload nginx || sudo systemctl start nginx)"
    fi

    # 7. Clean up old source directories
    local old_path
    old_path="$(old_paths "$node")"
    if [ "$old_path" != "$REMOTE_DIR" ]; then
        ssh $SSH_USER@"$node" "[ -d '$old_path' ] && sudo rm -rf '$old_path' && echo '  [$node] Cleaned up $old_path' || true"
    fi

    # 8. Verify
    sleep 2
    echo "  [$node] Verifying..."
    verify_node "$node"
}

# --- Main ---

VERSION=$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null || echo "unknown")
echo "MindRouter Sidecar Deployment (v$VERSION)"
echo ""

if [ "${1:-}" = "--verify" ]; then
    echo "Verification only:"
    for node in $ALL_NODES; do
        verify_node "$node"
    done
    exit 0
fi

# Deploy specific node or all nodes
TARGETS="${1:-$ALL_NODES}"
FAILED=""

for node in $TARGETS; do
    if deploy_node "$node"; then
        echo "  [$node] DONE"
    else
        echo "  [$node] FAILED"
        FAILED="$FAILED $node"
    fi
done

echo ""
echo "=========================================="
echo "  Deployment summary"
echo "=========================================="
if [ -n "$FAILED" ]; then
    echo "  FAILED:$FAILED"
    exit 1
else
    echo "  All nodes deployed successfully"
fi
