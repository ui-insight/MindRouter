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
#   - /dev/ipmi0 for server power monitoring (host `ipmi` group + --group-add,
#     because the container runs as non-root appuser)
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
# Identity for the aspen jump; override if your key lives elsewhere.
ASPEN_KEY="${ASPEN_KEY:-$HOME/.ssh/id_aspen}"

ALL_NODES="aspen1 aspen2 aspen3 aspen4 aspen5 aspen300 marten lynx calvin eunice webbyg1 webbyg2 neuromancer wintermute"

# SSH target. Most nodes resolve by bare hostname from the operator's
# ~/.ssh/config; wintermute does not, and is only reachable through the
# gateway, so it carries its FQDN and a ProxyJump below.
node_host() {
    case "$1" in
        aspen[1-5]|aspen300) echo "$1.hpc.uidaho.edu" ;;
        wintermute) echo "wintermute.nkn.uidaho.edu" ;;
        *)          echo "$1" ;;
    esac
}

# Host port the sidecar container publishes, and therefore the port the node's
# reverse proxy points at. The aspens were standardised on 18207; every other
# node uses 18007. Deploying with the wrong one starts the container somewhere
# the node's own nginx is not looking, which takes the node offline in
# MindRouter with a perfectly healthy container running.
sidecar_port() {
    case "$1" in
        aspen[1-5]|aspen300) echo "18207" ;;
        *)          echo "18007" ;;
    esac
}

# Fully-qualified name, used for TLS probes. The node certificates are issued
# for the hostname, so a probe against a bare name or 127.0.0.1 fails
# verification even when the proxy is healthy — which reads as an outage.
node_fqdn() {
    case "$1" in
        aspen[1-5]|aspen300|marten|lynx|webbyg1|webbyg2) echo "$1.hpc.uidaho.edu" ;;
        calvin|eunice|aurora|neuromancer|wintermute) echo "$1.nkn.uidaho.edu" ;;
        *) echo "$1" ;;
    esac
}

# Extra ssh/scp options per node. Deliberately empty for the existing nodes —
# they already work via ~/.ssh/config and changing that would risk breaking a
# working deploy path for a cosmetic gain.
ssh_opts() {
    case "$1" in
        # The aspens do not accept SSH from outside the cluster (port 22 times
        # out), so a bare `ssh sheneman@aspenN` — what this script used to do —
        # reports every one of them as an unreachable, dead sidecar.
        aspen[1-5]|aspen300) echo "-o ProxyJump=$SSH_USER@lynx.hpc.uidaho.edu -i $ASPEN_KEY" ;;
        wintermute) echo "-o ProxyJump=mindrouter@mindrouter.uidaho.edu" ;;
        *)          echo "" ;;
    esac
}

# Wrappers so every remote call picks up the host mapping and jump host.
nssh() { local n="$1"; shift; ssh $(ssh_opts "$n") "$SSH_USER@$(node_host "$n")" "$@"; }
nscp() { local n="$1"; local dst="$1"; shift; local last="${@: -1}"; set -- "${@:1:$#-1}";
         scp -q $(ssh_opts "$n") "$@" "$SSH_USER@$(node_host "$n"):$last"; }

# Per-node sidecar keys.
#
# SECURITY: this script no longer carries any keys. The twelve per-node keys
# that USED to live in an in-repo `case` table here (marten, aspen1-5, lynx,
# webbyg1/2, calvin, eunice, neuromancer) were committed to git history and
# MUST BE TREATED AS COMPROMISED — a sidecar key is not read-only, it also
# gates the node's model pull/delete endpoints. ROTATE ALL TWELVE as an
# operational step: mint a fresh key per node, update the MindRouter backend
# registration, and supply it to this script out of band via one of the two
# paths below. (git history still contains the old values; rotation is what
# neutralises them, not deleting the table.)
#
# Key resolution order (no in-repo fallback remains):
#
#   1. $SIDECAR_KEY_<NODE>            e.g. SIDECAR_KEY_WINTERMUTE=...
#   2. $SIDECAR_KEY_FILE              lines of "<node> <key>", default
#                                     /etc/mindrouter/sidecar-keys (mode 0600)
#
# If neither yields a key, node_key fails hard rather than deploying with a
# known-bad or absent secret.
SIDECAR_KEY_FILE="${SIDECAR_KEY_FILE:-/etc/mindrouter/sidecar-keys}"

node_key() {
    local envvar
    envvar="SIDECAR_KEY_$(printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_')"
    if [ -n "${!envvar:-}" ]; then echo "${!envvar}"; return 0; fi

    if [ -r "$SIDECAR_KEY_FILE" ]; then
        local k
        k=$(awk -v n="$1" '$1 == n { print $2; exit }' "$SIDECAR_KEY_FILE")
        if [ -n "$k" ]; then echo "$k"; return 0; fi
    fi

    echo "ERROR: no key for '$1'; set SIDECAR_KEY_$(printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_') or add it to \$SIDECAR_KEY_FILE ($SIDECAR_KEY_FILE)" >&2
    return 1
}

# Old sidecar paths to clean up
old_paths() {
    case "$1" in
        aspen[1-5]) echo "/scratch/mindrouter2/sidecar" ;;
        # Never had a pre-standardisation location; returning REMOTE_DIR makes
        # the cleanup step a no-op by construction rather than by luck.
        wintermute) echo "$REMOTE_DIR" ;;
        *)          echo "/space/mindrouter/sidecar" ;;
    esac
}

verify_node() {
    local node="$1"
    local key
    if ! key="$(node_key "$node")"; then
        printf "  %-14s NO KEY (set SIDECAR_KEY_<NODE> or add to \$SIDECAR_KEY_FILE)\n" "$node"
        return 1
    fi

    # Internal (container direct)
    local iport
    iport="$(sidecar_port "$node")"
    local internal
    internal=$(nssh "$node" \
        "curl -sf -m5 -H 'X-Sidecar-Key: $key' http://127.0.0.1:$iport/health 2>&1" || echo "UNREACHABLE")

    # Via the reverse proxy. Some nodes terminate TLS on 8007 (neuromancer,
    # wintermute) and some serve it plain, so try https first and fall back —
    # probing only http reports a working TLS node as UNREACHABLE.
    #    The TLS probe must use the FQDN: the node certificate is issued for
    #    the hostname, so hitting 127.0.0.1 fails verification even when the
    #    proxy is perfectly healthy.
    local fqdn
    fqdn="$(node_fqdn "$node")"
    local external
    external=$(nssh "$node" \
        "curl -sf -m5 -H 'X-Sidecar-Key: $key' https://$fqdn:8007/health 2>/dev/null \
         || curl -sf -m5 -H 'X-Sidecar-Key: $key' http://127.0.0.1:8007/health 2>&1" || echo "UNREACHABLE")

    # Server power
    local power
    power=$(nssh "$node" \
        "curl -sf -m10 -H 'X-Sidecar-Key: $key' https://$fqdn:8007/gpu-info 2>/dev/null \
         || curl -sf -m10 -H 'X-Sidecar-Key: $key' http://127.0.0.1:8007/gpu-info 2>/dev/null" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sp=d.get('server_power',{}); print(sp.get('instantaneous_watts', sp.get('error','N/A')))" 2>/dev/null \
        || echo "N/A")

    printf "  %-14s internal=%-40s nginx=%-40s power=%s W\n" "$node" "$internal" "$external" "$power"
}

deploy_node() {
    local node="$1"
    local key
    if ! key="$(node_key "$node")"; then
        return 1
    fi

    echo ""
    echo "=========================================="
    echo "  Deploying sidecar to $node"
    echo "=========================================="

    # 1. Copy sidecar files to /opt/mindrouter/sidecar/
    echo "  [$node] Copying files..."
    nscp "$node" "$SCRIPT_DIR/gpu_agent.py" \
                 "$SCRIPT_DIR/Dockerfile.sidecar" \
                 "$SCRIPT_DIR/requirements.txt" \
                 "$SCRIPT_DIR/VERSION" \
                 "~/"
    nssh "$node" "sudo mkdir -p $REMOTE_DIR && sudo cp ~/gpu_agent.py ~/Dockerfile.sidecar ~/requirements.txt ~/VERSION $REMOTE_DIR/"

    # 2. Create Docker network (10.200.0.0/24)
    #    neuromancer already has a 'mindrouter' network on this subnet (shared with voice containers)
    local net_name="$DOCKER_NETWORK"
    if nssh "$node" "sudo docker network inspect mindrouter >/dev/null 2>&1" && \
       [ "$node" != "" ] && \
       ! nssh "$node" "sudo docker network inspect $DOCKER_NETWORK >/dev/null 2>&1"; then
        # An existing 'mindrouter' network on the same subnet — reuse it
        net_name="mindrouter"
        echo "  [$node] Reusing existing 'mindrouter' Docker network..."
    else
        echo "  [$node] Ensuring Docker network ($DOCKER_SUBNET)..."
        nssh "$node" "sudo docker network inspect $DOCKER_NETWORK >/dev/null 2>&1 || \
            sudo docker network create --driver bridge \
            --subnet $DOCKER_SUBNET --gateway $DOCKER_GATEWAY \
            $DOCKER_NETWORK"
    fi

    # 3. Build Docker image
    echo "  [$node] Building Docker image..."
    nssh "$node" "cd $REMOTE_DIR && sudo docker build --network host --no-cache \
        -t mindrouter-sidecar:latest -f Dockerfile.sidecar . >/dev/null 2>&1"

    # 4. Stop and remove old container
    echo "  [$node] Removing old container..."
    nssh "$node" "sudo docker rm -f gpu-sidecar 2>/dev/null || true"

    # 5. Start new container
    #    The image runs as non-root appuser (uid 1000) since the 2.9.19 hardening
    #    pass, and /dev/ipmi0 is root:root 0600 on every node, so mapping the
    #    device in is NOT enough: ipmitool inside the container fails with
    #    "Could not open device" and the registry silently falls back to the
    #    GPU-power sum (that is how the aspens under-reported chassis power by
    #    ~2x from June to August 2026). Give the host an `ipmi` group that owns
    #    the device — via a udev rule so it survives reboots, plus an immediate
    #    chgrp/chmod — and add that gid to the container process.
    local ipmi_flag=""
    if nssh "$node" "test -c /dev/ipmi0" 2>/dev/null; then
        nssh "$node" "getent group ipmi >/dev/null || sudo groupadd -r ipmi; \
            echo 'KERNEL==\"ipmi[0-9]*\", GROUP=\"ipmi\", MODE=\"0660\"' | sudo tee /etc/udev/rules.d/99-mindrouter-ipmi.rules >/dev/null && \
            sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=ipmi; \
            sudo chgrp ipmi /dev/ipmi0 && sudo chmod 0660 /dev/ipmi0"
        local ipmi_gid
        ipmi_gid=$(nssh "$node" "getent group ipmi | cut -d: -f3")
        if [ -n "$ipmi_gid" ]; then
            ipmi_flag="--device /dev/ipmi0:/dev/ipmi0 --group-add $ipmi_gid"
        else
            echo "  [$node] WARNING: could not create/read the ipmi group; mapping device without group (power will fall back to GPU sum)"
            ipmi_flag="--device /dev/ipmi0:/dev/ipmi0"
        fi
    else
        echo "  [$node] WARNING: /dev/ipmi0 not found, skipping IPMI"
    fi

    echo "  [$node] Starting container (key=${key:0:8}...)..."
    nssh "$node" "sudo docker run -d \
        --name gpu-sidecar \
        --restart unless-stopped \
        --gpus all \
        $ipmi_flag \
        --network $net_name \
        -p 127.0.0.1:$(sidecar_port "$node"):8007 \
        -e SIDECAR_SECRET_KEY='$key' \
        mindrouter-sidecar:latest"

    # 6. Install/configure reverse proxy (:8007 → :18007)
    #    webbyg2 uses Apache (httpd) for multiple services — skip nginx there
    if nssh "$node" "test -f /etc/httpd/conf.d/sidecar-proxy.conf" 2>/dev/null; then
        echo "  [$node] Apache sidecar proxy already configured, skipping nginx..."
        nssh "$node" "sudo systemctl reload httpd 2>/dev/null || true"
    elif nssh "$node" "sudo grep -qi ' ssl' /etc/nginx/conf.d/sidecar-proxy.conf" 2>/dev/null; then
        # The node already terminates TLS on 8007. The template in this repo is
        # plain HTTP, so copying it over would silently downgrade the node and
        # break the https:// sidecar_url MindRouter has registered for it.
        echo "  [$node] Existing TLS sidecar proxy kept (template is plaintext)."
        nssh "$node" "sudo nginx -t >/dev/null 2>&1 && sudo systemctl reload nginx"
    else
        echo "  [$node] Configuring nginx..."
        nssh "$node" "command -v nginx >/dev/null 2>&1 || sudo dnf install -y nginx >/dev/null 2>&1"
        nscp "$node" "$SCRIPT_DIR/mindrouter-sidecar-nginx.conf" "/tmp/mindrouter-sidecar.conf"
        nssh "$node" "sudo cp /tmp/mindrouter-sidecar.conf /etc/nginx/conf.d/sidecar-proxy.conf && \
            sudo sed -i 's/127\\.0\\.0\\.1:18007/127.0.0.1:$(sidecar_port "$node")/' /etc/nginx/conf.d/sidecar-proxy.conf && \
            sudo nginx -t 2>/dev/null && sudo systemctl enable nginx >/dev/null 2>&1 && \
            (sudo systemctl is-active nginx >/dev/null 2>&1 && sudo systemctl reload nginx || sudo systemctl start nginx)"
    fi

    # 7. Clean up old source directories
    local old_path
    old_path="$(old_paths "$node")"
    if [ "$old_path" != "$REMOTE_DIR" ]; then
        nssh "$node" "[ -d '$old_path' ] && sudo rm -rf '$old_path' && echo '  [$node] Cleaned up $old_path' || true"
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
    # Optional node list after --verify, so a single node can be checked
    # without walking all of them over SSH.
    shift
    echo "Verification only:"
    for node in ${*:-$ALL_NODES}; do
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
