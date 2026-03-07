#!/usr/bin/env bash
#
# Generate MariaDB TDE encryption keys for primary and archive databases.
# Uses file_key_management plugin format: keyfile encrypted with AES-256-CBC.
#
# Usage: bash scripts/generate_encryption_keys.sh
#
# Idempotent — skips generation if key files already exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

generate_keys() {
    local name="$1"
    local dir="${PROJECT_ROOT}/mariadb/encryption/${name}"

    if [[ -f "${dir}/keyfile.enc" && -f "${dir}/keyfile.key" ]]; then
        echo "[skip] ${name} keys already exist at ${dir}"
        return 0
    fi

    echo "[gen]  Generating ${name} encryption keys..."
    mkdir -p "$dir"

    # Generate the key-encryption-key (used to encrypt the keyfile at rest)
    openssl rand -hex 32 > "${dir}/keyfile.key"

    # Generate the data encryption key (AES-256 = 32 bytes = 64 hex chars)
    local data_key
    data_key="$(openssl rand -hex 32)"

    # Create plaintext keyfile: <key_id>;<hex_key>
    local plaintext_keyfile
    plaintext_keyfile="$(mktemp)"
    echo "1;${data_key}" > "$plaintext_keyfile"

    # Encrypt the keyfile with the key-encryption-key
    openssl enc -aes-256-cbc -md sha512 -pbkdf2 \
        -pass "file:${dir}/keyfile.key" \
        -in "$plaintext_keyfile" \
        -out "${dir}/keyfile.enc"

    # Clean up plaintext
    rm -f "$plaintext_keyfile"

    # Restrict permissions
    chmod 600 "${dir}/keyfile.key" "${dir}/keyfile.enc"

    echo "[done] ${name} keys written to ${dir}"
}

generate_keys "primary"
generate_keys "archive"

echo ""
echo "=== Key generation complete ==="
echo ""
echo "IMPORTANT: Back up these key files to a secure offline location."
echo "Without them, encrypted database data is permanently unrecoverable."
echo ""
echo "  mariadb/encryption/primary/keyfile.key"
echo "  mariadb/encryption/primary/keyfile.enc"
echo "  mariadb/encryption/archive/keyfile.key"
echo "  mariadb/encryption/archive/keyfile.enc"
