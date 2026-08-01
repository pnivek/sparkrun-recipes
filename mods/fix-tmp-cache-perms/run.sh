#!/bin/bash
# fix-tmp-cache-perms — fix /tmp/.cache ownership in sparkrun containers.
#
# Known issue: sparkrun containers run as the host user, but /tmp/.cache/ can be
# root-owned from previous container runs. This breaks FlashInfer JIT compilation
# and vLLM model info caching, both of which try to write logs/cache there.
#
# https://forums.developer.nvidia.com/t/sparkrun-central-command/360832?page=5

set -e

if [ -d /tmp/.cache ]; then
    chown -R $(id -u):$(id -g) /tmp/.cache/ 2>/dev/null || true
    echo "[fix-tmp-cache-perms] /tmp/.cache ownership fixed"
fi
