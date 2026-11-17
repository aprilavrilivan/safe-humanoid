#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python scripts/manipulation/wipe_smoke.py "$@"
