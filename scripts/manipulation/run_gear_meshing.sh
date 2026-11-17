#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python scripts/manipulation/isaac_smoke.py --task gear_meshing "$@"
