#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python scripts/manipulation/automate_smoke.py "$@"
