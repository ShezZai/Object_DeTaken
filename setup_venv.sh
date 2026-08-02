#!/usr/bin/env bash
# Thin wrapper: the real, cross-platform logic lives in setup_venv.py.
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "${SCRIPT_DIRECTORY}/setup_venv.py"
