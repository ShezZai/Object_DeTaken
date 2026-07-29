#!/usr/bin/env bash
# Create a local virtual environment and install requirements.txt into it.
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIRECTORY="${SCRIPT_DIRECTORY}/.venv"
PYTHON="${PYTHON:-python3}"

if [ ! -d "${VENV_DIRECTORY}" ]; then
    echo "Creating virtual environment at ${VENV_DIRECTORY}"
    "${PYTHON}" -m venv "${VENV_DIRECTORY}"
else
    echo "Virtual environment already exists at ${VENV_DIRECTORY}"
fi

echo "Installing dependencies from requirements.txt"
"${VENV_DIRECTORY}/bin/pip" install --upgrade pip
"${VENV_DIRECTORY}/bin/pip" install -r "${SCRIPT_DIRECTORY}/requirements.txt"

echo
echo "Done. Activate the environment with:"
echo "  source .venv/bin/activate"
