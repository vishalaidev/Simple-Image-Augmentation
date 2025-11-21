#!/usr/bin/env bash
set -euo pipefail

APP_NAME="djsvs-ai-augment"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using interpreter: ${PYTHON_BIN}"

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "pyinstaller not found. Install with: ${PYTHON_BIN} -m pip install pyinstaller"
  exit 1
fi

pyinstaller --noconfirm \
  --noconsole \
  --onefile \
  --hidden-import PIL._tkinter_finder \
  --name "${APP_NAME}" \
  augment.py

echo "Built binary: dist/${APP_NAME}"
