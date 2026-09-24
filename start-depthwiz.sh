#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NODE_BIN="${NODE_BIN:-node}"
if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "ERROR: Node.js is required but was not found on PATH."
  exit 1
fi

exec "$NODE_BIN" "$SCRIPT_DIR/start-depthwiz.js" "$@"
