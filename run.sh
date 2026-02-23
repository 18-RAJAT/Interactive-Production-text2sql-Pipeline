#!/bin/bash
set -e

PYTHON="${PYTHON:-python3}"
VENV_DIR="venv"
FRONTEND_DIR="frontend"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

cleanup() {
    if [ -n "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup INT TERM

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "[1/5] Virtual environment exists."
fi

source "$VENV_DIR/bin/activate"

echo "[2/5] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e .

echo "[3/5] Installing frontend dependencies..."
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    npm install --silent
else
    echo "  node_modules exists, skipping."
fi
cd ..

echo "[4/5] Starting API server on $API_HOST:$API_PORT..."
$PYTHON -m uvicorn api.serve:app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!
sleep 3

if kill -0 "$API_PID" 2>/dev/null; then
    echo "  API server running (PID $API_PID)"
else
    echo "  ERROR: API server failed to start."
    exit 1
fi

echo "[5/5] Starting frontend on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR"
NEXT_PUBLIC_API_URL="http://localhost:$API_PORT" npx next dev -p "$FRONTEND_PORT" &
FRONTEND_PID=$!
cd ..

echo ""
echo "==================================="
echo "  API:      http://localhost:$API_PORT"
echo "  Frontend: http://localhost:$FRONTEND_PORT"
echo "  Press Ctrl+C to stop."
echo "==================================="

wait