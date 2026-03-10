#!/bin/bash
# Stop vLLM Server Script
# Gracefully stops vLLM processes
#
# Usage:
#   bash stop_vllm_server.sh [PORT]
#
# Examples:
#   bash stop_vllm_server.sh        # Stops server on default port 8000
#   bash stop_vllm_server.sh 8001   # Stops server on port 8001

PORT=${1:-8000}

echo "=============================================="
echo "Stopping vLLM Server"
echo "=============================================="

# Find and kill vLLM processes on the specified port
echo "Looking for vLLM processes on port $PORT..."

# Method 1: Find by port
PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "Found processes on port $PORT: $PIDS"
    for PID in $PIDS; do
        echo "Killing process $PID..."
        kill $PID 2>/dev/null || true
    done
    sleep 2

    # Force kill if still running
    for PID in $PIDS; do
        if ps -p $PID > /dev/null 2>&1; then
            echo "Force killing process $PID..."
            kill -9 $PID 2>/dev/null || true
        fi
    done
fi

# Method 2: Find by process name (vllm.entrypoints)
VLLM_PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true)
if [ -n "$VLLM_PIDS" ]; then
    echo "Found vLLM server processes: $VLLM_PIDS"
    for PID in $VLLM_PIDS; do
        echo "Killing vLLM process $PID..."
        kill $PID 2>/dev/null || true
    done
    sleep 2

    # Force kill if still running
    for PID in $VLLM_PIDS; do
        if ps -p $PID > /dev/null 2>&1; then
            echo "Force killing process $PID..."
            kill -9 $PID 2>/dev/null || true
        fi
    done
fi

# Check if any processes are still running
sleep 1
REMAINING=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo ""
    echo "[WARN] Some processes may still be running on port $PORT"
    echo "PIDs: $REMAINING"
    echo "You may need to kill them manually: kill -9 $REMAINING"
else
    echo ""
    echo "[OK] vLLM server stopped successfully"
fi

echo "=============================================="
