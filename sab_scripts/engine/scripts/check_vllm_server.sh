#!/bin/bash
# vLLM Server Health Check Script
# Verifies server status and tests completions
#
# Usage:
#   bash check_vllm_server.sh [HOST] [PORT]
#
# Examples:
#   bash check_vllm_server.sh                    # Uses localhost:8000
#   bash check_vllm_server.sh gpu-node-01 8000   # Specific host

HOST=${1:-localhost}
PORT=${2:-8000}
BASE_URL="http://${HOST}:${PORT}/v1"

echo "=============================================="
echo "vLLM Server Health Check"
echo "=============================================="
echo "Checking: $BASE_URL"
echo ""

# Check if server is responding
echo "1. Checking server health..."
if curl -s --max-time 5 "${BASE_URL}/models" > /dev/null 2>&1; then
    echo "   [OK] Server is responding"
else
    echo "   [FAIL] Server is not responding"
    echo "   Make sure the vLLM server is running at $BASE_URL"
    exit 1
fi

# Get available models
echo ""
echo "2. Available models:"
MODELS=$(curl -s "${BASE_URL}/models" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "$MODELS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for model in data.get('data', []):
        print(f\"   - {model.get('id', 'unknown')}\")
except:
    print('   [WARN] Could not parse model list')
" 2>/dev/null || echo "   [WARN] Could not parse model list"
fi

# Test a simple completion
echo ""
echo "3. Testing chat completion..."
MODEL_ID=$(curl -s "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('data'):
        print(data['data'][0].get('id', ''))
except:
    pass
" 2>/dev/null)

if [ -n "$MODEL_ID" ]; then
    RESPONSE=$(curl -s --max-time 60 "${BASE_URL}/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_ID\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word.\"}],
            \"max_tokens\": 10,
            \"temperature\": 0.1
        }" 2>/dev/null)

    if echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    print(f'   [OK] Response: {content[:50]}...' if len(content) > 50 else f'   [OK] Response: {content}')
except Exception as e:
    print(f'   [FAIL] Could not parse response')
    sys.exit(1)
" 2>/dev/null; then
        :
    else
        echo "   [FAIL] Completion test failed"
        echo "   Response: $RESPONSE"
    fi
else
    echo "   [SKIP] No model available for testing"
fi

echo ""
echo "=============================================="
echo "Health check complete"
echo ""
echo "To use this server for inference, set:"
echo "  export VLLM_API_BASE=$BASE_URL"
echo "  export VLLM_MODEL_NAME=$MODEL_ID"
echo "=============================================="
