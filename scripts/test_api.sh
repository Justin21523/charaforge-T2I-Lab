#!/bin/bash
# scripts/test_api.sh - CharaForge T2I API Testing Script

set -euo pipefail

# Configuration
API_URL=${API_URL:-"http://localhost:8000"}
OUTPUT_DIR=${OUTPUT_DIR:-"test_outputs"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=${3:-""}
    local expected_codes=${4:-"200"}
    local test_name=$5

    log_info "Testing: $test_name"

    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_URL$endpoint")
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    IFS=',' read -ra codes <<< "$expected_codes"
    for code in "${codes[@]}"; do
        if [ "$http_code" -eq "$code" ]; then
            log_success "$test_name (HTTP $http_code)"
            return 0
        fi
    done

    log_error "$test_name (HTTP $http_code, expected: $expected_codes)"
    echo "Response: $body"
    return 1
}

echo "🧪 CharaForge T2I API Test Suite"
echo "================================"
echo "API URL: $API_URL"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

TESTS_PASSED=0
TESTS_TOTAL=0

run_test() {
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    if "$@"; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
}

# Basic endpoints
run_test test_endpoint "GET" "/api/v1/health" "" "200" "Health Check"
run_test test_endpoint "GET" "/" "" "200" "Root Endpoint"

# Routers that should always be present
run_test test_endpoint "GET" "/api/v1/models" "" "200" "Models List"
run_test test_endpoint "GET" "/api/v1/datasets/root" "" "200" "Datasets Root"
run_test test_endpoint "GET" "/api/v1/datasets/list" "" "200" "Datasets List"
run_test test_endpoint "GET" "/api/v1/lora/list" "" "200" "LoRA List"
run_test test_endpoint "GET" "/api/v1/lora/status" "" "200" "LoRA Status"

# LoRA load should fail gracefully for unknown ids (no model download required).
run_test test_endpoint "POST" "/api/v1/lora/load" '{"lora_id":"missing_lora","weight":1.0}' "404" "LoRA Load (missing)"

# T2I generation may return 503 when no local base model is available.
generation_data='{
  "prompt": "A simple test image",
  "model_type": "sd15",
  "width": 256,
  "height": 256,
  "steps": 1,
  "batch_size": 1
}'
run_test test_endpoint "POST" "/api/v1/t2i/generate" "$generation_data" "200,503" "T2I Generate (200 or 503)"

# Training submission may return 400 when dataset does not exist (expected).
training_data='{
  "project_name": "api_test_training",
  "dataset_path": "does_not_exist",
  "instance_prompt": "a test image",
  "model_type": "sd15",
  "num_train_epochs": 1
}'
run_test test_endpoint "POST" "/api/v1/finetune/lora/train" "$training_data" "200,400" "LoRA Train Submit (200 or 400)"

# Performance
echo ""
log_info "Running performance tests..."

log_info "Testing: API Response Time"
start_time=$(date +%s%N)
curl -s "$API_URL/api/v1/health" > /dev/null
end_time=$(date +%s%N)
response_time_ms=$(( (end_time - start_time) / 1000000 ))

if [ $response_time_ms -lt 1000 ]; then
    log_success "API Response Time: ${response_time_ms}ms (good)"
elif [ $response_time_ms -lt 5000 ]; then
    log_warn "API Response Time: ${response_time_ms}ms (acceptable)"
else
    log_error "API Response Time: ${response_time_ms}ms (slow)"
fi

log_info "Testing: Basic Load (10 concurrent /api/v1/health requests)"
for i in {1..10}; do
    curl -s "$API_URL/api/v1/health" &
done
wait
log_success "Basic Load Test completed"

echo ""
echo "================================"
echo "🧪 API Test Results"
echo "================================"
echo "Tests Passed: $TESTS_PASSED/$TESTS_TOTAL"

if [ $TESTS_PASSED -eq $TESTS_TOTAL ]; then
    echo "Status: ✅ ALL TESTS PASSED"
    exit 0
elif [ $TESTS_PASSED -ge $((TESTS_TOTAL * 80 / 100)) ]; then
    echo "Status: 🟡 MOST TESTS PASSED"
    exit 1
else
    echo "Status: ❌ MANY TESTS FAILED"
    exit 2
fi
