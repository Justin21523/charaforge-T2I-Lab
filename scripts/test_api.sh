#!/bin/bash
# scripts/test_api.sh - CharaForge T2I API Testing Script

set -euo pipefail

# Configuration
API_URL=${API_URL:-"http://localhost:8000"}
OUTPUT_DIR=${OUTPUT_DIR:-"test_outputs"}
API_KEY=${API_KEY:-""}
API_KEY_HEADER=${API_KEY_HEADER:-"X-API-Key"}
TEST_MODELS_SCAN_ASYNC=${TEST_MODELS_SCAN_ASYNC:-"0"}

CURL_AUTH_ARGS=()
if [ -n "$API_KEY" ]; then
    CURL_AUTH_ARGS=(-H "${API_KEY_HEADER}: ${API_KEY}")
fi

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
        response=$(curl -s -w "\n%{http_code}" "${CURL_AUTH_ARGS[@]}" "$API_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            "${CURL_AUTH_ARGS[@]}" \
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

test_async_t2i_flow() {
    local payload=$1

    log_info "Testing: T2I Async Job Flow"

    local submit_response
    submit_response=$(curl -s -w "\n%{http_code}" -X "POST" \
        -H "Content-Type: application/json" \
        "${CURL_AUTH_ARGS[@]}" \
        -d "$payload" \
        "$API_URL/api/v1/t2i/submit")

    local submit_code
    submit_code=$(echo "$submit_response" | tail -n1)
    local submit_body
    submit_body=$(echo "$submit_response" | head -n -1)

    if [ "$submit_code" -ne 200 ]; then
        log_error "T2I Submit (HTTP $submit_code)"
        echo "Response: $submit_body"
        return 1
    fi

    local job_id
    job_id=$(
        python -c 'import json
import sys

raw = sys.stdin.read()
try:
    data = json.loads(raw) if raw else {}
except Exception:
    data = {}
print(data.get("job_id", "") or "")' <<<"$submit_body"
    )

    if [ -z "$job_id" ]; then
        log_error "T2I Submit missing job_id"
        echo "Response: $submit_body"
        return 1
    fi

    log_success "T2I Submit (job_id=$job_id)"

    local status_response
    status_response=$(curl -s -w "\n%{http_code}" "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/t2i/status/$job_id")
    local status_code
    status_code=$(echo "$status_response" | tail -n1)
    local status_body
    status_body=$(echo "$status_response" | head -n -1)

    if [ "$status_code" -ne 200 ]; then
        log_error "T2I Status (HTTP $status_code)"
        echo "Response: $status_body"
        return 1
    fi

    local status
    status=$(
        python -c 'import json
import sys

raw = sys.stdin.read()
try:
    data = json.loads(raw) if raw else {}
except Exception:
    data = {}
print(data.get("status", "") or "")' <<<"$status_body"
    )

    if [ -z "$status" ]; then
        log_error "T2I Status missing status field"
        echo "Response: $status_body"
        return 1
    fi

    log_success "T2I Status (status=$status)"

    # Optional: wait briefly for completion; if it doesn't finish, treat as warning.
    local final_status="$status"
    for _ in {1..15}; do
        if [[ "$final_status" == "succeeded" || "$final_status" == "failed" || "$final_status" == "canceled" ]]; then
            break
        fi
        sleep 1
        status_body=$(curl -s "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/t2i/status/$job_id")
        final_status=$(
            python -c 'import json
import sys

raw = sys.stdin.read()
try:
    data = json.loads(raw) if raw else {}
except Exception:
    data = {}
print(data.get("status", "") or "")' <<<"$status_body"
        )
    done

    if [[ "$final_status" == "queued" || "$final_status" == "running" ]]; then
        log_warn "T2I Job not finished yet (status=$final_status); attempting cancel"
        local cancel_response
        cancel_response=$(curl -s -w "\n%{http_code}" -X "POST" "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/t2i/cancel/$job_id")
        local cancel_code
        cancel_code=$(echo "$cancel_response" | tail -n1)
        local cancel_body
        cancel_body=$(echo "$cancel_response" | head -n -1)
        if [ "$cancel_code" -eq 200 ]; then
            log_success "T2I Cancel requested"
        else
            log_warn "T2I Cancel failed (HTTP $cancel_code)"
            echo "Response: $cancel_body"
        fi
    else
        log_success "T2I Job finished (status=$final_status)"
    fi

    return 0
}

test_async_models_scan_flow() {
    local payload=$1

    log_info "Testing: Models Scan Async Job Flow"

    local submit_response
    submit_response=$(curl -s -w "\n%{http_code}" -X "POST" \
        -H "Content-Type: application/json" \
        "${CURL_AUTH_ARGS[@]}" \
        -d "$payload" \
        "$API_URL/api/v1/models/scan/submit")

    local submit_code
    submit_code=$(echo "$submit_response" | tail -n1)
    local submit_body
    submit_body=$(echo "$submit_response" | head -n -1)

    if [ "$submit_code" -eq 403 ]; then
        log_warn "Models scan submit forbidden; skipping (provide an admin key to test this flow)"
        return 0
    fi
    if [ "$submit_code" -eq 409 ]; then
        log_warn "Models scan job already active; skipping (wait for the active scan to finish/cancel)"
        return 0
    fi
    if [ "$submit_code" -ne 200 ]; then
        log_error "Models scan submit (HTTP $submit_code)"
        echo "Response: $submit_body"
        return 1
    fi

    local job_id
    job_id=$(
        python -c 'import json
import sys

raw = sys.stdin.read()
try:
    data = json.loads(raw) if raw else {}
except Exception:
    data = {}
print(data.get("job_id", "") or "")' <<<"$submit_body"
    )

    if [ -z "$job_id" ]; then
        log_error "Models scan submit missing job_id"
        echo "Response: $submit_body"
        return 1
    fi

    log_success "Models scan submit (job_id=$job_id)"

    local status_response
    status_response=$(curl -s -w "\n%{http_code}" "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/models/scan/status/$job_id")
    local status_code
    status_code=$(echo "$status_response" | tail -n1)
    local status_body
    status_body=$(echo "$status_response" | head -n -1)

    if [ "$status_code" -ne 200 ]; then
        log_error "Models scan status (HTTP $status_code)"
        echo "Response: $status_body"
        return 1
    fi

    local status
    status=$(
        python -c 'import json
import sys

raw = sys.stdin.read()
try:
    data = json.loads(raw) if raw else {}
except Exception:
    data = {}
print(data.get("status", "") or "")' <<<"$status_body"
    )

    if [ -z "$status" ]; then
        log_error "Models scan status missing status field"
        echo "Response: $status_body"
        return 1
    fi

    log_success "Models scan status (status=$status)"

    # If the scan doesn't finish quickly, cancel it to avoid long-running I/O during smoke tests.
    local final_status="$status"
    for _ in {1..20}; do
        if [[ "$final_status" == "succeeded" || "$final_status" == "failed" || "$final_status" == "canceled" ]]; then
            break
        fi
        sleep 0.5
        status_body=$(curl -s "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/models/scan/status/$job_id")
        final_status=$(
            python -c 'import json
import sys

raw = sys.stdin.read()
try:
    data = json.loads(raw) if raw else {}
except Exception:
    data = {}
print(data.get("status", "") or "")' <<<"$status_body"
        )
    done

    if [[ "$final_status" == "queued" || "$final_status" == "running" ]]; then
        log_warn "Models scan job not finished yet (status=$final_status); attempting cancel"
        local cancel_response
        cancel_response=$(curl -s -w "\n%{http_code}" -X "POST" "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/models/scan/cancel/$job_id")
        local cancel_code
        cancel_code=$(echo "$cancel_response" | tail -n1)
        local cancel_body
        cancel_body=$(echo "$cancel_response" | head -n -1)
        if [ "$cancel_code" -eq 200 ]; then
            log_success "Models scan cancel requested"
        else
            log_warn "Models scan cancel failed (HTTP $cancel_code)"
            echo "Response: $cancel_body"
        fi

        final_status="$status"
        for _ in {1..40}; do
            sleep 0.25
            status_body=$(curl -s "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/models/scan/status/$job_id")
            final_status=$(
                python -c 'import json
import sys

raw = sys.stdin.read()
try:
    data = json.loads(raw) if raw else {}
except Exception:
    data = {}
print(data.get("status", "") or "")' <<<"$status_body"
            )
            if [[ "$final_status" == "succeeded" || "$final_status" == "failed" || "$final_status" == "canceled" ]]; then
                break
            fi
        done
    fi

    if [[ "$final_status" == "succeeded" || "$final_status" == "failed" || "$final_status" == "canceled" ]]; then
        log_success "Models scan job finished (status=$final_status)"
    else
        log_warn "Models scan job still not terminal (status=$final_status)"
    fi

    local jobs_response
    jobs_response=$(curl -s -w "\n%{http_code}" "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/models/scan/jobs")
    local jobs_code
    jobs_code=$(echo "$jobs_response" | tail -n1)
    if [ "$jobs_code" -eq 200 ]; then
        log_success "Models scan jobs list OK"
    else
        log_warn "Models scan jobs list failed (HTTP $jobs_code)"
    fi

    local delete_response
    delete_response=$(curl -s -w "\n%{http_code}" -X "DELETE" "${CURL_AUTH_ARGS[@]}" "$API_URL/api/v1/models/scan/jobs/$job_id")
    local delete_code
    delete_code=$(echo "$delete_response" | tail -n1)
    local delete_body
    delete_body=$(echo "$delete_response" | head -n -1)
    if [ "$delete_code" -eq 200 ]; then
        log_success "Models scan job record deleted"
    else
        log_warn "Models scan job delete failed (HTTP $delete_code)"
        echo "Response: $delete_body"
    fi

    return 0
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
run_test test_async_t2i_flow "$generation_data"

# Models scan async job flow (optional; set TEST_MODELS_SCAN_ASYNC=1 and use an admin key)
scan_data='{"replace":false}'
if [ "$TEST_MODELS_SCAN_ASYNC" = "1" ]; then
    run_test test_async_models_scan_flow "$scan_data"
else
    log_warn "Skipping models scan async flow (set TEST_MODELS_SCAN_ASYNC=1 to enable)"
fi

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
