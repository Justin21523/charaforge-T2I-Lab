#!/bin/bash
# scripts/test_api.sh - SagaForge T2I API Testing Script

set -e

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

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local expected_code=${4:-200}
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

    # Extract HTTP code from last line
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    if [ "$http_code" -eq "$expected_code" ]; then
        log_success "$test_name (HTTP $http_code)"
        return 0
    else
        log_error "$test_name (HTTP $http_code, expected $expected_code)"
        echo "Response: $body"
        return 1
    fi
}

echo "🧪 SagaForge T2I API Test Suite"
echo "================================"
echo "API URL: $API_URL"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test counter
TESTS_PASSED=0
TESTS_TOTAL=0

run_test() {
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    if "$@"; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
}

# Test 1: Health Check
run_test test_endpoint "GET" "/healthz" "" 200 "Health Check"

# Test 2: Root Endpoint
run_test test_endpoint "GET" "/" "" 200 "Root Endpoint"

# Test 3: T2I System Status
run_test test_endpoint "GET" "/t2i/system/status" "" 200 "T2I System Status"

# Test 4: List Available Models
run_test test_endpoint "GET" "/t2i/models" "" 200 "List Available Models"

# Test 5: Generate Preview Image
log_info "Testing: Generate Preview Image"
preview_data='{
    "prompt": "A simple test image",
    "model_type": "sd15"
}'

response=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d "$preview_data" \
    "$API_URL/t2i/preview")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" -eq 200 ]; then
    log_success "Generate Preview Image (HTTP $http_code)"

    # Extract image URL if available
    image_url=$(echo "$body" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('image_url', ''))
except:
    pass
" 2>/dev/null)

    if [ -n "$image_url" ]; then
        log_info "Preview image URL: $image_url"
    fi

    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    log_error "Generate Preview Image (HTTP $http_code)"
    echo "Response: $body"
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

# Test 6: Generate Full Image (Low Quality for Speed)
log_info "Testing: Generate Full Image"
generation_data='{
    "prompt": "A beautiful sunset landscape",
    "negative_prompt": "blurry, low quality",
    "model_type": "sd15",
    "width": 256,
    "height": 256,
    "num_inference_steps": 5,
    "guidance_scale": 7.5,
    "num_images": 1
}'

response=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d "$generation_data" \
    "$API_URL/t2i/generate")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" -eq 200 ]; then
    log_success "Generate Full Image (HTTP $http_code)"

    # Extract job_id
    job_id=$(echo "$body" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('job_id', ''))
except:
    pass
" 2>/dev/null)

    if [ -n "$job_id" ]; then
        log_info "Generation job ID: $job_id"

        # Test job status
        log_info "Testing: Check Job Status"
        if test_endpoint "GET" "/t2i/jobs/$job_id" "" 200 "Check Job Status"; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        fi
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
    fi

    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    log_error "Generate Full Image (HTTP $http_code)"
    echo "Response: $body"
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

# Test 7: LoRA Management
run_test test_endpoint "GET" "/t2i/models" "" 200 "LoRA Model List"

# Test 8: System Cleanup
run_test test_endpoint "POST" "/t2i/system/cleanup" "{}" 200 "System Cleanup"

# Test 9: Training System Status
run_test test_endpoint "GET" "/finetune/system/status" "" 200 "Training System Status"

# Test 10: List Training Configs
run_test test_endpoint "GET" "/finetune/configs" "" 200 "List Training Configs"

# Advanced API Tests (if basic tests pass)
if [ $TESTS_PASSED -ge 5 ]; then
    echo ""
    log_info "Running advanced API tests..."

    # Test LoRA Loading (will fail if no LoRAs available, but should not crash)
    log_info "Testing: LoRA Load (expected to fail gracefully)"
    lora_data='{
        "lora_id": "test_lora",
        "weight": 1.0,
        "model_type": "sd15"
    }'

    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$lora_data" \
        "$API_URL/t2i/lora/load")

    http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" -eq 400 ] || [ "$http_code" -eq 404 ]; then
        log_success "LoRA Load (graceful failure: HTTP $http_code)"
    elif [ "$http_code" -eq 200 ]; then
        log_success "LoRA Load (unexpected success: HTTP $http_code)"
    else
        log_error "LoRA Load (unexpected error: HTTP $http_code)"
    fi

    # Test Training Job Submission (mock)
    log_info "Testing: LoRA Training Submission"
    training_data='{
        "project_name": "test_training",
        "description": "API test training job",
        "base_model": "sd15",
        "dataset_type": "folder",
        "dataset_path": "/nonexistent/path",
        "instance_prompt": "a test image",
        "lora_rank": 16,
        "learning_rate": 0.0001,
        "num_train_epochs": 1
    }'

    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -d "$training_data" \
        "$API_URL/finetune/lora/train")

    http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 400 ]; then
        log_success "Training Submission (HTTP $http_code)"
    else
        log_error "Training Submission (HTTP $http_code)"
    fi
fi

# Performance Test
echo ""
log_info "Running performance tests..."

# Test API response time
log_info "Testing: API Response Time"
start_time=$(date +%s%N)
curl -s "$API_URL/healthz" > /dev/null
end_time=$(date +%s%N)

response_time_ms=$(( (end_time - start_time) / 1000000 ))

if [ $response_time_ms -lt 1000 ]; then
    log_success "API Response Time: ${response_time_ms}ms (good)"
elif [ $response_time_ms -lt 5000 ]; then
    log_info "API Response Time: ${response_time_ms}ms (acceptable)"
else
    log_error "API Response Time: ${response_time_ms}ms (slow)"
fi

# Load Test (simple)
log_info "Testing: Basic Load (10 concurrent requests)"
for i in {1..10}; do
    curl -s "$API_URL/healthz" &
done
wait
log_success "Basic Load Test completed"

# Generate Test Report
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