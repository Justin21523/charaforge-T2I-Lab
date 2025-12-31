#!/bin/bash
# scripts/start_t2i_system.sh - System startup (deprecated)

set -euo pipefail

echo "This helper script is deprecated."
echo ""
echo "To comply with ~/Desktop/data_model_structure.md (AI_WAREHOUSE 3.0), use:"
echo "  1) Start Redis"
echo "     - docker run -p 6379:6379 --name redis -d redis:7"
echo "     - OR: redis-server"
echo "  2) Start API (terminal A)"
echo "     - bash scripts/start_api.sh"
echo "  3) Start Worker (terminal B)"
echo "     - bash scripts/start_worker.sh"
echo ""
echo "Health check:"
echo "  http://localhost:8000/api/v1/health"
