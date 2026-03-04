#!/usr/bin/env bash
# weekly_retrain.sh — Cron-callable script for weekly FCVAE model retraining.
#
# Checks API health, triggers a retrain via POST /v1/retrain, polls until
# completion, and logs results as JSON. Designed for unattended cron execution.
#
# Environment variables:
#   FCVAE_API_URL   Base URL of the scoring API (default: http://localhost:8000)
#   FCVAE_API_KEY   Optional Bearer token for authenticated endpoints
#
# Usage:
#   ./scripts/weekly_retrain.sh                          # defaults
#   FCVAE_API_URL=http://prod:8000 ./scripts/weekly_retrain.sh
#
# Exit codes:
#   0  All combos retrained successfully (or skipped with insufficient_data)
#   1  Health check failed — API unreachable or unhealthy
#   2  Retrain trigger failed (409 conflict, 503 data store disabled, etc.)
#   3  Polling timed out (job still running after POLL_TIMEOUT_SECONDS)
#   4  Retrain job completed with failures (at least one combo errored)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

API_URL="${FCVAE_API_URL:-http://localhost:8000}"
API_KEY="${FCVAE_API_KEY:-}"

POLL_INTERVAL_SECONDS=30
POLL_TIMEOUT_SECONDS=1800   # 30 minutes

# ── Helpers ──────────────────────────────────────────────────────────────────

log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
}

# Build curl auth header if API key is set
curl_auth_args=()
if [[ -n "$API_KEY" ]]; then
    curl_auth_args=(-H "Authorization: Bearer ${API_KEY}")
fi

api_get() {
    local path="$1"
    curl -sf --max-time 10 "${curl_auth_args[@]+"${curl_auth_args[@]}"}" "${API_URL}${path}"
}

api_post() {
    local path="$1"
    local body="$2"
    curl -sf --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        "${curl_auth_args[@]+"${curl_auth_args[@]}"}" \
        -d "$body" \
        "${API_URL}${path}"
}

# ── Step 1: Health Check ─────────────────────────────────────────────────────

log "Starting weekly retrain cycle"
log "API: ${API_URL}"

health_response=$(api_get "/health" 2>&1) || {
    log "ERROR: Health check failed — API unreachable or returned error"
    log "Response: ${health_response:-<empty>}"
    exit 1
}

health_status=$(echo "$health_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")

if [[ "$health_status" != "healthy" ]]; then
    log "ERROR: API reports status='${health_status}'"
    log "Response: ${health_response}"
    exit 1
fi

log "Health check passed (status=${health_status})"

# ── Step 2: Trigger Retrain ──────────────────────────────────────────────────

retrain_body='{"mode":"weekly","lookback_days":7,"auto_reload":true}'

log "Triggering retrain: ${retrain_body}"

retrain_response=$(api_post "/v1/retrain" "$retrain_body" 2>&1) || {
    log "ERROR: Retrain trigger failed"
    log "Response: ${retrain_response:-<empty>}"
    exit 2
}

job_id=$(echo "$retrain_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "")

if [[ -z "$job_id" ]]; then
    log "ERROR: Could not parse job_id from retrain response"
    log "Response: ${retrain_response}"
    exit 2
fi

log "Retrain job started: job_id=${job_id}"

# ── Step 3: Poll Until Completion ────────────────────────────────────────────

elapsed=0

while true; do
    sleep "$POLL_INTERVAL_SECONDS"
    elapsed=$((elapsed + POLL_INTERVAL_SECONDS))

    status_response=$(api_get "/v1/retrain/status/${job_id}" 2>&1) || {
        log "WARNING: Status poll failed (elapsed=${elapsed}s), retrying..."
        continue
    }

    job_status=$(echo "$status_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")

    if [[ "$job_status" == "completed" || "$job_status" == "failed" ]]; then
        break
    fi

    log "Polling... status=${job_status} elapsed=${elapsed}s"

    if [[ "$elapsed" -ge "$POLL_TIMEOUT_SECONDS" ]]; then
        log "ERROR: Polling timed out after ${elapsed}s (job still ${job_status})"
        exit 3
    fi
done

# ── Step 4: Log Results ──────────────────────────────────────────────────────

duration=$(echo "$status_response" | python3 -c "import sys,json; d=json.load(sys.stdin).get('duration_seconds'); print(f'{d:.1f}s' if d else 'N/A')" 2>/dev/null || echo "N/A")

log "Retrain ${job_status} in ${duration} (job_id=${job_id})"

# Log full result as formatted JSON
echo "$status_response" | python3 -m json.tool 2>/dev/null || echo "$status_response"

# ── Step 5: Determine Exit Code ──────────────────────────────────────────────

if [[ "$job_status" == "failed" ]]; then
    log "ERROR: Retrain job completed with failures"
    exit 4
fi

log "Weekly retrain cycle complete"
exit 0
