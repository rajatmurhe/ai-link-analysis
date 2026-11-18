#!/usr/bin/env bash
# Runs sample requests against a running server and stores the responses in docs/samples/.
# Usage: ./scripts/run_samples.sh [base_url]
set -u
BASE="${1:-http://localhost:4000}"
OUT="$(dirname "$0")/../docs/samples"
mkdir -p "$OUT"

req() {  # name, json-body
  echo "POST /link-analysis $2"
  curl -s -X POST "$BASE/link-analysis" -H "Content-Type: application/json" -d "$2" \
    -o "$OUT/$1.json" -w "  -> HTTP %{http_code} in %{time_total}s\n"
}

curl -s "$BASE/health" -o "$OUT/health.json" -w "GET /health -> HTTP %{http_code}\n"
req basecamp            '{"url": "https://basecamp.com"}'
req python-org          '{"url": "python.org"}'
req allbirds            '{"url": "https://www.allbirds.com"}'
req basecamp-cached     '{"url": "https://basecamp.com/"}'
req error-invalid-url   '{"url": "ftp://example.com"}'
req error-ssrf-metadata '{"url": "http://169.254.169.254/latest/meta-data/"}'
req error-ssrf-localhost '{"url": "http://localhost/admin"}'
req error-unreachable   '{"url": "https://this-domain-does-not-exist-12345.com"}'
req error-not-html      '{"url": "https://httpbin.org/image/png"}'
req error-missing-field '{}'
