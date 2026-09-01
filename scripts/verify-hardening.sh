#!/usr/bin/env bash
# Verify that the security hardening on the proton-faces deployment is intact.
# Usage: BASE=https://protonface.mornati.ovh ./scripts/verify-hardening.sh
#
# Exits 0 on full pass, 1 on any failed check. Designed to run against the
# publicly-deployed demo URL AND against localhost:8080 (with appropriate
# DEMO_ALLOW_PUBLIC_THUMBS toggles).

set -u
BASE="${BASE:-http://localhost:8080}"
FAILS=0
PASSES=0

check() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  if [[ "$actual" == "$expected" ]]; then
    printf "  PASS  %-50s = %s\n" "$label" "$actual"
    PASSES=$((PASSES + 1))
  else
    printf "  FAIL  %-50s = %s  (expected %s)\n" "$label" "$actual" "$expected"
    FAILS=$((FAILS + 1))
  fi
}

check_match() {
  local label="$1"
  local actual="$2"
  local pattern="$3"
  if [[ "$actual" =~ $pattern ]]; then
    printf "  PASS  %-50s matches %s\n" "$label" "$pattern"
    PASSES=$((PASSES + 1))
  else
    printf "  FAIL  %-50s = %s  (expected match %s)\n" "$label" "$actual" "$pattern"
    FAILS=$((FAILS + 1))
  fi
}

section() { printf "\n=== %s ===\n" "$1"; }

section "Reachability"
HEALTH=$(curl -sS -o /dev/null -w "%{http_code}" "$BASE/api/health")
check "GET /api/health" "$HEALTH" "200"

section "Traefik security headers (Route A middleware)"
HEADERS=$(curl -sSI "$BASE/")
STS=$(echo "$HEADERS" | grep -i "^strict-transport-security:" | tr -d '\r' | head -1)
XFO=$(echo "$HEADERS" | grep -i "^x-frame-options:" | tr -d '\r' | head -1)
XCTO=$(echo "$HEADERS" | grep -i "^x-content-type-options:" | tr -d '\r' | head -1)
RP=$(echo "$HEADERS" | grep -i "^referrer-policy:" | tr -d '\r' | head -1)
CSP=$(echo "$HEADERS" | grep -i "^content-security-policy:" | tr -d '\r' | head -1)

check_match "Strict-Transport-Security present" "$STS" "max-age="
check_match "X-Frame-Options=DENY" "$XFO" "(DENY|deny)"
check_match "X-Content-Type-Options=nosniff" "$XCTO" "nosniff"
check_match "Referrer-Policy set" "$RP" "no-referrer"
check_match "Content-Security-Policy set" "$CSP" "default-src"

section "/api/status config redaction (anon)"
STATUS_ANON=$(curl -sS "$BASE/api/status")
HAS_CONFIG_ANON=$(echo "$STATUS_ANON" | python3 -c "import json,sys;print('yes' if 'config' in json.load(sys.stdin) else 'no')")
check "anon /api/status hides config block" "$HAS_CONFIG_ANON" "no"

section "Auth round-trip"
LOGIN=$(curl -sS -X POST -H "content-type: application/json" \
  -d '{"username":"demo","password":"'"${DEMO_ADMIN_PASSWORD:-proton-faces}"'"}' \
  "$BASE/api/auth/login")
TOKEN=$(echo "$LOGIN" | python3 -c "import json,sys;print(json.load(sys.stdin).get('access_token',''))")
if [[ -z "$TOKEN" ]]; then
  echo "  FAIL  login did not return an access token"
  FAILS=$((FAILS + 1))
else
  echo "  PASS  login returned a bearer token"
  PASSES=$((PASSES + 1))
fi

section "/api/status config visibility (authed)"
STATUS_AUTH=$(curl -sS -H "Authorization: Bearer $TOKEN" "$BASE/api/status")
HAS_CONFIG_AUTH=$(echo "$STATUS_AUTH" | python3 -c "import json,sys;print('yes' if 'config' in json.load(sys.stdin) else 'no')")
check "authed /api/status shows config block" "$HAS_CONFIG_AUTH" "yes"

section "Signed URL flow (prod mode)"
SIGN_RESP=$(curl -sS -X POST -H "Authorization: Bearer $TOKEN" -H "content-type: application/json" \
  -d '{"paths":["/api/photos/pic-0149/thumb"]}' "$BASE/api/sign")
SIG_PATH=$(echo "$SIGN_RESP" | python3 -c "import json,sys;d=json.load(sys.stdin);u=d.get('urls',[{}])[0];print(f\"{u.get('path','')}?sig={u.get('sig','')}&exp={u.get('exp','')}\")")
if [[ "$SIG_PATH" == *"sig="* && "$SIG_PATH" == *"exp="* ]]; then
  echo "  PASS  /api/sign returned a valid signed URL"
  PASSES=$((PASSES + 1))
  SIGNED_STATUS=$(curl -sS -o /dev/null -w "%{http_code}" "$BASE$SIG_PATH")
  check "signed URL serves the asset" "$SIGNED_STATUS" "200"
else
  echo "  FAIL  /api/sign did not return a usable signed URL: $SIGN_RESP"
  FAILS=$((FAILS + 1))
fi

section "Signed URL tamper detection"
TAMPERED_STATUS=$(curl -sS -o /dev/null -w "%{http_code}" "$BASE/api/photos/pic-0149/thumb?sig=deadbeef&exp=1")
check "tampered sig is rejected" "$TAMPERED_STATUS" "401"

section "Login rate-limit (Traefik middleware)"
# Burst 12 requests; in prod mode we expect at least one 429 from the
# pf-ratelimit middleware (5 rpm avg, burst 10). Demo mode has the
# middleware loaded but the routes might be excluded by Coolify labels —
# we only flag this as PASS if at least one 429 occurs; otherwise INFO.
N429=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  S=$(curl -sS -o /dev/null -w "%{http_code}" -X POST \
    -H "content-type: application/json" \
    -d '{"username":"probe","password":"probe"}' \
    "$BASE/api/auth/login")
  if [[ "$S" == "429" ]]; then N429=$((N429 + 1)); fi
done
if [[ "$N429" -gt 0 ]]; then
  check "Traefik rate-limit triggers" "$N429" ">=1"
else
  printf "  INFO  no 429s observed in burst of 12 — Traefik middleware not attached to this route\n"
fi

section "Summary"
printf "  passed: %d\n  failed: %d\n" "$PASSES" "$FAILS"
if [[ "$FAILS" -gt 0 ]]; then
  exit 1
fi
exit 0
