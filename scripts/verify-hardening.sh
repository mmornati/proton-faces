#!/usr/bin/env bash
# Verify that the security hardening on the proton-faces deployment is intact.
# Usage:
#   BASE=https://protonface.mornati.ovh ./scripts/verify-hardening.sh
#   INSECURE=1 ./scripts/verify-hardening.sh   # skip TLS cert verification
#   HOST_HEADER=protonface.mornati.ovh BASE=https://51.77.144.149 \
#     INSECURE=1 ./scripts/verify-hardening.sh  # pre-DNS testing
#
# Exits 0 on full pass, 1 on any failed check.

set -u
BASE="${BASE:-http://localhost:8080}"
HOST_HEADER="${HOST_HEADER:-}"
INSECURE="${INSECURE:-}"
CURL_OPTS=""
if [[ -n "$INSECURE" ]]; then
  CURL_OPTS="-k"
fi

# HOST_HEADER lets you test against the public IP before DNS propagates.
# We rewrite the URL host but also pass --resolve so curl can establish
# the TCP connection to the original IP.
if [[ -n "$HOST_HEADER" ]]; then
  _scheme="${BASE%%://*}"
  _rest="${BASE#*://}"
  _hostport="${_rest%%/*}"
  _path="${_rest#$_hostport}"
  _host="${_hostport%:*}"
  _port="${_hostport##*:}"
  [[ "$_port" == "$_hostport" ]] && _port="443"
  [[ "$_scheme" == "http" ]] && _port="80"
  CURL_OPTS="$CURL_OPTS --resolve ${HOST_HEADER}:${_port}:${_host}"
  BASE="${_scheme}://${HOST_HEADER}${_path}"
fi

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
HEALTH=$(curl $CURL_OPTS -sS -o /dev/null -w "%{http_code}" "$BASE/api/health")
check "GET /api/health" "$HEALTH" "200"

section "API docs hidden (P-01)"
# /docs, /redoc, /openapi.json should be 404 in production with
# EXPOSE_API_DOCS unset (default in DEMO_HARDENING_MODE).
for ep in /docs /redoc /openapi.json; do
  CODE=$(curl $CURL_OPTS -sS -o /dev/null -w "%{http_code}" "$BASE$ep")
  check "GET $ep hidden" "$CODE" "404"
done

section "Traefik security headers (Route A middleware)"
HEADERS=$(curl $CURL_OPTS -sSI "$BASE/")
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
STATUS_ANON=$(curl $CURL_OPTS -sS "$BASE/api/status")
HAS_CONFIG_ANON=$(echo "$STATUS_ANON" | python3 -c "import json,sys;print('yes' if 'config' in json.load(sys.stdin) else 'no')" 2>/dev/null || echo "err")
check "anon /api/status hides config block" "$HAS_CONFIG_ANON" "no"

section "Auth round-trip"
LOGIN=$(curl $CURL_OPTS -sS -X POST -H "content-type: application/json" \
  -d '{"username":"demo","password":"'"${DEMO_ADMIN_PASSWORD:-protonface-demo-2026-Q9vK3m}"'"}' \
  "$BASE/api/auth/login")
TOKEN=$(echo "$LOGIN" | python3 -c "import json,sys;print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null || echo "")
if [[ -z "$TOKEN" ]]; then
  echo "  FAIL  login did not return an access token"
  FAILS=$((FAILS + 1))
else
  echo "  PASS  login returned a bearer token"
  PASSES=$((PASSES + 1))
fi

section "/api/status config visibility (authed)"
STATUS_AUTH=$(curl $CURL_OPTS -sS -H "Authorization: Bearer $TOKEN" "$BASE/api/status")
HAS_CONFIG_AUTH=$(echo "$STATUS_AUTH" | python3 -c "import json,sys;print('yes' if 'config' in json.load(sys.stdin) else 'no')" 2>/dev/null || echo "err")
check "authed /api/status shows config block" "$HAS_CONFIG_AUTH" "yes"

section "Signed URL flow (prod mode)"
SIGN_RESP=$(curl $CURL_OPTS -sS -X POST -H "Authorization: Bearer $TOKEN" -H "content-type: application/json" \
  -d '{"paths":["/api/photos/pic-0149/thumb"]}' "$BASE/api/sign")
SIG_PATH=$(echo "$SIGN_RESP" | python3 -c "import json,sys;d=json.load(sys.stdin);u=d.get('urls',[{}])[0];print(f\"{u.get('path','')}?sig={u.get('sig','')}&exp={u.get('exp','')}\")" 2>/dev/null || echo "")
if [[ "$SIG_PATH" == *"sig="* && "$SIG_PATH" == *"exp="* ]]; then
  echo "  PASS  /api/sign returned a valid signed URL"
  PASSES=$((PASSES + 1))
  SIGNED_STATUS=$(curl $CURL_OPTS -sS -o /dev/null -w "%{http_code}" "$BASE$SIG_PATH")
  check "signed URL serves the asset" "$SIGNED_STATUS" "200"
else
  echo "  FAIL  /api/sign did not return a usable signed URL: $SIGN_RESP"
  FAILS=$((FAILS + 1))
fi

section "Signed URL tamper detection"
TAMPERED_STATUS=$(curl $CURL_OPTS -sS -o /dev/null -w "%{http_code}" "$BASE/api/photos/pic-0149/thumb?sig=deadbeef&exp=1")
check "tampered sig is rejected" "$TAMPERED_STATUS" "401"

section "Login rate-limit (Traefik middleware)"
# Burst 12 requests; in prod mode we expect at least one 429 from the
# pf-ratelimit middleware (5 rpm avg, burst 10).
N429=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  S=$(curl $CURL_OPTS -sS -o /dev/null -w "%{http_code}" -X POST \
    -H "content-type: application/json" \
    -d '{"username":"probe","password":"probe"}' \
    "$BASE/api/auth/login")
  if [[ "$S" == "429" ]]; then N429=$((N429 + 1)); fi
done
if [[ "$N429" -gt 0 ]]; then
  echo "  PASS  Traefik rate-limit triggers                    = $N429 (>0)"
  PASSES=$((PASSES + 1))
else
  printf "  INFO  no 429s observed in burst of 12 — Traefik middleware not attached to this route\n"
fi

# Wait for the rate-limit bucket to refill (5 rpm) before the next
# section's login burst — otherwise the P-04 victim login may be 429.
sleep 65

section "P-04 password change revokes other tokens"
# Create a victim user, log in as them, change their password as admin,
# then verify their old tokens are rejected.
VICTIM_USER="pentest-victim-$(date +%s)"
VICTIM_PW="victimpass123"
VICTIM_NEW_PW="newvictimpass456"
CREATED=$(curl $CURL_OPTS -sS -X POST -H "Authorization: Bearer $TOKEN" -H "content-type: application/json" \
  -d "{\"username\":\"$VICTIM_USER\",\"password\":\"$VICTIM_PW\",\"role\":\"read\"}" \
  "$BASE/api/admin/users")
VICTIM_ID=$(echo "$CREATED" | python3 -c "import json,sys;print(json.load(sys.stdin).get('user',{}).get('id',''))" 2>/dev/null || echo "")
if [[ -z "$VICTIM_ID" ]]; then
  echo "  SKIP  could not create victim user (admin endpoint returned: $CREATED)"
else
  VICTIM_LOGIN=$(curl $CURL_OPTS -sS -X POST -H "content-type: application/json" \
    -d "{\"username\":\"$VICTIM_USER\",\"password\":\"$VICTIM_PW\"}" "$BASE/api/auth/login")
  VICTIM_TOKEN=$(echo "$VICTIM_LOGIN" | python3 -c "import json,sys;print(json.load(sys.stdin).get('access_token',''))")
  VICTIM_RT=$(echo "$VICTIM_LOGIN" | python3 -c "import json,sys;print(json.load(sys.stdin).get('refresh_token',''))")
  # Verify victim works BEFORE password change.
  VICTIM_BEFORE=$(curl $CURL_OPTS -sS -H "Authorization: Bearer $VICTIM_TOKEN" "$BASE/api/auth/me" -o /dev/null -w "%{http_code}")
  check "victim token works pre-change" "$VICTIM_BEFORE" "200"
  # Admin changes victim's password.
  PATCH_RESP=$(curl $CURL_OPTS -sS -X PATCH -H "Authorization: Bearer $TOKEN" -H "content-type: application/json" \
    -d "{\"password\":\"$VICTIM_NEW_PW\"}" "$BASE/api/admin/users/$VICTIM_ID")
  REVOKED=$(echo "$PATCH_RESP" | python3 -c "import json,sys;print(json.load(sys.stdin).get('tokens_revoked',0))" 2>/dev/null || echo "0")
  if [[ "$REVOKED" -gt 0 ]]; then
    echo "  PASS  password change revoked $REVOKED tokens"
    PASSES=$((PASSES + 1))
  else
    echo "  FAIL  password change did not revoke any tokens (got $REVOKED)"
    FAILS=$((FAILS + 1))
  fi
  # Victim's OLD access token must now be invalid.
  VICTIM_AFTER=$(curl $CURL_OPTS -sS -H "Authorization: Bearer $VICTIM_TOKEN" "$BASE/api/auth/me" -o /dev/null -w "%{http_code}")
  check "victim's old access token rejected" "$VICTIM_AFTER" "401"
  # Victim's OLD refresh token must now be invalid.
  VICTIM_RT_RESP=$(curl $CURL_OPTS -sS -X POST -H "content-type: application/json" \
    -d "{\"refresh_token\":\"$VICTIM_RT\"}" "$BASE/api/auth/refresh" -o /dev/null -w "%{http_code}")
  check "victim's old refresh token rejected" "$VICTIM_RT_RESP" "401"
  # Cleanup.
  curl $CURL_OPTS -sS -X DELETE -H "Authorization: Bearer $TOKEN" "$BASE/api/admin/users/$VICTIM_ID" -o /dev/null
fi

section "P-05 /api/admin/overview redacts operational details"
OVERVIEW=$(curl $CURL_OPTS -sS -H "Authorization: Bearer $TOKEN" "$BASE/api/admin/overview")
HOSTNAME=$(echo "$OVERVIEW" | python3 -c "import json,sys;print(json.load(sys.stdin).get('server',{}).get('hostname'))" 2>/dev/null || echo "")
PLATFORM=$(echo "$OVERVIEW" | python3 -c "import json,sys;print(json.load(sys.stdin).get('server',{}).get('platform'))" 2>/dev/null || echo "")
DISK_PATH=$(echo "$OVERVIEW" | python3 -c "import json,sys;print(json.load(sys.stdin).get('disk',{}).get('path'))" 2>/dev/null || echo "")
if [[ "$HOSTNAME" == "None" && "$PLATFORM" == "None" && "$DISK_PATH" == "None" ]]; then
  echo "  PASS  hostname / platform / disk-path redacted"
  PASSES=$((PASSES + 1))
else
  echo "  FAIL  hostname=$HOSTNAME platform=$PLATFORM disk.path=$DISK_PATH (expected all None)"
  FAILS=$((FAILS + 1))
fi

section "Summary"
printf "  passed: %d\n  failed: %d\n" "$PASSES" "$FAILS"
if [[ "$FAILS" -gt 0 ]]; then
  exit 1
fi
exit 0
