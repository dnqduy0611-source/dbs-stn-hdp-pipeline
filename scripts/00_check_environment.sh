#!/usr/bin/env bash
set -euo pipefail

required_commands=(
  mrconvert
  mrinfo
  dwidenoise
  mrdegibbs
  dwifslpreproc
  dwi2response
  dwi2fod
  tckgen
  tckedit
  tckinfo
  flirt
  eddy
  topup
  bet
  antsRegistration
  antsApplyTransforms
  N4BiasFieldCorrection
)

echo "== DBS-STN HDP pipeline environment check =="
echo "Date: $(date -Is)"
echo "Host: $(hostname)"
echo "Shell: ${SHELL:-unknown}"
echo

missing=0
for cmd in "${required_commands[@]}"; do
  if command -v "$cmd" >/dev/null 2>&1; then
    printf "OK   %-28s %s\n" "$cmd" "$(command -v "$cmd")"
  else
    printf "MISS %-28s not found in PATH\n" "$cmd"
    missing=1
  fi
done

echo
echo "== Versions =="
mrconvert -version 2>/dev/null | head -n 1 || true
flirt -version 2>/dev/null | head -n 1 || true
antsRegistration --version 2>/dev/null | head -n 2 || true

echo
echo "FSLDIR=${FSLDIR:-}"

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Environment check failed: at least one required command is missing."
  exit 1
fi

echo
echo "Environment check passed."
