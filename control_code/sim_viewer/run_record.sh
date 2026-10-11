#!/usr/bin/env bash
# Wrapper: the script lives in scripts/ (kept here so the documented ./run_record.sh still works)
exec "$(cd "$(dirname "$0")" && pwd)/scripts/run_record.sh" "$@"
