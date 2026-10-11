#!/usr/bin/env bash
# Wrapper: the script lives in scripts/ (kept here so the documented ./run_sim_viewer.sh still works)
exec "$(cd "$(dirname "$0")" && pwd)/scripts/run_sim_viewer.sh" "$@"
