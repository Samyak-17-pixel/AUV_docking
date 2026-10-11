#!/usr/bin/env bash
# Wrapper: the script lives in scripts/ (kept here so the documented ./run_camera_sim.sh still works)
exec "$(cd "$(dirname "$0")" && pwd)/scripts/run_camera_sim.sh" "$@"
