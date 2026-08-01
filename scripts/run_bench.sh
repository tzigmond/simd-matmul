#!/usr/bin/env bash
# Runs every kernel at a given size and prints a markdown results table with
# speedup relative to the naive baseline. Usage: scripts/run_bench.sh [N] [threads]
set -euo pipefail

N="${1:-1024}"
THREADS="${2:-$(nproc)}"
BIN=build/benchmark

if [[ ! -x "$BIN" ]]; then
    echo "building..." >&2
    make all >/dev/null
fi

declare -A GF
for k in naive blocked avx2 aligned threaded openblas; do
    if [[ "$k" == "threaded" ]]; then
        out=$(OMP_NUM_THREADS="$THREADS" "$BIN" "$N" "$k" 2>/dev/null)
    else
        out=$("$BIN" "$N" "$k" 2>/dev/null)
    fi
    echo "$out" | grep -qE "correctness: (PASS|SKIPPED)" || { echo "FAIL: $k did not verify" >&2; exit 1; }
    GF[$k]=$(echo "$out" | awk '/^best:/ {print $2}')
done

base=${GF[naive]}
echo "Matrix size N=$N, threaded uses $THREADS threads. Best of 5 runs."
echo
echo "| Kernel | GFLOP/s | Speedup vs naive |"
echo "|--------|---------|------------------|"
for k in naive blocked avx2 aligned threaded openblas; do
    sp=$(awk -v g="${GF[$k]}" -v b="$base" 'BEGIN{printf "%.1fx", g/b}')
    printf "| %-9s | %7s | %-16s |\n" "$k" "${GF[$k]}" "$sp"
done
