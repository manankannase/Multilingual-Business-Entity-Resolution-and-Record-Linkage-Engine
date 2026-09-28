#!/bin/bash
# Pipeline status: queued/running jobs, finished stages, and the tail of each stage's newest .out / .err log.
#   bash status.sh          # summary
#   bash status.sh xenc     # full newest logs of one stage
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.conf"
LOGS="$BER_WORK_DIR/logs"
if [ -n "$1" ]; then
    out=$(ls -t "$LOGS"/ber_"$1"_*.out 2>/dev/null | head -1); err="${out%.out}.err"
    echo "######## $out"; cat "$out"; echo; echo "######## $err"; cat "$err" 2>/dev/null
    exit 0
fi
echo "== jobs"; squeue -u "$USER" -o "%.10i %.14j %.9T %.10M %.12l %R" 2>/dev/null | grep -E "JOBID|ber_" || true
echo "== finished stages"
for s in prep cand dense feat xenc llm stage2; do
    [ -f "$BER_WORK_DIR/.done_$s" ] && echo "  $s  done $(date -r "$BER_WORK_DIR/.done_$s" '+%d %b %H:%M')"
done
for s in prep cand dense feat xenc llm stage2; do
    out=$(ls -t "$LOGS"/ber_"$s"_*.out 2>/dev/null | head -1)
    [ -z "$out" ] && continue
    err="${out%.out}.err"
    echo; echo "== $s: $(basename "$out")"; tail -n 4 "$out"
    if [ -s "$err" ]; then
        echo "   -- $(basename "$err") ($(grep -ciE 'error|traceback|failed|killed|oom' "$err") error-like lines), last lines:"
        tail -n 3 "$err" | sed 's/^/   /'
    fi
done
