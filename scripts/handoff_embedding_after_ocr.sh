#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 ORCHESTRATOR_PID ENRICHMENT_PID EXPECTED_VIDEOS FAISS_DIR OCR_RESULT_COPY" >&2
    exit 2
fi

orchestrator_pid="$1"
enrichment_pid="$2"
expected_videos="$3"
faiss_dir="$4"
ocr_result_copy="$5"

orchestrator_state="$(awk '/^State:/ {print $2}' "/proc/${orchestrator_pid}/status" 2>/dev/null || true)"
if [[ "$orchestrator_state" != "T" && "$orchestrator_state" != "t" ]]; then
    echo "Orchestrator PID ${orchestrator_pid} is not stopped." >&2
    exit 1
fi

echo "[handoff] waiting for OCR subprocess from enrichment PID ${enrichment_pid}"
ocr_pid=""
while kill -0 "$enrichment_pid" 2>/dev/null; do
    while IFS= read -r child_pid; do
        child_cmd="$(tr '\0' ' ' < "/proc/${child_pid}/cmdline" 2>/dev/null || true)"
        if [[ "$child_cmd" == *"BackEnd.app.pipeline.ocr_worker"* ]]; then
            ocr_pid="$child_pid"
            break 2
        fi
    done < <(pgrep -P "$enrichment_pid" || true)
    sleep 5
done

if [[ -z "$ocr_pid" ]]; then
    echo "Enrichment PID ${enrichment_pid} ended before starting OCR." >&2
    exit 1
fi

readarray -d '' -t ocr_args < "/proc/${ocr_pid}/cmdline"
ocr_result_path=""
for ((index = 0; index < ${#ocr_args[@]}; index++)); do
    if [[ "${ocr_args[$index]}" == "--result-path" ]]; then
        ocr_result_path="${ocr_args[$((index + 1))]}"
        break
    fi
done
if [[ -z "$ocr_result_path" ]]; then
    echo "Cannot resolve OCR result path from PID ${ocr_pid}." >&2
    exit 1
fi

kill -STOP "$enrichment_pid"
echo "[handoff] enrichment parent stopped; OCR PID ${ocr_pid} continues"

while [[ -e "/proc/${ocr_pid}/stat" ]]; do
    ocr_state="$(awk '{print $3}' "/proc/${ocr_pid}/stat" 2>/dev/null || true)"
    if [[ "$ocr_state" == "Z" ]]; then
        break
    fi
    sleep 5
done

for _ in {1..20}; do
    if [[ -s "$ocr_result_path" ]]; then
        break
    fi
    sleep 1
done
if [[ ! -s "$ocr_result_path" ]]; then
    echo "OCR exited without a non-empty result: ${ocr_result_path}." >&2
    exit 1
fi

mkdir -p "$(dirname "$ocr_result_copy")"
cp "$ocr_result_path" "$ocr_result_copy"
echo "[handoff] durable OCR result copied to ${ocr_result_copy}"

python -m BackEnd.app.pipeline.embedding_only \
    --faiss-dir "$faiss_dir" \
    --stopped-orchestrator-pid "$orchestrator_pid" \
    --ocr-result-path "$ocr_result_copy" \
    --expected-completed-videos "$expected_videos"
