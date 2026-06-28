#!/bin/bash
# FutureNav Evaluation on RxR (habitat 0.1.7 compatible, multi-GPU)
# Usage: bash eval/eval_rxr.sh <model_path> [split] [procs_per_gpu] [max_episodes]
# Example: bash eval/eval_rxr.sh /path/to/checkpoint val_unseen 1 0
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# === Arguments ===
MODEL_PATH=${1:?"Usage: bash eval/eval_rxr.sh <model_path> [split] [procs_per_gpu] [max_episodes]"}
SPLIT=${2:-"val_unseen"}
PROCS_PER_GPU=${3:-1}
MAX_EPISODES=${4:-0}

# === Paths ===
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
NAVID_HOME=${NAVID_HOME:-}

CONFIG_PATH="${PROJECT_DIR}/eval/config/futurenav_rxr.yaml"

# === Environment ===
if [ -n "${CONDA_SH:-}" ]; then
    source "${CONDA_SH}"
fi
if [ -n "${CONDA_ENV:-}" ]; then
    conda activate "${CONDA_ENV}"
fi
if [ -n "${HABITAT_SIM_EXT:-}" ]; then
    export LD_LIBRARY_PATH=${HABITAT_SIM_EXT}:$LD_LIBRARY_PATH
fi
PYTHONPATH_ENTRIES="${PROJECT_DIR}/src:${PROJECT_DIR}:${PROJECT_DIR}/VLN_CE:${SCRIPT_DIR}"
if [ -n "${NAVID_HOME}" ]; then
    PYTHONPATH_ENTRIES="${PYTHONPATH_ENTRIES}:${NAVID_HOME}"
fi
export PYTHONPATH=${PYTHONPATH_ENTRIES}:${PYTHONPATH:-}

# === GPU detection ===
GPUS_AVAILABLE=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
CHUNKS=$(( GPUS_AVAILABLE * PROCS_PER_GPU ))

# === Output path ===
MODEL_NAME=$(basename "$MODEL_PATH")
if [[ "$MODEL_NAME" =~ ^checkpoint-[0-9]+$ ]]; then
    EXP_NAME=$(basename "$(dirname "$MODEL_PATH")")
    SAVE_PATH="${PROJECT_DIR}/outputs/eval/rxr/${EXP_NAME}/${MODEL_NAME}/${SPLIT}"
else
    SAVE_PATH="${PROJECT_DIR}/outputs/eval/rxr/${MODEL_NAME}/${SPLIT}"
fi
mkdir -p "${SAVE_PATH}"

echo "================================================"
echo "  FutureNav RxR Evaluation (habitat 0.1.7)"
echo "  Model:  ${MODEL_PATH}"
echo "  Config: ${CONFIG_PATH}"
echo "  Split:  ${SPLIT}"
echo "  GPUs:   ${GPUS_AVAILABLE}"
echo "  Chunks: ${CHUNKS}"
echo "  Output: ${SAVE_PATH}"
echo "================================================"

cd "${PROJECT_DIR}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_ID=$(( IDX % GPUS_AVAILABLE ))
    echo "Starting chunk ${IDX}/${CHUNKS} on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_DIR}/run.py \
        --exp-config $CONFIG_PATH \
        --split-num $CHUNKS \
        --split-id $IDX \
        --model-path $MODEL_PATH \
        --result-path $SAVE_PATH \
        --exp-save data \
        --num-history 8 \
        --max-steps 400 \
        --max-episodes $MAX_EPISODES \
        --gpu-id $GPU_ID &
    sleep 5
done

echo "All ${CHUNKS} processes launched, waiting..."
wait

echo ""
echo "================================================"
echo "All ${CHUNKS} chunks completed!"
echo "================================================"

# Merge results if merge script exists
if [ -f "${SCRIPT_DIR}/merge_results.py" ]; then
    python ${SCRIPT_DIR}/merge_results.py --result-path ${SAVE_PATH}
else
    python -c "
import json, glob, os, numpy as np, math
path = '${SAVE_PATH}'
results = []
for f in sorted(glob.glob(os.path.join(path, 'summary_split_*.json'))):
    with open(f) as fh:
        data = json.load(fh)
        results.extend(data.get('results', []))
if not results:
    log_dir = os.path.join(path, 'log')
    if os.path.exists(log_dir):
        for f in sorted(os.listdir(log_dir)):
            if f.endswith('.json'):
                with open(os.path.join(log_dir, f)) as fh:
                    results.append(json.load(fh))
if results:
    def safe(v):
        return 0 if (math.isinf(v) or math.isnan(v)) else v
    print(f'Total episodes: {len(results)}')
    print(f'SR:  {sum(safe(r.get(\"success\", 0)) for r in results)/len(results):.4f}')
    print(f'SPL: {sum(safe(r.get(\"spl\", 0)) for r in results)/len(results):.4f}')
    print(f'DTG: {sum(safe(r.get(\"distance_to_goal\", 0)) for r in results)/len(results):.2f}')
    print(f'OS:  {sum(safe(r.get(\"oracle_success\", 0)) for r in results)/len(results):.4f}')
    print(f'PL:  {sum(safe(r.get(\"path_length\", 0)) for r in results)/len(results):.2f}')
    merged = {'total_episodes': len(results), 'results': results}
    with open(os.path.join(path, 'merged_results.json'), 'w') as fh:
        json.dump(merged, fh, indent=2)
"
fi
