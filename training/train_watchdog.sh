#!/bin/bash
# Training watchdog - uses 'top' to get REAL memory usage including Metal/GPU

MAX_USED_GB=64
CHECK_INTERVAL=1

cd /Users/z/work/zen/zen-coder/training
export HF_HUB_DISABLE_XET=1

log() { echo "[$(date '+%H:%M:%S')] $1"; }

get_mem_gb() {
    # Parse PhysMem from top - this shows REAL usage including Metal
    top -l 1 -s 0 | grep PhysMem | sed 's/.*: \([0-9]*\)G used.*/\1/'
}

log "Watchdog started - limit ${MAX_USED_GB}GB (using top PhysMem)"

TRAINING_RUNNING=0

while true; do
    # Check if training is running
    if pgrep -f "mlx_lm.*lora" > /dev/null 2>&1 || pgrep -f "train_full.py" > /dev/null 2>&1; then
        TRAINING_RUNNING=1
    else
        if [ "$TRAINING_RUNNING" -eq 1 ]; then
            log "Training process exited - restarting..."
        else
            log "Starting training..."
        fi
        TRAINING_RUNNING=1
        /Users/z/work/zen/.venv/bin/python -u train_full.py --resume >> training_run.log 2>&1 &
        TRAIN_PID=$!
        log "Training started with PID $TRAIN_PID"
        sleep 10  # Wait longer for mlx_lm to initialize
    fi

    MEM=$(get_mem_gb)
    log "Memory: ${MEM}GB / ${MAX_USED_GB}GB"

    if [ "$MEM" -ge "$MAX_USED_GB" ]; then
        log "⚠️ KILLING - memory ${MEM}GB >= ${MAX_USED_GB}GB"
        pkill -9 -f "mlx_lm"
        pkill -9 -f "train_full"
        TRAINING_RUNNING=0
        sleep 10
        continue
    fi

    sleep $CHECK_INTERVAL
done
