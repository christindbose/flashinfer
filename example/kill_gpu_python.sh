#!/bin/bash
# Script to kill all Python processes using the GPU belonging to current user

USER=$(whoami)
echo "Finding GPU Python processes for user: $USER"

# Get PIDs of processes using the GPU from nvidia-smi
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')

if [ -z "$GPU_PIDS" ]; then
    echo "No GPU processes found."
    exit 0
fi

echo "GPU PIDs found: $GPU_PIDS"

KILLED=0
for PID in $GPU_PIDS; do
    # Check if this PID belongs to the current user and is a Python process
    PROC_USER=$(ps -o user= -p $PID 2>/dev/null | tr -d ' ')
    PROC_CMD=$(ps -o comm= -p $PID 2>/dev/null)
    
    if [ "$PROC_USER" = "$USER" ]; then
        if [[ "$PROC_CMD" == *"python"* ]]; then
            echo "Killing Python process: PID=$PID, CMD=$PROC_CMD"
            kill -9 $PID 2>/dev/null
            KILLED=$((KILLED + 1))
        else
            echo "Skipping non-Python process: PID=$PID, CMD=$PROC_CMD"
        fi
    else
        echo "Skipping process owned by $PROC_USER: PID=$PID"
    fi
done

echo "Killed $KILLED Python GPU process(es)."
