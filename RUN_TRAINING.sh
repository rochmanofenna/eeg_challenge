#!/bin/bash
# ONE COMMAND TO RULE THEM ALL

# This pod is configured as Shard 0 (processes R1, R4, R7, R10)
# For other pods, change SHARD_ID to 1 or 2

export SHARD_ID=0  # CHANGE THIS ON EACH POD: 0, 1, or 2
./launch_3pod_training.sh
