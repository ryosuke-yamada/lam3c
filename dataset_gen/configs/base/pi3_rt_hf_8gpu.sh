#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/pi3_roomtours_common.sh"
PI3_PBS_QUEUE=rt_HF
PI3_PBS_PROJECT=gag51492
PI3_PBS_SELECT=1
PI3_PBS_WALLTIME=24:00:00
PI3_NUM_GPUS=8
PI3_NUM_SHARDS=8
