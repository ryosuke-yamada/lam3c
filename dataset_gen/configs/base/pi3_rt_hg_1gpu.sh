#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/pi3_roomtours_common.sh"
PI3_PBS_QUEUE=rt_HG
PI3_PBS_PROJECT=gag51402
PI3_PBS_SELECT=1:ngpus=1
PI3_PBS_WALLTIME=96:00:00
PI3_NUM_GPUS=1
PI3_NUM_SHARDS=1
