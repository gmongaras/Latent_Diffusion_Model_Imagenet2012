#!/bin/bash

# Number of nodes
nnodes=1
# Number of tasks per node (gpus per node)
nproc_per_node=8
# Path to torchrun
torchrunpath=REPLACE_WITH_PATH_TO_TORCHRUN



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 srun $torchrunpath \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
src/train.py