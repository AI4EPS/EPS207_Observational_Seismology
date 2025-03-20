#!/bin/bash
#SBATCH --job-name=cctorch
##SBATCH --nodes=2
##SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1

nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nnodes=${#nodes[@]}
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Using $nnodes nodes: $SLURM_JOB_NODELIST
echo Master IP: $head_node $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
    --nnodes $nnodes \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    run.py --generate-pair  --path-data=./tests/slurm/temp_good --pair-list=./tests/slurm/eventIDs_good_LV_north.txt
