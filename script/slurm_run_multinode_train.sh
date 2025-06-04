#!/bin/bash
#SBATCH --job-name=code_reranker_multinode_training
#SBATCH --nodes=8
#SBATCH --partition=batch-auto-restart
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=UNLIMITED
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

set -e

export REPO_DIR=$PWD # Working directory
DIR="${REPO_DIR}/slurm"

### LOGS CONFIGURATION ###
mkdir -p $DIR/logs/srun/
mkdir -p $DIR/logs/sbatch/
mkdir -p $DIR/logs/nccl/${SLURM_JOB_NAME}

# parse job name into directory and log name
log_dir="${SLURM_JOB_NAME%/*}"
log_name="${SLURM_JOB_NAME##*/}"

# create logging directories and set log filename
if [ "$log_dir" != "${SLURM_JOB_NAME}" ]; then
    mkdir -p $DIR/logs/sbatch/"$log_dir"
    mkdir -p $DIR/logs/srun/"$log_dir"
    log_filename=$DIR/logs/srun/"$log_dir/${log_name}-${SLURM_JOB_ID}.log"
else
    log_filename=$DIR/logs/srun/"${log_name}-${SLURM_JOB_ID}.log"
fi

# prints message for the sbatch
echo "Job name: " ${SLURM_JOB_NAME}
echo "Arguments:" ${options}
echo "Log file:" ${log_filename}

### NETWORK CONFIGURATION FOR DISTRIBUTED TRAINING ###
# configure distributed operation
export MASTER_PORT=23456
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### LOAD REQUIRED MODULES OR CONTAINER IMAGES FOR YOUR RUN HERE

# distributed training arguments
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
ENVS="CUDA_DEVICE_MAX_CONNECTIONS=1,JOB_ID=${SLURM_JOB_ID},
    MASTER_ADDR=${MASTER_ADDR},MASTER_PORT=${MASTER_PORT},WORLD_SIZE=${WORLD_SIZE},REPO_DIR=${REPO_DIR},NUM_GPUS=8,NUM_NODES=${SLURM_NNODES}
"
ENVS=$(echo "$ENVS" | tr -d '[:space:]')

echo "Environment variables for the run: ${ENVS}"

# start training
srun -l \
     --export=${ENVS} \
     --output=${log_filename} sh -c "bash src/run_train_reranker_multi_node.sh"
set +x