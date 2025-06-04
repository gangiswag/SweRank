#!/bin/bash

### SETUP YOUR ENVIRONMENT IF USING SLURM CLUSTER: ex) conda activate mytrainingenv

# Define model, dataset paths, and output directory
BASE_MODEL="/mnt/clovanap/checkpoints/hf/boost/CodeRankLLM"
TRAIN_DATA_PATH="${REPO_DIR}/datasets/reranking_function_localization-varying_window_size_train.jsonl"  # Train Dataset --> Hugging Face dataset or Local dataset
EVAL_DATA_PATH="${REPO_DIR}/datasets/reranking_function_localization_eval.jsonl"  # Eval Dataset --> Hugging Face dataset or Local dataset
OUTPUT_DIR="${REPO_DIR}/models/code_localization_reranker_full"  # Directory to save the trained model
BEIR_DATA_DIR="${REPO_DIR}/datasets/beir/"

deepspeed_config="${REPO_DIR}/src/reranker/train_configs/zero3_bf16.json"
run_cmd="${REPO_DIR}/src/reranker/train_ranking.py"
TRAIN_ARGS="--model_name_or_path ${BASE_MODEL} 
    --deepspeed_config ${deepspeed_config}
    --train_dataset_path ${TRAIN_DATA_PATH}
    --eval_dataset_path ${EVAL_DATA_PATH}
    --beir_data_path ${BEIR_DATA_DIR}
    --per_device_eval_batch_size 1
    --num_train_epochs 1
    --seed 42
    --per_device_train_batch_size 1
    --eval_steps 1000
    --gradient_checkpointing
    --gradient_accumulation_steps 2
    --lr_scheduler_type cosine
    --num_warmup_steps 50
    --output_dir ${OUTPUT_DIR}
    --noisy_embedding_alpha 5
    --objective generation
    --first_identifier_only
    --use_liger_kernel"

JOB_ID="{Your job id}"
MASTER_ADDR="{Master node address}"
MASTER_ADDR="{Master node port}"
NNODES="{Number of nodes}"
NUM_GPUS="{Number of gpus per node}"

# multi-node distributed training arguments
HOST_NODE_ADDR=${MASTER_ADDR}:${MASTER_PORT}
DISTRIBUTED_ARGS="--nnodes=$NUM_NODES
    --nproc-per-node=$NUM_GPUS
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR" 

# launch training with DeepSpeed configuration
torchrun ${DISTRIBUTED_ARGS} \
    ${run_cmd} ${TRAIN_ARGS}