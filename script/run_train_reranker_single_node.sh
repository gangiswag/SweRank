#!/bin/bash

set -e 

NUM_GPUS=4

export REPO_DIR=$PWD

# Define model, dataset paths, and output directory
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

TRAIN_DATA_PATH="${REPO_DIR}/datasets/reranking_function_localization_train.jsonl"  # Train Dataset --> Hugging Face dataset or Local dataset
EVAL_DATA_PATH="${REPO_DIR}/datasets/reranking_function_localization_train.jsonl"  # Eval Dataset --> Hugging Face dataset or Local dataset

OUTPUT_DIR="${REPO_DIR}/models/swerank_test"  # Directory to save the trained model
BEIR_DATA_DIR="${REPO_DIR}/datasets/beir/"

deepspeed_config="${REPO_DIR}/src/reranker/train_configs/zero3_bf16.json"
RUN_CMD="${REPO_DIR}/src/reranker/train_ranking.py"
TRAIN_ARGS="--model_name_or_path ${BASE_MODEL} 
    --deepspeed_config ${deepspeed_config}
    --train_dataset_path ${TRAIN_DATA_PATH}
    --eval_dataset_path ${EVAL_DATA_PATH}
    --beir_data_path ${BEIR_DATA_DIR}
    --per_device_eval_batch_size 1
    --num_train_epochs 1
    --seed 42
    --per_device_train_batch_size 4
    --eval_steps 1000
    --gradient_checkpointing
    --gradient_accumulation_steps 4
    --lr_scheduler_type cosine
    --num_warmup_steps 50
    --output_dir ${OUTPUT_DIR}
    --noisy_embedding_alpha 5
    --objective generation
    --use_liger_kernel"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --multi_gpu \
    ${RUN_CMD} ${TRAIN_ARGS}