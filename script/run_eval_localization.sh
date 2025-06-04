#!/bin/bash

export REPO_DIR="$(pwd)"
export DATASET_DIR="${REPO_DIR}/datasets"
export OUTPUT_DIR="${REPO_DIR}/outputs"

MODE=${1:-"reranker"} # code-retriever, reranker
RETRIEVER_MODEL_NAME=${2:-"CodeRankEmbed"}
DATASET=${3:-"swe-bench-lite"}

# Model paths
CodeRankLLM="nomic-ai/CodeRankLLM"
GPT="gpt-4.1"

# Reranker model configs
RERANKER_MODEL_PATH=$GPT
RERANKER_TAG=$(basename $RERANKER_MODEL_PATH)

# Reranker output configs
DATA_TYPE="${RETRIEVER_MODEL_NAME}_${RERANKER_TAG}"
RERANKER_OUTPUT_DIR="${OUTPUT_DIR}/${DATA_TYPE}"

if [ "$MODE" == "reranker" ]; then
    python src/refactored_eval_localization.py \
        --model $RETRIEVER_MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --reranker_output_dir $RERANKER_OUTPUT_DIR \
        --dataset_dir $DATASET_DIR \
        --dataset $DATASET
else
    python src/refactored_eval_localization.py \
        --model $RETRIEVER_MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --dataset_dir $DATASET_DIR \
        --dataset $DATASET
fi
