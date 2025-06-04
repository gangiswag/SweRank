#!/bin/bash

set -e

export REPO_DIR="$(pwd)"
export DATASET_DIR="${REPO_DIR}/datasets"
export OUTPUT_DIR="${REPO_DIR}/outputs"
export EVAL_DIR="${REPO_DIR}/eval_results"

retriever=${1:-"swerankembed_ft"}
dataset=${2:-"swe-bench-lite"}
split=${3:-"test"}
level=${4:-"function"}
eval_mode=${5:-"default"}

# Default reranking parameters
TOP_K=100
WINDOW_SIZE=10
STEP_SIZE=5

### RETRIEVER OUTPUT PATTERN: model=CodeRankEmbed_dataset=swe-bench-lite_split=test_level=function_eval_mode=default_results.json

# Refer to reranker/utils/llm_utils.py for the supported list of openai models
OPENAI_API_KEY="your_api_key"

RERANKER_TAG="gpt-4.1"
RETRIEVER_OUTPUT_DIR="${OUTPUT_DIR}/model=${retriever}_dataset=${dataset}_split=${split}_level=${level}_evalmode=${eval_mode}_results.json"
DATA_TYPE="${retriever}_${RERANKER_TAG}"

RERANKER_MODEL_PATH=${RERANKER_TAG}

# Run the reranker (includes conversion step)
echo "Running reranker with model: ${RERANKER_MODEL_PATH}"
python src/rerank.py \
    --model ${RERANKER_MODEL_PATH} \
    --dataset_dir ${DATASET_DIR} \
    --dataset_name ${dataset} \
    --retriever_output_dir ${RETRIEVER_OUTPUT_DIR} \
    --data_type ${DATA_TYPE} \
    --output_dir ${OUTPUT_DIR} \
    --eval_dir ${EVAL_DIR} \
    --top_k ${TOP_K} \
    --window_size ${WINDOW_SIZE} \
    --step_size ${STEP_SIZE} \
    --api_key ${OPENAI_API_KEY} # API key for OpenAI model call

echo "Reranking completed!"