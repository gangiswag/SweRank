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

export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

### RETRIEVER OUTPUT PATTERN: model=CodeRankEmbed_dataset=swe-bench-lite_split=test_level=function_evalmode=default_results.json

# Model paths
CodeRankLLM="nomic-ai/CodeRankLLM"

# Reranker model configs
RERANKER_MODEL_PATH=$CodeRankLLM
RERANKER_TAG=$(basename $RERANKER_MODEL_PATH)

# Reranker output configs
RETRIEVER_OUTPUT_DIR="${OUTPUT_DIR}/model=${retriever}_dataset=${dataset}_split=${split}_level=${level}_evalmode=${eval_mode}_results.json"
DATA_TYPE="${retriever}_${RERANKER_TAG}"

# Run the reranker (includes conversion step)
echo "Running reranker with model: ${RERANKER_MODEL_PATH}"
echo "Reranker tag: ${RERANKER_TAG}"
python src/rerank.py \
    --model ${RERANKER_MODEL_PATH} \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_name ${dataset} \
    --retriever_output_dir ${RETRIEVER_OUTPUT_DIR} \
    --data_type ${DATA_TYPE} \
    --output_dir "${OUTPUT_DIR}" \
    --eval_dir "${EVAL_DIR}" \
    --top_k "${TOP_K}" \
    --window_size "${WINDOW_SIZE}" \
    --step_size "${STEP_SIZE}" \
    --use_parallel_reranking

echo "Reranking completed!"