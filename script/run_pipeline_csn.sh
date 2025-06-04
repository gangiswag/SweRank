#!/bin/bash

export REPO_DIR="$(pwd)"
export DATASET_DIR="${REPO_DIR}/datasets"
export OUTPUT_DIR="${REPO_DIR}/outputs"
export EVAL_DIR="${REPO_DIR}/evaluations"

# Default reranking parameters
TOP_K=100
WINDOW_SIZE=10
STEP_SIZE=5
SKIP_RETRIEVER=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval_dir)
            EVAL_DIR="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --step_size)
            STEP_SIZE="$2"
            shift 2
            ;;
        --skip_retriever)
            SKIP_RETRIEVER=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${DATASET_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${EVAL_DIR}"

if [ "${SKIP_RETRIEVER}" -eq 0 ]; then

    # Run retrieval evaluation to generate rank files
    echo "Running retrieval evaluation..."
    python evaluations/eval_csn.py \
        --device "cuda" \
        --dataset_dir "${DATASET_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size 64
else
    echo "Skipping retriever step..."
    if [ ! -d "${DATASET_DIR}" ]; then
        echo "Error: Dataset directory ${DATASET_DIR} does not exist!"
        echo "Please provide a valid dataset directory when skipping retriever step."
        exit 1
    fi
fi

# Run the reranker (includes conversion step)
echo "Running reranker..."
python rerank.py \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_dir "${EVAL_DIR}" \
    --top_k "${TOP_K}" \
    --window_size "${WINDOW_SIZE}" \
    --step_size "${STEP_SIZE}"

echo "Pipeline completed! Results are in:"
echo "- Datasets: ${DATASET_DIR}"
echo "- Reranking outputs: ${OUTPUT_DIR}"
echo "- Evaluation results: ${EVAL_DIR}" 