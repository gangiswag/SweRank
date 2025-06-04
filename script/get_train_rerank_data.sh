#!/bin/bash

set -e

export REPO_DIR=$PWD

OUT_PATH=${REPO_DIR}/datasets
CONTRASTIVE_MINED_PATH=${1:-$OUT_PATH} # root path of contrastive mined data 
SKIP_FILTER=${2:-0} # whether to skip filtering or not
mkdir -p ${OUT_PATH}

if [[ $SKIP_FILTER -ne 1 ]]; then
    echo "Filtering contrastive mined data..."

    python src/collect/filter_dataset.py \
        --file_path ${CONTRASTIVE_MINED_PATH} \
        --file_prefix repo_contrastive_mined_
else
    echo "Skipping data filtering"
fi

echo "Constructing reranker training data..."

WINDOW_SIZE=10
DATA_FILE_NAME="repo_contrastive_mined_filtered.jsonl"
echo "Filtered dataset path: ${CONTRASTIVE_MINED_PATH}/${DATA_FILE_NAME}"
python src/reranker/get_rerank_train_data.py \
    --data_type local \
    --data_path ${CONTRASTIVE_MINED_PATH}/${DATA_FILE_NAME} \
    --out_path ${OUT_PATH} \
    --window_size ${WINDOW_SIZE} \
    --random_seed 42 \
    --eval_ratio 0.01 \
    --shuffle_context \
    --first_identifier_only
    # --varying_window_size # for augmentation to scale up dataset size