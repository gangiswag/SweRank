#!/bin/bash

export DATASET_DIR="${REPO_DIR}/datasets"
export OUTPUT_DIR="${REPO_DIR}/outputs"
export EVAL_DIR="${REPO_DIR}/evaluations"

repo_info_path="pypi_rankings.jsonl"
valid_repo_out_path="valid_top_pypi_gitrepos.jsonl"
python get_valid_urls.py \
    --repo_info_path ${repo_info_path} \
    --out_path ${valid_repo_out_path}

prs_path=prs
tasks_path=tasks
git_token=${GITHUB_TOKEN}

python generate_tasks.py \
    --repo_info_path ${valid_repo_out_path} \
    --prs_path ${prs_path} \
    --tasks_path ${tasks_path} \
    --git_token ${git_token} \
    --start_idx 51 \
    --num_repos_to_process 500

python build_dataset_ft.py \
    --instances_path ${tasks_path} \
    --output_path ${DATASET_DIR} \
    --eval_path ${EVAL_DIR}