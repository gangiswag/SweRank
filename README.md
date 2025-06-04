# SweRank

```
export PYTHONPATH={SweRank_repo_path}:$PYTHONPATH
```

## Reranker

### Preparing data
Pass the parent path of the contrastive mined data shards as the first argument to training data construction script
```
bash script/get_rerank_train_data.sh ${CONTRASTIVE_MINED_PATH}
```

This will filter mined data based on predefined heuristics and convert them into reranking llm training format. The outputs will be saved under `SweRank/datasets`

### Training
For single node training, run
```
bash script/run_train_reranker_single_node.sh
```
Notice that you have to specify the number of gpus to be used for training.

For multi node training, run
```
bash script/run_train_reranker_multi_node.sh
```
You have to set your distributed arguments for multi node training within the above script.
We have a sample script to run multinode training in slurm cluster at `script/slurm_run_multinode_train.sh`
Make sure to load required modules within the example script for the run as well.

### Inference
We use a single json file with all the retriever results as the input for reranking run. To make a reranking run,

```
bash script/run_rerank.sh ${retriever_name} ${dataset_name} # CodeRankEmbed swe-bench-lite
```

Notice that `RERANKER_TAG` is used to name the saving file of reranking results, and `RERANKER_MODEL_PATH` is the path of the reranker.

To make a reranking run with GPT models, 
```
bash script/run_rerank_gpt.sh ${retriever_name} ${dataset_name}
```

### Localization evlauation
To run localization evaluation with reranker,
```
bash script/run_eval_localization.sh ${mode} ${retriever_name} ${dataset_name}

bash script/run_eval_localization.sh reranker ${retriever_name} ${dataset_name} # mode=reranker
```

Set `RERANKER_TAG` in the script to the one you used for the above reranking run to evaluate the localization accuruacy of your trained reranker model.