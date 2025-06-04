import os
import json
import torch
import time
import argparse
from pathlib import Path
from reranker import rerank_llm, convert_format
from datasets import load_dataset

import ray
NUM_GPUS_PER_JOB=1

def evaluate_results(eval_dir, dataset_name, qrels_path, results_path):
    """
    Evaluate reranking results using MRR@k metrics
    """
    # Load qrels
    with open(qrels_path) as f:
        qrels = {}
        for i, line in enumerate(f):
            if i == 0 and line.lower().startswith("query-id"):
                continue
            
            qid, docid, score = line.strip().split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(score)
    
    # Load results
    with open(results_path) as f:
        results = json.load(f)
    
    # Calculate MRR@k for different k values
    metrics = [1, 3, 5, 10, 20, 100]
    mrr_at_k = {}
    
    for k in metrics:
        mrr_sum = 0.0
        num_queries = 0
        
        for qid in qrels:
            if qid in results:
                sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
                
                for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
                    if doc_id in qrels[qid] and qrels[qid][doc_id] > 0:
                        mrr_sum += 1.0 / rank
                        break
                num_queries += 1
        
        mrr = mrr_sum / num_queries if num_queries > 0 else 0.0
        mrr_at_k[k] = mrr
    
    # Save evaluation results
    os.makedirs(os.path.join(eval_dir, "eval_results"), exist_ok=True)
    eval_path = os.path.join(eval_dir, "eval_results", f"{dataset_name}_eval.json")
    with open(eval_path, "w") as f:
        json.dump(mrr_at_k, f, indent=4)
    
    return mrr_at_k

def run_convert_and_rerank(args):
    """
    First convert results and then run the reranker on retriever outputs.
    The retriever stores datasets in BEIR format at:
    - CSN: {args.dataset_dir}/code_datasets/csn_{lang}
    - SWE-bench: {args.dataset_dir}/swe-bench-lite-function_{instance_id}
    """

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    code_datasets_dir = os.path.join(args.dataset_dir, "code_datasets")

    if os.path.exists(code_datasets_dir):
        # datasets = os.listdir(code_datasets_dir)
        datasets = ["csn_ruby"]
    else:
        datasets = []
    
    if os.path.exists(args.dataset_dir):
        if 'loc-bench' in args.dataset_name:
            loc_bench_dataset = load_dataset("czlll/Loc-Bench_V1")['test']
            valid_datasets = [item['instance_id'] for item in loc_bench_dataset]
            for qid in valid_datasets:
                if os.path.exists(os.path.join(args.dataset_dir, f"{args.dataset_name}-function_{qid}")):
                    datasets.append(f"{args.dataset_name}-function_{qid}")
        else:
            datasets.extend([d for d in os.listdir(args.dataset_dir) if d.startswith(f"{args.dataset_name}-function_")])

    if not os.path.exists(args.retriever_output_dir):
        raise Exception(f"Retriever output doesn't exist at: {args.retriever_output_dir}")

    if "csn" == args.dataset_name:
        if args.data_type:
            args.data_type=f"{args.data_type}_csn"
        else:
            args.data_type="csn"
        rerank_type="code"
        prompt_type = "docstring"
    elif args.dataset_name in ["swe-bench-lite", "loc-bench"]:
        if args.data_type:
            args.data_type=f"{args.data_type}_{args.dataset_name}"
        else:
            args.data_type=args.dataset_name
        
        rerank_type="code"
        prompt_type = "github_issue"
    else:
        raise Exception("Invalid dataset for code reranking run")

    convert_format.convert_results_single_pass(
        prefix=args.dataset_name,
        data_dir=args.dataset_dir,
        data_type=args.data_type,
        output_path=args.output_dir,
        retriever_results_path=args.retriever_output_dir,
        top_k=args.top_k,
        rerank_type="code"
    )

    # Skip existing instances
    dataset_rem = [d for d in datasets if not os.path.exists(os.path.join(args.output_dir, args.data_type, d))]
    datasets = dataset_rem
    print(f"Number of remaining dataset: {len(datasets)}")

    # Create API configs
    api_config = {
        "keys": args.api_key,
        "api_type": args.api_key,
        "api_base": args.api_key,
        "api_version": args.api_key,
    }

    # Then run reranker
    print(f"Initializing reranker...")

    if args.use_parallel_reranking:
        world_size = torch.cuda.device_count()
        if world_size > 1 :
            print(f"Initiating parallel reranking...")
            os.environ['RAY_DEDUP_LOGS'] = "1"
            ray.init(num_cpus=world_size*4, num_gpus=world_size, ignore_reinit_error=True)

            world_size = world_size//args.tensor_parallel_size
            NUM_GPUS_PER_JOB=args.tensor_parallel_size

            @ray.remote(num_gpus=NUM_GPUS_PER_JOB)
            def process_dataset_parallel(model, output_path, data_dir, dataset, data_type, eval_dir, use_logits, use_alpha, llm_top_k, window_size, step_size, batched, context_size, rerank_type="text", code_prompt_type="docstring", api_config={}, device_rank=None):
                rerank_llm.process_dataset(model, output_path, data_dir, dataset, data_type, eval_dir, use_logits, use_alpha, llm_top_k, window_size, step_size, batched, context_size, rerank_type, code_prompt_type, api_config, device_rank)

            shard_size = len(datasets)//world_size
            sharded_inputs = []

            ### Split up the inputs
            for rank in range(world_size):
                if rank == world_size-1:
                    curr_sharded_inputs = datasets[shard_size*rank:]
                else:
                    curr_sharded_inputs = datasets[shard_size*rank: shard_size*(rank+1)]
                sharded_inputs.append(curr_sharded_inputs)

            futures = [process_dataset_parallel.remote(
                model=args.model, 
                output_path=args.output_dir, 
                data_dir=args.dataset_dir if args.dataset_name in ["swe-bench-lite", "loc-bench" ]else code_datasets_dir,
                dataset=sharded_inputs[i], 
                data_type=args.data_type, 
                eval_dir=args.eval_dir,
                use_logits=0, 
                use_alpha=0, 
                llm_top_k=args.top_k, 
                window_size=args.window_size, 
                step_size=args.step_size, 
                batched=0, 
                context_size=32768, 
                rerank_type=rerank_type, 
                code_prompt_type=prompt_type,
                api_config=api_config,
                device_rank=i,
            ) for i in range(world_size)]
            ray.get(futures)
        else:
            print(f"WARNING: Parallel reranking requires more than one device, but {world_size} device detected.")
    else:
        rerank_llm.process_dataset(
            model=args.model, 
            output_path=args.output_dir, 
            data_dir=args.dataset_dir if args.dataset_name in ["swe-bench-lite", "loc-bench" ]else code_datasets_dir,
            dataset=datasets, 
            data_type=args.data_type, 
            eval_dir=args.eval_dir,
            use_logits=0, 
            use_alpha=0, 
            llm_top_k=args.top_k, 
            window_size=args.window_size, 
            step_size=args.step_size, 
            batched=0, 
            context_size=32768, 
            rerank_type=rerank_type, 
            code_prompt_type=prompt_type,
            api_config=api_config,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help="reranker model path")
    parser.add_argument("--dataset_dir", type=str, required=True,
                      help="Directory containing retriever outputs in BEIR format")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to store reranker outputs")
    parser.add_argument("--eval_dir", type=str, required=True,
                      help="Directory to store evaluation results")
    parser.add_argument("--retriever_output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100,
                      help="Number of candidates to rerank")
    parser.add_argument("--window_size", type=int, default=10,
                      help="Window size for reranking")
    parser.add_argument("--step_size", type=int, default=5,
                      help="Step size for reranking")
    parser.add_argument("--dataset_name", type=str, default="swe-bench-lite")
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--use_parallel_reranking", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Multi-gpu inference if set > 1")

    # For API calls
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_type", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_version", type=str, default=None)
    args = parser.parse_args()
    
    run_convert_and_rerank(args)

if __name__ == "__main__":
    main() 