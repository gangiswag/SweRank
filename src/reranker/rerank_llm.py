import os
import ray
import json
import datetime
import time
from time import strftime
from time import gmtime
import logging
from beir.datasets.data_loader import GenericDataLoader
from src.reranker.utils.result import Result, ResultsLoader
from src.reranker.utils.llm_util import evaluate_results, get_results_to_eval, save_rerank_results, rerank_beir_outputs_llm, load_reranker, save_histories

def get_mrr_at_k(eval_dir, dataset_name, data_type, qrels_path, results):
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
    os.makedirs(os.path.join(eval_dir, f"{data_type}_eval_results"), exist_ok=True)
    eval_path = os.path.join(eval_dir, f"{data_type}_eval_results", f"{dataset_name}_eval.json")
    with open(eval_path, "w") as f:
        json.dump(mrr_at_k, f, indent=4)
    
    return mrr_at_k

def rerank_beir_outputs(reranker, output_path, data_dir, dataset, data_type, eval_dir, use_logits, use_alpha, llm_top_k, window_size, step_size, batched, rerank_type="text"):
    try:
        # Load dataset based on type
        if rerank_type == "code":
            data_path = os.path.join(data_dir, dataset)
        else:  # text reranking
            data_path = os.path.join(data_dir, "beir", dataset)
            
        # Handle dataset loading
        split = "dev" if dataset == "msmarco" else "test"
        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset} from {data_path}: {e}")
            return

        # Load converted retriever results
        try:
            if rerank_type == "code":
                results_output_path = os.path.join(output_path, dataset, f'{data_type}_rank_100.json')
            else:
                results_output_path = os.path.join(output_path, "beir", dataset, f'{data_type}_rank_100.json')
                
            results_loader = ResultsLoader(results_output_path)
            results_to_rerank = results_loader.get_results(with_context=True)
        except Exception as e:
            print(f"Error loading results from {results_output_path}: {e}")
            return

        # Reranking
        try:
            reranked_results, histories = rerank_beir_outputs_llm(
                reranker, results_to_rerank, use_logits=use_logits, use_alpha=use_alpha, 
                top_k=llm_top_k, window_size=window_size, step_size=step_size, 
                batched=batched
            )
        
            # Evaluate results
            converted_results = get_results_to_eval(reranked_results)
            
            if rerank_type == "code":
                pass
                # mrr_at_k = evaluate_results(dataset, qrels, converted_results, rerank_type="code")
                # print("\nMean Reciprocal Rank (MRR) at different cutoffs:")
                # for k, mrr in mrr_at_k.items():
                #     print(f"MRR@{k}: {mrr:.4f}")
            else:
                ndcg, _map, recall, precision = evaluate_results(dataset, qrels, converted_results)
                print(f"\nNDCG (Normalized Discounted Cumulative Gain):\n {ndcg}")
                print(f"\nRecall:\n {recall}\n")

            # Save rerank results to appropriate directory
            if rerank_type == "code":
                save_path = os.path.join(output_path, data_type, dataset)
            else:
                save_path = os.path.join(output_path, "beir", dataset)
                
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save rerank results and run histories
            rerank_results = save_rerank_results(save_path, dataset, converted_results, llm_top_k, 
                              use_logits, use_alpha, is_llm_result=True)
            save_histories(save_path, dataset, histories, llm_top_k, 
                              use_logits, use_alpha, is_llm_result=True)
            print(f"Reranked results saved successfully for dataset {dataset}")

            # Evaluate results
            print(f"Evaluating results for {dataset}...")
            qrels_path = os.path.join(data_path, "qrels", "test.tsv")

            mrr_at_k = get_mrr_at_k(eval_dir, dataset, data_type, qrels_path, rerank_results)
            print(f"Evaluation results for {dataset}:")
            for k, mrr in mrr_at_k.items():
                print(f"MRR@{k}: {mrr:.4f}")
            
        except Exception as e:
            print(f"Error during reranking process: {e}")
            raise
            
    except Exception as e:
        print(f"Unexpected error in rerank_beir_outputs: {e}")
        raise

def process_dataset(model, output_path, data_dir, dataset, data_type, eval_dir, use_logits, use_alpha, llm_top_k, window_size, step_size, batched, context_size, rerank_type="text", code_prompt_type="docstring", api_config={}, device_rank=None):
    logger = logging.getLogger(f"{f'[DEVICE {device_rank}]' if device_rank is not None else ''} Reranking logger")
    reranker = load_reranker(model, use_logits, use_alpha, window_size, batched, context_size, rerank_type, code_prompt_type, api_config)

    start_time = time.perf_counter()
    full_len = len(dataset)

    for i, item in enumerate(dataset):
        print(f"{f'[DEVICE {device_rank}]' if device_rank is not None else ''} Running reranking on {item}")
        rerank_beir_outputs(reranker, output_path, data_dir, item, data_type, eval_dir, use_logits, use_alpha, llm_top_k, window_size, step_size, batched, rerank_type)

        # Logging
        num_processed = i+1
        curr_time = time.perf_counter()
        num_samples_left = full_len - num_processed

        time_elapsed_in_secs = curr_time - start_time
        secs_per_iter = time_elapsed_in_secs/num_processed
        estimated_time_to_process = num_samples_left*secs_per_iter
        logger.warning(f"{f'[DEVICE {device_rank}]' if device_rank is not None else ''} RERANKING LOG {strftime('%m-%d %H:%M:%S', gmtime())} rerank_llm.py] {num_processed}/{full_len} [{strftime('%H:%M:%S', gmtime(int(time_elapsed_in_secs)))}<{strftime('%H:%M:%S', gmtime(int(estimated_time_to_process)))},    {secs_per_iter:.2f}s/it]")