import os
import json
import logging
import argparse
from time import time
from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from tqdm import tqdm
from transformers import AutoTokenizer
from copy import deepcopy
import torch
import csv
from sentence_transformers import SentenceTransformer
from collections import defaultdict

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
        
def get_top_docs(results: dict, corpus: dict, task_id: str, topk: int = 100) -> list[str]:
    if task_id not in results: return []
    doc_scores = results[task_id]
    doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    doc_scores_sorted = doc_scores_sorted[:topk]
    doc_code_snippets = [code_id for code_id, score in doc_scores_sorted]
    return doc_code_snippets

def save_beir_results_to_tsv(results, output_file):
    # Create a defaultdict to store the results
    formatted_results = defaultdict(dict)

    # Process the results
    for query_id, doc_scores in results.items():
        for doc_id, score in doc_scores.items():
            formatted_results[query_id][doc_id] = score

    # Write to TSV
    with open(output_file, 'w', newline='') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter='\t')
        tsvwriter.writerow(['Query ID', 'Corpus ID', 'Relevance Score'])

        for query_id, doc_scores in formatted_results.items():
            for doc_id, score in doc_scores.items():
                tsvwriter.writerow([query_id, doc_id, score])

    print(f"Results saved to {output_file}")

def save_beir_results_to_tsv_list(all_results, output_file):
    results_dct = {}
    for result in all_results:
        for k, v in result.items():
            if k in results_dct:
                import pdb;pdb.set_trace()
            results_dct[k] = v 
    save_beir_results_to_tsv(results_dct, output_file) 

def main(args):

    contrast_encoder = models.SentenceBERT()
    contrast_encoder.q_model = SentenceTransformer(args.model, trust_remote_code= True).to(torch.bfloat16)
    contrast_encoder.doc_model = SentenceTransformer(args.model, trust_remote_code= True).to(torch.bfloat16)
    contrast_encoder.q_model.max_seq_length = args.sequence_length
    contrast_encoder.doc_model.max_seq_length = args.sequence_length

    model = DRES(contrast_encoder, batch_size=args.batch_size, corpus_chunk_size=512*9999)
    retriever = EvaluateRetrieval(model, score_function="dot")

    model_name = args.model.split("/")[-1]
    args.level = "function"
    
    args.output_file = f"{args.output_dir}/model={model_name}_dataset={args.dataset}_split={args.split}_level={args.level}_eval_mode=default_output.json"
    args.results_file = f"{args.output_dir}/model={model_name}_dataset={args.dataset}_split={args.split}_level={args.level}_eval_mode=default_results.json"
    args.rerank_input_file = f"{args.output_dir}/model={model_name}_dataset={args.dataset}_split={args.split}_level={args.level}_eval_mode=default_rerank_input.tsv"
    
    if os.path.exists(args.results_file):
        os.remove(args.results_file)

    if args.tok:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code = True)
        query_tokens = []
        corpus_tokens = []
    

    if args.dataset.startswith("swe-bench"):
        all_eval_results = []
        
        if args.dataset.startswith("swe-bench"):
            if 'lite' in args.dataset.lower():
                swebench = load_dataset("princeton-nlp/SWE-bench_Lite")[args.split]
            elif 'verified' in args.dataset.lower():
                swebench = load_dataset("princeton-nlp/SWE-bench_Verified")[args.split]
            else:
                swebench = load_dataset("princeton-nlp/SWE-bench")[args.split]
            all_top_docs = [[] for _ in swebench]
            if args.split == 'test':
                prefx = f"{args.dataset}" 
            else: 
                prefx = f"{args.dataset}-{args.split}"
            if args.level == 'file':
                prefx += '_'
            else:
                prefx += f'-{args.level}_'
        else:
            raise ValueError(f"`dataset` should starts with either 'swe-bench'.")
        
        instance_list = [i for i in os.listdir(args.dataset_dir) if i.startswith(prefx)]

        for ins_dir in tqdm(instance_list):
            logging.info("Instance Repo: {}".format(ins_dir))
            # load data and perform retrieval
            corpus, queries, qrels = GenericDataLoader(
                data_folder=os.path.join(args.dataset_dir, ins_dir)
            ).load(split="test")
            if args.add_prefix and args.query_prefix != '':
                queries = {k : f'{args.query_prefix}: {v}' for k, v in queries.items()}
            
            logging.info(f"Instance #{ins_dir}: #{len(corpus)} corpus, #{len(queries)} queries")
            
            if args.tok:
                for v in queries.values():
                    query_tokens.append(tokenizer(v, padding=True, truncation=False, return_tensors="pt")['input_ids'].shape[1])

                for v in corpus.values():
                    corpus_tokens.append(tokenizer(v['text'], padding=True, truncation=False, return_tensors="pt")['input_ids'].shape[1])

                continue
            
            dir_path = os.path.join(args.output_dir, ins_dir)
            os.makedirs(dir_path, exist_ok=True)
            rerank_tsv_file = os.path.join(dir_path, "rank.tsv")
            
            start_time = time()
            if len(queries) == 1:
                queries.update({"dummy": "dummy"})
            results = retriever.retrieve(corpus, queries)
            save_beir_results_to_tsv(results, rerank_tsv_file)
            if "dummy" in queries:
                queries.pop("dummy")
                results.pop("dummy")
            end_time = time()
            logging.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

            # get topk retrieved docs
            if args.dataset.startswith("swe-bench"):
                indices = [i for i,ex in enumerate(swebench) if ex["instance_id"] in queries]
                for index in indices:
                    instance_id = swebench[index]["instance_id"]
                    all_top_docs[index] = get_top_docs(results, corpus, instance_id)
            
            else:
                raise ValueError(f"`dataset` should starts with either 'swe-bench'.")

            # evaluate retrieval results
            if len(qrels) == 0:
                logging.info("No qrels found for this dataset.")
                return
            logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
            eval_results = {
                "ndcg": ndcg, "mrr": mrr,
                "recall": recall, "precision": precision,
                "time": end_time - start_time
            }
            logging.info(f"Instance #{ins_dir}: {eval_results}")
            all_eval_results.append(eval_results)
            
            with open(args.output_file + "_all", "w") as f:
                json.dump(all_eval_results, f)
        
        if args.tok:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            # Create the plot
            plt.figure(figsize=(10, 6))
            sns.histplot(query_tokens, bins=50, kde=True, stat='probability', label='queries distribution')

            # Set the axis labels
            plt.xlabel('tokens length')
            plt.ylabel('frequency')
            plt.legend()

            # Display the plot
            plt.savefig('queries_tok.png')
            plt.close()
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            sns.histplot(corpus_tokens, bins=50, kde=True, log_scale = (True, False), stat='probability', label='corpus distribution')

            # Set the axis labels
            plt.xlabel('tokens length')
            plt.ylabel('frequency')
            plt.legend()

            # Display the plot
            plt.savefig('corpus_tok.png')
            plt.close()
            
            print('queries mean tokens ', np.mean(query_tokens))
            print('corpus mean tokens ', np.mean(corpus_tokens))
            return

        if args.dataset.startswith("swe-bench"):
            swebench = swebench.add_column("docs", all_top_docs)
            swebench.to_json(args.results_file)

        avg_eval_results = {}
        for k,v_dict in all_eval_results[0].items():
            if isinstance(v_dict, dict):
                avg_v_dict = {}
                for vk,vv in v_dict.items():
                    avg_vv = sum([e[k][vk] for e in all_eval_results])/len(all_eval_results)
                    avg_v_dict[vk] = avg_vv
                avg_eval_results.update(avg_v_dict)
            elif isinstance(v_dict, float):
                avg_v = sum([e[k] for e in all_eval_results])/len(all_eval_results)
                avg_eval_results[k] = avg_v
            else:
                raise ValueError
        print("Average Eval Results: ", avg_eval_results)
        with open(args.output_file, "w") as f:
            json.dump(avg_eval_results, f)
        
    else:
        raise ValueError(f"`dataset` should starts with either 'swe-bench'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="swe-bench-lite",
                        help="Dataset to use for evaluation")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use for evaluation")
    parser.add_argument("--model", type=str, default="cornstack/CodeRankEmbed", help="Sentence-BERT model to use")
    parser.add_argument("--tokenizer", type=str, default= "Snowflake/snowflake-arctic-embed-m-long", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for retrieval")
    parser.add_argument("--sequence_length", type=int, default=1024, help="Sequence length for retrieval")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--dataset_dir", type=str, default="datasets",
                        help="Specify the filepath where the dataset is storied")
    parser.add_argument('--tok', default= False, type = bool)
    parser.add_argument('--add_prefix', default= True, type = bool)
    parser.add_argument('--query_prefix', default= 'Represent this query for searching relevant code', type = str)
    parser.add_argument('--document_prefix', default= '', type = str)
    args = parser.parse_args()

    main(args)