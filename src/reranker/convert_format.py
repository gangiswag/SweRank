from beir.datasets.data_loader import GenericDataLoader
import csv
import re
import os
import json
from collections import defaultdict
from .utils.result import Result, ResultsWriter
from datasets import load_dataset

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_json(file_path, read_mode='r'):
        data = []
        with open(file_path, read_mode, encoding="utf-8") as f:
            json_data = re.sub(r"}\s*{", "},{", f.read())
            data.extend(json.loads("["+json_data+"]"))
        f.close()
        return data

def convert_solutions_dict(dataset, key = 'model_patch'):
    return {elem['instance_id']: elem[key] for elem in dataset}

def convert_results_single_pass(prefix, output_path, data_dir, data_type, retriever_results_path, top_k, rerank_type="code"):
    retrieval_results = load_json(retriever_results_path)
    print("Converting data for reranking run...")
    converted_cnts = 0

    if 'loc-bench' in prefix:
        loc_bench_dataset = load_dataset("czlll/Loc-Bench_V1")['test']
        valid_datsets = [item['instance_id'] for item in loc_bench_dataset]

    for item in retrieval_results:
        qid = item['instance_id']
        docs = item['docs']

        if 'loc-bench' in prefix and qid not in valid_datsets:
            continue

        dataset_name = f"{prefix}-function_{qid}"
        print(os.path.join(data_dir, dataset_name))
        
        dataset_dir = os.path.join(data_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"{dataset_dir} doesn't exist. Skipping instance: {dataset_name}")
            continue
        else:
            corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join(data_dir, dataset_name)).load(split="test")


        num_docs = len(docs)
        pseudo_scores = list(reversed(range(1,num_docs+1)))
        dataset_output_path = os.path.join(output_path, dataset_name)

        results = {f"{qid}":{}}
        for doc_id, score in zip(docs, pseudo_scores):
            results[qid][doc_id] = score

        print("Converting to reranker results")
        # Remove dummy entries if present (for code reranking)
        if 'dummy' in results:
            results.pop('dummy')
            
        results_to_rerank = to_reranker_results(results, queries, corpus, top_k)

        # Ensure output directory exists
        os.makedirs(dataset_output_path, exist_ok=True)
        print(dataset_output_path)
        
        results_output_path = os.path.join(dataset_output_path, f'{data_type}_rank_{top_k}.json')
        results_writer = ResultsWriter(results_to_rerank)
        results_writer.write_in_json_format(results_output_path)
        print(f"Results saved to {results_output_path}")

        converted_cnts += 1
    print(f"Number of valid instances prepared for reranking: {converted_cnts}")


def convert_results(output_path, data_dir, dataset, data_type, top_k, rerank_type="text"):
    """Convert ranking results to format suitable for reranking
    
    Args:
        rerank_type (str): Whether this is for "text" or "code" reranking
    """
    print(f"Loading {dataset} dataset")
    
    try:
        # Load datasets based on type
        if rerank_type == "code":
            if dataset in ('swebench_function', 'swebench_file'):
                return convert_results_swebench(output_path, data_dir, dataset, data_type, top_k)
            
            data_path = os.path.join(data_dir, dataset)
            print(f"DATA PATH: {data_path}")
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
            
            # Load rank data
            rank_path = os.path.join(output_path, dataset, "rank.tsv")
            dataset_output_path = os.path.join(output_path, dataset)
            
        else:  # text reranking
            out_dir = os.path.join(data_dir, "beir")
            data_path = os.path.join(out_dir, dataset)
            
            split = "dev" if dataset == "msmarco" else "test"
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
            
            # Load rank data
            dataset_output_path = os.path.join(output_path, "beir", dataset)
            rank_path = os.path.join(dataset_output_path, "rank.tsv")

        print("Loading rank data")
        if not os.path.exists(rank_path):
            print(f"Rank file not found: {rank_path}")
            return

        results = {}
        with open(rank_path, 'r') as rank_file:
            csv_reader = csv.reader(rank_file, delimiter="\t", quotechar='|')
            if rerank_type == "code":
                next(csv_reader)  # Skip header for code files
            for row in csv_reader:
                qid = str(row[0])
                pid = str(row[1])
                score = float(row[2])
                if qid not in results:
                    results[qid] = {}
                results[qid][pid] = score

        print("Converting to reranker results")
        # Remove dummy entries if present (for code reranking)
        if 'dummy' in results:
            results.pop('dummy')
            
        results_to_rerank = to_reranker_results(results, queries, corpus, top_k)

        # Ensure output directory exists
        os.makedirs(dataset_output_path, exist_ok=True)
        
        results_output_path = os.path.join(dataset_output_path, f'{data_type}_rank_{top_k}.json')
        results_writer = ResultsWriter(results_to_rerank)
        results_writer.write_in_json_format(results_output_path)
        print(f"Results saved to {results_output_path}")
        
    except Exception as e:
        print(f"Error in convert_results: {e}")
        raise

def convert_results_swebench(output_path, data_dir, dataset, data_type, top_k):
    """Special handling for swebench datasets"""
    prefx = f"csn_{dataset.split('_')[1]}"
    instance_list = [instance for instance in os.listdir(data_dir) if instance.startswith(prefx)]
    
    for dataset_instance in instance_list:
        print(f"Loading {dataset_instance} dataset")
        data_path = os.path.join(data_dir, data_type, dataset_instance)

        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

            print("Loading rank data")
            rank_path = os.path.join(output_path, data_type, dataset_instance, "rank.tsv")
            
            results = {}
            with open(rank_path, 'r') as rank_file:
                csv_reader = csv.reader(rank_file, delimiter="\t", quotechar='|')
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    qid = str(row[0])
                    pid = str(row[1])
                    score = float(row[2])
                    if qid not in results:
                        results[qid] = {}
                    results[qid][pid] = score

            print("Converting to reranker results")
            results_to_rerank = to_reranker_results(results, queries, corpus, top_k)

            dataset_output_path = os.path.join(output_path, data_type, dataset_instance)
            os.makedirs(dataset_output_path, exist_ok=True)
            
            results_output_path = os.path.join(dataset_output_path, f'{data_type}_rank_{top_k}.json')
            results_writer = ResultsWriter(results_to_rerank)
            results_writer.write_in_json_format(results_output_path)
            print(f"Results saved to {results_output_path}")
            
        except Exception as e:
            print(f"Error processing {dataset_instance}: {e}")
            continue

def to_reranker_results(results, queries, corpus, top_k):
    """Convert results to format needed by reranker"""
    retrieved_results_with_text = []
    for qid, docs_scores in results.items():
        query_text = queries[qid]
        for doc_id, score in docs_scores.items():
            doc_text = corpus[doc_id]
            result_with_text = {
                'qid': qid,
                'query_text': query_text,
                'doc_id': doc_id,
                'doc_text': doc_text,
                'score': score
            }
            retrieved_results_with_text.append(result_with_text)

    hits_by_query = defaultdict(list)
    for result in retrieved_results_with_text:
        content_string = ''
        if isinstance(result['doc_text'], dict):
            if result['doc_text'].get('title'):
                content_string += result['doc_text']['title'] + ". "
            content_string += result['doc_text']['text']
        else:
            content_string = result['doc_text']
            
        hits_by_query[result['query_text']].append({
            'qid': result['qid'],
            'docid': result['doc_id'],
            'score': result['score'],
            'content': content_string
        })

    results_to_rerank = []
    for query_text, hits in hits_by_query.items():
        sorted_hits = sorted(hits, reverse=True, key=lambda x: x['score'])[:top_k]
        result = Result(query=query_text, hits=sorted_hits)
        results_to_rerank.append(result)
    
    return results_to_rerank