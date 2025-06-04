from datasets import load_dataset, Dataset, concatenate_datasets
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
import json
import wandb
from beir.retrieval import models
from sentence_transformers import SentenceTransformer
import torch
import ray 
import fire
import os

def count_lines(file_name, chunk_size=1024*1024):
    count = 0
    with open(file_name, 'rb') as f:
        while chunk := f.read(chunk_size):
            count += chunk.count(b'\n')
    return count

class RetrieverInference:
    def __init__(self, file_name):
        contrast_encoder = models.SentenceBERT()
        st_model = SentenceTransformer("cornstack/CodeRankEmbed", trust_remote_code= True).to(torch.bfloat16)
        st_model.max_seq_length = 1024
        contrast_encoder.q_model = st_model
        contrast_encoder.doc_model = st_model
        model = DRES(contrast_encoder, batch_size= 256, corpus_chunk_size=512*9999)
        self.retriever = EvaluateRetrieval(model, score_function="dot", k_values = [100])
        self.file_name = file_name
        
    
    def __call__(self, item):
        inst = {k : v[0] for k, v in item.items()}
        
        queries = {'query_0' : f'Represent this query for searching relevant code: {inst["query"]}', 'dummy': 'dummy'}
        corpus = {}
        
        for i, doc in enumerate(inst['negative_codes']):
            corpus[f'doc_{i}'] = {
                "title": "",
                "text": doc
            }
        
        corpus['doc_gt'] = {"title": "", "text": inst['positive_code']}

        results = self.retriever.retrieve(corpus, queries)
        
        if 'doc_gt' not in results['query_0']:
            inst['positive_code_score'] = 0
            inst['positive_code_rank'] = 99999
            sorted_res = []
        else:
            sorted_res = [(inst['positive_code'], 'doc_gt', results['query_0']['doc_gt'])]
        
        for doc_id in results['query_0'].keys():
            if doc_id != 'doc_gt':
                idx = doc_id.split('_')[-1]
                sorted_res.append((inst['negative_codes'][int(idx)], doc_id, results['query_0'][doc_id]))
                
        sorted_res = sorted(sorted_res, key= lambda x: x[2], reverse= True)       
        inst['negative_codes'] = []
        inst['negative_ids'] = []
        inst['negative_code_scores'] = []
        inst['negative_code_rank'] = []
        for i, x in enumerate(sorted_res):
            if x[1] != 'doc_gt':
                inst['negative_codes'].append(x[0])
                inst['negative_ids'].append(x[1])
                inst['negative_code_scores'].append(x[2])
                inst['negative_code_rank'].append(i)
            else:
                inst['positive_code_score'] = x[2]
                inst['positive_code_rank'] = i
        
        with open(self.file_name, 'a') as f:
            f.write(json.dumps(inst) + '\n')

def main(start_idx, end_idx, num_workers_per_gpu = 1, micro_batch_size = 100, overwrite = False):
    wandb.init(
        project="code",
        entity="ragllm"
    )

    num_gpus = torch.cuda.device_count()
    
    ds = [load_dataset("cornstack/repo_contrastive_premined", split = 'train', streaming= True)] if start_idx == 0 else []
    ds = ds + [load_dataset(f"cornstack-dev/repo_contrastive_premined_{i}", split = 'train', streaming= True) for i in range(max(1, start_idx), end_idx + 1)]
    ds = concatenate_datasets(ds)

    data_file_name = f"repo_contrastive_mined_{start_idx}_{end_idx}.jsonl"
    
    if overwrite and os.path.exists(data_file_name):
        os.remove(data_file_name)
    
    already_processed = 0
    if os.path.exists(data_file_name):
        already_processed = count_lines(data_file_name)

    print('ALREADY PROCESSED ', already_processed)
    @ray.remote(num_gpus=1/num_workers_per_gpu)
    class InferenceActor:
        def __init__(self):
            self.inference = RetrieverInference(data_file_name)
        
        def process_item(self, item):
            return self.inference(item)
    
    if not ray.is_initialized():
            ray.init()

    actors = [InferenceActor.remote() for _ in range(num_gpus * num_workers_per_gpu)]
    futures = []
    batch_ds = []
    num_processed = 0
    
    for inst in ds:
        num_processed += 1
        if num_processed <= already_processed:
            continue
        
        batch_ds.append(inst)
        if len(batch_ds) == micro_batch_size:
            batch_ds = Dataset.from_list(batch_ds)
            batch_ds = ray.data.from_huggingface(batch_ds)
            
            for batch in batch_ds.iter_batches(batch_size= 1):
                actor_id = len(futures) % len(actors)
                futures.append(actors[actor_id].process_item.remote(batch))
            
            results = ray.get(futures)
            wandb.log({"num_processed": num_processed})
            batch_ds = []
    
    if batch_ds:
        batch_ds = Dataset.from_list(batch_ds)
        batch_ds = ray.data.from_huggingface(batch_ds)
        
        for batch in batch_ds.iter_batches(batch_size= 1):
            actor_id = len(futures) % len(actors)
            futures.append(actors[actor_id].process_item.remote(batch))
        
        results = ray.get(futures)
        wandb.log({"num_processed": num_processed})
    
    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)