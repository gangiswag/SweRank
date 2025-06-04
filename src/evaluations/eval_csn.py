import os
import torch
import random
import logging
import argparse
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
import fire

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def evaluate(device, lang, batch_size, dataset_dir, output_dir):
    model = SentenceTransformer( "cornstack/CodeRankEmbed", trust_remote_code= True).to(device).to(torch.bfloat16)
    
    corpus, queries, qrels = GenericDataLoader(
                data_folder=os.path.join(dataset_dir, "code_datasets", f'csn_{lang}')
            ).load(split="test")
    

    query_examples = [(k, f'Represent this query for searching relevant code: {v}') for k, v in queries.items()]
    code_examples = [(k, v['text']) for k, v in corpus.items()]
        
    qs = [ex[1] for ex in query_examples]
    cs = [ex[1] for ex in code_examples]
    
    nl_vecs = model.encode(qs, show_progress_bar= True, batch_size= batch_size)
    code_vecs = model.encode(cs, show_progress_bar= True, batch_size= batch_size)


    scores = np.matmul(nl_vecs, code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    print(f"nl_vecs_shape: {nl_vecs.shape} "
          f"\t code_vecs_shape: {code_vecs.shape} "
          f"\t score_matrix_shape: {scores.shape}")

    nl_ids = [ex[0] for ex in query_examples]
    code_ids = [ex[0] for ex in code_examples]

    # Save retrieval results in rank format
    rank_file = os.path.join(output_dir, f'csn_{lang}', 'rank.tsv')
    os.makedirs(os.path.dirname(rank_file), exist_ok=True)
    
    with open(rank_file, 'w') as f:
        for query_idx, (query_id, sort_id) in enumerate(zip(nl_ids, sort_ids)):
            for rank, doc_idx in enumerate(sort_id[:1000]):  # Save top 1000 results
                doc_id = code_ids[doc_idx]
                score = float(scores[query_idx][doc_idx])
                f.write(f"{query_id}\t{doc_id}\t{score}\n")

    ranks = []
    for url, sort_id in zip(nl_ids, sort_ids):
        rank = 0
        find = False
        for i, idx in enumerate(sort_id[:1000]):
            if find is False:
                rank += 1
            if code_ids[idx] == list(qrels[url].keys())[0]:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    mrr = float(np.mean(ranks))
    return mrr


def main(device = 'cuda', output_dir = 'results', dataset_dir = 'datasets', batch_size = 64):
    set_seed(42)
    Path(f'{output_dir}/csn').mkdir(parents=True, exist_ok=True)

    for lang in ['python', 'java', 'ruby', 'php', 'javascript', 'go']:
        mrr = evaluate(device, lang, batch_size, dataset_dir, output_dir)
        print(f'{lang} MRR', mrr)

        result_data = {
            "language": lang,
        }

        with open(f"{output_dir}/csn/overall_results.jsonl", 'a') as f:
            f.write(json.dumps({**result_data, **{'mrr': mrr}}) + "\n")

if __name__ == "__main__":
    fire.Fire(main)
