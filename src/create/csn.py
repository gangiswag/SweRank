import json
import argparse
import pandas as pd
from pathlib import Path 
from utils import save_tsv_dict, save_file_jsonl, NL2CodeDataset
import os
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets")

    args = parser.parse_args()
    
    # Create code_datasets directory
    code_datasets_dir = os.path.join(args.dataset_dir, "code_datasets")
    os.makedirs(code_datasets_dir, exist_ok=True)
    
    # Change to dataset directory for download
    orig_dir = os.getcwd()
    os.chdir(code_datasets_dir)
    
    # Download and extract dataset
    commands = [
        "wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip",
        "unzip dataset.zip",
        "rm -r dataset.zip",
        "mv dataset CSN",
        "cd CSN && bash run.sh",
        "cd .."
    ]

    print("Downloading and preparing CSN dataset...")
    for command in commands:
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command '{command}': {e}")
            os.chdir(orig_dir)
            return

    for lang in ['python', 'java', 'ruby', 'php', 'javascript', 'go']:
        print(f"Processing {lang}...")
        path = Path(os.path.join(code_datasets_dir, f'csn_{lang}'))
        path.mkdir(parents=True, exist_ok=True)
        qrels_path = Path(os.path.join(path, 'qrels'))
        qrels_path.mkdir(parents=True, exist_ok=True)
        
        query_dataset = NL2CodeDataset(f'CSN/{lang}/test.jsonl', None)
        code_dataset = NL2CodeDataset(f'CSN/{lang}/codebase.jsonl', prefix=None)
        
        queries, docs, qrels = [], [], []
        url2id = {}
        i = 0
        for url, example in code_dataset.url2example.items():
            url2id[url] = i
            docs.append({'_id': f'{url2id[url]}_code', 'text': example['code'], 'title': example['title'], 'metadata': {}})
            i += 1
            
        for url, example in query_dataset.url2example.items():
            queries.append({'_id': f'{url2id[url]}_query', 'text': example['nl'], 'metadata': {}})
            qrels.append({"query-id": f'{url2id[url]}_query', "corpus-id": f'{url2id[url]}_code', "score": 1})
        
        save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
        save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))
        qrels_path = os.path.join(path, "qrels", "test.tsv")
        save_tsv_dict(qrels, qrels_path, ["query-id", "corpus-id", "score"])

    # Cleanup
    subprocess.run('rm -r CSN', shell=True, check=True)
    subprocess.run('rm -r _MACOSX', shell=True, check=True)
    
    os.chdir(orig_dir)

if __name__ == "__main__":
    main()