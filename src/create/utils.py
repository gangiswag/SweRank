import jsonlines
import csv
import os
from tqdm import tqdm 
import json 
def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def save_tsv_dict(data, fp, fields):
    # build dir
    dir_path = os.path.dirname(fp)
    os.makedirs(dir_path, exist_ok=True)
    
    # writing to csv file
    with open(fp, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter='\t',)
        writer.writeheader()
        writer.writerows(data)

def cost_esitmate(path):
    corpus = load_jsonlines(os.path.join(path, "corpus.jsonl"))
    queries = load_jsonlines(os.path.join(path, "queries.jsonl"))
    num_corpus_words = 0
    num_queries_words = 0
    for item in tqdm(corpus):
        num_corpus_words += len(item["text"].split(" "))
    for item in tqdm(queries):
        num_queries_words += len(item["text"].split(" "))
    print(len(corpus))
    print(len(queries))
    print(num_corpus_words)
    print(num_queries_words)


def convert_nl2code_examples_to_features(js, prefix = None):
    #TODO: use actual code and not code tokens
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code = f'{prefix}: {code}' if prefix is not None else code

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl = f'{prefix}: {nl}' if prefix is not None else nl

    return code, nl, (js['url'] if "url" in js else js["retrieval_idx"])

class NL2CodeDataset:
    def __init__(self, file_path=None, prefix = None):
        data = []
        self.url2example = {}
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    try:
                        js = json.loads(line)
                    except:
                        continue
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)
        for js in data:
            code, nl, url = convert_nl2code_examples_to_features(js, prefix)
            self.url2example[url] = {'code': code, 'nl': nl, 'title': js.get('func_name', '')}
    
    def __len__(self):
        return len(self.urls)