import os
import ujson
import fire
import numpy as np
from tqdm import tqdm

NEGATIVE_INDICES=range(3,25+1)

def load_jsonl(file_path: str):
    return [ujson.loads(line) for line in open(file_path, 'r')]
def save_jsonl(data: list[dict], file_path: str):
    print("Saving file")
    with open(file_path, 'w') as f:
        for item in tqdm(data):
            ujson.dump(item, f)
            f.write('\n')
    f.close()

def filter_dataset(dataset: list[dict]):
    ret = []
    for item in tqdm(dataset):
        positive_code_rank = item['positive_code_rank']
        
        if positive_code_rank > 10 or positive_code_rank < 0:
            continue

        negative_code_rank = item['negative_code_rank']
        negative_pass_idx = []
        for i, rank in enumerate(negative_code_rank):
            if rank in NEGATIVE_INDICES:
                negative_pass_idx.append(i)
        for k, v in item.items():
            if "negative_" in k:
                item[k] = np.array(v)[negative_pass_idx].tolist()
        ret.append(item)
    return ret

def main(file_path: str=os.environ['PWD'], file_prefix: str="repo_contrastive_mined_"):
    if not os.path.exists(file_path):
        raise Exception(f"Path {file_path} doesn't exist. Pass the path of the directory contatining contrastive mined data prefiex with '{file_prefix}*'")
    
    ret = []
    for shard_name in os.listdir(file_path):
        if shard_name.startswith(file_prefix) and shard_name.endswith(".jsonl") and "filtered" not in shard_name:
            print(f"Loading shard: {shard_name}")
            dataset = load_jsonl(os.path.join(file_path, shard_name))
            ret += filter_dataset(dataset)
            print(f"Curr size: {len(ret)}")

    save_file_name = "repo_contrastive_mined_filtered.jsonl"
    save_full_path = os.path.join(file_path, save_file_name)
    print(f"Finished filtering all shard. Saving to {save_full_path}")
    save_jsonl(ret, save_full_path)


if __name__ == '__main__':
    fire.Fire(main)