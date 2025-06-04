import os
import json
from tqdm import tqdm
import fire

def load_jsonl(file_path):
    return [json.loads(line) for line in open(os.path.join(file_path), 'r')]

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)
    f.close()

def main(repo_info_path="pypi_rankings.jsonl", out_path="valid_top_pypi_gitrepos.jsonl"):
    repo_info_path = "pypi_rankings.jsonl"
    data = load_jsonl(repo_info_path)
    valid_repo = []
    for item in data:
        if item['github']:
            valid_repo.append(item['github'])

    # extract git repo from urls
    repo_names = []
    for url in tqdm(valid_repo):
        if "https://github.com/" not in url:
            continue
        else:
            try:
                url = url[url.index("https://github.com") + len("https://github.com"):]
                splitted = [x.strip() for x in url.split('/') if x.strip() != ""]
                repo_name = f"{splitted[0].strip()}/{splitted[1].strip()}"
                repo_names.append(repo_name)
            except Exception as e:
                print(f"Error while parsing git-repo name from {url}: skipping {url}")
    repo_names = sorted(list(set(repo_names)))
    print(f"Total number of valid repos: {len(repo_names)}")


    out_file_path = "valid_top_pypi_gitrepos.jsonl"
    save_json(repo_names, out_file_path)
if __name__ == '__main__':
    fire.Fire(main)