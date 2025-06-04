import os
import json
import fire
from get_tasks_pipeline import main as get_tasks_pipeline
from tqdm import tqdm

def load_jsonl(file_path):
    return [json.loads(line) for line in open(os.path.join(file_path), 'r')]

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    f.close()
    return data

def main(repo_info_path, prs_path, tasks_path, git_token, start_idx=0, num_repos_to_process=-1):
    assert git_token, "Pass the git_token for api access"
    os.environ['GITHUB_TOKENS'] = git_token

    data = load_json(repo_info_path)
    print(f"Total number of repos: {len(data)}")

    print(f"Staring pulling PRs from repo #{start_idx+1}")

    if not os.path.exists(prs_path):
        os.mkdir(prs_path)
    if not os.path.exists(tasks_path):
        os.mkdir(tasks_path)

    for git_repo in tqdm(data[start_idx:start_idx+num_repos_to_process]):
        get_tasks_pipeline(
            repos = [git_repo],
            path_prs=prs_path,
            path_tasks=tasks_path,
        )

if __name__ =='__main__':
    fire.Fire(main)