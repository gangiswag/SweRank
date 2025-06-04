import os
import re
import ujson
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ALPH_START_IDX = ord('A') - 1
SYS_MSG = "You are CodeRanker, an intelligent code reviewer that can analyze GitHub issues and rank code functions based on their relevance to contain the faults causing the GitHub issue."

USR_PREFIX_ALPHA = "I will provide you with {num} code functions, each indicated by a alphabetical identifier []. Rank the code functions based on their relevance to contain the faults causing the GitHub issue: {query}.\n"
EX_ORDERING_ALPHA="[D] > [B]"

USR_PREFIX_NUMERIC = "I will provide you with {num} code functions, each indicated by a numerical identifier []. Rank the code functions based on their relevance to contain the faults causing the GitHub issue: {query}.\n"
EX_ORDERING_NUMERIC = "[4] > [2]"

USR_SUFFIX = "GitHub Issue: {query}.\nRank the {num} code functions above based on their relevance to contain the faults causing the GitHub issue. All the code functions should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}. Only respond with the ranking results, do not say any word or explain."

def save_jsonl(data: list[dict], file_path: str):
    with open(file_path, 'w') as f:
        for item in data:
            ujson.dump(item, f)
            f.write('\n')
    f.close()

def load_jsonl(data_path: str):
    return [ujson.loads(line) for line in open(data_path, 'r')]

def replace_number(s: str, use_alpha: bool) -> str:
    if use_alpha:
        return re.sub(r"\[([A-z]+)\]", r"(\1)", s)
    else:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

def get_input_context(positive_code: str, positive_code_rank: int,
                       negative_codes: list[str], negative_code_rank: list[int], use_alpha: bool):
    context_dict = {negative_code_rank[i]: negative_code for i, negative_code in enumerate(negative_codes)}
    context_dict[positive_code_rank] = positive_code

    input_context = ""
    num_identifiers = len(negative_codes) + 1
    for i in range(num_identifiers):
        identifier = chr(ALPH_START_IDX + i+1) if use_alpha else str(i+1)
        input_context += f"[{identifier}] {replace_number(context_dict[i], use_alpha)}\n"

    return input_context

def get_model_response(positive_code_rank: int, negative_code_rank: list[int], use_alpha: bool, first_identifier_only: bool):
    identifiers_order = []

    # Positive identifier comes first
    identifier = chr(ALPH_START_IDX + positive_code_rank+1) if use_alpha else str(positive_code_rank+1)
    identifiers_order.append(f"[{identifier}]")

    if first_identifier_only:
        return "".join(identifiers_order)

    random.shuffle(negative_code_rank)

    for content_id in negative_code_rank:
        identifier = chr(ALPH_START_IDX + content_id+1) if use_alpha else str(content_id+1)
        identifiers_order.append(f"[{identifier}]")
    
    return " > ".join(identifiers_order)

def construct_conversations(
        item: dict, 
        tokenizer: PreTrainedTokenizerBase, 
        max_content_len: int, 
        max_len_per_code: int,
        use_alpha: bool, 
        window_size: int, 
        first_identifier_only: bool, 
        shuffle_context: bool):
    query = item['query']

    negative_codes = item['negative_codes']
    negative_code_rank = item['negative_code_rank']

    positive_code = item['positive_code']
    positive_code_rank = item['positive_code_rank']

    merged_code_rank = negative_code_rank + [positive_code_rank]
    full_rank =  [sorted(merged_code_rank).index(x) for x in merged_code_rank]
    
    positive_code_rank = full_rank[-1]
    negative_code_rank = full_rank[:-1]

    # Negatives random sampling to fit the window size
    if len(negative_codes) > window_size-1:
        negative_code_idx = range(len(negative_codes))
        negative_sample_idx = random.sample(negative_code_idx, k=window_size-1)

        negative_codes = np.array(negative_codes)[negative_sample_idx].tolist()
        negative_code_rank = np.array(negative_code_rank)[negative_sample_idx].tolist()

        # Determine the ranks of the samples
        merged_code_rank = negative_code_rank + [positive_code_rank]
        full_rank =  [sorted(merged_code_rank).index(x) for x in merged_code_rank]
        
        positive_code_rank = full_rank[-1]
        negative_code_rank = full_rank[:-1]
    
    # Random shuffling the input context order
    if shuffle_context: 
        merged_code_rank = negative_code_rank + [positive_code_rank]
        random.shuffle(merged_code_rank)

        positive_code_rank = merged_code_rank[-1]
        negative_code_rank = merged_code_rank[:-1]        

    max_len_per_code = 1024
    tokenized_content = tokenizer.tokenize(positive_code, add_special_tokens=False)
    content_tokens = tokenized_content[:max_len_per_code]
    positive_code = tokenizer.convert_tokens_to_string(content_tokens)

    for i, negative_code in enumerate(negative_codes):
        tokenized_content = tokenizer.tokenize(negative_code, add_special_tokens=False)
        content_tokens = tokenized_content[:max_len_per_code]
        negative_code = tokenizer.convert_tokens_to_string(content_tokens)
        negative_codes[i] = negative_code

    # Construct model prompt
    ret = []
    ret.append({ # Add system message
        "from": "system",
        "value": SYS_MSG,
    })

    input_context = get_input_context(
        positive_code=positive_code,
        positive_code_rank=positive_code_rank,
        negative_codes=negative_codes,
        negative_code_rank=negative_code_rank,
        use_alpha=use_alpha,)

    user_msg = ""
    if use_alpha:
        user_msg += USR_PREFIX_ALPHA.format(num=len(negative_codes)+1, query=query) + input_context + USR_SUFFIX.format(query=query, num=len(negative_codes)+1, example_ordering=EX_ORDERING_ALPHA)
    else:
        user_msg += USR_PREFIX_NUMERIC.format(num=len(negative_codes)+1, query=query) + input_context + USR_SUFFIX.format(query=query, num=len(negative_codes)+1, example_ordering=EX_ORDERING_NUMERIC)

    ret.append({
        "from": "user",
        "value": user_msg,
    })

    # Construct model response
    model_response = get_model_response(
        positive_code_rank=positive_code_rank, negative_code_rank=negative_code_rank, use_alpha=use_alpha, first_identifier_only=first_identifier_only)
    ret.append({
        "from": "gpt",
        "value": model_response,
    })
    
    user_msg_token_cnt = len(tokenizer.tokenize(user_msg, add_special_tokens=False))
    model_response_token_cnt = len(tokenizer.tokenize(model_response, add_special_tokens=False))

    # Truncate query
    if user_msg_token_cnt > max_content_len:
        over_token_cnt = user_msg_token_cnt - max_content_len
        tokenized_query = tokenizer.tokenize(query, add_special_tokens=False)
        tokenized_query = tokenized_query[:len(tokenized_query) - over_token_cnt // 2]
        query = tokenizer.convert_tokens_to_string(content_tokens)
        
        user_msg = ""
        if use_alpha:
            user_msg += USR_PREFIX_ALPHA.format(num=len(negative_codes)+1, query=query) + input_context + USR_SUFFIX.format(query=query, num=len(negative_codes)+1, example_ordering=EX_ORDERING_ALPHA)
        else:
            user_msg += USR_PREFIX_NUMERIC.format(num=len(negative_codes)+1, query=query) + input_context + USR_SUFFIX.format(query=query, num=len(negative_codes)+1, example_ordering=EX_ORDERING_NUMERIC)
        user_msg_token_cnt = len(tokenizer.tokenize(user_msg, add_special_tokens=False))
        ret[1]['value'] = user_msg
        assert user_msg_token_cnt + model_response_token_cnt < max_content_len
        
    return ret, positive_code_rank, negative_code_rank, user_msg_token_cnt + model_response_token_cnt

def main(args):
    random.seed(args.random_seed)

    if args.data_type == 'local':
        dataset = load_jsonl(args.data_path)
    elif args.data_type == 'hf':
        dataset = list(load_dataset(args.data_path, split="train"))
    else:
        raise Exception(f"Invalid data_type: {args.data_type}")
    print(f"Retriever results size: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("Constructing reranker training and eval data")
    train_ret = []
    eval_ret = []
    eval_size = int(len(dataset)*args.eval_ratio)

    random.shuffle(dataset)

    positive_ranks = []

    token_cnts = []
    sys_msg_token_cnt = len(tokenizer.tokenize(SYS_MSG, add_special_tokens=False))
    max_content_len = args.max_len

    if args.varying_window_size:
        window_sizes = [args.window_size-5,args.window_size-3,args.window_size]
    else:
        window_sizes = [args.window_size]
    for i, item in enumerate(tqdm(dataset)):
        if len(item['negative_codes']) < 5:
            continue
        for k in window_sizes:
            ret, positive_window_rank, negative_window_ranks,  token_cnt = construct_conversations(
                    item=item, 
                    tokenizer=tokenizer,
                    max_content_len=max_content_len,
                    max_len_per_code=args.max_len_per_code,
                    use_alpha=args.use_alpha, 
                    window_size=k,
                    first_identifier_only=args.first_identifier_only,
                    shuffle_context=args.shuffle_context)
            negative_labels = [get_model_response(negative_rank, negative_window_ranks, args.use_alpha, args.first_identifier_only) for negative_rank in negative_window_ranks]

            if i < eval_size:
                eval_ret.append({
                    "id": item['instance_id'],
                    "repo": item['repo'],
                    "conversations": ret,
                    "neagtive_labels": negative_labels,})
            else:
                train_ret.append({
                    "id": item['instance_id'],
                    "repo": item['repo'],
                    "conversations": ret,
                    "neagtive_labels": negative_labels,})
            token_cnts.append(token_cnt+sys_msg_token_cnt)
            positive_ranks.append(positive_window_rank)

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # Save training data
    out_file_name_prefix = "reranking_function_localization"
    if args.varying_window_size:
        out_file_name_prefix += "-varying_window_size"
    out_file_name = f"{out_file_name_prefix}_train.jsonl"
    out_full_path = os.path.join(args.out_path, out_file_name)
    print(f"Saving output to {out_full_path}")
    save_jsonl(train_ret, out_full_path)

    # Save eval data
    out_file_name = f"{out_file_name_prefix}_eval.jsonl"
    out_full_path = os.path.join(args.out_path, out_file_name)
    print(f"Saving output to {out_full_path}")
    save_jsonl(eval_ret, out_full_path)

    out_plot_name = f"{out_file_name_prefix}_positive_window_rank.png"
    out_plot_full_path = os.path.join(args.out_path, out_plot_name)
    plt.hist(positive_ranks,bins=args.window_size)
    plt.savefig(out_plot_full_path)
    plt.clf()

    out_plot_name = f"{out_file_name_prefix}_token_cnts.png"
    out_plot_full_path = os.path.join(args.out_path, out_plot_name)
    plt.hist(token_cnts)
    plt.savefig(out_plot_full_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, required=True, choices=['local', 'hf'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)

    parser.add_argument('--model_name', type=str, default='cornstack/CodeRankLLM', required=False)
    parser.add_argument('--window_size', type=int, default=10, required=False)
    parser.add_argument('--eval_ratio', type=float, default=0.01, required=False)

    parser.add_argument('--max_len', type=int, default=16384, required=False)
    parser.add_argument('--max_len_per_code', type=int, default=1024, required=False)

    parser.add_argument('--random_seed', type=int, default=42, required=False)

    parser.add_argument('--use_alpha', action="store_true")
    parser.add_argument('--shuffle_context', action="store_true")
    parser.add_argument('--varying_window_size', action="store_true")
    parser.add_argument('--first_identifier_only', action="store_true")

    args = parser.parse_args()
    main(args)