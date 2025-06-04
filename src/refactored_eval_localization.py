import json
import os
import sys
import pandas as pd

from typing import Optional
from torch import Tensor
import torch
from datasets import load_dataset
import collections
import re
import fire
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader
from datasets import load_dataset
import copy

def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]

def load_json(file_path, read_mode='r'):
        data = []
        with open(file_path, read_mode, encoding="utf-8") as f:
            json_data = re.sub(r"}\s*{", "},{", f.read())
            data.extend(json.loads("["+json_data+"]"))
        f.close()
        return data

def get_sorted_documents_func(results, sorted_documents_per_query):

    for query_id, doc_scores in results.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_documents_per_query[query_id] = [doc_id for doc_id, score in sorted_docs]

    return sorted_documents_per_query


filtered_instances=['pytest-dev__pytest-5227',
 'sympy__sympy-15345',
 'sympy__sympy-21614',
 'scikit-learn__scikit-learn-13439',
 'sympy__sympy-11400',
 'sympy__sympy-19487',
 'sympy__sympy-15308',
 'django__django-12915',
 'sympy__sympy-20590',
 'sympy__sympy-17022',
 'django__django-11099',
 'django__django-13220',
 'django__django-11964',
 'matplotlib__matplotlib-25332',
 'django__django-10914',
 'django__django-14915',
 'django__django-11049',
 'django__django-11564',
 'sympy__sympy-17655',
 'sympy__sympy-16106',
 'sympy__sympy-12171',
 'django__django-15400',
 'django__django-14411',
 'sympy__sympy-21055',
 'django__django-15213',
 'django__django-15902',
 
 'tobymao__sqlglot-4477',
 'pyodide__pyodide-1215',
 'django__django-17811',
 'tobymao__sqlglot-4497',
 'optuna__optuna-5778',
 'sagemath__sage-36814',
 'tobymao__sqlglot-4457',
 'tobymao__sqlglot-4505',
 'tobymao__sqlglot-4430',
 'raft-tech__TANF-app-3234',
'AllenNeuralDynamics__foraging-behavior-browser-106',
'Australian-Text-Analytics-Platform__atap-corpus-slicer-3',
 ]

def _dcg(target: Tensor) -> Tensor:
    batch_size, k = target.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
    return (target / torch.log2(rank_positions + 1)).sum(dim=-1)


def div_no_nan(a: Tensor, b: Tensor, na_value: Optional[float] = 0.) -> Tensor:
    return (a / b).nan_to_num_(nan=na_value, posinf=na_value, neginf=na_value)


def normalized_dcg(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]
    ideal_target = ideal_target[:, :k]
    return div_no_nan(_dcg(pred_target), _dcg(ideal_target)).mean(0)


def recall_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    relevant = (pred_target == 1).sum(dim=-1)  # 计算预测中相关文档的个数
    total_relevant = (ideal_target == 1).sum(dim=-1)  # 计算所有相关文档的个数
    recall = div_no_nan(relevant, total_relevant, na_value=0.)  # 计算 Recall@k
    return recall.mean(0)


def acc_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    print(pred_target)
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    ideal_target = ideal_target[:, :k]
    
    relevant = (pred_target == 1).sum(dim=-1)  # 计算预测中相关文档的个数
    total_relevant = (ideal_target == 1).sum(dim=-1)  # 计算所有相关文档的个数

    comparison = relevant == total_relevant
    return comparison.sum()/relevant.shape[0]


def precision_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    relevant = (pred_target == 1).sum(dim=-1)  # 计算预测中相关文档的个数
    precision = relevant / k  # 计算 Precision@k
    return precision.mean(0)


def average_precision_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    batch_size, k_val = pred_target.shape
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    ideal_target = ideal_target[:, :k]
    
    precisions = []
    for i in range(batch_size):
        ap = 0.0
        relevant_count = 0
        for j in range(k):
            if pred_target[i, j] == 1:  # 如果是相关文档
                relevant_count += 1
                ap += relevant_count / (j + 1)  # 计算 Precision@j
        # if relevant_count > 0:
        ap = ap/k
        precisions.append(ap)
    
    return torch.tensor(precisions).mean()


def load_gt_dict(gt_file, level):
    gt_datas = load_jsonl(gt_file)
    # gt_data = [data for data in gt_datas if data['instance_id']==instance_id][0]
    
    gt_dict = {}
    for gt_data in gt_datas:
        gt_locs = []
        instance_id = gt_data['instance_id']
        file_changes = gt_data['file_changes']
        for file_change in file_changes:
            if level == 'file':
                gt_locs.append(file_change['file'])
            elif level == 'module':
                changes = file_change['changes']
                if 'edited_modules' in changes:
                    gt_locs.extend(changes['edited_modules'])
            elif level == 'function':
                changes = file_change['changes']
                if 'edited_entities' in changes:
                    gt_locs.extend(changes['edited_entities'])

        gt_dict[gt_data['instance_id']] = gt_locs
    return gt_dict


def extract_file_path(changed_funcs):
    for k, v in changed_funcs.items():
        changed_files = []
        seen_files = set()
        for vv in v:
            match = re.match(r"(.+\.py)(/.*)?", vv)
            if match:
                if match.group(1) not in seen_files:
                    changed_files.append(match.group(1))
                    seen_files.add(changed_files[-1])
            else:
                import pdb;pdb.set_trace()  
        changed_funcs[k] = changed_files
    
    return changed_funcs


def convert_solutions_dict(dataset, key = 'model_patch'):
    return {elem['instance_id']: elem[key] for elem in dataset}


METRIC_FUNC = {
    'ndcg': normalized_dcg,
    'recall': recall_at_k,
    'acc': acc_at_k,
    'precision': precision_at_k,
    'map': average_precision_at_k
}
METRIC_NAME = {
    'ndcg': 'NDCG',
    'recall': 'Recall',
    'acc': 'Acc',
    'precision': 'P',
    'map': 'MAP'
}


def cal_metrics_w_file(gt_file, loc_file, key,
                level,
                k_values, # < 100
                metrics=['acc', 'ndcg', 'precision', 'recall', 'map'],
                filter_list=filtered_instances,
                selected_list=None,
                # merge_init = True,
                ):
    assert key in ['found_files', 'found_modules', 'found_entities', 'docs']
    
    max_k = max(k_values)
    # loc_output = load_jsonl(loc_file)
    gt_dict = load_gt_dict(gt_file, level)
    if key == 'docs' and level == 'file':
        pred_dict = extract_file_path(convert_solutions_dict(load_jsonl(loc_file), key='docs'))
    elif key == 'docs':
        pred_dict = convert_solutions_dict(load_jsonl(loc_file), key='docs')
        for ins in pred_dict:
            pred_funcs = pred_dict[ins]
            pred_modules = []
            for i, pl in enumerate(pred_funcs):
                fle, func_n = pl.split('.py/')
                if level == 'function':
                    if func_n.endswith('.__init__'):
                        func_n = func_n[:(len(func_n)-len('.__init__'))]
                    pred_funcs[i] = f"{fle}.py:{func_n.strip('/').replace('/', '.')}"
                elif level == 'module':
                    module_name = f"{fle}.py:{func_n.strip('/').split('/')[0]}"
                    if module_name not in pred_modules:
                        pred_modules.append(module_name)
                    pred_dict[ins] = pred_modules
    else:
        pred_dict = convert_solutions_dict(load_jsonl(loc_file), key=key)
        
    _gt_labels = []
    _pred_labels = []
    
    # for loc in loc_output:
    for instance_id in gt_dict.keys():
        # instance_id = loc['instance_id']
        if filter_list and instance_id in filter_list: continue # filter
        if selected_list and instance_id not in selected_list: continue
        if not gt_dict[instance_id]: continue
        
        if instance_id not in pred_dict:
            pred_locs = []
        else:
            pred_locs = pred_dict[instance_id][: max_k]
                
        gt_labels = [0 for _ in range(max_k)]
        pred_labels = [0 for _ in range(max_k)]

        for i in range(len(gt_dict[instance_id])):
            if i < max_k:
                gt_labels[i] = 1
        
        for i, l in enumerate(pred_locs):
            if l in gt_dict[instance_id]:
                pred_labels[i] = 1
                
        _gt_labels.append(gt_labels)
        _pred_labels.append(pred_labels)
    
    _pred_target = torch.tensor(_pred_labels)
    _ideal_target = torch.tensor(_gt_labels)
    
    result = {}
    for metric in metrics:
        assert metric in METRIC_FUNC.keys()
        
        metric_func = METRIC_FUNC[metric]
        name = METRIC_NAME[metric]
        for k in k_values:
            value = metric_func(_pred_target, _ideal_target, k=k)
            result[f'{name}@{k}'] = round(value.item(), 4)
            
    return result


def eval_w_file(gt_file, loc_file, level2key_dict, selected_list=None, k_values_list=None):
    if not k_values_list:
        k_values_list = [
            [1, 3, 5],
            [5, 10],
            [5, 10]
        ]
    file_res = cal_metrics_w_file(gt_file, loc_file, 
                            level2key_dict['file'], level='file', k_values=k_values_list[0],
                            selected_list=selected_list)
    module_res = cal_metrics_w_file(gt_file, loc_file, 
                            level2key_dict['module'], level='module', k_values=k_values_list[1],
                            selected_list=selected_list)
    function_res = cal_metrics_w_file(gt_file, loc_file, 
                            level2key_dict['function'], level='function', k_values=k_values_list[2],
                            selected_list=selected_list)

    all_df = pd.concat([pd.DataFrame(res, index=[0])
                          for res in [file_res, module_res, function_res]], 
                        axis=1, 
                        keys=['file', 'module', 'function'])
    return all_df


def load_qrels(ds_name, split, dataset_dir):
    data_dir = dataset_dir
    if split != 'test': 
        prefix = f'{ds_name}-{split}'
    else:
        prefix = ds_name
    instance_list = [i for i in os.listdir(f"{data_dir}/") if i.startswith(f"{prefix}-function_")]
    if 'loc-bench' in ds_name: 
        dataset = load_dataset("czlll/Loc-Bench_V1")[split]
        filtered_ids = [f'{prefix}-function_{i}' for i in dataset['instance_id']] 
        instance_list = [i for i in instance_list if i in filtered_ids]
        assert len(instance_list) == len(filtered_ids)
    qrels = [GenericDataLoader(
                data_folder=os.path.join(f"{data_dir}", ins_dir)
            ).load(split="test")[2] for ins_dir in instance_list]
    return qrels


def cal_metrics_w_dataset(loc_file, key,
                eval_level,
                dataset, split, 
                k_values,
                metrics,
                selected_list=None,
                qrels=None,
                reranker_results=None,
                ):
    assert key in ['found_files', 'found_modules', 'found_entities', 'docs']
    max_k = max(k_values)
    
    # load localization labels
    gt_dict = collections.defaultdict(list)
    if 'czlll' in dataset:
        bench_data = load_dataset(dataset, split=split)
        for instance in bench_data:
            if eval_level == 'file':
                for func in instance['edit_functions']:
                    fn = func.split(':')[0]
                    if fn not in gt_dict[instance['instance_id']]:
                        gt_dict[instance['instance_id']].append(fn)
            elif eval_level == 'module':
                for func in instance['edit_functions']:
                    fn = func.split(':')[0]
                    mname = func.split(':')[-1].split('.')[0]
                    mid = f'{fn}:{mname}'
                    if mid not in gt_dict[instance['instance_id']]:
                        gt_dict[instance['instance_id']].append(mid)
            elif eval_level == 'function':
                gt_dict[instance['instance_id']].extend(instance['edit_functions'])
    else:
        for qrel in qrels:
            instance_id = list(qrel.keys())[0]
            if eval_level == 'file': 
                
                for func in set(qrel[instance_id].keys()):
                    fn = func.split('.py')[0] + '.py'
                    if fn not in gt_dict[instance_id]:
                        gt_dict[instance_id].append(fn)
            elif eval_level == 'module':
                for func in set(qrel[instance_id].keys()):
                    fn = func.split('.py/')[0] + '.py'
                    mname = func.split('.py/')[-1].split('/')[0]
                    mid = f'{fn}:{mname}'
                    if mid not in gt_dict[instance_id]:
                        gt_dict[instance_id].append(mid)
            elif eval_level == 'function':
                for func in set(qrel[instance_id].keys()):
                    fle, func_n = func.split('.py/')
                    if func_n.endswith('.__init__'):
                        func_n = func_n[:(len(func_n)-len('.__init__'))]
                    fn = f"{fle}.py:{func_n.strip('/').replace('/', '.')}"
                    if fn not in gt_dict[instance_id]:
                        gt_dict[instance_id].append(fn)
        
        
        
    
    # load predicted localization results
    if key == 'docs' and eval_level == 'file':
        if reranker_results:
            pred_dict = extract_file_path(reranker_results)
        else:
            pred_dict = extract_file_path(convert_solutions_dict(load_jsonl(loc_file), key='docs'))
    elif key == 'docs':
        if reranker_results:
            pred_dict = reranker_results
        else:
            pred_dict = convert_solutions_dict(load_jsonl(loc_file), key='docs')
        for ins in pred_dict:
            pred_funcs = pred_dict[ins]
            pred_modules = []
            for i, pl in enumerate(pred_funcs):
                fle, func_n = pl.split('.py/')
                if eval_level == 'function':
                    if func_n.endswith('.__init__'):
                        func_n = func_n[:(len(func_n)-len('.__init__'))]
                    pred_funcs[i] = f"{fle}.py:{func_n.strip('/').replace('/', '.')}"
                elif eval_level == 'module':
                    module_name = f"{fle}.py:{func_n.strip('/').split('/')[0]}"
                    if module_name not in pred_modules:
                        pred_modules.append(module_name)
                    pred_dict[ins] = pred_modules
    else:
        # pred_dict = collections.defaultdict(list)
        if reranker_results:
            pred_dict = reranker_results
        else:
            pred_dict = convert_solutions_dict(load_jsonl(loc_file), key=key)
        # for ins in _pred_dict:
        #     if eval_level == 'file':
        #         for func in _pred_dict[ins]:
        #             fn = func.split(':')[0]
        #             if fn not in pred_dict[ins]:
        #                 pred_dict[ins].append(fn)
        #     elif eval_level == 'module':
        #         for func in _pred_dict[ins]:
        #             mname = func.split(':')[-1].split('.')[0]
        #             if mname not in pred_dict[ins]:
        #                 pred_dict[ins].append(mname)
        #     elif eval_level == 'function':
        #         pred_dict[ins].extend(_pred_dict[ins])
            
        
    _gt_labels = []
    _pred_labels = []
    
    # for loc in loc_output:
    for instance_id in gt_dict.keys():
        if selected_list and instance_id not in selected_list: continue
        if not gt_dict[instance_id]: continue
        
        if instance_id not in pred_dict:
            pred_locs = []
        else:
            pred_locs = pred_dict[instance_id][: max_k]
                
        gt_labels = [0 for _ in range(max_k)]
        pred_labels = [0 for _ in range(max_k)]

        for i in range(len(gt_dict[instance_id])):
            if i < max_k:
                gt_labels[i] = 1
        
        for i, l in enumerate(pred_locs):
            if l in gt_dict[instance_id]:
                pred_labels[i] = 1
                
        _gt_labels.append(gt_labels)
        _pred_labels.append(pred_labels)
    
    _pred_target = torch.tensor(_pred_labels)
    _ideal_target = torch.tensor(_gt_labels)
    
    result = {}
    for metric in metrics:
        assert metric in METRIC_FUNC.keys()
        
        metric_func = METRIC_FUNC[metric]
        name = METRIC_NAME[metric]
        for k in k_values:
            value = metric_func(_pred_target, _ideal_target, k=k)
            result[f'{name}@{k}'] = round(value.item(), 4)
            
    return result




def evaluate_results(model='coderankembed_loc', output_dir="outputs",reranker_output_dir=None, dataset_dir="datasets",
                     dataset='czlll/SWE-bench_Lite', split='test',
                     selected_list=None,
                     metrics=['acc'], 
                     k_values_list=None):
    if not k_values_list:
        if 'swe' in dataset:
            k_values_list = [
                [1, 3, 5],
                [5, 10],
                [5, 10]
            ]
        else:
            k_values_list = [
                [5, 10],
                [10, 15],
                [10, 15]
            ]
    
    ds_name = dataset
 
    if 'czlll' not in dataset:
        qrels = load_qrels(ds_name, split, dataset_dir)
    else:
        qrels = None

    prefx = f'{output_dir}/model={model}_dataset={ds_name}_split={split}_level=function_evalmode=default'
    loc_file = Path(f'{prefx}_results.json')
    
    if reranker_output_dir:
        reranker_results = {}
        reranker_output_dir = f"{reranker_output_dir}_{ds_name}"
        data_dir = dataset_dir
        if split != 'test': 
            prefix = f'{ds_name}-{split}'
        else:
            prefix = ds_name
        instance_list = [i for i in os.listdir(f"{data_dir}/") if i.startswith(f"{prefix}-function_")]
        if 'loc-bench' in ds_name: 
            loc_dataset = load_dataset("czlll/Loc-Bench_V1")[split]
            filtered_ids = [f'{prefix}-function_{i}' for i in loc_dataset['instance_id']] 
            instance_list = [i for i in instance_list if i in filtered_ids]
            assert len(instance_list) == len(filtered_ids)
        for ins_dir in instance_list:
            retrieved_results = load_json(os.path.join(reranker_output_dir, ins_dir, "rerank_100_llm_gen_num.json"))[0]
            get_sorted_documents_func(retrieved_results, reranker_results)
    else:
        reranker_results = None

    file_res = cal_metrics_w_dataset(loc_file, 'docs', 'file', dataset, split, 
                            metrics=metrics,
                            k_values=k_values_list[0],
                            selected_list=selected_list,
                            qrels=qrels,
                            reranker_results=copy.deepcopy(reranker_results))
    module_res = cal_metrics_w_dataset(loc_file, 'docs', 'module', dataset, split, 
                            metrics=metrics,
                            k_values=k_values_list[1],
                            selected_list=selected_list,
                            qrels=qrels,
                            reranker_results=copy.deepcopy(reranker_results))
    function_res = cal_metrics_w_dataset(loc_file, 'docs', 'function', dataset, split, 
                            metrics=metrics,
                            k_values=k_values_list[2],
                            selected_list=selected_list,
                            qrels=qrels,
                            reranker_results=copy.deepcopy(reranker_results))

    all_df = pd.concat([pd.DataFrame(res, index=[0])
                          for res in [file_res, module_res, function_res]], 
                        axis=1, 
                        keys=['file', 'module', 'function'])
    print(all_df)

if __name__ == '__main__':
    fire.Fire(evaluate_results)