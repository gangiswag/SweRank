from swebench.harness.run_evaluation import get_predictions_from_file
import numpy as np
import json
from tabulate import tabulate
from fire import Fire
from pathlib import Path
import unidiff
import os
import csv
from beir.datasets.data_loader import GenericDataLoader
from get_repo_structure.get_patch_info import *

SWE_BENCH_MAP = {'swe-bench-lite': 'princeton-nlp/SWE-bench_Lite', 
                 'swe-bench-verified': 'princeton-nlp/SWE-bench_Verified', 
                 'swe-bench': 'princeton-nlp/SWE-bench'}

#ids where there exists at least one original function in the repo edited for swe-bench-lite test split
CHANGED_IDS = set(['sympy__sympy-15011', 'mwaskom__seaborn-3190', 'django__django-15498', 'sympy__sympy-13043', 'django__django-16229', 'sympy__sympy-24066', 'django__django-13315', 'astropy__astropy-14995', 'django__django-13158', 'sympy__sympy-20442', 'matplotlib__matplotlib-23314', 'psf__requests-2674', 'django__django-14608', 'django__django-15061', 'astropy__astropy-12907', 'pytest-dev__pytest-5413', 'django__django-15814', 'pylint-dev__pylint-7993', 'django__django-13265', 'matplotlib__matplotlib-25442', 'django__django-14787', 'django__django-15252', 'scikit-learn__scikit-learn-11281', 'django__django-13590', 'sphinx-doc__sphinx-8506', 'sympy__sympy-13773', 'pytest-dev__pytest-7490', 'django__django-12113', 'sympy__sympy-14774', 'sympy__sympy-18835', 'sympy__sympy-24213', 'pydata__xarray-4248', 'django__django-16820', 'sphinx-doc__sphinx-8595', 'django__django-15388', 'scikit-learn__scikit-learn-10508', 'django__django-16379', 'matplotlib__matplotlib-23964', 'pydata__xarray-5131', 'pytest-dev__pytest-7168', 'scikit-learn__scikit-learn-14894', 'matplotlib__matplotlib-23563', 'sphinx-doc__sphinx-10451', 'django__django-13925', 'sympy__sympy-14817', 'scikit-learn__scikit-learn-25500', 'django__django-11620', 'pallets__flask-4992', 'sympy__sympy-13437', 'sympy__sympy-20154', 'scikit-learn__scikit-learn-11040', 'sympy__sympy-15609', 'psf__requests-3362', 'psf__requests-2148', 'sympy__sympy-13647', 'pytest-dev__pytest-5103', 'sympy__sympy-23117', 'django__django-11999', 'django__django-12497', 'django__django-16816', 'django__django-11422', 'sympy__sympy-22840', 'pytest-dev__pytest-11148', 'sympy__sympy-16988', 'matplotlib__matplotlib-23562', 'pytest-dev__pytest-5495', 'sympy__sympy-14308', 'django__django-13660', 'django__django-16910', 'pylint-dev__pylint-6506', 'scikit-learn__scikit-learn-10949', 'django__django-11133', 'sympy__sympy-21379', 'sphinx-doc__sphinx-7686', 'sympy__sympy-16792', 'sympy__sympy-20322', 'pytest-dev__pytest-7432', 'django__django-16139', 'matplotlib__matplotlib-22711', 'astropy__astropy-7746', 'sympy__sympy-15678', 'scikit-learn__scikit-learn-13241', 'pydata__xarray-4094', 'django__django-15202', 'django__django-12453', 'django__django-11001', 'astropy__astropy-14365', 'django__django-13321', 'sympy__sympy-12454', 'scikit-learn__scikit-learn-12471', 'sympy__sympy-18057', 'sympy__sympy-12419', 'sympy__sympy-16503', 'sympy__sympy-13031', 'django__django-12708', 'sympy__sympy-22714', 'pallets__flask-4045', 'django__django-10924', 'sphinx-doc__sphinx-10325', 'scikit-learn__scikit-learn-14983', 'scikit-learn__scikit-learn-13584', 'matplotlib__matplotlib-23299', 'django__django-11039', 'django__django-12908', 'scikit-learn__scikit-learn-13496', 'django__django-15819', 'django__django-14016', 'django__django-15738', 'django__django-11905', 'django__django-11019', 'sphinx-doc__sphinx-7738', 'sphinx-doc__sphinx-11445', 'django__django-14534', 'scikit-learn__scikit-learn-25747', 'sympy__sympy-18698', 'sympy__sympy-13471', 'django__django-12308', 'django__django-14997', 'sympy__sympy-21171', 'django__django-13710', 'django__django-11848', 'matplotlib__matplotlib-25079', 'matplotlib__matplotlib-24334', 'sympy__sympy-13146', 'django__django-16527', 'django__django-14730', 'sympy__sympy-18532', 'django__django-15789', 'django__django-14238', 'pylint-dev__pylint-7114', 'django__django-16595', 'mwaskom__seaborn-2848', 'scikit-learn__scikit-learn-13142', 'django__django-12589', 'sphinx-doc__sphinx-8721', 'scikit-learn__scikit-learn-25570', 'django__django-14752', 'django__django-12700', 'pylint-dev__pylint-7080', 'scikit-learn__scikit-learn-25638', 'pallets__flask-5063', 'django__django-11179', 'django__django-12983', 'django__django-15347', 'scikit-learn__scikit-learn-13497', 'sympy__sympy-21614', 'sympy__sympy-20212', 'sphinx-doc__sphinx-8435', 'django__django-15851', 'django__django-11583', 'sympy__sympy-12236', 'django__django-11910', 'matplotlib__matplotlib-23913', 'sympy__sympy-15345', 'scikit-learn__scikit-learn-15512', 'django__django-13964', 'sympy__sympy-11897', 'sympy__sympy-18189', 'sympy__sympy-19007', 'sympy__sympy-18087', 'sympy__sympy-13177', 'django__django-12125', 'matplotlib__matplotlib-26011', 'django__django-16046', 'django__django-11815', 'django__django-12747', 'django__django-16400', 'django__django-15695', 'pytest-dev__pytest-8906', 'django__django-15996', 'sphinx-doc__sphinx-8273', 'sympy__sympy-22005', 'django__django-17051', 'django__django-16873', 'sphinx-doc__sphinx-8474', 'django__django-12470', 'django__django-12284', 'pytest-dev__pytest-5221', 'django__django-11630', 'pytest-dev__pytest-5692', 'matplotlib__matplotlib-24265', 'pytest-dev__pytest-11143', 'matplotlib__matplotlib-26020', 'django__django-11797', 'django__django-17087', 'django__django-12856', 'django__django-13033', 'django__django-13658', 'django__django-12184', 'django__django-16408', 'django__django-14382', 'django__django-11742', 'mwaskom__seaborn-3407', 'sympy__sympy-14396', 'scikit-learn__scikit-learn-13779', 'pydata__xarray-3364', 'matplotlib__matplotlib-24149', 'django__django-13448', 'sympy__sympy-14317', 'sympy__sympy-12481', 'scikit-learn__scikit-learn-14087', 'sympy__sympy-24152', 'django__django-14667', 'matplotlib__matplotlib-25498', 'django__django-13757', 'psf__requests-1963', 'django__django-13028', 'pytest-dev__pytest-6116', 'sympy__sympy-19254', 'sympy__sympy-13971', 'scikit-learn__scikit-learn-15535', 'sympy__sympy-21627', 'sympy__sympy-21612', 'sympy__sympy-17139', 'matplotlib__matplotlib-24970', 'scikit-learn__scikit-learn-10297', 'django__django-15320', 'django__django-15781', 'django__django-16041', 'pylint-dev__pylint-7228', 'django__django-13447', 'django__django-13933', 'sympy__sympy-15346', 'sympy__sympy-24909', 'matplotlib__matplotlib-23476', 'django__django-13401', 'sympy__sympy-18621', 'sympy__sympy-23191', 'sympy__sympy-20639', 'django__django-14017', 'pytest-dev__pytest-7373', 'sphinx-doc__sphinx-8282', 'pydata__xarray-4493', 'sympy__sympy-13915', 'matplotlib__matplotlib-18869', 'django__django-13551', 'matplotlib__matplotlib-22835', 'sympy__sympy-13480', 'django__django-14155', 'django__django-14672', 'pylint-dev__pylint-5859', 'matplotlib__matplotlib-23987', 'sympy__sympy-11870', 'django__django-15790', 'mwaskom__seaborn-3010', 'sympy__sympy-18199', 'django__django-11283', 'sphinx-doc__sphinx-8801', 'matplotlib__matplotlib-25433', 'pytest-dev__pytest-8365', 'pytest-dev__pytest-7220', 'django__django-14855', 'django__django-12286', 'sphinx-doc__sphinx-8713', 'psf__requests-2317', 'astropy__astropy-14182', 'django__django-13768', 'django__django-13230', 'sympy__sympy-24102', 'sympy__sympy-21847', 'sympy__sympy-14024', 'psf__requests-863', 'scikit-learn__scikit-learn-14092', 'django__django-14999', 'sympy__sympy-13895', 'astropy__astropy-6938', 'sphinx-doc__sphinx-7975', 'sympy__sympy-16281', 'django__django-16255', 'sympy__sympy-23262', 'sympy__sympy-20049', 'sphinx-doc__sphinx-8627', 'django__django-14580', 'matplotlib__matplotlib-25311'])


def topk_accuracy(predictions_dict, label_dict, k = 5, level = 'file'):
    topk = np.zeros(k)
    tot = 0
    for key in label_dict.keys():
        if key in predictions_dict and key in label_dict and key in CHANGED_IDS:
            predictions, label = predictions_dict[key], label_dict[key]
            for i in range(min(len(predictions), k)):
                    if label.issubset(set(predictions[:i + 1])):
                        topk[i:] += 1.0
                        break 
        
            tot += 1
    assert tot == 274
    topk = topk / tot * 100
    i_lst = [0, 1, 2] if level == 'file' else [0,1,2,4,9]
    return tabulate([[f'top-{i + 1}', round(acc, 1)] for i, acc in enumerate(topk) if i in i_lst], 
                    headers=['top-k', 'accuracy (%)'], tablefmt="grid")



            
def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def convert_solutions_dict(dataset, key = 'model_patch'):
    return {elem['instance_id']: elem[key] for elem in dataset}

def tabulate_dict(dict, headers = ['metric', 'value']):
    return tabulate([[k, v] for k, v in dict.items()], 
                    headers=headers, tablefmt="grid")


def parse_agentless_funcs(agentless_results):
    parsed_res = {}
    for i in range(len(agentless_results)):
        data = agentless_results[i]
        file_res = []
        for j in range(len(data['found_related_locs'])):
            file_name = data['found_files'][j]
            for retrieved in data['found_related_locs'][j]:
                for ret in retrieved.split('\n'):
                    if ret.startswith('function: '):
                        file_res.append(f"{file_name}/{ret.split('function: ')[-1].strip().replace('.', '/')}")
        parsed_res[data['instance_id']] = file_res            
    return parsed_res

def file_localization_results(agentless_path = None,
                              crag_bench_path_results = None, ds_name = "swe-bench-lite", split = 'test', k = 5):
    solutions =  convert_solutions_dict(get_predictions_from_file("gold", SWE_BENCH_MAP[ds_name], split))

    gold_files_changed = {key: 
        {patch_file.source_file.split("a/", 1)[-1]
        for patch_file in unidiff.PatchSet(v)} for key, v in solutions.items()
        }
    
    if agentless_path is not None:
        data = read_jsonl(agentless_path)
        localized_files = convert_solutions_dict(data, key = 'found_files')
    else:
        localized_files = crag_bench_path_results
        
    print(f'Top-{k} accuracy for File Localization: ')
    print(topk_accuracy(localized_files, gold_files_changed, k))

def parse_agentless_repair(data):
    localized_functions = {}
    for dat in data:
        patch_info = patch_to_dict(dat['model_patch'])
        changed_funcs = set()
        for fle, hunks in patch_info.items():
            for hunk in hunks:
                if hunk['function_changed'] and hunk['newly_added'] is False:
                    if hunk['class_changed']:
                        changed_funcs.add(f'{fle}/{hunk["class_changed"]}/{hunk["function_changed"]}')
                    else:
                        changed_funcs.add(f'{fle}/{hunk["function_changed"]}')
        
        localized_functions[dat['instance_id']] = changed_funcs
    return localized_functions
        

def function_localization_results(agentless_path = None,
                              crag_bench_path_results = None, ds_name = "swe-bench-lite", split = 'test', k = 5, changed_functions = None,  reranker_results = {}):
    def load_json(file_path, read_mode='r'):
        data = []
        with open(file_path, read_mode, encoding="utf-8") as f:
            json_data = re.sub(r"}\s*{", "},{", f.read())
            data.extend(json.loads("["+json_data+"]"))
        f.close()
        return data
    if agentless_path is not None:
            data = read_jsonl(agentless_path)
            localized_functions = parse_agentless_funcs(data)
    elif reranker_results != {}:
        localized_functions = reranker_results
    else:
        localized_functions = convert_solutions_dict(load_json(crag_bench_path_results), key = 'docs')

    print(f'Top-{k} accuracy for Function Localization: ')
    print(topk_accuracy(localized_functions, changed_functions, k, level = 'function'))
    
    

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


def load_beir_results_from_tsv(input_file):
    results = defaultdict(dict)
    with open(input_file, 'r', newline='') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        next(tsvreader) 

        for row in tsvreader:
            query_id, doc_id, score = row[0], row[1], float(row[2])
            results[query_id][doc_id] = score

    return results

def get_sorted_documents(results, sorted_documents_per_query):

    for query_id, doc_scores in results.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_doc_ids = [doc_id for doc_id, score in sorted_docs]
        
        res = []
        seen = set()
        for doc_id in sorted_doc_ids:
            match = re.match(r"(.+\.py)(/.*)?", doc_id)
            if match.group(1) not in seen:
                res.append(match.group(1))
                seen.add(match.group(1))
            
        
        sorted_documents_per_query[query_id] = res

    return sorted_documents_per_query

def get_sorted_documents_func(results, sorted_documents_per_query):

    for query_id, doc_scores in results.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_documents_per_query[query_id] = [doc_id for doc_id, score in sorted_docs]

    return sorted_documents_per_query
            

def main(level = 'file', mode = 'agentless', model = 'CodeRankEmbed', ds_name = 'swe-bench-lite', split = 'test', k = 10, agentless_path = '', retriever_output_dir = 'results', dataset_output_dir = 'datasets'):
    if ds_name != 'swe-bench-lite' and split != 'test':
        raise NotImplementedError('only evaluation on the swe-bench-lite test split is supported now!!!')
    
    agentless_path = None
    crag_bench_path_results = None
    beir_res = None

    if not mode.startswith('agentless'):
        if level == 'file':
            level = 'function'
            actually_file = True 
        else:
            actually_file = False
            
    if mode == 'agentless':
        agentless_path = Path(agentless_path)
    
    elif mode == 'code-retriever':
        prefx = f'{retriever_output_dir}/model={model}_dataset={ds_name}_split={split}_level={level}_eval_mode=default'
        crag_bench_path_results = Path(f'{prefx}_results.json')
        try:
            beir_res = read_jsonl(f'{prefx}_output.json')
        except:
            beir_res = None
    
    if not ((mode.startswith('agentless') and level == 'file')) :
        if split == 'test':
            prefx = f"{ds_name}" 
        else: 
           prefx = f"{ds_name}-{split}"
        
        if level == 'file':
            prefx += '_'
        else:
            prefx += f'-{level}_'
        
        instance_list = [i for i in os.listdir(dataset_output_dir) if i.startswith(prefx)]

        reranker_results = {}
        if mode == 'reranker':
            for ins_dir in instance_list:
                retrieved_results = load_beir_results_from_tsv(f'swe-bench-rerank-tsv-folder/csn_{ins_dir.split("swe-bench-lite-")[1]}.tsv')
                get_sorted_documents_func(retrieved_results, reranker_results)


        if (mode.startswith('agentless') and level == 'function') or not actually_file:
            qrels = [GenericDataLoader(
                data_folder=os.path.join(dataset_output_dir, ins_dir)
            ).load(split="test")[2] for ins_dir in instance_list]
            
            changed_functions = {list(qrel.keys())[0] : set(qrel[list(qrel.keys())[0]].keys()) for qrel in qrels}
            if beir_res is not None:
                print(beir_res)
            
            function_localization_results(agentless_path, crag_bench_path_results, 
                                    ds_name= ds_name, split= split, k = k, changed_functions= changed_functions, reranker_results = reranker_results)
        
        elif mode == 'reranker':
            pass
        
        else:
            crag_bench_path_results = extract_file_path(convert_solutions_dict(read_jsonl(crag_bench_path_results), key = 'docs'))
            level = 'file'

    if level == 'file':
        if mode == 'reranker':
            raise NotImplementedError('Re-ranker localization evaluation to be added in')


        file_localization_results(agentless_path, crag_bench_path_results, 
                                ds_name= ds_name, split= split, k = k)



if __name__ == "__main__":
    Fire(main)