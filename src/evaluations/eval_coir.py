import coir
from coir.evaluation import COIR
from sentence_transformers import SentenceTransformer
from beir.retrieval import models
import fire
import numpy as np
import torch
from utils import Retriever

 #ran with 512 seq length for fair comparison against baselines following CoIR paper although higher seq length may yield better pfm
def main(tasks = 'all', output_dir = 'results', batch_size = 256, max_seq_length = 512):
    st = SentenceTransformer("cornstack/CodeRankEmbed", trust_remote_code= True).to(torch.bfloat16)
    st.max_seq_length = max_seq_length
    contrast_encoder = Retriever(st, add_prefix= True)

    if tasks == 'all':
        tasks = ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt",
                                        "codefeedback-st","codetrans-contest","synthetic-text2sql",
                                        "cosqa","codesearchnet","codesearchnet-ccr"]
    else:
        tasks = tasks.split()
    
    for task in tasks:
        if task in ['apps', 'cosqa', "synthetic-text2sql"]:
            contrast_encoder.add_prefix = True 
        else:
            contrast_encoder.add_prefix = False 
        
        # Initialize evaluation
        evaluation = COIR(tasks= coir.get_tasks(tasks= [task]),batch_size=batch_size)

        # Run evaluation
        results = evaluation.run(contrast_encoder, output_folder=f"{output_dir}/coir")
        print(results)
    
if __name__ == "__main__":
    fire.Fire(main)