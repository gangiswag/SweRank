import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from ftfy import fix_text

max_psg_num = 20
START_IDX = ord('A')
IGNORE_INDEX = -100

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return input_ids, labels, sources_tokenized["input_ids_lens"]

def degeneration_preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    negative_targets: Sequence[Sequence[str]],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing including negative examples."""
    # Process positive examples first (similar to original preprocess)
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    # Process negative examples
    negative_labels = []
    for neg_examples in negative_targets:
        # Tokenize each negative example
        neg_tokenized = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            ).input_ids[0]
            for text in neg_examples
        ]
        # Pad all negative examples to the same length
        max_neg_len = max(len(x) for x in neg_tokenized)
        padded_neg = [
            torch.nn.functional.pad(
                x, 
                (0, max_neg_len - len(x)), 
                value=tokenizer.pad_token_id
            ) for x in neg_tokenized
        ]
        # Stack negative examples for this instance
        neg_stack = torch.stack(padded_neg)
        negative_labels.append(neg_stack)

    return input_ids, labels, negative_labels, sources_tokenized["input_ids_lens"]

class RankingDataset(Dataset):
    def __init__(self, raw_data, model_tokenizer, type) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.tokenizer.padding_side="left"
        self.type = type
        self.system_message_supported = "system" in self.tokenizer.chat_template
    
    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        sys_msg = conversation[0]['value']
        input_context = conversation[1]['value']
        target_generation = conversation[2]["value"]

        if self.system_message_supported:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": input_context}
            ]
        else:
            messages = [
                {"role": "user", "content": sys_msg + "\n " + input_context}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "["
        prompt = fix_text(prompt)

        if self.type == "train":
            label_map = {}
            label_rank = 0
            for token in target_generation:
                if token.isalpha():
                    label_map[token] = label_rank
                    label_rank += 1
            
            label = [label_map[chr(c)] for c in range(START_IDX, START_IDX+len(label_map))]

        elif self.type == "eval":
            label = [self.raw_data[index]["id"]] + self.raw_data[index]["docids"] + self.raw_data[index]["scores"]
        else:
            raise Exception("Invalid run type specified for Dataset. Choose from ['train', 'eval']")
        return prompt, label
    
    def __len__(self):
        return len(self.raw_data)

class GenerationDataset(Dataset):
    def __init__(self, raw_data, model_tokenizer, combined=False, first_only=False) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.combined = combined
        self.first_only = first_only
        self.system_message_supported = "system" in self.tokenizer.chat_template
    
    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        sys_msg = conversation[0]['value']
        input_context = conversation[1]['value']
        label = conversation[2]["value"]
        if not self.first_only:
            label += self.tokenizer.eos_token
        
        if self.system_message_supported:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": input_context}
            ]
        else:
            messages = [
                {"role": "user", "content": sys_msg + "\n " + input_context}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = fix_text(prompt)
        if self.combined:
            label_map = {}
            label_rank = 0
            for token in conversation[2]["value"]:
                if token.isalpha():
                    label_map[token] = label_rank
                    label_rank += 1
            
            rank_label = [label_map[chr(c)] for c in range(START_IDX, START_IDX+len(label_map))]
            return prompt, label, rank_label
        else:
            return prompt, label
    
    def __len__(self):
        return len(self.raw_data)

class DegenerationDataset(Dataset):
    def __init__(self, raw_data, model_tokenizer, combined=False, first_only=False) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.combined = combined
        self.first_only = first_only
        self.system_message_supported = "system" in self.tokenizer.chat_template

        if self.combined:
            raise Exception("Ranking loss is not supported with degeneration. Set combined=False to use degeneration loss")
    
    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        sys_msg = conversation[0]['value']
        input_context = conversation[1]['value']
        label = conversation[2]["value"]

        negative_labels = self.raw_data[index]["negative_candidates"]

        if not self.first_only:
            label += self.tokenizer.eos_token
        
        if self.system_message_supported:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": input_context}
            ]
        else:
            messages = [
                {"role": "user", "content": sys_msg + "\n " + input_context}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = fix_text(prompt)
        
        return prompt, label, negative_labels
    
    def __len__(self):
        return len(self.raw_data)

def ranking_collate_fn(data, tokenizer):
    prompts, labels = list(zip(*data))
    tokenized_inputs = tokenizer(prompts, padding="longest", truncation=False, return_tensors="pt")
    return tokenized_inputs, labels

def generation_collate_fn(data, tokenizer):
    prompts, labels = list(zip(*data))
    tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return tokenized_inputs, labels

def combined_collate_fn(data, tokenizer):
    prompts, labels, rank_labels = list(zip(*data))
    tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return tokenized_inputs, labels, rank_labels, source_lens

def degeneration_collate_fn(data, tokenizer):
    """Collate function for degeneration training."""
    prompts, labels, negative_labels = list(zip(*data))
    
    # Process inputs and positive labels
    tokenized_inputs, labels, neg_labels, source_lens = degeneration_preprocess(
        prompts, labels, negative_labels, tokenizer
    )
    
    # Pad the main inputs and labels
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    
    # Pad negative labels to max batch size and max sequence length
    max_neg_examples = max(neg.shape[0] for neg in neg_labels)
    max_neg_length = max(neg.shape[1] for neg in neg_labels)
    
    padded_neg_labels = []
    for neg in neg_labels:
        # Pad to max number of negative examples
        current_neg_examples, current_length = neg.shape
        padding_examples = max_neg_examples - current_neg_examples
        padding_length = max_neg_length - current_length
        
        # Pad both dimensions
        padded_neg = torch.nn.functional.pad(
            neg,
            (0, padding_length, 0, padding_examples),
            value=tokenizer.pad_token_id
        )
        padded_neg_labels.append(padded_neg)
    
    # Stack all padded negative labels
    negative_labels_tensor = torch.stack(padded_neg_labels)
    
    return tokenized_inputs, labels, negative_labels_tensor
