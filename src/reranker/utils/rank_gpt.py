import json
import time
import random
import openai
import tiktoken
import unicodedata
from tqdm import tqdm
from enum import Enum
from ftfy import fix_text
from typing import Any, Dict, List, Optional, Tuple, Union

from .rankllm import PromptMode, RankLLM
from .result import Result

ALPH_START_IDX = ord('A') - 1

class SafeOpenai(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        rerank_type: str = "text",
        code_prompt_type: str = "docstring",
        keys=None,
        key_start_id=None,
        proxy=None,
        api_type: str = None,
        api_base: str = None,
        api_version: str = None,
    ) -> None:
        """
        Creates instance of the SafeOpenai class, a specialized version of RankLLM designed for safely handling OpenAI API calls with
        support for key cycling, proxy configuration, and Azure AI conditional integration.

        Parameters:
        - model (str): The model identifier for the LLM (model identifier information can be found via OpenAI's model lists).
        - context_size (int): The maximum number of tokens that the model can handle in a single request.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
        - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
        the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
        by default.
        - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
        - keys (Union[List[str], str], optional): A list of OpenAI API keys or a single OpenAI API key.
        - key_start_id (int, optional): The starting index for the OpenAI API key cycle.
        - proxy (str, optional): The proxy configuration for OpenAI API calls.
        - api_type (str, optional): The type of API service, if using Azure AI as the backend.
        - api_base (str, optional): The base URL for the API, applicable when using Azure AI.
        - api_version (str, optional): The API version, necessary for Azure AI integration.

        Raises:
        - ValueError: If an unsupported prompt mode is provided or if no OpenAI API keys / invalid OpenAI API keys are supplied.

        Note:
        - This class supports cycling between multiple OpenAI API keys to distribute quota usage or handle rate limiting.
        - Azure AI integration is depends on the presence of `api_type`, `api_base`, and `api_version`.
        """
        super().__init__(
            model, context_size, prompt_mode, num_few_shot_examples)
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide OpenAI Keys.")
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. Only RANK_GPT is supported."
            )
        self.system_message_supported = True
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None
        self._rerank_type = rerank_type
        self._code_prompt_type = code_prompt_type
        self._acc_cost = 0
        self._curr_cost = 0

        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        if proxy:
            openai.proxy = proxy
        openai.api_key = self._keys[self._cur_key_id]
        self.use_azure_ai = False

        if all([api_type, api_base, api_version]):
            # See https://learn.microsoft.com/en-US/azure/ai-services/openai/reference for list of supported versions
            openai.api_version = api_version
            openai.api_type = api_type
            openai.api_base = api_base
            self.use_azure_ai = True

    class CompletionMode(Enum):
        UNSPECIFIED = 0
        CHAT = 1
        TEXT = 2

    def _call_completion(
        self,
        *args,
        completion_mode: CompletionMode,
        return_text=False,
        reduce_length=False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        cost=0
        while True:
            try:
                if completion_mode == self.CompletionMode.CHAT:
                    completion = openai.chat.completions.create(
                        *args, **kwargs, timeout=30
                    )
                elif completion_mode == self.CompletionMode.TEXT:
                    completion = openai.Completion.create(*args, **kwargs)
                else:
                    raise ValueError(
                        "Unsupported completion mode: %V" % completion_mode
                    )
                cost = (self.cost_per_1k_token(input_token=True)*completion.usage.prompt_tokens + self.cost_per_1k_token(input_token=False)*completion.usage.completion_tokens)/1000
                break
            except Exception as e:
                print("Error in completion call")
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print("reduce_length")
                    return "ERROR::reduce_length"
                if "The response was filtered" in str(e):
                    print("The response was filtered")
                    return "ERROR::The response was filtered"
                self._cur_key_id = (self._cur_key_id + 1) % len(self._keys)
                openai.api_key = self._keys[self._cur_key_id]
                time.sleep(0.1)
        if return_text:
            completion = (
                completion.choices[0].message.content
                if completion_mode == self.CompletionMode.CHAT
                else completion.choices[0].text
            )
        return completion, cost

    def run_llm(
        self, prompt: str, current_window_size: Optional[int] = None, use_logits: bool = False, use_alpha: bool = False
    ) -> Tuple[str, int]:
        model_key = "model"
        if self._model == "o4-mini":
            response, cost = self._call_completion(
                messages=prompt,
                completion_mode=SafeOpenai.CompletionMode.CHAT,
                return_text=True,
                **{
                    model_key: self._model,
                    "reasoning_effort": "low"},
            )
            self._acc_cost += cost
            self._curr_cost += cost
            try:
                encoding = tiktoken.get_encoding(self._model)
            except:
                encoding = tiktoken.get_encoding("cl100k_base")

            # Update history
            self._history.append({
                "prompt": prompt,
                "response": response,
            })
            return response, len(encoding.encode(response))
        else:
            response, cost = self._call_completion(
                messages=prompt,
                temperature=0,
                completion_mode=SafeOpenai.CompletionMode.CHAT,
                return_text=True,
                **{model_key: self._model},
            )
            self._acc_cost += cost
            self._curr_cost += cost
            try:
                encoding = tiktoken.get_encoding(self._model)
            except:
                encoding = tiktoken.get_encoding("cl100k_base")

            # Update history
            self._history.append({
                "prompt": prompt,
                "response": response,
            })
            return response, len(encoding.encode(response))

    def _add_prefix_prompt(self, use_alpha, query: str, num: int) -> str:
        if self._rerank_type == "code":
            if self._code_prompt_type == "docstring":
                return self._add_prefix_prompt_doc_string(use_alpha, query, num)
            elif self._code_prompt_type == "github_issue":
                return self._add_prefix_prompt_github_issue(use_alpha, query, num)
            else:
                raise ValueError(f"Invalid code_prompt_type: {self._code_prompt_type}")
        else:  # text reranking
            if use_alpha:
                return f"I will provide you with {num} passages, each indicated by a alphabetical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
            else:
                return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, use_alpha, query: str, num: int) -> str:
        if self._rerank_type == "code":
            if self._code_prompt_type == "docstring":
                return self._add_post_prompt_doc_string(use_alpha, query, num)
            elif self._code_prompt_type == "github_issue":
                return self._add_post_prompt_github_issue(use_alpha, query, num)
            else:
                raise ValueError(f"Invalid code_prompt_type: {self._code_prompt_type}")
        else:  # text reranking
            if use_alpha:
                example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
            else:
                example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
            return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."

    def _add_prefix_prompt_doc_string(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            return f"I will provide you with {num} code functions, each indicated by an alphabetical identifier []. Rank the code functions based on their relevance to the functionality described by the following doc string: {query}.\n"
        else:
            return f"I will provide you with {num} code snippets, each indicated by a numerical identifier []. Rank the code snippets based on their relevance to the functionality described by the following doc string: {query}.\n"

    def _add_post_prompt_doc_string(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        else:
            example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"Doc String: {query}.\nRank the {num} code snippets above based on their relevance to the functionality described by the doc string. All the code snippets should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}. Only respond with the ranking results, do not say any word or explain."

    def _add_prefix_prompt_github_issue(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            prefix_prompt = f"I will provide you with {num} code functions, each indicated by a alphabetical identifier []."
            prefix_prompt += f" Rank the code functions based on their relevance to contain the faults causing the GitHub issue: {query}.\n"
        else:
            prefix_prompt = f"I will provide you with {num} code functions, each indicated by a numerical identifier []."
            prefix_prompt += f" Rank the code functions based on their relevance to contain the faults causing the GitHub issue: {query}.\n"
        return prefix_prompt

    def _add_post_prompt_github_issue(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        else:
            example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"GitHub Issue: {query}. \nRank the {num} code functions above based on their relevance to contain the faults causing the GitHub issue. All the code functions should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}. Only respond with the ranking results, do not say any word or explain."

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            try:
                encoder = tiktoken.get_encoding(self._model)
            except:
                encoder = tiktoken.get_encoding("cl100k_base")

            _output_token_estimate = (
                len(
                    encoder.encode(
                        " > ".join([f"[{i+1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate
    
    def get_total_output_tokens(self, use_alpha: bool, current_window_size: Optional[int] = None) -> int:
        """Get total number of output tokens"""
        base_tokens = self.num_output_tokens(current_window_size)
        return base_tokens

    def create_prompt_batched(self):
        pass

    def run_llm_batched(self):
        pass

    def _add_few_shot_examples_messages(self, messages):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
        return messages

    def create_prompt(
        self, result: Result, use_alpha: bool, rank_start: int, rank_end: int) -> Tuple[List[Dict[str, str]], int]:
            return self.create_rank_gpt_prompt(result, use_alpha, rank_start, rank_end)

    def create_rank_gpt_prompt(self, result: Result, use_alpha: bool, rank_start: int, rank_end: int) -> Tuple[List[Dict[str, str]], int]:
        query = result.query
        max_query_len = int(len(query)//4)+1 # Consider round down
        num = len(result.hits[rank_start:rank_end])
        max_doc_length = 1024 if self._rerank_type == "code" else 300
        min_doc_length = 300
        while True:
            messages = list()
            if self._system_message:
                messages.append({"role": "system", "content": self._system_message})
            messages = self._add_few_shot_examples_messages(messages)
            truncated_query = query[:int(max_query_len*4)]
            prefix = self._add_prefix_prompt(use_alpha, query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                if self._rerank_type == "code":
                    content = hit["content"]
                    content = content.replace("Title: Content: ", "")
                    truncated_content = content[:int(max_doc_length*4)]
                    identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)
                    input_context += f"[{identifier}] {self._replace_number(truncated_content, use_alpha)}\n"
                else:
                    content = hit["content"].replace("Title: Content: ", "").strip()
                    content = " ".join(content.split()[:max_doc_length])
                    identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)
                    input_context += f"[{identifier}] {self._replace_number(content, use_alpha)}\n"
            input_context += self._add_post_prompt(use_alpha, truncated_query, num)
            input_context = fix_text(input_context)
            messages.append({"role": "user", "content": input_context})

            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.get_total_output_tokens(rank_end - rank_start):
                break
            else:
                prefix_len = self.get_num_tokens(prefix)
                if (int(max_query_len*4) + prefix_len) > (self.max_tokens() - min_doc_length *(rank_end - rank_start) - self.get_total_output_tokens(use_alpha, rank_end - rank_start)):
                    # Query truncation to ensure min doc length for each candidate document/code
                    offset = num_tokens - (self.max_tokens() - self.get_total_output_tokens(use_alpha, rank_end - rank_start))
                    max_query_len -= (offset//2 + 1)
                else:
                    # Document truncation
                    max_doc_length -= max(
                        1,
                        (
                            num_tokens - self.max_tokens() + self.get_total_output_tokens(rank_end - rank_start)
                        ) // ((rank_end - rank_start) * 4),
                    )
        return messages, num_tokens

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """Returns the number of tokens used by a list of messages in prompt."""
        if self._model in [ "gpt-4o-mini", "gpt-4o", "gpt-4.1", "o1", "o1-mini", "o3-mini", "o4-mini" ]:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message, tokens_per_name = 0, 0

        try:
            encoding = tiktoken.get_encoding(self._model)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        if isinstance(prompt, list):
            for message in prompt:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    try: 
                        num_tokens += len(encoding.encode(value))
                    except Exception as e:
                        num_tokens += (len(value)//4)
                    if key == "name":
                        num_tokens += tokens_per_name
        else:
            num_tokens += len(encoding.encode(prompt))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    
    def cost_per_1k_token(self, input_token: bool) -> float:
        # Brought in from https://openai.com/pricing on 2023-07-30
        cost_dict = {
            "gpt-4o-mini": 0.00015 if input_token else 0.00060,
            "gpt-4o": 0.0025 if input_token else 0.01,
            "gpt-4.1": 0.002 if input_token else 0.008,
            "o1": 0.015 if input_token else 0.06,
            "o1-mini": 0.0011 if input_token else 0.0044,
            "o3-mini": 0.0011 if input_token else 0.0044,
            "o4-mini": 0.0011 if input_token else 0.0044,
        }
        return cost_dict[self._model]

    def get_name(self) -> str:
        return self._model