import gc
import abc
import numpy as np
import math
from typing import Iterable
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import re

# For DFS
def softmax_bias(answers,temperature=1):

    sums = 0.0
    answers = [ 10**((cont/temperature)/400) for cont in answers]
    for cont in answers:
        assert type(cont) == float or type(cont) == int
        sums += cont
    answers = [ cont/sums for cont in answers]
    return np.array(answers)

def compute_epsilon_new_node(p_new_node):
    '''
    根据公式换算delta
    '''
    delta = 400 * math.log10(p_new_node /(1-p_new_node))
    return 1000 + delta

# For prediction parsing, into ReACT format
def react_parser(string):
    thought = [string[string.find("Thought: ") + len("Thought: "): string.find("\nAction: ")]]
    action = [string[string.find("Action: ") + len("Action: "): string.find("\nAction Input: ")]]
    action_input = [string[string.find("Action Input: ") + len("Action Input: "):]]
    return thought[0], action[0], action_input[0]

# For toolllama's predictions 
def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

import outlines.text.generate as generate

@torch.inference_mode()
def generate_stream(
    model, model2, tokenizer, params, device, context_len=8192, stream_interval=2, force_generate=False, regexps=None
):
    prompt = params["prompt"]
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    past_key_values = out = None

    
    if regexps is None:
        regexps = r'.*'
    c3 = generate.regex(model2, regexps, max_tokens=max_new_tokens)
    output = c3(prompt)
    output = re.sub(' +', ' ', output)
    i = len(tokenizer.tokenize(output))
    stopped = True

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()

# For IO presentation
class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""
    
    @abc.abstractmethod
    def return_output(self, output_stream):
        """Return output."""

class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)
    
    def return_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                pre = now
        return " ".join(output_text)
