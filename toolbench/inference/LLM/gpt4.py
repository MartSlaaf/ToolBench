
from toolbench.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

#!/usr/bin/env python
# coding=utf-8
import time
from termcolor import colored
from typing import Optional, List
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from toolbench.utils import process_system_message
from toolbench.model.model_adapter import get_conversation_template
from toolbench.inference.utils import SimpleChatIO, generate_stream, react_parser

from peft import PeftModel

import os
import grazie
from grazie.api.client.chat.prompt import ChatPrompt, TextChatMessage
from grazie.api.client.endpoints import GrazieApiGatewayUrls
from grazie.api.client.gateway import GrazieApiGatewayClient, GrazieAgent
from grazie.api.client.llm_parameters import LLMParameters
from grazie.api.client.parameters import Parameters
from grazie.api.client.profiles import Profile

import json

import re

def strip_quotes(s):
    if s.startswith(("'", '"', '`')) and s.endswith(("'", '"', '`')) and s[0] == s[-1]:
        return s[1:-1]
    return s

def split_input(input_text):
    # Initialize variables to hold the different parts
    comments = ""
    function_name = ""
    parameters = {}

    # Split the input by lines
    lines = input_text.split('\n')

    # Regex patterns to match function call and parameters
    func_pattern = re.compile(r'^(.+)\(')  # Pattern to find function name
    param_pattern = re.compile(r'(\w+)\s*=\s*([^\s,\)]+)')  # Pattern to find named parameters

    for line in lines:
        # Strip leading/trailing whitespace
        stripped_line = line.strip()

        # If the line is a comment or has a comment part
        comment_index = stripped_line.find('#')
        if comment_index != -1:
            # Add the comment part to the comments string
            comments += stripped_line[comment_index:] + "\n"
            # Remove comment part from the line
            stripped_line = stripped_line[:comment_index].strip()

        # If the line is not empty after removing the comment
        if stripped_line:
            # Check for function call
            func_match = func_pattern.search(stripped_line)
            if func_match and not function_name:
                function_name = func_match.group(1)
            
            # Extract named parameters
            for param_match in param_pattern.finditer(stripped_line):
                param_name = param_match.group(1)
                param_value = strip_quotes(param_match.group(2))
                # Attempt to convert numerical values
                if param_value.replace('.', '', 1).isdigit():
                    param_value = float(param_value) if '.' in param_value else int(param_value)
                parameters[param_name] = param_value

    # Remove the last newline character from comments
    comments = comments.rstrip("\n")

    return comments, function_name, parameters

class GPT4:
    def __init__(self) -> None:
        self.profile = Profile.OPENAI_GPT_4
        self.grazie_jwt_token = os.getenv("GRAZIE_JWT_TOKEN")
        self.base_prompt = f"""You are a helpful AI assistant that operates inside the PyCharm IDE to help user to help him work on his project. However, you have no direct access to the chat with a user. Instead, your task is to select and call functions according to the user request. Any of your answers should be a function call. 
To call a function, you will use the python-like syntax. For example, to call a function named list-directory you will output solely list-directory(address='/mnt'). The system will give you the response of the environment.
You have one special function: 'Finish'. You should call them to interact with the user, to either say that you can not give the final answer, or that you have successfully done the final task.
Additional functions that are relevant for the user request will be provided after the user request.
Don't forget:
1. Any of your calls should be a call of a function. You can output python-like comments behind #, but they will not be considered as input
2. The process can have as many steps as you want. After calling a function, you can assess the result and call another function. After calling 'Give up and restart' you can start from scratch.
3. The user questions are related to the project, not the functions and APIs you see. Functions available to you are covered from user, and only available to you.
3. To give the final answer you should always use the 'Finish' function."""
        self.change_messages([])
        
    def add_message(self, message):
        content = str(message['content'])
        if message['role'] == 'system':
            self.prompt.add_system(content)
        elif message['role'] == 'user':
            self.prompt.add_user(content)
        elif message['role'] == 'assistant':
            self.prompt.add_assistant(content)
        elif message['role'] == 'function':
            self.prompt.add_system(content)
        else:
            raise(Exception(f'Unknown role: {message["role"]}'))

    def change_messages(self, messages):
        self.prompt = ChatPrompt()
        for message in messages:
            self.add_message(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    @staticmethod
    def _prepare_functions(api_list):
        new_description_list = []
        for api in api_list:
            if api['name'] == 'Finish':
                # another specific format, so it's a hardcoded stuff for now...
                required_parameters = [{'name': 'return_type', 'description': 'should be either give_answer or give_up_and_restart depending on desired end scenario.'}]
                optional_parameters = [{'name': 'final_answer', 'description': 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"'}]
            else:
                required_parameters = [{'name': param['name'], 'description': param['description']} for param in api['required_parameters']]
                optional_parameters = [{'name': param['name'], 'description': param['description'], 'default': param['default']} for param in api['optional_parameters']]
            new_description_list.append({'name': api['name'], 'description': api['description'], 'required_parameters': required_parameters, 'optional_parameters': optional_parameters})
        return new_description_list

    
    def parse(self, functions, process_id, **args):

        # hijack the first message. Somehow it's out of control of the class now, so we return this control back.
        self.prompt.messages[0] = TextChatMessage(role=self.prompt.messages[0].role, text=self.base_prompt)
        
        # add currently available functions as the last message. TODO: is this the proper place to add them? looks legit to me, as we want to modify them on the go.
        self.prompt.add_system('Available functions: ' + str(self._prepare_functions(functions['api_list'])))
        
        # print(json.dumps(self.prompt.get_messages(), indent=4))

        grazie_client = GrazieApiGatewayClient(grazie_agent=GrazieAgent("grazie-toolformers", "v1.0"),
                                               grazie_jwt_token=self.grazie_jwt_token,
                                               auth_type=grazie.api.client.gateway.AuthType.SERVICE,
                                               url=GrazieApiGatewayUrls.STAGING )

        predictions = grazie_client.chat(chat=self.prompt, 
                                         profile=self.profile, 
                                         parameters={LLMParameters.Temperature: Parameters.FloatValue(1), 
                                                     # LLMParameters.Functions: Parameters.JsonValue.from_functions(*gpt_function_definitions), 
                                                     # LLMParameters.FunctionCall: Parameters.JsonValue("auto" if use_func else "none"), 
                                                     LLMParameters.Length: Parameters.IntValue(1024*8)})
        


        # react format prediction
        thought, action, action_input = split_input(predictions.content)
        # action = action.replace('_', ' ') #TODO: discuss with Evgeniia, function names should follow some simpler pattern, I guess. Or we can do some pattern matching?
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": json.dumps(action_input)
            }
        }
        return message, 0, 10


if __name__ == "__main__":
    # can accept all huggingface LlamaModel family
    llm = ToolLLaMA("decapoda-research/llama-7b-hf")
    messages = [
        {'role': 'system', 'content': '''You are AutoGPT, you can use many tools(functions) to do
the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is , you can\'t go
back to the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\nLet\'s Begin!\nTask description: Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number=24. Each
step, you are only allowed to choose two of the left numbers to obtain a new number. For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.\nRemember:\n1.all of the number must be used , and must be used ONCE. So Only when left numbers is exact 24, you will win. So you don\'t succeed when left number = [24, 5]. You succeed when left number = [24]. \n2.all the try takes exactly 3 steps, look
at the input format'''}, 
{'role': 'user', 'content': '\nThe real task input is: [1, 2, 4, 7]\nBegin!\n'}
]
    functions = [{'name': 'play_24', 'description': '''make your current conbine with the format "x operation y = z (left: aaa) " like "1+2=3, (left: 3 5 7)", then I will tell you whether you win. This is the ONLY way
to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1''','parameters':{'type': 'object', 'properties':{}}}]#, 'parameters': {'type': 'object', 'properties': {'input': {'type': 'string', 'description': 'describe what number you want to conbine, and how to conbine.'}}, 'required': ['input']}}]

    llm.change_messages(messages)
    output = llm.parse(functions=functions)
    print(output)