import json
import threading
import subprocess
import os
import json

def run_command(command):
    with open(os.devnull, 'w') as fp:
        process = subprocess.Popen(command, shell=True, stdout=fp, stderr=fp)
    process.wait()

class TaskManager:
    def __init__(self, command):
        self.command = command
        self.process = None

    def run(self):
        with open(os.devnull, 'w') as fp:
            self.process = subprocess.Popen(self.command, shell=True, stdout=fp, stderr=fp)
        self.process.wait()

    def terminate(self):
        if self.process:
            self.process.terminate()

# def run_python_script():
#     print('***'*30)
#     subprocess.run(['export', 'PYTHONPATH=./', 'CUDA_VISIBLE_DEVICES=0,1', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python', 'python', 'toolbench/inference/qa_pipeline.py', '--tool_root_dir', 'data/toolenv/tools/',
#     '--backbone_model', 'toolllama',
#     '--model_path codellama/CodeLlama-7b-Instruct-hf', '--lora', '--lora_path', '/mnt/data/mart/toolcodellama_lora_constant/checkpoint-843',
#     '--max_observation_length', '1024',
#     '--observ_compress_method', 'truncate',
#     '--method', 'DFS_woFilter_w2',
#     '--input_query_file', 'data/instruction/inference_query_demo.json',
#     '--output_answer_file', 'data/answer/toolllama_lora_dfs',
#     '--toolbench_key', '$TOOLBENCH_KEY', 
#     '--get_local_api'], shell=True)
#     # print('***'*30)
#     # for line in process.stdout:
#     #     print('***'*30)
#     #     print(line.decode().strip())
#     # print('<<<'*30)

from omegaconf import DictConfig, OmegaConf
import hydra
import os

def manage_tasks(name, git, questionnaire, model_addr, results_prefix, lora_addr=None, model_class='toolllama'):
    # Clone the repository
    subprocess.run(f"mkdir repos; cd repos; git clone {git} {name}", shell=True)

    # Start the continuous task in a separate thread
    # continuous_task_thread = threading.Thread(target=run_command, args=(f'~/projects/ideformer-plugin/ide-former-plugin/runPluginStarter.sh ~/projects/repos/{name} 127.0.0.1 5000',))
    # continuous_task_thread.start()

    task_manager = TaskManager(f'sh ../ideformer-plugin/ide-former-plugin/runPluginStarter.sh "$PWD/repos/{name}" 127.0.0.1 5000')
    continuous_task_thread = threading.Thread(target=task_manager.run)
    continuous_task_thread.start()

    os.makedirs('test_inputs', exist_ok=True)

    with open(f'test_inputs/test_input_{name}.json', 'w') as f:
        json.dump([{'query': q['question'], 'query_id': i, 'api_list':[]} for i, q in enumerate(questionnaire)], f)
    
    os.makedirs(f'{results_prefix}_results/{name}_outputs', exist_ok=True)

    starter_command = f'''export PYTHONPATH=./
    CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model {model_class} --model_path {model_addr}{' --lora --lora_path' if lora_addr is not None else ''} {lora_addr if lora_addr is not None else ''} \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file  test_inputs/test_input_{name}.json\
    --output_answer_file {results_prefix}_results/{name}_outputs \
    --toolbench_key $TOOLBENCH_KEY \
    --get_local_api'''
    
    print(starter_command)

    subprocess.run(starter_command, shell=True) # practically, why do I call it as a tool, when I can call it as a function?
    
    task_manager.terminate()
    continuous_task_thread.join()

from datetime import datetime

@hydra.main(version_base=None, config_path=".", config_name="test_config")
def main(cfg):
    model_addr = cfg.get('model_addr')
    lora_addr = cfg.get('lora_addr', None)
    model_class = cfg.get('model_class', None)
    results_prefix = cfg.get('results_prefix', datetime.now().strftime('%d.%m.%Y.%H.%M.%S'))
    task_addr = cfg.get('task_addr', 'tasks.json')

    with open(task_addr, 'r') as file:
        tasks = json.load(file)

    # global_answers = {}
    
    for task in tasks:
        # name, results = 
        manage_tasks(**task, model_addr=model_addr, lora_addr=lora_addr, results_prefix=results_prefix, model_class=model_class)
        # global_answers[name] = results
    
    # with open('global_report.json', 'w') as f:
    #     json.dump(global_answers, f)



if __name__ == '__main__':
    main()