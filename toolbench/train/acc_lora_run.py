
from toolbench.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from peft import prepare_model_for_kbit_training
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from accelerate import Accelerator

dummy_accelerator = Accelerator()
current_device = dummy_accelerator.process_index


from datasets import load_dataset
from peft import LoraConfig, get_peft_model

import transformers

import pandas as pd
import yaml
import numpy as np
from datasets import concatenate_datasets, Dataset
import re
import argparse # move to hydra later, I guess...

from train import make_supervised_data_module

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from omegaconf import DictConfig, OmegaConf
import hydra
from metadict import MetaDict

@hydra.main(version_base=None, config_path=".", config_name="tuning_config")
def train(cfg):
    cfg = MetaDict(OmegaConf.to_object(cfg))
    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        model_max_length=cfg.model.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    # prepare datasets
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=cfg.data_args)

    # prepare model
    bnb_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16
                                )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        # load_in_8bit=True,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={'':current_device}
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg.lora_args.lora_r, 
        lora_alpha=cfg.lora_args.lora_alpha, 
        target_modules=cfg.lora_args.lora_target_modules,
        lora_dropout=cfg.lora_args.lora_dropout, 
        bias=cfg.lora_args.lora_bias, 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # --logging_steps 1 \
    # --gradient_checkpointing True \

    training_args = transformers.TrainingArguments(
        output_dir=cfg.training.save_location, 
        learning_rate=5e-5,
        num_train_epochs=cfg.training.num_train_epochs, 
        warmup_ratio=cfg.training.warmup_ratio, 
        save_strategy='epoch', 
        evaluation_strategy='epoch',
        # save_strategy='steps', 
        # evaluation_strategy='steps',
        # eval_steps=1,
        # save_steps=1,
        weight_decay=cfg.training.weight_decay,
        # auto_find_batch_size=True,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        eval_accumulation_steps=1,
        fp16=True,
        # save_total_limit=4,
        logging_steps=25,
        # optim="paged_adamw_8bit",
        lr_scheduler_type = cfg.training.lr_scheduler_type,
        ddp_find_unused_parameters=False,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        **data_module
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint,)






if __name__ == "__main__":

    train()


    # # Verifying the datatypes.
    # dtypes = {}
    # for _, p in model.named_parameters():
    #     dtype = p.dtype
    #     if dtype not in dtypes:
    #         dtypes[dtype] = 0
    #     dtypes[dtype] += p.numel()
    # total = 0
    # for k, v in dtypes.items():
    #     total += v
    # for k, v in dtypes.items():
    #     print(k, v, v / total)

    

    # training_args = transformers.TrainingArguments(
    #     # auto_find_batch_size=True,
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=32,
    #     num_train_epochs=1,
    #     learning_rate=2e-4,
    #     fp16=True,
    #     save_total_limit=4,
    #     logging_steps=25,
    #     output_dir="./outputs",
    #     save_strategy='epoch',
    #     optim="paged_adamw_8bit",
    #     lr_scheduler_type = 'cosine',
    #     warmup_ratio = 0.05,
    #     ddp_find_unused_parameters=False,
    # )

    # trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=data["train"],
    #     args=training_args,
    #     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # )
    # model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # trainer.train()

    