
import os
import json
import random
import numpy as np

import gc
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_json(file):
    lines = []
    if file.endswith(".jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                lines.append(json.loads(line))
    else:
        with open(file, "r", encoding="utf-8") as f:
            lines = json.load(f)
    return lines


def setup_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # maybe slower training


def clean_gpu(llm=None):
    if llm is not None:
        del llm.llm_engine.model_executor
        del llm
    gc.collect()
    torch.cuda.empty_cache()


def merge_save_model(model_name_or_path, peft_model_path, output_dir, model_max_length, logger):
    logger.info(f"Loading base model: {model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float16,
        device_map='cpu'
    )

    logger.info(f"Loading Peft: {peft_model_path}")
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    logger.info("Running merge_and_unload")
    model = model.merge_and_unload()

    logger.info(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        model_max_length=model_max_length
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Saving model and tokenizer in {output_dir}")
    model.save_pretrained(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")

    del model, tokenizer
    clean_gpu()

