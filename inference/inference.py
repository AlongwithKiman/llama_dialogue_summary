
from flask import Flask, request, jsonify
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer, default_data_collator, Trainer, TrainingArguments, TrainerCallback
import datasets
from contextlib import nullcontext
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from time import time
from flask_cors import CORS
import re



with open('config.json', 'r') as f:
    config_data = json.load(f)


model_config = config_data["model_config"]
model_id = model_config["model_id"]
pretrained = model_config["pretrained"]
quant_bit = model_config["load_in_bit"]
test_data_path = model_config["test_data_path"]
num_test_data = model_config["num_test_data"]
result_save_path = model_config["result_save_path"]
if pretrained:
    model_path = model_config["model_path"]
else:
    model_path = model_id


tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=2048)

if quant_bit == 16:
    model = LlamaForCausalLM.from_pretrained(model_path,\
                                        device_map='auto',\
                                        torch_dtype=torch.float16,\
                                        cache_dir=model_config["cache_dir"],\
                                        # token=model_config["token"]
                                        )
elif quant_bit == 8:
    model = LlamaForCausalLM.from_pretrained(model_path,\
                                        load_in_8bit=True,\
                                        device_map='auto',\
                                        torch_dtype="auto",\
                                        cache_dir=model_config["cache_dir"],\
                                        # token=model_config["token"]
                                        )
elif quant_bit == 4:
    model = LlamaForCausalLM.from_pretrained(model_path,\
                                        load_in_4bit=True,\
                                        device_map='auto',\
                                        torch_dtype="auto",\
                                        cache_dir=model_config["cache_dir"],\
                                        # token=model_config["token"]
                                        )
    
model.eval()


def parse(dialog):
    pattern = re.compile(r'요약:(.*?)\[INST\]', re.DOTALL)
    match = pattern.search(dialog)
    
    if match:
        summary = match.group(1).strip()
        return summary
    else:
        return "error"

if __name__ == "__main__":
    with open(file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_data = test_data[:num_test_data]
    result = []
    for _data in test_data:
        _dialogue = _data["dialogue"]
        prompt = f"[INST]아래 대화를 한줄로 요약해주세요. \n\n### 대화:{_dialogue}\n[/INST]\n\n\n### 요약:"
        model_input = tokenizer(prompt, return_tensors="pt").to("cuda")


        start = time()
        with torch.no_grad():
            output = model.generate(**model_input, max_new_tokens=100)[0]
            answer = tokenizer.decode(output, skip_special_tokens = True)
            answer = parse(answer)
            num_tokens = len(output)
            _data["infer"] = answer

        print(_data)
        result.append(_data)
        
    os.makedirs(output_file_path, exist_ok=True)    
    output_file_path = os.path.join(result_save_path, "infer_result.json")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=4)








