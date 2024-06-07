import json
import copy
import torch
import datasets
import time
import re
from transformers import LlamaForCausalLM, AutoTokenizer
from konlpy.tag import Mecab
from KoBERTScore import BERTScore
from openai import OpenAI

with open('config.json', 'r') as f:
    config_data = json.load(f)

test_config = config_data["test_config"]
model_config = config_data["model_config"]
quant_bit = model_config["load_in_bit"]
elapsed_time = []
OPENAI_API_KEY = config_data["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)



def get_test_dataset(test_size=0.1):
    json_dataset = datasets.load_dataset("json", data_files=test_config["test_file_path"], split="train")
    splitted_dataset = json_dataset.train_test_split(test_size=test_size, shuffle=False)
    _, test_dataset = splitted_dataset['train'], splitted_dataset['test']

    return test_dataset

with open(test_config["test_file_path"], "r") as test_file:
  test_json = [i for i in json.load(test_file) if len(i['summary']) > 10 ]

def get_model_output(model, tokenizer, input):
    start = time.time()
    result_strings = []
    
    for i in input:
        eval_prompt = f"[INST]아래는 고객과 상담원의 대화입니다. 아래 대화를 한줄로 요약해주세요. \n\n### 대화:{i['dialogue']}\n[/INST]\n\n\n### 요약:"
        model_input= tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        model.eval()
        with torch.no_grad():
            result_strings.append(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True)[len(eval_prompt):])
    end = time.time()
    elapsed_time.append((end - start))
    result_strings_processed = [i[:i.find("[INST]")] for i in result_strings]
    return result_strings_processed

def evaluate_model_output(output):
    f1 = {"max":0, "min":1, "mean":0}

    bertscore = BERTScore("beomi/kcbert-base", best_layer=4)
    
    references = [i["summary"] for i in test_json]
    candidates = output

    results = bertscore(references, candidates, batch_size=128)

    f1["max"] = max(results)
    f1["min"] = min(results)
    f1["mean"] = sum(results) / len(results)

    return f1

def check_if_truncated(consultation):
    if len(re.split(r'[ \n]+', consultation)) >= 4096:
        return "길이 초과"

    try:
      response = client.chat.completions.create(model="gpt-3.5-turbo",
      messages=[
                  {"role": "system", "content": "You should give a short summary for a consultation"},
          {"role": "user", "content": f"{consultation}\n 다음 문장이 중간에 끊겼으면 O, 끊겼으면 X를 print해줘.."},
      ])
      return response.choices[0].message.content
      
    except:
      return "error"



def has_coda(word):
  return (ord(word[-1]) - 44032) % 28 != 0

def fix_JKS_JX(string):
  mecab = Mecab()

  spaces = [index for index, char in enumerate(string) if char.isspace()]
  tokens = mecab.pos(string,flatten = True)

  for i in range(len(tokens) - 1):
    first, second = tokens[i], tokens[i + 1]

    if first[1].startswith(("N","XSN")):
      # 종성 있는 명사: 주격조사 "이", 보조사 "은"
      if has_coda(first[0]):
        if second[0] == "가":
          tokens[i + 1] = ("이", "JKS")
        elif second[0] == "는":
          tokens[i + 1] = ("은", "JX")

      # 종성 없는 명사: 주격조사 "가" 보조사 "는"
      if not has_coda(first[0]):
        if second[0] == "이":
          tokens[i + 1] = ("가", "JKS")
        elif second[0] == "은":
          tokens[i + 1] = ("는", "JX")
    
  fixed_string = "".join([token[0] for token in tokens])

  for index in spaces:
    fixed_string = fixed_string[:index] + ' ' + fixed_string[index:]

  return fixed_string


def postprocess_output(output):
    #조사, 보조사 교정
  processed_output = copy.deepcopy(output)
  processed_output = fix_JKS_JX(processed_output)

  return processed_output

if __name__ == "__main__":
    model_id = model_config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=4096)
    if quant_bit == 16:
        model = LlamaForCausalLM.from_pretrained(model_id,\
                                            device_map='auto',\
                                            torch_dtype=torch.float16,\
                                            cache_dir=model_config["cache_dir"])
    elif quant_bit == 8:
        model = LlamaForCausalLM.from_pretrained(model_id,\
                                            load_in_8bit=True,\
                                            device_map='auto',\
                                            cache_dir=model_config["cache_dir"])
    elif quant_bit == 4:
        model = LlamaForCausalLM.from_pretrained(model_id,\
                                            load_in_4bit=True,\
                                            device_map='auto',\
                                            cache_dir=model_config["cache_dir"])

    model_output = get_model_output(model, tokenizer, test_json)

    processed_output = postprocess_output(model_output)

    f1 = evaluate_model_output(processed_output)
