from tqdm import tqdm
from itertools import chain
import datasets
import json
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from torch.utils.data import Dataset
with open('config.json', 'r') as f:
    config_data = json.load(f)

model_id=config_data["model_config"]["model_id"]
model_token = config_data["model_config"]["token"]

class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result


def get_batched_dataset(json_data_path, chunk_size):
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=4096)
    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                dialog=sample["dialogue"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    def process_dataset(dataset, tokenizer, chunk_size, prompt):
      dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
      dataset = dataset.map(
          lambda sample: tokenizer(sample["text"]),
          batched=False,
          remove_columns=list(dataset.features)
      ).map(Concatenator(chunk_size=chunk_size), batched=True)
      return dataset

    json_dataset = datasets.load_dataset("json", data_files=json_data_path, split="train")

    prompt = (
        f"[INST]아래 대화를 한줄로 요약해주세요. \n\n### 대화:\n{{dialog}}[/INST]\n\n\n### 요약:{{summary}}"
    )



    train_data = process_dataset(json_dataset, tokenizer, chunk_size, prompt)

    return train_data