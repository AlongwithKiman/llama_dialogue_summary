import os
import json
import torch
from transformers import LlamaForCausalLM, AutoModelForCausalLM, LlamaTokenizer,AutoTokenizer, default_data_collator, Trainer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
import datasets
from dataloader import get_batched_dataset
from contextlib import nullcontext
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

with open('config.json', 'r') as f:
    config_data = json.load(f)

peft_config = config_data["peft_config"]
dataset_config = config_data["dataset_config"]
train_config = config_data["train_config"]  
model_config = config_data["model_config"]
quant_bit = model_config["load_in_bit"]

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

def create_peft_config(model):
    get_peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=peft_config["r"],
        lora_alpha=peft_config["lora_alpha"],
        lora_dropout=peft_config["lora_dropout"],
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model, False)
    
    model = get_peft_model(model, get_peft_config)
    return model, get_peft_config    



if __name__ == "__main__":
    model_id=model_config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=4096, token = model_config["token"])
    print(model_config["cache_dir"])
    print(os.getcwd())
    if quant_bit == 16:
        model = AutoModelForCausalLM.from_pretrained(model_id,\
                                            device_map=device,\
                                            torch_dtype=torch.float16,\
                                            cache_dir=model_config["cache_dir"],\
                                            token=model_config["token"]
        )
    elif quant_bit == 8:
        model = AutoModelForCausalLM.from_pretrained(model_id,\
                                            load_in_8bit=True,\
                                            device_map=device,\
                                            torch_dtype=torch.float16,\
                                            cache_dir=model_config["cache_dir"],\
                                            token=model_config["token"]
        )
    elif quant_bit == 4:
        model = AutoModelForCausalLM.from_pretrained(model_id,\
                                            load_in_4bit=True,\
                                            device_map=device,\
                                            torch_dtype=torch.float16,\
                                            cache_dir=model_config["cache_dir"],\
                                            token=model_config["token"]
        )

    print("Model Loaded")

    train = get_batched_dataset(dataset_config["dataset_path"], chunk_size=dataset_config["chunk_size"])
    print("dataset loaded")
    model.train()



    # create peft config
    model, lora_config = create_peft_config(model)
    print("peft config applied")
    enable_profiler = False
    output_dir = train_config["output_dir"]

    config = {
        'lora_config': lora_config,
        'learning_rate': train_config["learning_rate"],
        'num_train_epochs': train_config["num_train_epochs"],
        'gradient_accumulation_steps': train_config["gradient_accumulation_steps"],
        'per_device_train_batch_size': train_config["per_device_train_batch_size"],
        'gradient_checkpointing': train_config["gradient_checkpointing"],
    }

    # Set up profiler
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)

        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler

            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()



    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch_fused",
        max_steps=total_steps if enable_profiler else -1,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    )

    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            data_collator=default_data_collator,
            callbacks=[profiler_callback] if enable_profiler else [],
        )

    print("starting train")
    # Start training
    trainer.train()

    if os.path.exists(train_config["model_ckpt_save_dir"]) is False:
        os.mkdir(train_config["model_ckpt_save_dir"])
        
    model.save_pretrained(train_config["model_ckpt_save_dir"], safe_serialization = False)