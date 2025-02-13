import json

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments, \
    EarlyStoppingCallback

from datasets import Dataset
import os

def load_jsonl(path):
    """
    数据加载
    :param path: 数据路径
    :return: DataFrame格式的数据
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        return pd.DataFrame(data)


def preprocess(item, tokenizer, max_length=2048):
    """
    数据预处理
    :param item: 数据
    :param tokenizer: 分词器
    :param max_length: 最大长度
    :return: input_ids, attention_mask, labels
    """
    input_ids, attention_mask, labels = [],[],[]

    # 系统提示词
    system_message = "You are helpful assistant"
    # 用户输入提示词
    user_message = item['instruction'] + item['input']
    # 预期的回复
    assistant_message = json.dumps({'is_fraud': item['label']}, ensure_ascii=False)

    instruction = tokenizer(f'<|im_start|>system\n {system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\nassistant\n', add_special_tokens=False)
    response = tokenizer(assistant_message, add_special_tokens=False)
    input_ids = instruction['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = instruction['attention_mask'] + response['attention_mask'] + [1]
    # -100 是一个特殊标记 用于指示指令部分的token不应参与损失计算
    # 不关注输入 + 输出的input_ids = 只关注输出
    labels = [-100] * len(instruction['input_ids']) + response['input_ids'] + [tokenizer.pad_token_id]

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length]
    }

def load_dataset(train_path, eval_path, tokenizer):
    train_df = load_jsonl(train_path)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(lambda x: preprocess(x, tokenizer), remove_columns=train_ds.column_names)

    eval_df = load_jsonl(eval_path)
    eval_ds = Dataset.from_pandas(eval_df)
    eval_dataset = eval_ds.map(lambda x: preprocess(x, tokenizer), remove_columns=eval_ds.column_names)

    return train_dataset, eval_dataset

def load_model(model_path, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    return model.to(device), tokenizer

def build_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  #全部为线性层
        inference_mode=False,  #训练模型
        r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )

def build_train_arguments(output_path):
    return TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=4, # 每个设备的训练批次大小
        gradient_accumulation_steps=4, # 梯度累计的步骤数
        logging_steps=10,
        logging_first_step=True, # 是否在训练的第一步就记录日志
        logging_dir= os.path.join(output_path, "logs"),
        log_level="debug",
        log_level_replica="info", # 多卡训练时其它设备上训练进程的日志级别
        num_train_epochs=3, #训练的总轮数
        per_device_eval_batch_size=8, # 每个设备的预测批次大小
        eval_strategy="steps", # 设置评估策略为steps
        eval_on_start=False, # 在训练开始时就进行模型评估 True会报错 默认为False
        eval_steps=100, # 设置评估的步数
        save_steps=100, # 每多少步保存
        learning_rate=1e-4, # 学习率
        save_on_each_node=True, # 是否每个节点上都保存check_point
        load_best_model_at_end=True, # 训练结束时加载最佳模型
        remove_unused_columns=False, # 是否溢出数据集中模型训练未使用到的列
        dataloader_drop_last=True, # 抛弃最后一批迭代数据 因为最后一批可能不满足一批 影响训练效果
        gradient_checkpointing=True, # 启用梯度检查点节省显存
    )

def build_trainer(model, tokenizer, train_args, lora_config, train_dataset, eval_dataset):
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return Trainer(
        model=peft_model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # 早停回调
        use_cache=False
    )
