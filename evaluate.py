import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import pandas as pd
from tqdm import tqdm

test_data_path = "./datasets/train_test/test0819.jsonl"
model_path = r"D:\Pretrained_models\Qwen\Qwen2-1___5B-Instruct"


def load_model(model_path, checkpoint_path='', device='cuda'):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # 加载LoRA权重
    if checkpoint_path:
        model = PeftModel.from_pretrained(model, checkpoint_path).to(device)

    return model, tokenizer

def safe_loads(text, default_value=None, debug = False):
    print(f"safe_loads and text is {text}") if debug else None
    json_string = re.sub(r'^```json\n(.*)\n```$', r'\1', text.strip(), flags=re.DOTALL)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"invalid json: {json_string}")
        return default_value


def predict(model, tokenizer, content, device='cuda', debug=False):
    prompt = f"下面是一段对话文本，请分析对话内容是否有诈骗风险，只以json格式输出你的判断结果(is_fraud:true/false)。\n\n{content}"

    # 返回的格式是 [batch_size, sequence_length]，其中 batch_size 是输入的样本数量，sequence_length 是输入文本的 token 数量（经过 tokenizer 处理后的文本
    inputs = tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": prompt,
        }],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True).to(device)

    default_response = {"is_fraud":False}
    gen_kwargs = {"max_new_tokens":2048, "do_sample":True, "top_k":1}

    # 将模型加载到显存
    model.to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return safe_loads(response, default_response)

def run_test(model, tokenizer, test_data, device='cuda', debug=False):
    real_labels = []
    pred_labels = []


    for i, item in tqdm(enumerate(test_data), total=len(test_data), desc="Processing", unit="item"):
        if debug:
            print(item['input'])

        dialog_input = item['input']
        real_label = item['label']

        prediction = predict(model, tokenizer, dialog_input, device=device, debug=debug)

        if debug:
            print(f"prediction is {prediction}")

        pred_label = prediction['is_fraud']

        real_labels.append(real_label)
        pred_labels.append(pred_label)

        if debug and i % (len(test_data) // 20 + 1) == 0:
            print(f"percent: {(i * 100) / len(test_data):.2f}%")

    return real_labels, pred_labels

def precision_recall(true_labels, pred_labels, labels=None, debug=False):
    cm = confusion_matrix(true_labels, pred_labels,labels=labels)
    tn, fp, fn, tp = cm.ravel()
    print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}") if debug else None

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall

def evaluate(model_path, check_point_path, dataset, device='cuda', debug=False):
    model, tokenizer = load_model(model_path, checkpoint_path=check_point_path, device=device)
    true_labels, pred_labels = run_test(model, tokenizer, dataset, device=device, debug=debug)
    precision, recall = precision_recall(true_labels, pred_labels, labels=[True, False], debug=True)
    print(f"precision:{precision}, recall:{recall}")
