"""
工具函数
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_dir, device='cuda'):
    """
    加载模型和分词器
    :param model_dir: 模型所在位置
    :param device: 训练设备
    :return: model，tokenizer
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    return model, tokenizer


def predict(model, tokenizer, prompt, device='cuda', debug=True):
    """
    文本推理
    :param model: 模型
    :param tokenizer: 分词器
    :param prompt: 提示词
    :param device: cuda or cpu
    :param debug: 是否调试
    """

    message = [{
        "role": "system",
        "content": "you are a helpful assistant."
    },
        {
            "role": "user",
            "content": prompt
        }]

    # 使用 apply_chat_template 返回文本格式的提示词
    text = tokenizer.apply_chat_template(message, tokenizer=True, add_generation_prompt=True)

    if debug: print(f"input: {text}")

    # 根据返回的类型判断如何构造 model_inputs
    if isinstance(text, list) and len(text) > 0 and isinstance(text[0], int):
        # 如果 text 是 token id 列表，直接转换为 tensor
        model_inputs = {"input_ids": torch.tensor([text]).to(device)}
    elif isinstance(text, str):
        # 如果 text 是字符串，则进行编码
        model_inputs = tokenizer([text], return_tensors="pt")
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    elif isinstance(text, list) and all(isinstance(x, str) for x in text):
        # 如果 text 是字符串列表
        model_inputs = tokenizer(text, return_tensors="pt")
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    else:
        raise ValueError("Unsupported type for text generated by apply_chat_template")

    print(f"input_ids:{model_inputs}") if debug else None

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    print(f"generated_ids:{generated_ids}") if debug else None

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs["input_ids"], generated_ids)]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = r'D:\Pretrained_models\Qwen\Qwen2-7B-Instruct'
    model, tokenizer = load_model(model_dir=model_dir, device=device)
    # for name,param in model.named_parameters():
    #     print(name, param.device)

    prompt = "请简短介绍下大语言模型"
    result = predict(model=model, tokenizer=tokenizer, prompt=prompt, device=device, debug=True)
    print(result)
