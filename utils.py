"""
工具函数
"""
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def load_model(model_dir, device='cuda'):
    """
    加载模型和分词器
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++66666666
    :param model_dir:
    :param device:
    :return:
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device=device,
        trust_remote_code=True
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
        "content":"you are a helpful assistant."
    },
    {
        "role": "user",
        "content":prompt
    }]

    text = tokenizer.apply_chat_template(message, tokenizer=False, add_generation_prompt=True)

    print(f"input: {text}") if debug else None

    model_inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"input_ids:{model_inputs}") if debug else None

    generated_ids = model.generate(model_inputs=model_inputs, max_new_token=512)
    print(f"generated_ids:{generated_ids}") if debug else None

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]