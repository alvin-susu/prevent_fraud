# !/bin/sh
export CUDA_VISIBLE_DEVICES=0
export API_PORT=8001
nohup llamafactory-cli api D:/pycharm_project/prevent_fraud/qwen2_inference.yaml > D:/pycharm_project/prevent_fraud/log/anti_fraud.log 2>&1 &