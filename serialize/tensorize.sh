#!/bin/bash

# python tensorize.py VLLM_ARGS

python tensorize.py --model openai-community/gpt2 --gpu-memory-utilization 0.8

pip install -r requirements-hpu.txt
python3 setup.py develop 
cd serialize
pip install -r requirements.txt
export VLLM_SKIP_WARMUP=true

# python tensorize.py --model openai-community/gpt2 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 serialize
# python tensorize.py --model openai-community/gpt2 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 deserialize

# python -m examples.other.tensorize_vllm_model  --model openai-community/gpt2 --dtype float32 serialize    --serialized-directory ./facebook-model    --suffix v1
# python -m examples.other.tensorize_vllm_model  --model openai-community/gpt2 --dtype float32 deserialize --path-to-tensors /mnt/vllm/facebook-model/vllm/openai-community/gpt2/v1/model.tensors

# python -m examples.other.tensorize_vllm_model  --model openai-community/gpt2 --dtype float16 deserialize --path-to-tensors /mnt/vllm/serialize/models/gpt2/gpt2.model


# python -m examples.other.tensorize_vllm_model  --model mistralai/Mistral-7B-v0.1 --dtype float32 serialize    --serialized-directory ./facebook-model    --suffix v1
# python -m examples.other.tensorize_vllm_model  --model mistralai/Mistral-7B-v0.1 --dtype float32 deserialize --path-to-tensors /mnt/vllm/facebook-model/vllm/mistralai/Mistral-7B-v0.1/v1/model.tensors



python tensorize.py --model openai-community/gpt2 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 deserialize --path-to-tensors serialized/vllm/openai-community/gpt2/model.tensors --keyfile serialized/vllm/openai-community/gpt2/model.key 


python tensorize.py --model openai-community/gpt2 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 deserialize --path-to-tensors nvidia/gpt2/model.tensors --keyfile nvidia/gpt2/model.key 
python tensorize.py --model openai-community/gpt2 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 serialize --serialized-directory serialized


                "--path-to-tensors", "serialized/vllm/openai-community/gpt2/model.tensors",
                "--keyfile", "serialized/vllm/openai-community/gpt2/model.key", 