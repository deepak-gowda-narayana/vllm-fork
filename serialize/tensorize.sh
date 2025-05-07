#!/bin/bash

# python tensorize.py VLLM_ARGS

python tensorize.py --model openai-community/gpt2 --gpu-memory-utilization 0.8

vllm-hpu-extension @ git+https://github.com/HabanaAI/vllm-hpu-extension.git@3e0fb39198a592566af045c6509dbca9234502c1
cherry-pick b67b45750ee1da1c3e7cd51e5992fd30d557e2db



# Install checkout to particular commit and cherry pick
https://github.com/HabanaAI/vllm-hpu-extension.git
git checkout 3e0fb39198a592566af045c6509dbca9234502c1 -b custom-with-patch
git cherry-pick f7209791be3d6ead39cfcf57b9f46515c82693da

# Go to vllm-fork  and apply our patch
# Remove vllm-hpu extension install from requiremts/hpu.txt file 
cd vllm-fork
pip install -r requirements-hpu.txt
python3 setup.py develop 

cd vllm-hpu-extension
python3 setup.py develop 

cd vllm-fork
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


python tensorize.py --model openai-community/gpt2 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 deserialize --path-to-tensors gpt2/model.tensors --keyfile gpt2/model.key 
python tensorize.py --model openai-community/gpt2 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 serialize --serialized-directory serialized


python tensorize.py --model Qwen/Qwen2.5-VL-3B-Instruct --max-num-seqs 128 --dtype float32 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 serialize --serialized-directory serialized
python tensorize.py --model Qwen/Qwen2.5-VL-3B-Instruct --max-num-seqs 128 --dtype float32 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 deserialize --path-to-tensors serialized/vllm/Qwen/Qwen2.5-VL-3B-Instruct/model.tensors --keyfile serialized/vllm/Qwen/Qwen2.5-VL-3B-Instruct/model.key 



python tensorize.py --model mistralai/Mistral-7B-v0.1 --max-num-seqs 128 --dtype float16 --gpu_memory_utilization 0.98 --num-lookahead-slots 1 serialize --serialized-directory serialized
python tensorize.py --model mistralai/Mistral-7B-v0.1 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 deserialize --path-to-tensors serialized/vllm/mistralai/Mistral-7B-v0.1/model.tensors --keyfile serialized/vllm/mistralai/Mistral-7B-v0.1/model.key 



                "--path-to-tensors", "serialized/vllm/openai-community/gpt2/model.tensors",
                "--keyfile", "serialized/vllm/openai-community/gpt2/model.key", 




vllm-hpu-extension @ git+https://github.com/HabanaAI/vllm-hpu-extension.git@f1f6624



python tensorize.py --model mistralai/Mistral-7B-v0.1 --max-num-seqs 128 --dtype bfloat16 --use-v2-block-manager --gpu_memory_utilization 0.98 --num-lookahead-slots 1 deserialize --path-to-tensors nvidia/gpt2/model.tensors --keyfile nvidia/gpt2/model.key 

mistralai/Mistral-7B-v0.1


 File "/mnt/vllm-habana/vllm/model_executor/models/gpt2.py", line 272, in forward
[rank0]:    hidden_states = self.transformer(input_ids, positions,
                                         intermediate_tensors, inputs_embeds)
File "/mnt/vllm-habana/vllm/model_executor/models/gpt2.py", line 230, in forward
[rank0]:     hidden_states = layer(hidden_states)
 File "/mnt/vllm-habana/vllm/model_executor/models/gpt2.py", line 168, in forward
[rank0]:     attn_output = self.attn(hidden_states=hidden_states)
 File "/mnt/vllm-habana/vllm/model_executor/models/gpt2.py", line 98, in forward
[rank0]:     attn_output = self.attn(q, k, v)
