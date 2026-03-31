#!/bin/bash
# SpinQuant + LoRDS evaluation
# Usage: bash scripts/eval_spinquant_lords.sh <model_path> <w_bits> <a_bits> <kv_bits>
# Example: bash scripts/eval_spinquant_lords.sh /data2/mengfanxu/huggingface/Meta-Llama-3-8B 4 8 16
#          bash scripts/eval_spinquant_lords.sh /data2/mengfanxu/huggingface/Meta-Llama-3-8B 4 4 4

torchrun --nnodes=1 --nproc_per_node=1 --master_port=${MASTER_PORT:-29500} ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_lords \
--lords_steps 500 \
--lords_lr 1e-2 \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize 128 \
--rotate
