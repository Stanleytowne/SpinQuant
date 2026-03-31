#!/bin/bash
# SpinQuant + LoRDS evaluation
#
# Usage:
#   # Quantize + eval (first run)
#   bash scripts/eval_spinquant_lords.sh <model> <w_bits> <a_bits> <kv_bits> [extra_args...]
#
#   # Save quantized model for reuse
#   bash scripts/eval_spinquant_lords.sh <model> 4 8 16 --save_qmodel_path lords_w4.pth
#
#   # Load saved model (skip re-quantization)
#   bash scripts/eval_spinquant_lords.sh <model> 4 4 4 --load_qmodel_path lords_w4.pth
#
# Example:
#   bash scripts/eval_spinquant_lords.sh /data2/mengfanxu/huggingface/Meta-Llama-3-8B 4 8 16

MODEL=$1; W=$2; A=$3; KV=$4
shift 4  # remaining args passed through (e.g. --save_qmodel_path, --load_qmodel_path)

torchrun --nnodes=1 --nproc_per_node=1 --master_port=${MASTER_PORT:-29500} ptq.py \
--input_model $MODEL \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $W \
--a_bits $A \
--k_bits $KV \
--v_bits $KV \
--w_lords \
--lords_steps 500 \
--lords_lr 1e-2 \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize 128 \
--rotate \
$@
