#!/bin/bash
# SpinQuant + RTN baseline evaluation
#
# Usage:
#   bash scripts/eval_spinquant_baseline.sh <model> <w_bits> <a_bits> <kv_bits> [extra_args...]
#
#   # Save/load quantized model
#   bash scripts/eval_spinquant_baseline.sh <model> 4 8 16 --save_qmodel_path rtn_w4.pth
#   bash scripts/eval_spinquant_baseline.sh <model> 4 4 4 --load_qmodel_path rtn_w4.pth

MODEL=$1; W=$2; A=$3; KV=$4
shift 4

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
--w_rtn \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize 128 \
--rotate \
$@
