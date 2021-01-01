#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9999 \
  --nproc_per_node=4 \
  main.py \
  --pred_step 1 \
  --hyperbolic \
  --hyperbolic_version 1 \
  --distance squared \
  --network_feature resnet18 \
  --dataset finegym \
  --seq_len 5 \
  --num_seq 6 \
  --ds 3 \
  --batch_size 32 \
  --img_dim 128 \
  --epochs 200 \
  --fp16 \
  --fp64_hyper \
  --num_workers 15 \
  --lr 0.0001 \
  --prefix finetune_self_finegym_hyperbolic \
  --cross_gpu_score \
  --pretrain /path/to/pretrained/on/kinetics/checkpoint.pth.tar \
  --path_dataset /path/to/datasets/FineGym/ \
  --path_data_info /path/to/data/info