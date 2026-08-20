#!/bin/bash
# CPU Fine-Tuning menggunakan llama.cpp
# Pastikan sudah build llama.cpp sebelumnya
./llama.cpp/llama-finetune \
  --model-base ./models/base-model \
  --train-data ./training/dataset/training_data.jsonl \
  --threads $(nproc) \
  --ctx 512
