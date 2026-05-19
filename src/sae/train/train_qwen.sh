#!/bin/bash

declare -a arr=("5 10 15 20 25 30 32")

for i in "${arr[@]}"
do
   .venv/bin/python3 ./src/sae/train/train_sae_qwen.py --layer "$i"
done
