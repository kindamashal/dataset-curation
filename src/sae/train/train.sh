#!/bin/sh

declare -a arr=("5" "15" "20" "35" "40" "50")

for i in "${arr[@]}"
do
   .venv/bin/python3 ./src/sae/train/train_sae.py --layer "$i"
done