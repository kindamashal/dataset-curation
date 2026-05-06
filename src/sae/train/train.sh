#!/bin/bash

declare -a arr=("10")

for i in "${arr[@]}"
do
   .venv/bin/python3 ./src/sae/train/train_sae.py --layer "$i"
done
