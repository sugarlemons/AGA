#!/bin/bash -ex

# gsimclr
for seed in 0 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=$1 python gsimclr.py --DS $2 --lr 0.01 --local --num-gc-layers 3 --aug $3 --dnodes-degree $4 --pedges-degree $5 --subgraph-degree $6 --mask-nodes-degree $7 --batch-size $8 --seed $seed
done
