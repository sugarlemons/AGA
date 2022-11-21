## Dependencies

### Option 1: ###
Please setup the environment following Requirements in this [repository](https://github.com/chentingpc/gfn#requirements).
Typically, you might need to run the following commands:
```
pip install torch==1.4.0
pip install torch-scatter==1.1.0 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==0.4.4 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==1.1.0 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric==1.1.0
pip install torch-vision==0.5.0
```

### Option 2: ###
You also can create a conda environment with the command:
```
conda env create -f environment.yml
conda activate a3gcl
```

Then, you need to create two directories for pre-trained models and finetuned results to avoid errors:

```
cd ./pre-training
mkdir models
mkdir logs
cd ..
cd ./funetuning
mkdir logs
cd ..
```

## Pre-training & finetuning

Take NCI1 as an example:

### Pre-training: ###

```
cd ./pre-training
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 0
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 1
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 2
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 3
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 4
```

### Finetuning: ###

```
cd ./funetuning
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 4
```

Five suffixes stand for five runs (with mean & std reported).

```lr``` in pre-training should be tuned from {0.01, 0.001, 0.0001} and ```model_epoch``` in finetuning (this means the epoch checkpoint loaded from pre-trained model) from {20, 40, 60, 80, 100}.

## Acknowledgements

The backbone implementation is reference to https://github.com/chentingpc/gfn.
