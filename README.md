## A3-GCL
This paper is under review by ICASSP 2023. The code is based on [GraphCL](https://github.com/Shen-Lab/GraphCL).
### Overview
The figure below is the framework of A3-GCL. We first get graphs of different views via graph data augmentations. To maximize the agreement between positive pairs, we train a GNN encoder and a projection head via contrastive loss. We add actively adaptive augmentation to training. AGC is calculated during training, which can fill the gap between pre-training and downstream tasks.
![](https://raw.githubusercontent.com/sugarlemons/A3-GCL/main/method.png)
### Experiments
#### Datasets
* [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/)
* [MNIST](https://data.dgl.ai/dataset/benchmarking-gnns/MNIST.pkl)
* [CIFAR10](https://data.dgl.ai/dataset/benchmarking-gnns/CIFAR10.pkl)
#### How to run
For unsupervised and semi-supervised learning on different datasets, refer to READMEs respectively.
