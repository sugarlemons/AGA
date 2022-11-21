## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)==1.6.0

Then, you need to create a directory for recoreding finetuned results to avoid errors:

```
mkdir logs
mkdir figs
```

## Training & Evaluation

For training without calculating AGC:
```
sh go.sh $GPU_ID $DATASET_NAME $AUGMENTATION $dnodes_degree $pedges_degree $subgraph_degree $mask_nodes_degree $batch_size
```
For training with calculating AGC via single augmentation:
```
sh go_single_aug.sh $GPU_ID $DATASET_NAME $AUGMENTATION $batch_size
```
For training with calculating AGC via combined augmentations:
```
sh go_two_aug.sh $GPU_ID $DATASET_NAME $AUGMENTATION $batch_size
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be four augmentations```dnodes, pedges, subgraph, mask_nodes```and six combined augmentations```dp, sd, dm, sp, pm, sm```. ```dnodes_degree```,```pedges_degree```,```subgraph_degree```and```mask_nodes_degree```are strengths for augmentations respectively.

## Acknowledgements

The backbone implementation is reference to https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised.
