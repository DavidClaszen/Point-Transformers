# Pytorch Implementation of Various Point Transformers

This repo is a fork from qq456cvb's [Point-Transformers](https://github.com/qq456cvb/Point-Transformers). The main change is an additional dataloader that allows to load partial ModelNet40 data, created by and used for [PAPNet](https://github.com/TerenceHsu666/PAPNet).

The original repo is a PyTorch implementation of various point cloud transformer models:
- [PCT: Point Cloud Transformer (Meng-Hao Guo et al.)](https://arxiv.org/abs/2012.09688)
- [Point Transformer (Nico Engel et al.)](https://arxiv.org/abs/2011.00931)
- [Point Transformer (Hengshuang Zhao et al.)](https://arxiv.org/abs/2012.09164)


## Changes

- To use PAPNet style data, make the necessary data available and use the new argument `use_papnet_loader=True`
- Issues mentioned [here](https://github.com/qq456cvb/Point-Transformers/issues/47) have been fixed and implemented.
- Automatic Mixed Precision was added for a good efficiency boost.


## Classification
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.org/) and save in `modelnet40_normal_resampled`.

### Run
Change which method to use in `config/cls.yaml` and run
```
python train_cls.py
```

### Results
Only Menghao has been trained and tested for this fork and the originally reported performance could not be reproduced. The model was trained on an A100 with the following hyperparameters, everything else using the defaults. Best instance average of the classification results reached 91.9.

| Hyperparameter | Setting |
|--|--|
| Batch size | 256 |
| Learning Rate | 0.001 |
| Epochs |  250 |