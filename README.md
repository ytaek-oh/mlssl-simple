
# MLSSL-SIMPLE

This repo implements common semi-supervised learning algorithms under multi-label learning setup.
Currently, supporting algorithms and benchmark datasets are as following:
- Algorithms: [PseudoLabel](https://www.kaggle.com/blobs/download/forum-message-attachment-files/746/pseudo_label_final.pdf), [MeanTeacher](https://arxiv.org/abs/1703.01780) 
- Benchmarks: [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)


## Installation

The code is tested with:
- CUDA 10.2
- Python 3.6
- PyTorch 1.8.1
- Titan Xp GPU (Mem: 12GB) 


To install requirements:

### (Option 1) Install with conda

```setup
#  installing environments via conda
conda create -n mlssl python=3.6 -y
conda activate mlssl

# installing required library
pip install sklearn
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install future tensorboard

git clone https://github.com/ytaek-oh/mlssl-simple.git && cd mlssl-simple
```

### (Option 2) Install with Docker
```setup
# clone the repository
git clone https://github.com/ytaek-oh/mlssl-simple.git && cd mlssl-simple

# build environment from Dockfile
docker build -t mlssl:v0 docker/
docker run --gpus all -it --shm-size=24gb \
    -v {PATH_TO_REPOSITORY}/mlssl-simple:/home/appuser/mlssl-simple \
    -v {PATH_TO_DATASET}/datasets:/home/appuser/mlssl-simple/datasets \
    --name mlssl mlssl:v0
```

## Data Preparation
By default, `./datasets` will be the root directory of datasets. 
For Pascal VOC 2007 as an example, it will be automatically downloaded when running the code firstly, or you can manually download the data in structure like:
```data
{DATA_ROOT}/  # e.g) ./datasets/
  VOCdevkit/
    VOC2007/
      Annotations/
      ImageSets/
      JEPGImages/
      ...
```


## Training and test
You may refer to all of the pre-defined configurations from `defaults.py`.


To train the model(s), run the command as:

```train
# by default, train ResNet-50 with 10% of VOC labels
CUDA_VISIBLE_DEVICES=0,1 python train_pseudo_label.py

# train with 20% labels
CUDA_VISIBLE_DEVICES=0,1 python train_pseudo_label.py --percent-labels 20
```

- As note, two GPUs with 12GB memory are used for training PseudoLabel and MeanTeacher on ResNet-50 model. You may change the batch size to properly fit on your GPU setup. 


To test a model from the existing checkpoints, run the command as:
```test
python train_pseudo_label.py --eval-only True --weights {PATH_TO_CHECKPOINT}
```


## Reference

| Method | VOC (10% Labels) | VOC (20% Labels) |
|--|--|--|
| Supervised*  | 18.36 +- 0.65 | 28.84 +- 1.68 |
| PseudoLabel* | 27.44 +- 0.55 | 34.84 +- 1.88 |
| MeanTeacher* | 32.55 +- 1.48 | 39.62 +- 1.66 |
| Supervised  | 19.6 | 27.3 |
| PseudoLabel | 24.7 | 30.0 |
| MeanTeacher | 25.5 | 34.1 |

*: Performances reported in [UPS paper](https://arxiv.org/abs/2101.06329).


### Current issue: reproducing mAP performance of baseline SSL methods reported in the UPS paper above.  

- For our results, we report the mAP score from the final epoch, all of which are the best scores that we ever obtained through adjusting several parameters.
  - For SSL algorithms, serveral default parameters are as following. 
    - base_lr: 0.4, num_epochs: 270, momentum: 0.9, weight_decay: 1e-4, cosine learning rate scheduler with decay ratio (`--cos-ratio`).

- From our observation, the `--cos-ratio`, which is the parameter for the cosine learning scheduler seems to greatly affect the final mAP score on test set. (default value is 7, and this can be set as {POWER OF 2} - 1.)