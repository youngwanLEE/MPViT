# MPViT on ImageNet classification

## Main results on ImageNet-1K

:rocket: These all models are trained for 300 epochs on **ImageNet-1K** with the same training recipe as [DeiT](https://github.com/facebookresearch/deit) and [CoaT](https://github.com/mlpc-ucsd/CoaT).

| model | resolution | acc@1 | #params | FLOPs |                              weight                               |
|:---:  |  :---:     | :---: |   :---: | :---: |:-----------------------------------------------------------------:|
| MPViT-T |  224x224 | 78.2 | 5.8M   | 1.6G  | [weight](https://dl.dropbox.com/s/1cmquqyjmaeeg1n/mpvit_tiny.pth)  | 
| MPViT-XS|  224x224 | 80.9 | 10.5M  | 2.9G  | [weight](https://dl.dropbox.com/s/vvpq2m474g8tvyq/mpvit_xsmall.pth) |
| MPViT-S |  224x224 | 83.0 | 22.8M  | 4.7G  | [weight](https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth)  |
| MPViT-B |  224x224 | 84.3 | 74.8M  | 16.4G | [weight](https://dl.dropbox.com/s/la8w31m0apj2830/mpvit_base.pth) |



## Preparation

### Data
Please refer to [DeiT](https://github.com/facebookresearch/deit#data-preparation) for data preparation.

We recommend to symlink the path(`data/ImageNet`) to the ImageNet dataset path to `data/` as follows

```bash
# symlink the ImageNet dataset
cd MPViT
mkdir data
ln -s /path_to_imagenet_dataset data/ImageNet
```

Check the `data/ImageNet` path points to `/path_to_imagenet` correctly.
```bash
# ImageNet -> /path_to_imagenet_dataset
ll data
```

### Required packages installation
We use `Pytorch==1.7.0`, `torchvision==0.8.1`, `cuda==10.1` `einops` and pytorch-image-models (`timm`).

Please refer to the official [pytorch installation guide](https://pytorch.org/get-started/previous-versions/). 

```
# Install Pytorch and torchvision
conda install pytorch==1.7.0 torchvision==0.8.1 -c pytorch

# Install timm
pip install timm

# Install einops
pip install einops
```

## Evaluation
To evaluate a pre-trained MPViT on ImageNet val with a single GPU, run:


`MPViT-base`:

```bash
python main.py --eval --resume https://dl.dropbox.com/s/la8w31m0apj2830/mpvit_base.pth --model mpvit_base
```
This should give
```
* Acc@1 84.292 Acc@5 96.820 loss 0.916
  Accuracy of the network on the 50000 test images: 84.3%
```

Here you'll find the command-lines to reproduce the inference results for MPViT models.
<details>

<summary>
MPViT-Tiny
</summary>

```
python main.py --eval --resume https://dl.dropbox.com/s/1cmquqyjmaeeg1n/mpvit_tiny.pth --model mpvit_tiny
```
giving
```
* Acc@1 78.210 Acc@5 94.270 loss 1.112
  Accuracy of the network on the 50000 test images: 78.2%
```

</details>


<details>

<summary>
MPViT-Xsmall
</summary>

```
python main.py --eval --resume https://dl.dropbox.com/s/vvpq2m474g8tvyq/mpvit_xsmall.pth --model mpvit_xsmall
```
giving
```
* Acc@1 80.896 Acc@5 95.352 loss 1.039
Accuracy of the network on the 50000 test images: 80.9%
```

</details>

<details>

<summary>
MPViT-Small
</summary>

```
python main.py --eval --resume https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth --model mpvit_small
```
giving
```
* Acc@1 82.960 Acc@5 96.154 loss 0.893
Accuracy of the network on the 50000 test images: 83.0%
```

</details>


## Training from scratch
To train MPViT models on ImageNet on a single node with 8 gpus for 300 epochs, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py \ 
--model <mpvit_model> --data-path <imagenet-path> --batch-size <batch-size-per-gpu> --output <output-directory>
```


`MPViT-Tiny`:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model mpvit_tiny --batch-size 128 --data-path <imagenet-path> --output_dir <output-directory>
```

`MPViT-Xsmall`:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model mpvit_xsmall --batch-size 128 --data-path <imagenet-path> --output_dir <output-directory>
```

`MPViT-Small`:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model mpvit_small --batch-size 128 --drop-path 0.05 --data-path <imagenet-path> --output_dir <output-directory>
```

`MPViT-Base`:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model mpvit_base --batch-size 128 --drop-path 0.3 --data-path <imagenet-path> --output_dir <output-directory>
```
