# MPViT on top of detectron2


This folder contains RetinaNet and Mask R-CNN results on top of [Detectron2](https://github.com/facebookresearch/detectron2).

## Main results on RetinaNet and Mask R-CNN

:rocket: All model are trained using *ImageNet-1K* [pretrained weights](GET_STARTED.md).

:sunny: MS denotes the same multi-scale training augmentation as in [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py#L23) which follows the MS augmentation as in DETR and Sparse-RCNN.
Therefore, we also follows the official implementation of [DETR](https://github.com/facebookresearch/detr) and [Sparse-RCNN](https://github.com/PeizeSun/SparseR-CNN) which are also based on Detectron2.



Backbone | Method  | lr Schd | box mAP | mask mAP | #params | FLOPS |                                       weight                                      | 
|:---:     | :---:  | :---:   |   :---: | :---:    |   :---: | :---: |:---------------------------------------------------------------------------------:|
| MPViT-T | RetinaNet  | 1x     | 41.8 | -    | 17M | 196G |   <a href="https://dl.dropbox.com/s/0pep3jnx3zvt1zc/retinanet_mpvit_tiny_1x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/5fpuicgbk2i2sp2/retinanet_mpvit_tiny_1x_metrics.json">metrics</a>  | 
| MPViT-XS| RetinaNet  | 1x     | 43.8 | -    | 20M | 211G |    <a href="https://dl.dropbox.com/s/4oh8h8wag6yhrir/retinanet_mpvit_xs_1x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/2jm7b0uj5wfa45f/retinanet_mpvit_xs_1x_metrics.json">metrics</a>  | 
| MPViT-S | RetinaNet  | 1x     | 45.7 | -    | 32M | 248G |  <a href="https://dl.dropbox.com/s/cbcvz3y6t9hun6l/retinanet_mpvit_small_1x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/d9zyltgy4o6eb28/retinanet_mpvit_small_1x_metrics.json">metrics</a>  | 
| MPViT-B | RetinaNet  | 1x     | 47.0 | -    | 85M | 482G |   <a href="https://dl.dropbox.com/s/hznu2ljqbh0fr1z/retinanet_mpvit_base_1x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/kettv7sk5ett9qz/retinanet_mpvit_base_1x_metrics.json">metrics</a>  | 
| MPViT-T | RetinaNet  | MS+3x  | 44.4 | -    | 17M | 196G | <a href="https://dl.dropbox.com/s/o66ht73g1shpwhn/retinanet_mpvit_tiny_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/4slpgagl49vl37h/retinanet_mpvit_tiny_ms_3x_metrics.json">metrics</a> | 
| MPViT-XS| RetinaNet  | MS+3x  | 46.1 | -    | 20M | 211G |  <a href="https://dl.dropbox.com/s/8kxauovyyaq8x5b/retinanet_mpvit_xs_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/2n9pmm8nbb1ikry/retinanet_mpvit_xs_ms_3x_metrics.json">metrics</a>  | 
| MPViT-S | RetinaNet  | MS+3x  | 47.6 | -    | 32M | 248G | <a href="https://dl.dropbox.com/s/gh00mdtqxoic64e/retinanet_mpvit_small_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/zkmblogkjk9t347/retinanet_mpvit_small_ms_3x_metrics.json">metrics</a> | 
| MPViT-B | RetinaNet  | MS+3x  | 48.3 | -    | 85M | 482G | <a href="https://dl.dropbox.com/s/z7scimhn6dy06kh/retinanet_mpvit_base_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/d5n3ujikitnghvo/retinanet_mpvit_base_ms_3x_metrics.json">metrics</a> | 
| |
| MPViT-T | Mask R-CNN | 1x    | 42.2 | 39.0 | 28M | 216G |   <a href="https://dl.dropbox.com/s/pxregez7a3hdqzl/mask_rcnn_mpvit_tiny_1x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/juczvf6jlx131pn/mask_rcnn_mpvit_tiny_1x_metrics.json">metrics</a>  |
| MPViT-XS| Mask R-CNN | 1x    | 44.2 | 40.4 | 30M | 231G |    <a href="https://dl.dropbox.com/s/os9vk9co87ppg1y/mask_rcnn_mpvit_xs_1x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/4rhc3gzuhrp7b0a/mask_rcnn_mpvit_xs_1x_metrics.json">metrics</a>   |
| MPViT-S | Mask R-CNN | 1x    | 46.4 | 42.4 | 43M | 268G |  <a href="https://dl.dropbox.com/s/ucfwkf65qqklcqn/mask_rcnn_mpvit_small_1x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/9lyuwyc509q69o9/mask_rcnn_mpvit_small_1x_metrics.json">metrics</a>  |
| MPViT-B | Mask R-CNN | 1x    | 48.2 | 43.5 | 95M | 503G |   <a href="https://dl.dropbox.com/s/m7p17jp5qaf41lm/mask_rcnn_mpvit_base_1x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/v639wuwa08729mn/mask_rcnn_mpvit_base_1x_metrics.json">metrics</a>  |
| MPViT-T | Mask R-CNN | MS+3x | 44.8 | 41.0 | 28M | 216G | <a href="https://dl.dropbox.com/s/2wu26zurp5u5057/mask_rcnn_mpvit_tiny_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/6fz98386gix3nif/mask_rcnn_mpvit_tiny_ms_3x_metrics.json">metrics</a> |
| MPViT-XS| Mask R-CNN | MS+3x | 46.6 | 42.3 | 30M | 231G |  <a href="https://dl.dropbox.com/s/yw85vk53kcdi9ed/mask_rcnn_mpvit_xs_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/3prmnkynixtmw4f/mask_rcnn_mpvit_xs_ms_3x_metrics.json">metrics</a>  |
| MPViT-S | Mask R-CNN | MS+3x | 48.4 | 43.9 | 43M | 268G | <a href="https://dl.dropbox.com/s/b0fohmjmggahnny/mask_rcnn_mpvit_small_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/fcfpo2qcfzydsyc/mask_rcnn_mpvit_small_ms_3x_metrics.json">metrics</a> |
| MPViT-B | Mask R-CNN | MS+3x | 49.5 | 44.5 | 95M | 503G | <a href="https://dl.dropbox.com/s/9apn9ywk5ujk01s/mask_rcnn_mpvit_base_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/jcdh98hir236e9x/mask_rcnn_mpvit_base_ms_3x_metrics.json">metrics</a> |


## Usage
### Environment Preparation

We test all models using `pytorch==1.7.0` `detectron2==0.5` `cuda==10.1` on NVIDIA V100 GPUs.

For the installation `detectron2` library, please refer to the [Detectron2's INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

```bash
# Install `detectron2`
python -m pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

# Install `shapely`
conda install shapely
```

For the coco data preparation, please refer to the [CoaT's guide](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/detectron2#code-and-dataset-preparation).



### Evaluation

The following commands provide an example, Retinanet / Mask R-CNN with MPViT backbone using a single GPU.



You can download the trained checkpoint by yourself and use the weight path for this command.

or

You can use the checkpoint link directly in this command.
```bash
cd MPViT/detectron2
python train_net.py --config-file <config-file> --eval-only --num-gpus <num_gpus> MODEL.WEIGHTS <checkpoint_file_path or link>  
``` 

For RetinaNet with `MPViT-Small`:
```bash
python train_net.py --config-file configs/retinanet/retinanet_mpvit_small_ms_3x.yaml --eval-only --num-gpus 1 MODEL.WEIGHTS https://dl.dropbox.com/s/gh00mdtqxoic64e/retinanet_mpvit_small_ms_3x.pth
```
<details>
<summary>
This should give the following result:
</summary>

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl
47.5802,68.7466,51.2814,32.0966,51.8934,61.1945
```
</details>


For Mask R-CNN with `MPViT-Small`:

```bash
python train_net.py --config-file configs/maskrcnn/mask_rcnn_mpvit_small_ms_3x.yaml --eval-only  --num-gpus 1 MODEL.WEIGHTS https://dl.dropbox.com/s/b0fohmjmggahnny/mask_rcnn_mpvit_small_ms_3x.pth
```
<details>

<summary>
This should give the following result:
</summary>

```
Task: bbox
AP,AP50,AP75,APs,APm,APl
48.4422,70.5305,52.5705,32.4423,51.5775,62.6640
Task: segm
AP,AP50,AP75,APs,APm,APl
43.9366,67.6408,47.5103,25.2489,46.4255,62.0025
```

</details>   

### Training
The following command provides an example (Mask R-CNN, 8-GPU) to train the Mask R-CNN w/ MPViT backbone.
```bash
python train_net.py --config-file <config-file> --num-gpus <num_gpus>
```

For Mask R-CNN with `MPViT-Small`:

```bash
python train_net.py --config-file configs/maskrcnn/mask_rcnn_mpvit_small_ms_3x.yaml --num-gpus 8
```


[Detectron2's document](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) may help you for more details.


## Acknowledgment
Thanks to [Detectron2](https://github.com/facebookresearch/detectron2) for the RetinaNet and Mask R-CNN implementation.
