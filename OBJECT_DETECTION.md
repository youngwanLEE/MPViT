# MPViT on object detection and instance segmentation

We provide RetinaNet and Deformable DETR results for object detection and Mask R-CNN results for instance segmentation.

We implement RetinaNet and Mask R-CNN on top of [Detectron2](https://github.com/facebookresearch/detectron2) and Deformable DETR on top of the official [Deformable DETR code](https://github.com/fundamentalvision/Deformable-DETR).
 

## Main results on RetinaNet and Mask R-CNN

:rocket: All model are trained using *ImageNet-1K* [pretrained weights](GET_STARTED.md).

:sunny: MS denotes the same multi-scale training augmentation as in [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py#L23) which follows the MS augmentation as in DETR and Sparse-RCNN.
Therefore, we also follows the official implementation of [DETR](https://github.com/facebookresearch/detr) and [Sparse-RCNN](https://github.com/PeizeSun/SparseR-CNN) which are also based on Detectron2.

Please refer to [`detectron2/`](detectron2/) for the details.

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

### Deformable-DETR

All models are trained using the same training recipe.

Please refer to [`deformable_detr/`](deformable_detr) for the details.

backbone | box mAP | epochs |                                       link                                       |
 |:---:     | :---:  |:--------------------------------------------------------------------------------:| :---:  |
ResNet-50 | 44.5 | 50 |                                        -                                         | 
CoaT-lite S | 47.0 | 50 |    [link](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR)     | 
CoaT-S | 48.4 | 50|    [link](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR)     |
MPViT-S | 49.0 | 50 | [link](https://dl.dropbox.com/s/omzvc4jaqcag540/deformable_detr_mpvit_small.pth) |
