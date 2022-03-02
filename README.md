# [MPViT](https://arxiv.org/abs/2112.11010) : Multi-Path Vision Transformer for Dense Prediction

This repository inlcudes official implementations and model weights for [MPViT](https://arxiv.org/abs/2112.11010).

[[`Arxiv`](https://arxiv.org/abs/2112.11010)] [[`BibTeX`](#CitingMPViT)]

> **[MPViT : Multi-Path Vision Transformer for Dense Prediction](https://arxiv.org/abs/2112.11010)**<br>
> :classical_building:️️:school:[Youngwan Lee](https://github.com/youngwanLEE), :classical_building:️️Jonghee Kim, :school:[Jeff Willette](https://jeffwillette.github.io/), :school:[Sung Ju Hwang](http://www.sungjuhwang.com/) <br>
> ETRI:classical_building:️, KAIST:school: <br>


## News
 🎉 MPViT has been accepted in CVPR2022.

## Abstract

We explore **multi-scale patch embedding** and **multi-path structure**, constructing the **Multi-Path Vision
Transformer (MPViT)**. MPViT embeds features of the same size (i.e., sequence length) with patches of different scales
simultaneously by using overlapping convolutional patch embedding. Tokens of different scales are then independently fed
into the Transformer encoders via multiple paths and the resulting features are aggregated, enabling both fine and
coarse feature representations at the same feature level. Thanks to the diverse and multi-scale feature representations,
our MPViTs scaling from Tiny(5M) to Base(73M) consistently achieve superior performance over state-of-the-art Vision
Transformers on ImageNet classification, object detection, instance segmentation, and semantic segmentation. These
extensive results demonstrate that MPViT can serve as a versatile backbone network for various vision tasks.


<div align="center">
  <img src="https://dl.dropbox.com/s/qsp5scrd9okl3pw/mpvit_plot1.png" width="850px" />
</div>

<div align="center">
  <img src="https://dl.dropbox.com/s/dsaqd0cc9ryzqim/mpvit_plot2.png" width="850px" />
</div>

## Main results on ImageNet-1K

:rocket: These all models are trained on **ImageNet-1K** with the same training recipe as [DeiT](https://github.com/facebookresearch/deit) and [CoaT](https://github.com/mlpc-ucsd/CoaT).

| model | resolution | acc@1 | #params | FLOPs |                              weight                               |
|:---:  |  :---:     | :---: |   :---: | :---: |:-----------------------------------------------------------------:|
| MPViT-T |  224x224 | 78.2 | 5.8M   | 1.6G  | [weight](https://dl.dropbox.com/s/1cmquqyjmaeeg1n/mpvit_tiny.pth)  | 
| MPViT-XS|  224x224 | 80.9 | 10.5M  | 2.9G  | [weight](https://dl.dropbox.com/s/vvpq2m474g8tvyq/mpvit_xsmall.pth) |
| MPViT-S |  224x224 | 83.0 | 22.8M  | 4.7G  | [weight](https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth)  |
| MPViT-B |  224x224 | 84.3 | 74.8M  | 16.4G | [weight](https://dl.dropbox.com/s/la8w31m0apj2830/mpvit_base.pth) |


## Main results on COCO object detection

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

## Main results on ADE20K Semantic segmentation

All model are trained using ImageNet-1K pretrained weight.

Please refer to [`semantic_segmentation/`](semantic_segmentation) for the details.

| Backbone | Method | Crop Size | Lr Schd | mIoU | #params | FLOPs |                                   weight
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:--------------------------------------------------------------------------:|
| MPViT-S  | UperNet | 512x512 | 160K | 48.3 | 52M | 943G | [weight](https://dl.dropbox.com/s/5opqzboalok7lme/upernet_mpvit_small.pth) | 
| MPViT-B  | UperNet | 512x512 | 160K | 50.3 | 105M | 1185G | [weight](https://dl.dropbox.com/s/shr88fojdcqvhpr/upernet_mpvit_base.pth)  |

## Getting Started

:raised_hand: We use `pytorch==1.7.0` `torchvision==0.8.1` `cuda==10.1` libraries on NVIDIA V100 GPUs. If you use different versions of `cuda`, you may obtain different accuracies, but the differences are negligible.

- For **Image Classification**, please see [GET_STARTED.md](GET_STARTED.md).
- For **Object Detection and Instance Segmentation**, please
  see [OBJECT_DETECTION.md](OBJECT_DETECTION.md).
- For **Semantic Segmentation**, please
  see [semantic_segmentation/README.md](semantic_segmentation/README.md).

## Acknowledgement

This repository is built using the [Timm](https://github.com/rwightman/pytorch-image-models)
library, [DeiT](https://github.com/facebookresearch/deit), [CoaT](https://github.com/mlpc-ucsd/CoaT), [Detectron2](https://github.com/facebookresearch/detectron2), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) repositories.

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant
funded by the Korean government (MSIT) (No. 2020-0-00004, Development of Previsional Intelligence based on Long-term
Visual Memory Network and No. 2014-3-00123, Development of High Performance Visual BigData Discovery Platform for
Large-Scale Realtime Data Analysis).

## License

Please refer to [MPViT LSA](LICENSE.md).

## <a name="CitingMPViT"></a>Citing MPViT

```BibTeX
@inproceedings{lee2022mpvit,
      title={MPViT: Multi-Path Vision Transformer for Dense Prediction}, 
      author={Youngwan Lee and Jonghee Kim and Jeffrey Willette and Sung Ju Hwang},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2022}
}
```
