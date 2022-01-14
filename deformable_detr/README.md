# Deformable DETR with MPViT backbone 

This folder contains the Deformable DETR using MPViT as a backbone experiment using [Deformable DETR](https://arxiv.org/abs/2010.04159) framework. 
For fair comparison with [CoaT](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR), We also use the same [official implementation](https://github.com/fundamentalvision/Deformable-DETR) as [CoaT](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR) and follow its default settings (with multi-scale) in our experiments.


## Main Result
backbone | box mAP  | epochs |                                       link                                       |
 |:--------:| :---:  |:--------------------------------------------------------------------------------:| :---:  |
ResNet-50 |   44.5   | 50 |                                        -                                         | 
CoaT-lite S |   47.0   | 50 |    [link](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR)     | 
CoaT-S |   48.4   | 50|    [link](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR)     |
**MPViT-S** | **49.0** | 50 | [link](https://dl.dropbox.com/s/omzvc4jaqcag540/deformable_detr_mpvit_small.pth) |



## Usage
### Environment Preparation
Install required packages. See Deformable DETR's [original readme](https://github.com/fundamentalvision/Deformable-DETR) for more details.
   ```bash
   # Install the required packages.
   cd MPViT/deformable_detr
   pip install -r ./requirements.txt

   # Build and install MultiScaleDeformableAttention operator.
   # Note: 1. It may requires CUDA installation. In our environment, we install CUDA 11.3 
   #          which is compatible with CUDA 11.0 bundled with PyTorch and RTX 30 series graphic cards.
   #       2. If you found error "no kernel image is available for execution on the device" during training,
   #          please use `pip uninstall MultiScaleDeformableAttention` to remove the installed package,
   #          delete all build folders (e.g. ./build, ./dist and ./*.egg-info), and then re-run `./make.sh`.
   cd ./models/ops
   sh ./make.sh
   cd ../../
   ```

### Code and Dataset Preparation
Please follow the steps in the [CoaT's guide](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/detectron2#code-and-dataset-preparation) to download COCO 2017 dataset and extract. 

Here we simply create symbolic links for models and the dataset folder.

   ```bash
   # Create symbolic links.
   # Note: Here we directly create a symbolic link to COCO dataset which has set up for detectron2/. You may
   #       refer to the [corresponding readme](../detectron2/README.md) to download COCO dataset first. 
   mkdir -p ./data
   ln -sfT ../../detectron2/datasets/coco ./data/coco
   ```

### Evaluation
We provide the MPViT-Small checkpoint pre-trained on the ImageNet-1K dataset.

We compare MPViT-Small with CoaT-Lite Small and CoaT Small which are from [CoaT's official repo](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR). 


| Name            | AP | AP50 | AP75 | APS | APM | APL | URL                                                                                                                                                                                                                               |
|-----------------| --- | --- | --- | --- | --- | --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CoaT-Lite Small | 47.0 | 66.5 | 51.2 | 28.8 | 50.3 | 63.3  | [model](http://vcl.ucsd.edu/coat/pretrained/tasks/Deformable-DETR/coat_lite_small_deformable_detr_1801ee09.pth) &nbsp;/&nbsp; [log](http://vcl.ucsd.edu/coat/pretrained/tasks/Deformable-DETR/coat_lite_small_deformable_detr_1801ee09.txt)    |
| CoaT Small      | 48.4 | 68.5 | 52.4 | 30.1 | 51.8 | 63.8  | [model](http://vcl.ucsd.edu/coat/pretrained/tasks/Deformable-DETR/coat_small_deformable_detr_8a86ba55.pth) &nbsp;/&nbsp; [log](http://vcl.ucsd.edu/coat/pretrained/tasks/Deformable-DETR/coat_small_deformable_detr_8a86ba55.txt) |
| **MPViT-Small** | 49.0 | 68.7 | 53.7 | 31.7 | 52.4 | 64.5 |  [model](https://dl.dropbox.com/s/omzvc4jaqcag540/deformable_detr_mpvit_small.pth) &nbsp;/&nbsp; [log](https://www.dropbox.com/s/rndlrfns2nrvrr1/deformable_detr_mpvit_small_log.txt?dl=0)                                                     

The following commands provide an example (MPViT Small) to evaluate the pre-trained checkpoint.
   ```bash
    # Usage: Please see [Deformable DETR's document] for more details.
    cd MPViT/deformable_detr
    GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/mpvit_small_deformable_detr.sh --resume https://dl.dropbox.com/s/omzvc4jaqcag540/deformable_detr_mpvit_small.pth --eval
   ```

<details>
<summary>
This should give the following result:
</summary>

```bash
    # IoU metric: bbox
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
    # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.687
    # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.537
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.317
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.524
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.373
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.623
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.667
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.465
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.714
    # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.848
```
</details>


### Training
The following commands provide an example (MPViT-Small, 8-GPU) to train the Deformable DETR w/ MPViT backbone.
   ```bash
   # Usage: Please see Deformable DETR's document for more details.
   cd MPViT/deformable_detr
   GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/mpvit_small_deformable_detr.sh
   ```


## Acknowledgment
Thanks to Deformable DETR for its [official implementation](https://github.com/fundamentalvision/Deformable-DETR) and [CoaT](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR).

We borrow some codes from [CoaT](https://github.com/mlpc-ucsd/CoaT/tree/main/tasks/Deformable-DETR).