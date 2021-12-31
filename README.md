# Bird's Eye View Detection from Stereo Camera 

## Dataset Structure

[KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev) datasets. The structures of the datasets are shown in below. 

It's preferable to use the following command to download the dataset as we edited it to only download the testing images.

```bash
# install awscli "AWS Command Line" for python
!pip install awscli 

# Run dataset kitti download script.
!chmod +x ./W-Stereo-Disp/scripts/download_kitti.sh
! ./W-Stereo-Disp/scripts/download_kitti.sh

```
#### KITTI Object Detection
```
KITTI
    | testing
        | calib
          | 000000.txt
        | image_2
          | 000000.png
        | image_3
          | 000000.png
    | val.txt
```

## [1- Wasserstein Distances for Stereo Disparity Estimation](https://arxiv.org/abs/2007.03085) 

![Figure](figures/neurips2020-pipeline.png)


Clone Repo:

- git clone https://github.com/oforomar/BEV-Project.git --recurse-submodules

If PIL error occurem, Use -> pip install --upgrade pillow
