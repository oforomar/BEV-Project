# Bird's Eye View Detection from Stereo Camera 

## Clone this repository
```bash
!git clone https://github.com/oforomar/BEV-Project.git --recurse-submodules
```

## Dataset

[KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev) datasets. The structures of the datasets are shown in below. 

It's preferable to use the following command to download the dataset as we edited it to only download the testing images.

```bash
# install awscli "AWS Command Line" for python
!pip install awscli 

# Kitti download script
!chmod +x ./W-Stereo-Disp/scripts/download_kitti.sh
! ./W-Stereo-Disp/scripts/download_kitti.sh
```
You can use your own Indices for Inference or you can use the following command for using the first 15 images in the testing dataset for inferance   
```bash
# val.txt file contain the idx of the images from the dataset, which you want to be used in Inferance.
!mv val.txt KITTI/
```

#### KITTI Object Detection Structure
```
KITTI
    | val.txt
    | testing
        | calib
          | 000000.txt
        | image_2
          | 000000.png
        | image_3
          | 000000.png
```
## Install Dependencies:
```bash
!pip install -r requirements.txt
```

## [Wasserstein Distances for Stereo Disparity Estimation](https://arxiv.org/abs/2007.03085) 

![Figure](https://user-images.githubusercontent.com/54632431/147827857-3dc56611-7e92-4819-84e5-d12117e5b693.png)

### [Pretrained model](https://drive.google.com/drive/folders/1gePafBBvHJDm1b4EpTa34C3XqoPOz757)

Downloading Checkpoint and Best Model pre-trained, to be used as inferance. 
Checkpoints should be located W-Stereo-Disp/checkpoints/. 
Steps:  
    1-Create W-Stereo-Disp/checkpoints. 
    2-Download the files  
    3-Return to main project folder. 
```bash
!mkdir W-Stereo-Disp/checkpoints
%cd W-Stereo-Disp/checkpoints

# To Download checkpoint.pth.tar, model_best.pth.tar files
# checkpoint.pth.tar
!gdown --id 1K110r6n0kg_j3Xq6ThicwOXpYmiBjQ77
# model_best.pth.tar
!gdown --id 10GKd_H4qpdG4PxPVQ-Cjz9pfmdn78B9o
%cd ../..
```










If PIL error occurem, Use -> pip install --upgrade pillow
