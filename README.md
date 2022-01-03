[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tm4Eq3wh-A3wl9mpN30L8eVAo2Ea3Zd5?usp=sharing)

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

* Downloading Checkpoint and Best Model pre-trained, to be used as inferance    
* Checkpoints should be located W-Stereo-Disp/checkpoints/  
* Steps:  
    1-Create W-Stereo-Disp/checkpoints    
    2-Download the files    
    3-Return to main project folder    
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
### Generating depth maps

```bash
!python "W-Stereo-Disp/src/main_depth.py" -c "W-Stereo-Disp/src/configs/kitti_w1.config" \
    --bval 2 \
    --resume "W-Stereo-Disp/checkpoints/checkpoint.pth.tar" --pretrain "W-Stereo-Disp/checkpoints/model_best.pth.tar" --datapath  "KITTI/testing" \
    --data_list="KITTI/val.txt" --generate_depth_map
```

### Visualize depth maps (optional)
 
```bash
import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('KITTI/testing/depth_maps/000000.npy')
plt.imshow(img_array,cmap='gray')
plt.show()
```

## Pseudo Lidar V2

In this step the dataset used should be as follows:

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
        | depth_maps
          | 000000.npy
```
### Convert depth maps to Pseudo-LiDAR and Planes

Convert depth maps to Pseudo-Lidar Point Clouds

```bash
!python ./Pseudo_Lidar_V2/src/preprocess/generate_lidar_from_depth.py --calib_dir  "KITTI/testing/calib" \
    --depth_dir "KITTI/testing/depth_maps/"  \
    --save_dir  "KITTI/testing/velodyne/"
```

Predict Ground Planes from Point Clouds

```bash
!python ./Pseudo_Lidar_V2/src/preprocess/kitti_process_RANSAC.py --calib_dir  "KITTI/testing/calib" \
    --lidar_dir "KITTI/testing/velodyne" \
    --planes_dir  "KITTI/testing/planes"
```


## Avod

In this step the dataset used should be as follows (extra folders is not an issue):

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
        | planes
          | 000000.txt
        | velodyne
          | 000000.bin
```
Compile integral image library in wavedata
```bash
!chmod +x ./avod/scripts/install/build_integral_image_lib.bash
!sh ./avod/scripts/install/build_integral_image_lib.bash
```
If for any reason the above command didn't compile, remove CMakeCashe.txt in avod/wavedata/wavedata/tools/core

Avod uses Protobufs to configure model and training parameters. Before the framework can be used, the protos must be compiled (from top level avod folder):
```bash
!sh ./avod/avod/protos/run_protoc.sh
```

### [Pretrained model](https://drive.google.com/file/d/1wuMykUDx8tcCfxpqnprmzrgUyheQV42F/view)

Use the following command to download the checkpoints in checkpoints directoy
```bash
# Checkpoints download
%mkdir avod/avod/data/outputs/pyramid_cars_with_aug_example/checkpoints
%cd avod/avod/data/outputs/pyramid_cars_with_aug_example/checkpoints
!gdown --id 1wuMykUDx8tcCfxpqnprmzrgUyheQV42F
!unzip avod.zip
%rm -r avod.zip
%cd ../../../../..
```
Run Inferance
```bash
!python avod/avod/experiments/run_inference.py --checkpoint_name='pyramid_cars_with_aug_example' \
    --data_split='val' --ckpt_indices=120 --device='1'
```
The output of the above command will be the detected cars in txt files saved in avod/avod/data/outputs/pyramid_cars_with_aug_example/predictions/final_predictions_and_scores/val/120000

### Detection Formats

final_prediction of avod, actually box_3d: (N, 7)    
[x, y, z, l, w, h, ry, score,type]
```
-2.70278 1.18219 31.97493 1.69001 4.29071 1.59802 -1.40545 0.99985 0
```

KITTI:
[type, truncation, occlusion, alpha(viewing angle), (x1, y1, x2, y2), (h, w, l), (x, y, z), ry, score]
```
Car -1 -1 -1 488.679 171.776 591.806 209.057 1.69 1.598 4.291 -2.703 1.182 31.975 -1.405 1.0
```

### Converting from avod format to Kitti format
```bash
!python to_kitti_format.py --avod_label_path "avod/avod/data/outputs/pyramid_cars_with_aug_example/predictions/final_predictions_and_scores/val/120000" \
        --save_path KITTI/testing/label_2
```

## Kitti Detect and Visualition

In this step the dataset used should be as follows (extra folders is not an issue):

```
KITTI
    | val.txt
    | testing
        | calib
          | 000000.txt
        | image_2
          | 000000.png
        | planes
          | 000000.txt
        | velodyne
          | 000000.bin
        | label_2
          | 000000.txt
          
```

Changing where you are
```bash
%cd kitti_object_vis/
```

Imports
```bash
%matplotlib inline
import matplotlib.pyplot as plt
import cv2
from kitti_object import kitti_object, show_lidar_with_depth, show_lidar_on_image, \
                         show_image_with_boxes, show_lidar_topview_with_boxes
```

```bash
from xvfbwrapper import Xvfb
vdisplay = Xvfb(width=1920, height=1080)
vdisplay.start()
from mayavi import mlab
mlab.init_notebook('ipy')
```

Detecting 
```bash
dataset = kitti_object("../KITTI", "testing")
# Number of image you want to visualize its BEV
data_idx = 5
objects = dataset.get_label_objects(data_idx)
pc_velo = dataset.get_lidar(data_idx)
calib = dataset.get_calibration(data_idx)
img = dataset.get_image(data_idx)
img_height, img_width, _ = img.shape

fig_3d = mlab.figure(bgcolor=(0, 0, 0), size=(800, 450))
show_lidar_with_depth(pc_velo, objects, calib, fig_3d, True, img_width, img_height)
fig_3d
```

Visualzing 2D image with 3D boxes
```bash
_, img_bbox3d = show_image_with_boxes(img, objects, calib)
img_bbox3d = cv2.cvtColor(img_bbox3d, cv2.COLOR_BGR2RGB)

fig_bbox3d = plt.figure(figsize=(14, 7))
ax_bbox3d = fig_bbox3d.subplots()
ax_bbox3d.imshow(img_bbox3d)
plt.show()
```
Visualzing Bird's eye View
```bash
img_bev = show_lidar_topview_with_boxes(pc_velo, objects, calib)
fig_bev = plt.figure(figsize=(7, 14))
ax_bev = fig_bev.subplots()
ax_bev.imshow(img_bev)
plt.show()
```
Saving the output as images
```bash
import numpy as np
from PIL import Image
import os

depthimage_save_path = "../KITTI/testing/BEV_3D_images/" 
if not os.path.exists(depthimage_save_path):
    os.makedirs(depthimage_save_path)
    
im_3DBOX = Image.fromarray(img_bbox3d)
im_BEV = Image.fromarray(img_bev)
im_3DBOX.save(depthimage_save_path+"3D_BOX_"+str(data_idx)+".jpeg")
im_BEV.save(depthimage_save_path+"BEV_"+str(data_idx)+".jpeg")
```
