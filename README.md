# ASNet

Stereo Depth Estimation Based on Adaptive Stacks from Event Cameras

Unlike the frames from traditional frame-based cameras, events are sparse, so how to use events for depth estimation becomes a critical part of the entire process. In this work, we propose adaptive stacks method for events processing, which uses a set of weighted stacks for event stacking, and dynamically changes the size of each stack according to the events generation rate to achieve a good reconstruction of the scene information. The preprocessed event results are then fed into the corresponding neural network for depth estimation. Finally, the model is evaluated using the open-source event dataset MVSEC, and its metrics are compared with other event-based stereo depth estimation methods.

## Evaluation Results 
ASNet are trained and tested using  MVSEC datasets.  
In order to use ASNet, you need first download the MVSEC "indoor-flying' files in a folder like this:

|MVSEC
│ 
└───indoor_flying_left_x_map.txt
│ 
└───indoor_flying_left_y_map.txt
│ 
└───indoor_flying_right_x_map.txt
│ 
└───indoor_flying_right_y_map.txt
│
└───indoor_flying1_data.hdf5
│ 
└───indoor_flying1_gt.hdf5
│ 
└───indoor_flying2_data.hdf5
│ 
└───indoor_flying2_gt.hdf5
│
...



|forth
│ 
└───event7.aedat4



## Installation

### Requirements
The code is tested on:
- Ubuntu 18.04
- Python 3.6 
- PyTorch 1.4.0 
- Torchvision 0.5.0
- CUDA 10.0

### Setting up the environment

```shell
conda env create --file ASNet.yml
conda activate ASNet
```



### Training 

Set a variable for the dataset directory, e.g. ```DATAPATH="/Datasets/MVSEC/"```. Then, run ```train.py``` as below:


```shell
python train.py --dataset mvsec --datapath $DATAPATH --datasplit 1 --trainlist ./filenames/frame_train.txt --testlist ./filenames/frame_val.txt --epochs 100 --lrepochs "40,60,80,90:5"  --logdir ./Logdir/s1

```

### Prediction

The following script creates disparity maps for a specified model:

```shell

python prediction.py --dataset mvsec --datapath $DATAPATH --datasplit 1 --testlist ./filenames/framesampleTRAIN.txt --loadckpt ./Logdir/s1/best.ckpt

```
### Application
datapath is the folder where a single forth camrea file is stored: 
```shell
python prediction.py --dataset forth --datapath $datapath --datafilename event7.aedat4 --testlist ./filenames/frame_forth.txt --loadckpt ./Logdir/s1/best.ckpt --model ASnet
```