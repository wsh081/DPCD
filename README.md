
# Dual-path collaborative distillation
 - This project provides source code for our Dual-path collaborative distillation (DPCD).

## Installation

### Requirements


Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 12.1

PyTorch 2.1.1


## Perform  experiments on CIFAR-100 dataset
#### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

#### Training baselines
```
python train_baseline_cifar.py --arch wrn_16_2 --data ./data/  --gpu 0
```
More commands for training various architectures can be found in [train_baseline_cifar.sh]

#### Training teacher networks

You can use the following commands to train your own teacher network.
```
python train_teacher_cifar.py \
    --arch wrn_40_2_aux \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 1
```

#### Training student networks
(1) train baselines of student networks
```
python train_baseline_cifar.py --arch wrn_16_2 --data ./data/  --gpu 0
```


(2) train student networks with a pre-trained teacher network

Note that the specific teacher network should be pre-trained before training the student networks

```
python train_student_cifar.py \
    --tarch wrn_40_2_aux \
    --arch wrn_16_2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0
```

More commands for training various teacher-student pairs can be found in [train_student_cifar.sh]

####  Results of the same architecture style between teacher and student networks

|Teacher <br> Student | WRN-40-2 <br> WRN-16-2 | ResNet32×4  <br> ResNet8×4 | ResNet-56 <br> ResNet-20 | WRN-40-2  <br> WRN-40-1 | VGG13<br> VGG8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|:--------------------:|
| Teacher  |    75.61 | 79.42 | 72.34 | 75.61 | 74.64 |
| Student | 73.26| 72.50| 69.06| 71.98| 70.36 |
| DPCD | 77.30| 77.42| 72.58| 76.04| 75.83|
 


####  Results of different architecture styles between teacher and student networks

|Teacher <br> Student |ResNet32×4  <br>ShufffeNetV2  |  ResNet50  <br> MobileNetV2 | ResNet32x4 <br> ShuffleNetV1 | WRN-40-2<br> ShuffleNetV1 |
|:---------------:|:-----------------:|:-----------------:|:--------------------:|:--------------------:|
| Teacher  |    79.42|79.34 |79.42 |75.61   |
| Student | 71.82|  64.60 |70.50| 70.50 |
| DPCD | 78.91  | 71.38 | 77.87  |77.80 |



#### Obtain the student network weights

The pre-trained Student networks can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/19TFpc0HVcJ1ucofyTOQe-g) (Access code: qspu).

unzip to the `./checkpoint` folder

#### Testing student networks
The accuracy of the student network is recorded in the saved weight files. Of course, you can also evaluate the student network using the following script commands.
```

python test_model.py \
  --arch wrn_16_2_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_wrn_40_2_aux_arch_wrn_16_2_aux_dataset_cifar100_seed0/wrn_16_2_aux_best.pth.tar

```
More test commands for the student network can be found in [test_model.sh]


## Perform  experiments on ImageNet dataset

### Dataset preparation

- Download the ImageNet dataset to YOUR_IMAGENET_PATH and move validation images to labeled subfolders
    - The [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) may be helpful.

- Create a datasets subfolder and a symlink to the ImageNet dataset

```
$ ln -s PATH_TO_YOUR_IMAGENET ./data/
```
Folder of ImageNet Dataset:
```
data/ImageNet
├── train
├── val
```

### Training teacher networks
You can use the following commands to train your own teacher network.
```
python train_teacher_imagenet.py \
    --dist-url 'tcp://127.0.0.1:2222' \
    --data ./data/ImageNet/ \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --checkpoint-dir ./checkpoint/ \
    --gpu 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0 --manual_seed 1
```

### Training student networks

```
python train_student_imagenet.py \
    --data ./data/ImageNet/ \
    --arch resnet18_imagenet_aux \
    --tarch resnet34_imagenet_aux \
    --tcheckpoint ./checkpoint/train_teacher_imagenet_arch_resnet34_aux_dataset_imagenet_seed1/resnet34_imagenet_aux_best.pth.tar \
    --dist-url 'tcp://127.0.0.1:2222' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0 --manual_seed 1
```

####  Results on the teacher-student pair of ResNet-34 and ResNet-18 

| Accuracy |Teacher | Student  |  DPCD|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| Top-1 | 73.31  | 69.75 | 71.89| 
| Top-5 | 91.42  | 89.07 | 90.50 |


 
 
