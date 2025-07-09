
# using the teacher network of the version of a frozen backbone 

python train_student_cifar.py \
    --tarch wrn_40_2_aux \
    --arch wrn_16_2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0
python train_student_cifar.py \
    --tarch wrn_40_2_aux \
    --arch wrn_16_2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 2
python train_student_cifar.py --tarch wrn_40_2_aux --arch wrn_40_1_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0


python train_student_cifar.py --tarch wrn_40_2_aux --arch ShuffleV1_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch resnet32x4_aux --arch resnet8x4_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_aux_dataset_cifar100_seed0/resnet32x4_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch resnet56_aux --arch resnet20_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet56_aux_dataset_cifar100_seed0/resnet56_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch vgg13_bn_aux --arch mobilenetV2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_vgg13_bn_aux_dataset_cifar100_seed0/vgg13_bn_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0



python train_student_cifar.py --tarch ResNet50_aux --arch mobilenetV2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_ResNet50_aux_dataset_cifar100_seed0/ResNet50_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0



python train_student_cifar.py --tarch ResNet50_aux --arch vgg8_bn_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_ResNet50_aux_dataset_cifar100_seed0/ResNet50_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0


python train_student_cifar.py --tarch resnet32x4_aux --arch ShuffleV2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_aux_dataset_cifar100_seed0/resnet32x4_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0
python train_student_cifar.py --tarch resnet32x4_aux --arch ShuffleV1_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_aux_dataset_cifar100_seed0/resnet32x4_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0

# using the teacher network of the joint training version 
python train_student_cifar.py --tarch wrn_40_2_aux --arch wrn_40_1_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py \
    --tarch wrn_40_2_aux \
    --arch wrn_16_2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed1/wrn_40_2_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0


python train_student_cifar.py --tarch wrn_40_2_aux --arch ShuffleV1_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch resnet32x4_aux --arch resnet8x4_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_aux_dataset_cifar100_seed1/resnet32x4_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch resnet56_aux --arch resnet20_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet56_aux_dataset_cifar100_seed0/resnet56_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch vgg13_bn_aux --arch mobilenetV2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_vgg13_bn_aux_dataset_cifar100_seed0/vgg13_bn_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch ResNet50_aux --arch mobilenetV2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_ResNet50_aux_dataset_cifar100_seed0/ResNet50_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch resnet32x4_aux --arch ShuffleV2_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_aux_dataset_cifar100_seed0/resnet32x4_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0
python train_student_cifar.py --tarch vgg13_bn_aux --arch vgg8_bn_aux \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_vgg13_bn_aux_dataset_cifar100_seed0/vgg13_bn_aux_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python eval_rep.py \
    --arch wrn_16_2 \
    --dataset STL-10 \
    --data ./data/  \
    --s-path ./checkpoint/train_student_cifar_tarch_wrn_40_2_aux_arch_wrn_16_2_aux_dataset_cifar100_seed1/wrn_16_2_aux_best.pth.tar

python eval_rep.py \
    --arch wrn_16_2 \
    --dataset TinyImageNet \
    --data ./data/tiny-imagenet-200/  \
    --s-path ./checkpoint/train_student_cifar_tarch_wrn_40_2_aux_arch_wrn_16_2_aux_dataset_cifar100_seed0/wrn_16_2_aux.pth.tar