
# using the teacher network of the version of a frozen backbone 

python test_model.py \
  --arch wrn_16_2_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_wrn_40_2_aux_arch_wrn_16_2_aux_dataset_cifar100_seed0/wrn_16_2_aux_best.pth.tar


python test_model.py \
  --arch wrn_16_2_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_wrn_40_2_aux_arch_wrn_40_1_aux_dataset_cifar100_seed0/wrn_16_2_aux_best.pth.tar


python test_model.py \
  --arch ShuffleV1_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_wrn_40_2_aux_arch_ShuffleV1_aux_dataset_cifar100_seed0/ShuffleV1_aux_best.pth.tar


python test_model.py \
  --arch vgg8_bn_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_vgg13_bn_aux_arch_vgg8_bn_aux_dataset_cifar100_seed0/vgg8_bn_aux_best.pth.tar

python test_model.py \
  --arch mobilenetV2_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_vgg13_bn_aux_arch_mobilenetV2_aux_dataset_cifar100_seed0/mobilenetV2_aux_best.pth.tar

python test_model.py \
  --arch mobilenetV2_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_ResNet50_aux_arch_mobilenetV2_aux_dataset_cifar100_seed0/mobilenetV2_aux_best.pth.tar

python test_model.py \
  --arch resnet20_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_resnet56_aux_arch_resnet20_aux_dataset_cifar100_seed0/resnet20_aux_best.pth.tar


python test_model.py \
  --arch ShuffleV2_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_resnet32x4_aux_arch_ShuffleV2_aux_dataset_cifar100_seed0/ShuffleV2_aux_best.pth.tar

python test_model.py \
  --arch ShuffleV1_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_resnet32x4_aux_arch_ShuffleV1_aux_dataset_cifar100_seed0/ShuffleV1_aux_best.pth.tar

python test_model.py \
  --arch resnet8x4_aux \
  --student-weights ./checkpoint/train_student_cifar_tarch_resnet32x4_aux_arch_resnet8x4_aux_dataset_cifar100_seed0/resnet8x4_aux_best.pth.tar
