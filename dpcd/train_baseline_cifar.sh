python train_baseline_cifar.py --arch ShuffleV1 --data ./data/ --init-lr 0.01 --gpu 0
python train_baseline_cifar.py --arch ShuffleV2 --data ./data/ --init-lr 0.01 --gpu 0
python train_baseline_cifar.py --arch mobilenetV2 --data ./data/ --init-lr 0.01 --gpu 0


python train_baseline_cifar.py --arch wrn_16_2 --data ./data/  --gpu 0
python train_baseline_cifar.py --arch wrn_40_1 --data ./data/  --gpu 0
python train_baseline_cifar.py --arch resnet8x4 --data ./data/  --gpu 0
python train_baseline_cifar.py --arch resnet20 --data ./data/  --gpu 0
python train_baseline_cifar.py --arch vgg8 --data ./data/  --gpu 0

