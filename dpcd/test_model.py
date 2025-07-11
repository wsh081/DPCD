import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import models


def correct_num(output, target, topk=(1,)):
    """计算正确样本数量，而非百分比"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k)
    return res


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Testing')
    parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
    parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
    parser.add_argument('--arch', default='wrn_16_2_aux', type=str, help='student network architecture')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='the number of workers')
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--manual_seed', type=int, default=0)
    parser.add_argument('--student-weights', required=True, type=str, help='path to trained student weights')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 设置随机种子
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.set_printoptions(precision=4)

    num_classes = 100

    # 数据集准备
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])

    testset = torchvision.datasets.CIFAR100(
        root=args.data, train=False, download=True,
        transform=test_transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    # 初始化学生网络
    print(f'==> Building student model: {args.arch}')
    net = getattr(models, args.arch)(num_classes=num_classes).cuda()
    net = torch.nn.DataParallel(net)

    # 获取辅助分支数量
    input_sample = torch.randn(2, 3, 32, 32).cuda()
    _, ss_logits = net(input_sample)
    num_auxiliary_branches = len(ss_logits)
    print(f'Number of auxiliary branches: {num_auxiliary_branches}')

    # 加载预训练权重
    print(f'==> Loading student weights from: {args.student_weights}')
    checkpoint = torch.load(args.student_weights, map_location='cuda')

    # 处理权重键名不匹配问题
    new_state_dict = {}
    for key, value in checkpoint['net'].items():
        # 跳过num_batches_tracked参数
        if 'num_batches_tracked' in key:
            continue

        # 添加module前缀
        new_key = 'module.' + key
        new_state_dict[new_key] = value

    # 加载权重
    net.load_state_dict(new_state_dict, strict=False)

    # 测试网络性能
    net.eval()
    criterion = nn.CrossEntropyLoss()

    # 初始化统计变量 - 使用正确样本数而非百分比
    test_loss_cls = 0.
    ss_top1_correct = [0] * num_auxiliary_branches
    class_top1_correct = [0] * num_auxiliary_branches
    main_top1_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.cuda(), target.cuda()
            batch_size = inputs.size(0)
            total_samples += batch_size

            # 前向传播
            logits, ss_logits = net(inputs)

            # 计算损失
            loss_cls = criterion(logits, target)
            test_loss_cls += loss_cls.item() * batch_size

            # 计算每个辅助分支的准确率
            for i in range(len(ss_logits)):
                top1, _ = correct_num(ss_logits[i], target, topk=(1, 5))
                ss_top1_correct[i] += top1.item()

            # 计算每个分支的准确率
            for i in range(len(ss_logits)):
                top1, _ = correct_num(ss_logits[i], target, topk=(1, 5))
                class_top1_correct[i] += top1.item()

            # 计算主分支的准确率
            top1, _ = correct_num(logits, target, topk=(1, 5))
            main_top1_correct += top1.item()

            # 打印进度
            if batch_idx % 20 == 0:
                current_acc = 100.0 * main_top1_correct / total_samples
                print(f'Batch: {batch_idx}/{len(testloader)}, Top-1 Acc: {current_acc:.2f}%')

        # 计算最终准确率
        test_loss_cls /= total_samples
        ss_acc1 = [100.0 * ss_top1_correct[i] / total_samples for i in range(len(ss_logits))]
        class_acc1 = [100.0 * class_top1_correct[i] / total_samples for i in range(num_auxiliary_branches)] + [
            100.0 * main_top1_correct / total_samples]

        # 打印结果
        print('\nTest results:')
        print(f'Test loss: {test_loss_cls:.4f}')
        print(f'Top-1 ss_accuracy: {[round(x, 2) for x in ss_acc1]}')
        print(f'Top-1 class_accuracy: {[round(x, 2) for x in class_acc1]}')
        print(f'Final Top-1 Accuracy: {class_acc1[-1]:.2f}%')


if __name__ == '__main__':
    main()