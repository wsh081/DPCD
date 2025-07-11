import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
from collections import deque
import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds

from bisect import bisect_right
import time
import math
from sklearn.neighbors import KernelDensity



parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='wrn_16_2_aux', type=str, help='student network architecture')
parser.add_argument('--tarch', default='wrn_40_2_aux', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='wrn_40_2_aux.pth.tar', type=str, help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 180, 210], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=300, type=int, dest='sgdr_t', help='SGDR T_0')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=240, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'tarch' + '_' + args.tarch + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + \
          'seed' + str(args.manual_seed) + '.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'tarch' + '_' + args.tarch + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + \
          'seed' + str(args.manual_seed)

args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if args.resume is False and args.evaluate is False:
    with open(log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.set_printoptions(precision=4)

num_classes = 100
trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                  [0.2675, 0.2565, 0.2761])
                                         ]))

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                 [0.2675, 0.2565, 0.2761]),
                                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))

print('==> Building model..')
net = getattr(models, args.tarch)(num_classes=num_classes)
net.eval()
resolution = (1, 3, 32, 32)
print('Teacher Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
      % (args.tarch, cal_param_size(net) / 1e6, cal_multi_adds(net, resolution) / 1e9))
del (net)
net = getattr(models, args.arch)(num_classes=num_classes)
net.eval()
resolution = (1, 3, 32, 32)
print('Student Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
      % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, resolution) / 1e9))
del (net)

print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))
checkpoint = torch.load(args.tcheckpoint, map_location=torch.device('cpu'))

model = getattr(models, args.arch)
net = model(num_classes=num_classes).cuda()
net = torch.nn.DataParallel(net)

tmodel = getattr(models, args.tarch)
tnet = tmodel(num_classes=num_classes).cuda()
tnet.load_state_dict(checkpoint['net'])
tnet.eval()
tnet = torch.nn.DataParallel(tnet)

_, ss_logits = net(torch.randn(2, 3, 32, 32))
num_auxiliary_branches = len(ss_logits)
cudnn.benchmark = True


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)

        return loss



class DynamicDKD32(nn.Module):
    def __init__(self, history_size=2000, base_alpha=1.0, base_beta=6.0,warmup=20):
        super().__init__()
        self.conf_history = deque(maxlen=history_size)
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.ema_conf = None  # 新增EMA跟踪
        self.warmup = warmup

        # 分层阈值参数
        self.threshold_table = {
            0: (0.60, 0.65),
            200: (0.65, 0.70),
            500: (0.70, 0.75)

        }

    def _get_dynamic_threshold(self, step):
        """分层动态阈值(训练前期更宽松)"""
        # 获取当前阶段参数
        for key in sorted(self.threshold_table.keys(), reverse=True):
            if step >= key:
                init_th, quantile = self.threshold_table[key]
                break

        if len(self.conf_history) < 2000:
            return init_th
        return np.quantile(self.conf_history, quantile)

    def forward(self, logits_student, logits_teacher, target,epoch, temperature_tckd=3.0, temperature_nckd=3.0, global_step=0):
        # 计算教师置信度
        t_probs = F.softmax(logits_teacher, dim=1)
        batch_confs = t_probs[torch.arange(len(target)), target].detach().cpu().numpy()
        self.conf_history.extend(batch_confs.tolist())

        # 指数移动平均
        current_batch_conf = np.mean(batch_confs)
        if self.ema_conf is None:
            self.ema_conf = current_batch_conf
        else:
            self.ema_conf = 0.9 * self.ema_conf + 0.1 * current_batch_conf

        device = logits_student.device
        current_conf = torch.tensor(self.ema_conf, dtype=torch.float32, device=device)
        threshold = torch.tensor(self._get_dynamic_threshold(global_step),
                                 dtype=torch.float32, device=device)

        # 自适应灵敏度调整
        scale_factor = 12 * (1 - min(global_step / 1000, 1)) + 6  # 前期高灵敏度，后期稳定
        delta = threshold - current_conf
        adjust_ratio = torch.sigmoid(scale_factor * delta)

        # 动态参数调整
        alpha = self.base_alpha * (1 + 1.0 * adjust_ratio)
        beta = self.base_beta * (1 - 0.5 * adjust_ratio)


        alpha = torch.clamp(alpha, 0.5, 2.0)
        #
        # beta = torch.clamp(beta, 6.0, 10.0)
        #res324    res50
        # beta = torch.clamp(beta, 1.0, 4.0)
        #res56
        beta = torch.clamp(beta, 5.0, 8.0)
        #vgg wrn
       

        # 损失计算部分
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)


        pred_student = F.softmax(logits_student / temperature_tckd, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature_tckd, dim=1)

        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

        log_pred_student = torch.log(pred_student)
        tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='sum') * (temperature_tckd ** 2) / target.shape[0]


        pred_teacher_part2 = F.softmax(logits_teacher / temperature_nckd - 1000.0 * gt_mask, dim=1)
        log_pred_student_part2 = F.log_softmax(logits_student / temperature_nckd - 1000.0 * gt_mask, dim=1)
        nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum') * (temperature_nckd ** 2) / target.shape[0]

        loss = alpha * tckd_loss + beta * nckd_loss
        loss = min(epoch / self.warmup, 1.0) * loss
        return loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


# 获取非ground truth mask
def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


# 拼接mask
def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt








class CEDloss(nn.Module):
    def __init__(self,):
        super(CEDloss, self).__init__()


    def forward(self, logits_student, logits_teacher, target, alpha=1.0, beta=8.0, temperature=3.0, gamma=2.0):
        # 确保输入尺寸一致
        assert logits_student.shape == logits_teacher.shape, "Student and teacher logits must have same shape"

        # 生成mask（保持原始尺寸）
        gt_mask = _get_gt_mask(logits_student, target)  # [B, C]
        other_mask = _get_other_mask(logits_student, target)  # [B, C]

        # 计算原始概率分布（保持原始类别维度）
        pred_student_raw = F.softmax(logits_student / temperature, dim=1)  # [B, C]
        pred_teacher_raw = F.softmax(logits_teacher / temperature, dim=1)  # [B, C]

        # ========== 提前计算非目标类权重 ==========
        with torch.no_grad():
            # 获取非目标类概率（保持原始维度）
            p_student = pred_student_raw * other_mask  # [B, C]
            p_teacher = pred_teacher_raw * other_mask  # [B, C]

            # 计算一致性权重（逐类别处理）
            agreement = torch.sqrt(p_student * p_teacher + 1e-8)
            agreement_weights = agreement ** gamma  # [B, C]

        # ========== TCKD损失计算（使用cat_mask处理后的结果） ==========
        # 拼接处理后的预测结果
        pred_student = cat_mask(pred_student_raw, gt_mask, other_mask)  # [B, 2]
        pred_teacher = cat_mask(pred_teacher_raw, gt_mask, other_mask)  # [B, 2]

        # 计算TCKD损失
        log_pred_student = torch.log(pred_student + 1e-8)
        tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='sum') * (temperature ** 2) / target.shape[0]

        # ========== NCKD损失计算（使用原始维度处理） ==========
        # 生成非目标类的教师概率
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask,
            dim=1
        ) * other_mask  # [B, C]

        # 学生概率处理
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask,
            dim=1
        )  # [B, C]

        # 加权KL散度计算
        nckd_loss = (
                            F.kl_div(
                                log_pred_student_part2,
                                pred_teacher_part2.detach(),
                                reduction='none'
                            ).sum(dim=1)  # 按类别求和 → [B]
                            * (agreement_weights.sum(dim=1))  # 样本级权重求和 → [B]
                    ).sum() * (temperature ** 2) / target.shape[0]

        # 总损失
        loss = alpha * tckd_loss + beta * nckd_loss

        return loss


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch * all_iters_per_epoch) / (
                    args.warmup_epoch * all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def apply_gradient_projection(model, main_loss_weight=0.7):
    """
   
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # 分离主梯度（DynamicDKD32）和辅助梯度（DKDloss1）
        grad_main = param.grad.data * main_loss_weight
        grad_aux = param.grad.data - grad_main

        # 计算投影
        if grad_main.norm() > 1e-6:
            scale = (grad_aux.flatten().dot(grad_main.flatten())) / (grad_main.norm() ** 2 + 1e-8)
            grad_proj = grad_main + scale * grad_main
        else:
            grad_proj = grad_main  # 主梯度为零时，仅保留主方向

        param.grad.data.copy_(grad_proj)

def train(epoch, criterion_list, optimizer):
    train_loss = 0.
    train_loss_cls = 0.
    train_loss_div = 0.

    train_loss_crd = 0.

    ss_top1_num = [0] * num_auxiliary_branches
    ss_top5_num = [0] * num_auxiliary_branches
    class_top1_num = [0] * num_auxiliary_branches
    class_top5_num = [0] * num_auxiliary_branches
    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    criterion_crd = criterion_list[2]


    net.train()
    for batch_idx, (input, target) in enumerate(trainloader):
        batch_start_time = time.time()
        input = input.float().cuda()
        target = target.cuda()

        size = input.shape[1:]

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        optimizer.zero_grad()
        logits, ss_logits = net(input, grad=True)
        with torch.no_grad():
            t_logits, t_ss_logits = tnet(input)

        loss_cls = torch.tensor(0.).cuda()
        loss_div = torch.tensor(0.).cuda()

        loss_crd = torch.tensor(0.).cuda()


        for i in range(len(ss_logits)):
            loss_cls = loss_cls + criterion_cls(ss_logits[i], target)

        loss_cls = loss_cls + criterion_cls(logits, target)

        loss_div = loss_div + criterion_div(logits, t_logits.detach(), target,epoch)





        #

        #

        for i in range(len(ss_logits) - 1):
            for j in range(i, len(ss_logits)):
                loss_crd += criterion_crd(ss_logits[j], ss_logits[i], target)

        for i in range(len(ss_logits)):
            loss_crd += criterion_crd(logits, ss_logits[i], target)







#76.97


        loss = loss_cls+loss_div+ loss_crd




        loss.backward()


        optimizer.step()

        train_loss += loss.item() / len(trainloader)
        train_loss_cls += loss_cls.item() / len(trainloader)
        train_loss_div += loss_div.item() / len(trainloader)

        train_loss_crd += loss_crd.item() / len(trainloader)


        for i in range(len(ss_logits)):
            top1, top5 = correct_num(ss_logits[i], target, topk=(1, 5))
            ss_top1_num[i] += top1
            ss_top5_num[i] += top5

        for i in range(len(ss_logits)):
            top1, top5 = correct_num(ss_logits[i], target, topk=(1, 5))
            class_top1_num[i] += top1
            class_top5_num[i] += top5

        top1, top5 = correct_num(logits, target, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += target.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time() - batch_start_time, (top1_num / (total)).item()))

    ss_acc1 = [round((ss_top1_num[i] / (total)).item(), 4) for i in range(num_auxiliary_branches)]
    ss_acc5 = [round((ss_top5_num[i] / (total)).item(), 4) for i in range(num_auxiliary_branches)]
    class_acc1 = [round((class_top1_num[i] / (total)).item(), 4) for i in range(num_auxiliary_branches)] + [
        round((top1_num / (total)).item(), 4)]
    class_acc5 = [round((class_top5_num[i] / (total)).item(), 4) for i in range(num_auxiliary_branches)] + [
        round((top5_num / (total)).item(), 4)]

    print('Train epoch:{}\nTrain Top-1 ss_accuracy: {}\nTrain Top-1 class_accuracy: {}\n'.format(epoch, str(ss_acc1),
                                                                                                 str(class_acc1)))

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                '\nTrain Top-1 ss_accuracy: {}\nTrain Top-1 class_accuracy: {}\n'
                .format(epoch, lr, time.time() - start_time,
                        str(ss_acc1), str(class_acc1)))


def test(epoch, criterion_cls, net):
    global best_acc
    test_loss_cls = 0.

    ss_top1_num = [0] * (num_auxiliary_branches)
    ss_top5_num = [0] * (num_auxiliary_branches)
    class_top1_num = [0] * num_auxiliary_branches
    class_top5_num = [0] * num_auxiliary_branches

    top1_num = 0
    top5_num = 0
    top10_num = 0
    top50_num = 0

    total = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            batch_start_time = time.time()
            input, target = inputs.cuda(), target.cuda()

            size = input.shape[1:]

            logits, ss_logits = net(input)
            loss_cls = torch.tensor(0.).cuda()
            loss_cls = loss_cls + criterion_cls(logits, target)

            test_loss_cls += loss_cls.item() / len(testloader)

            batch_size = logits.size(0) // 4
            for i in range(len(ss_logits)):
                top1, top5 = correct_num(ss_logits[i], target, topk=(1, 5))
                ss_top1_num[i] += top1
                ss_top5_num[i] += top5

            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top10_num += top1
            top50_num += top5

            for i in range(len(ss_logits)):
                top1, top5 = correct_num(ss_logits[i], target, topk=(1, 5))
                class_top1_num[i] += top1
                class_top5_num[i] += top5
            #

            top1, top5 = correct_num(logits, target, topk=(1, 5))

            top1_num += top1
            top5_num += top5
            total += target.size(0)

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                epoch, batch_idx, len(testloader), time.time() - batch_start_time, (top1_num / (total)).item()))

        ss_acc1 = [round((ss_top1_num[i] / (total)).item(), 4) for i in range(len(ss_logits))]+ [
            round((top10_num / (total)).item(), 4)]
        # ss_acc5 = [round((ss_top5_num[i] / (total)).item(), 4) for i in range(len(ss_logits))]
        class_acc1 = [round((class_top1_num[i] / (total)).item(), 4) for i in range(num_auxiliary_branches)] + [
            round((top1_num / (total)).item(), 4)]

        # class_acc5 = [round((class_top5_num[i] / (total)).item(), 4) for i in range(num_auxiliary_branches)] + [
        #     round((top5_num / (total)).item(), 4)]



        with open(log_txt, 'a+') as f:
            f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 ss_accuracy: {}\nTop-1 class_accuracy: {}\n'
                    .format(epoch, test_loss_cls, str(ss_acc1), str(class_acc1)))
        print('test epoch:{}\nTest Top-1 ss_accuracy: {}\nTest Top-1 class_accuracy: {}\n'.format(epoch, str(ss_acc1),
                                                                                                  str(class_acc1)))

    return class_acc1[-1]


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()

    #
    criterion_div = DynamicDKD32()

    criterion_crd =CEDloss()




    if args.evaluate:
        print('load pre-trained weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        test(start_epoch, criterion_cls, net)
    else:
        print('Evaluate Teacher:')
        acc = test(0, criterion_cls, tnet)
        print('Teacher Acc:', acc)

        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)

        criterion_list.append(criterion_crd)

        criterion_list.cuda()

        if args.resume:
            print('load pre-trained weights from: {}'.format(
                os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls, net)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls, net)

        with open(log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + log_txt + ' ' + args.checkpoint_dir)
