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


class DecoupledTempLoss(nn.Module):
    def __init__(self, T_high=4.0, T_low=2.0, lambda_target=1.0, grad_clip=8.0):
        super().__init__()
        self.T_high = T_high  # 目标类温度
        self.T_low = T_low  # 非目标类温度
        self.lambda_target = lambda_target  # 目标类损失权重
        self.grad_clip = grad_clip

    def forward(self, student_logits, teacher_logits):
        batch_size, num_classes = teacher_logits.shape

        # === 生成教师目标掩码 ===
        target_idx = torch.argmax(teacher_logits, dim=-1)  # [B]
        target_mask = torch.zeros_like(teacher_logits, dtype=torch.bool)
        target_mask[torch.arange(batch_size), target_idx] = True  # [B,C]

        # === 教师Logits解耦与温度处理 ===
        # 目标类处理 (高温软化)
        teacher_target = teacher_logits[target_mask] / self.T_high  # [B]

        # 非目标类处理 (低温锐化)
        teacher_nontarget = teacher_logits[~target_mask] / self.T_low  # [B*(C-1)]

        # === 学生Logits解耦 ===
        student_target = student_logits[target_mask]  # [B]
        student_nontarget = student_logits[~target_mask]  # [B*(C-1)]

        # === 梯度防护 ===
        student_target = student_target.detach() + (student_target - student_target.detach())
        student_nontarget = student_nontarget.detach() + (student_nontarget - student_nontarget.detach())
        for tensor in [student_target, student_nontarget]:
            if tensor.grad is not None:
                tensor.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip, self.grad_clip))

        # === 分项损失计算 ===
        # 目标类KL散度 (高温软化后的教师 vs 学生)
        loss_target = F.kl_div(
            F.log_softmax(student_target, dim=-1),
            F.softmax(teacher_target, dim=-1),
            reduction='batchmean'
        )

        # 非目标类KL散度 (低温锐化的教师 vs 学生)
        loss_nontarget = F.kl_div(
            F.log_softmax(student_nontarget, dim=-1),
            F.softmax(teacher_nontarget, dim=-1),
            reduction='batchmean'
        )

        # 加权总损失
        total_loss = self.lambda_target * loss_target + (1 - self.lambda_target) * loss_nontarget
        return total_loss



class DynamicDKD32(nn.Module):
    def __init__(self, history_size=2000, base_alpha=1.0, base_beta=8.0,warmup=20):
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
#7733
        # alpha = self.base_alpha * (1 + 0.5 * adjust_ratio)
        # beta = self.base_beta * (1 - 1.0 * adjust_ratio)
#7675
        # alpha = self.base_alpha * (1 + 1.0 * adjust_ratio)
        # beta = self.base_beta * (1 - 1.0 * adjust_ratio)
#7634
        # alpha = self.base_alpha * (1 - 1.0 * adjust_ratio)
        # beta = self.base_beta * (1 + 1.0 * adjust_ratio)
#7637

        # alpha = self.base_alpha * (1 - 0.5  * adjust_ratio)
        # beta = self.base_beta * (1 + 1.0 * adjust_ratio)
#7633
        # alpha = self.base_alpha * (1 - 1.0 * adjust_ratio)
        # beta = self.base_beta * (1 + 0.5 * adjust_ratio)
#7654

        alpha = torch.clamp(alpha, 0.5, 2.0)
        #
        # beta = torch.clamp(beta, 6.0, 10.0)
        #res324    res50
        # beta = torch.clamp(beta, 1.0, 4.0)
        #res56
        beta = torch.clamp(beta, 5.0, 8.0)
        #vgg wrn
        # beta = torch.clamp(beta, 4.0, 7.0)

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
        # loss = min(epoch / self.warmup, 1.0) * loss
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


# 定义DKDloss类
class DKDloss(nn.Module):
    def __init__(self,warmup=20):
        super(DKDloss, self).__init__()
        self.warmup = warmup

    # 前向传播
    def forward(self, logits_student, logits_teacher, target,epoch, alpha=1.0, beta=8.0, temperature=3.0):
        # 获取ground truth mask和非ground truth mask
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)

        # 对学生和教师logits进行softmax
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

        # 拼接mask
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

        # 计算tckd_loss
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='sum')
                * (temperature ** 2) / target.shape[0]
        )

        # 计算nckd_loss
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
                * (temperature ** 2) / target.shape[0]
        )

        # 计算总的loss
        loss = alpha * tckd_loss + beta * nckd_loss

        return loss






class DKDloss1(nn.Module):
    def __init__(self,):
        super(DKDloss1, self).__init__()


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


class ConsensusEnhancedKD(nn.Module):
    def __init__(self, temp=3.0, gamma=2.0):
        super().__init__()
        self.temp = temp  # 温度系数
        self.gamma = gamma  # 共识增强强度

    def forward(self, student_logits, teacher_logits, targets):
        # 输入尺寸校验
        assert student_logits.shape == teacher_logits.shape

        # 生成目标类掩码
        batch_size, num_classes = student_logits.shape
        gt_mask = torch.zeros_like(student_logits).scatter(1, targets.unsqueeze(1), 1).bool()

        # 温度缩放概率
        s_probs = F.softmax(student_logits / self.temp, dim=1)
        t_probs = F.softmax(teacher_logits / self.temp, dim=1)

        # 共识权重计算（仅非目标类）
        with torch.no_grad():
            # 计算非目标类一致性
            non_target_mask = ~gt_mask
            agreement = torch.sqrt(s_probs * t_probs + 1e-8)
            weights = torch.ones_like(s_probs)
            weights[non_target_mask] = (agreement[non_target_mask] ** self.gamma)

        # 完整的KL散度计算（带权重）
        loss = (t_probs * (torch.log(t_probs + 1e-8) - torch.log(s_probs + 1e-8)) * weights).sum(dim=1)

        # 温度缩放补偿 & 批平均
        return (loss * (self.temp ** 2)).mean()
class DKDloss2(nn.Module):
    def __init__(self):
        super(DKDloss2, self).__init__()

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
    将DKDloss1的梯度投影到DynamicDKD32的主方向
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
    train_loss_w = 0.
    train_loss_crd = 0.
    train_loss_a = 0.
    train_loss_k = 0.
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
    criterion_w = criterion_list[2]
    criterion_crd = criterion_list[3]
    criterion_a = criterion_list[4]
    criterion_k = criterion_list[5]


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
        loss_w = torch.tensor(0.).cuda()

        loss_crd = torch.tensor(0.).cuda()
        loss_a = torch.tensor(0.).cuda()
        loss_k = torch.tensor(0.).cuda()

        for i in range(len(ss_logits)):
            loss_cls = loss_cls + criterion_cls(ss_logits[i], target)

        loss_cls = loss_cls + criterion_cls(logits, target)

        loss_div = loss_div + criterion_div(logits, t_logits.detach(), target,epoch)
        # loss_div = loss_div + criterion_div(logits, t_logits.detach())
        #
        for i in range(len(t_ss_logits)):
            loss_w = loss_w + criterion_w(ss_logits[i], t_ss_logits[i].detach())




        # for i in range(len(ss_logits)):
        #     loss_w += criterion_w(ss_logits[i], t_logits.detach(), target)
        # loss_w = loss_w + criterion_w(logits, t_logits.detach())


        #

        #

        for i in range(len(ss_logits) - 1):
            for j in range(i, len(ss_logits)):
                loss_crd += criterion_crd(ss_logits[j], ss_logits[i], target)

        for i in range(len(ss_logits)):
            loss_crd += criterion_crd(logits, ss_logits[i], target)
        # for i in range(len(ss_logits) - 1):
        #         loss_crd += criterion_crd(ss_logits[i+1], ss_logits[i], target)
        # loss_crd += criterion_crd(logits, ss_logits[-1], target)



        # #
        # for i in range(len(ss_logits) - 1):
        #     for j in range(i, len(ss_logits)):
        #         loss_a += criterion_crd(ss_logits[j], ss_logits[i])
        # for i in range(len(ss_logits)):
        #     loss_a += criterion_crd(logits, ss_logits[i])



#76.97


        loss = loss_cls+loss_div+ loss_crd
        # loss = loss_cls + loss_crd




        loss.backward()


        optimizer.step()

        train_loss += loss.item() / len(trainloader)
        train_loss_cls += loss_cls.item() / len(trainloader)
        train_loss_div += loss_div.item() / len(trainloader)
        train_loss_w += loss_w.item() / len(trainloader)
        train_loss_crd += loss_crd.item() / len(trainloader)
        train_loss_a += loss_a.item() / len(trainloader)
        train_loss_k += loss_k.item() / len(trainloader)

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
            # integrated_logits = torch.zeros_like(ss_logits[0])
            # for i in range(len(ss_logits)):
            #     integrated_logits += ss_logits[i]
            #
            # # 将主干网络的 logits 也加入加权
            #
            # integrated_logits += logits
            #
            # integrated_logits = integrated_logits / (len(ss_logits) + 1)
            #
            # top1, top5 = correct_num(integrated_logits, target, topk=(1, 5))
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
    # criterion_div = DistillKL(args.kd_T)
    #
    criterion_div = DynamicDKD32()
    #76.6
    # criterion_div = DKDloss()
    #76.24
    # criterion_div = DynamicDKD33()

    # criterion_w =DistillKL(args.kd_T)
    criterion_k = DistillKL(args.kd_T)
    criterion_w =DistillKL(args.kd_T)
    #77.33
    #
    # criterion_w = DKDloss()
    #76.86
    # criterion_w = DistillKL(args.kd_T)
    #75.94

    # criterion_crd =DistillKL(args.kd_T)
    #76.75
    criterion_crd =DKDloss1()
    #77.3
    # criterion_crd = DKDloss()
    #
    criterion_a =DistillKL(args.kd_T)



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
        criterion_list.append(criterion_w)
        criterion_list.append(criterion_crd)
        criterion_list.append(criterion_a)
        criterion_list.append(criterion_k)
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
