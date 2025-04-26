import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # 原论文中这里加入了一个 WarmUP
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

# 设置随机数种子, 从而可以复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
device = "cpu"

T = 4               # temperature : 知识蒸馏中的温度
ALPHA = 1.0         # alpha : TCKD 部分的loss weight
BETA = 2.0          # beta : NCKD 部分的loss weight
LOSS_CE = 1.0       # loss_ce : 交叉熵的loss weight

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class LeNetHalfChannel(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetHalfChannel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)   
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=3 * 12 * 12, out_features=10)   

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        
        return x
    

teacher_net = LeNet().to(device=device)
student_net = LeNetHalfChannel().to(device=device)

teacher_net.load_state_dict(torch.load('learning_project/model.pt'))

# 设置归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 获取数据集
# 这里直接读取 ch02 中下载好的数据
train_dataset = datasets.MNIST(root='../ch02/data/mnist/', train=True, download=False, transform=transform)  
test_dataset = datasets.MNIST(root='../ch02/data/mnist/', train=False, download=False, transform=transform)  # train=True训练集，=False测试集

# 设置DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

lr = 0.01
momentum = 0.5
num_epoch = 5
optimizer = torch.optim.SGD(student_net.parameters(),  lr=lr, momentum=momentum)  # lr学习率，momentum冲量

# 分别定义训练集和测试集上的最佳Acc, 使用 global 修饰为全局变量, 然后再训练期间更新
best_train_acc = 0
best_test_acc = 0

from tqdm import tqdm

def train(epoch):
    global best_train_acc

    # 设置学生模型为训练模式
    student_net.train()

    print('\nEpoch: %d' % epoch)

    train_loss = 0
    correct = 0
    total = 0

    # 使用 tqdm 包装 trainloader 以显示进度条
    with tqdm(train_loader, desc=f"Training Epoch {epoch}", total=len(train_loader)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits_student = student_net(inputs)
            with torch.no_grad():
                logits_teacher = teacher_net(inputs)

            # 硬损失
            ce_loss = nn.CrossEntropyLoss()(logits_student, targets)
            # 软损失
            kd_loss = dkd_loss(logits_student, logits_teacher, targets, ALPHA, BETA, T)
            total_loss = ALPHA * ce_loss + BETA * kd_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = logits_student.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 使用 set_postfix 更新进度条的后缀
            pbar.set_postfix(loss=train_loss / (batch_idx + 1), acc=f"{100. * correct / total:.1f}%")

    # 如果当前训练集上的准确率高于 best_test_acc，则更新 best_test_acc
    acc = 100 * correct / total
    if acc > best_train_acc:
        best_train_acc = acc
        
def test(net, epoch):
    global best_test_acc
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        # 使用 tqdm 包装 testloader 以显示进度条
        with tqdm(test_loader, desc=f"Testing Epoch {epoch}", total=len(test_loader)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):

                inputs, targets = inputs.to(device), targets.to(device)
                logits_student = net(inputs)

                loss = nn.CrossEntropyLoss()(logits_student, targets)

                test_loss += loss.item()
                _, predicted = logits_student.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 在 tqdm 进度条的后缀中显示当前损失和准确率
                pbar.set_postfix(loss=test_loss / (batch_idx + 1), acc=f"{100. * correct / total:.1f}%")

        # 计算当前测试集上的准确率
        acc = 100. * correct / total

        # 如果当前测试集上的准确率高于 best_test_acc，则更新 best_test_acc
        # 并且将学生模型保存下来
        if acc > best_test_acc:
            print('Saving..')
            torch.save(student_net, 'checkpoints/distillation_dkd.pt')
            best_test_acc = acc

for epoch in range(1, num_epoch + 1) :
    train(epoch)
    test(student_net, epoch)
    
print('best_Train_Acc = ', best_train_acc)
print('best_Test_Acc = ', best_test_acc)

