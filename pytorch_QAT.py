import torch
import torchvision
import torchvision.transforms as transforms
import torch.quantization
import random
import numpy as np
import torch.nn as nn
import copy
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
import torch.nn.functional as F
from torchvision import datasets
from tqdm.auto import tqdm

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.relu1 = nn.ReLU()
    
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
       
    def forward(self, x):
        x = self.quant(x)
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))

        x = x.contiguous().view(x.shape[0], -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x
    
def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: LambdaLR,
  callbacks = None
) -> None:
  model.train()

  for inputs, targets in tqdm(dataloader, desc='train', leave=False):
    # Move the data from CPU to GPU
    # inputs = inputs.to('mps')
    # targets = targets.to('mps')

    # Reset the gradients (from the last iteration)
    optimizer.zero_grad()

    # Forward inference
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward propagation
    loss.backward()

    # Update optimizer and LR scheduler
    optimizer.step()
    scheduler.step()

    if callbacks is not None:
        for callback in callbacks:
            callback()
    
@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  extra_preprocess = None
) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
    # Move the data from CPU to GPU
    # inputs = inputs.to('mps')
    if extra_preprocess is not None:
        for preprocess in extra_preprocess:
            inputs = preprocess(inputs)

    # targets = targets.to('mps')

    # Inference
    outputs = model(inputs)

    # Convert logits to class indices
    outputs = outputs.argmax(dim=1)

    # Update metrics
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item()

def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet()#.to(device=device)
print(model)

exit(0)

# 加载模型的状态字典
checkpoint = torch.load('learning_project/model.pt')
# checkpoint = torch.load('./model.pt')

# 加载状态字典到模型
model.load_state_dict(checkpoint)
fp32_model = copy.deepcopy(model)

# 设置归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 获取数据集
train_dataset = datasets.MNIST(root='../ch02/data/mnist', train=True, download=True, transform=transform)  
test_dataset = datasets.MNIST(root='../ch02/data/mnist', train=False, download=True, transform=transform)  # train=True训练集，=False测试集
# train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)  
# test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)  # train=True训练集，=False测试集
# 设置DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

fp32_model_accuracy = evaluate(fp32_model, test_loader)
fp32_model_size = get_model_size(fp32_model)
print(f"fp32 model has accuracy={fp32_model_accuracy:.5f}%")
print(f"fp32 model has size={fp32_model_size/MiB:.2f} MiB")

backend = torch.backends.quantized.supported_engines #运行时可以使用这句命令检查自己支持的后端
print(backend)

# 将模型量化为QAT模型
quant_model = copy.deepcopy(model)
quant_model.qconfig = torch.quantization.get_default_qat_qconfig(backend[-1])
torch.quantization.prepare_qat(quant_model, inplace=True)
print(quant_model)

exit(0)

quant_model_accuracy = evaluate(quant_model, test_loader)
print(f"quant model has accuracy={quant_model_accuracy:.5f}%")

num_finetune_epochs = 3 #   这里一般微调5～10个epoch
optimizer = torch.optim.SGD(quant_model.parameters(), lr=0.001, momentum=0.9) #学习率调低一些
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
criterion = nn.CrossEntropyLoss()
best_accuracy = 0
epoch = num_finetune_epochs
while epoch > 0:
    train(quant_model, train_loader, criterion, optimizer, scheduler)
    model_accuracy = evaluate(quant_model, test_loader)
    is_best = model_accuracy > best_accuracy
    best_accuracy = max(model_accuracy, best_accuracy)
    print(f'        Epoch {num_finetune_epochs-epoch} Accuracy {model_accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')
    epoch -= 1
    
from tqdm.auto import tqdm
quant = torch.quantization.QuantStub() 
dequant = torch.quantization.DeQuantStub()

def evaluate_quant(
  model: nn.Module,
  dataloader: DataLoader,
  extra_preprocess = None
) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0
  
  for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
    # Move the data from CPU to GPU
    # inputs = inputs.to('mps')
    if extra_preprocess is not None:
        for preprocess in extra_preprocess:
            inputs = quant(inputs)
            inputs = preprocess(inputs)

    # targets = targets.to('mps')

    # Inference
    outputs = model(inputs)
    # print(outputs)
    outputs = dequant(outputs)
    # Convert logits to class indices
    # print(outputs)
    outputs = outputs.to("cpu")
    # outputs = outputs.argmax(dim=1)
    outputs=torch.max(outputs,1)[1]

    # Update metrics
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item()


torch.quantization.convert(quant_model, inplace=True)
print(quant_model)

final_quant_model_accuracy = evaluate_quant(quant_model, test_loader)
print(f"final_quant model has accuracy={final_quant_model_accuracy:.5f}%")