from __future__ import print_function
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append('.')
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys
import wandb

from networks.vae import *
from utils.set import *
sys.path.append('.')

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--lr_schedule', type=str, required=False,
                    help='comma-separated list of epochs when learning '
                         'rate should drop')
parser.add_argument('--num_epochs', type=int, required=False,
                    help='number of epochs trained')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--seed', default=666, type=int, help='seed')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/mnist]')
parser.add_argument('--log_dir', type=str, default='../../deep_Mahalanobis_detector/data/wide_resnet_sgd_epoch300')
parser.add_argument('--batch_size', type=int, default=128,
                    help='number of examples/minibatch')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')

args = parser.parse_args()
wandb.init(config=args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()

log_dir = os.path.join(args.log_dir)
if os.path.exists(log_dir):
    time.sleep(5)
else:
    os.makedirs(log_dir)

print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

print("| Preparing CIFAR-10 dataset...")
sys.stdout.write("| ")
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
num_classes = 10

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('\n[Phase 2] : Model setup')
model = Wide_ResNet(28, 10, 0.3, 10)

if use_cuda:
    model.cuda()
    model = nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if args.lr is None:
    args.lr = 1e-1
if args.lr_schedule is None:
    args.lr_schedule = '150,225'
if args.num_epochs is None:
    args.num_epochs = 300
lr_drop_epochs = [int(epoch_str) for epoch_str in
                      args.lr_schedule.split(',')]

learning_rate = 1.e-3
learning_rate_min = 2.e-4

optimizer = AdamW([
    {'params': model.parameters()}
], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,
                                                       eta_min=learning_rate_min)
# Training


print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))


def run_iter(
        inputs: torch.Tensor,
        labels: torch.Tensor,
        iteration: int
):
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    optimizer.zero_grad()
    logits = model(inputs)

    # CONSTRUCT LOSS
    loss = F.cross_entropy(logits, labels, reduction='none')
    loss = loss.mean()
    loss.backward()
    optimizer.step()

    # LOGGING
    _, predicted = torch.max(logits.data, 1)
    correct = predicted.eq(labels.data).cpu().sum()
    accuracy = correct/inputs.size(0)
    wandb.log({'loss': loss.item()}, step=iteration)
    wandb.log({'accuracy': accuracy.item()}, step=iteration)
    print(f'ITER {iteration:06d}',
          f'accuracy: {accuracy.item() * 100:5.1f}%',
          f'loss: {loss.item():.2f}',
          sep='\t')

def test(epoch, test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\t Acc@1: %.2f%%" %(epoch, acc))

start_epoch, iteration = 0, 0
for epoch in range(start_epoch, args.num_epochs):
    model.train()  # now we set the model to train mode
    print(f'START EPOCH {epoch:04d} (lr={learning_rate:.0e})')
    for batch_index, (inputs, labels) in enumerate(train_loader):

        run_iter(inputs, labels, iteration)
        iteration += 1
    print(f'END EPOCH {epoch:04d}')
    scheduler.step()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if epoch % 10 == 9:
        # VALIDATION
        print('BEGIN VALIDATION')
        model.eval()
        test(epoch, test_loader, model)
        checkpoint_fname = os.path.join(log_dir, f'{epoch:04d}.ckpt.pth')
        torch.save(model, checkpoint_fname)

print('\n[Phase 4] : Testing model')
model.eval()
test(epoch, test_loader, model)
checkpoint_fname = os.path.join(log_dir, f'{epoch:04d}.ckpt.pth')
torch.save(model, checkpoint_fname)
