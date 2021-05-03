from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys

sys.path.append('.')
from evaluation import evaluation_class_dependent
from networks.vae import *
from networks.resnet import resnet50
from utils.set import *
from advex.attacks import *
from utils.normalize import *
normalize = CIFARNORMALIZE(32)
innormalize = CIFARINNORMALIZE(32)

def reconst_images(epoch=2, batch_size=64, batch_num=2, dataloader=None, model=None):
    cifar10_dataloader = dataloader

    model.eval()
    acc_avg = AverageMeter()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(cifar10_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                bs = X.size(0)
                _, _, xi = model(X)

                norm = torch.norm(torch.abs(normalize(X).view(X.size(0), -1)), p=2, dim=1)
                acc_xi = 1 - F.mse_loss(torch.div(normalize(xi), norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                        torch.div(normalize(X), norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                        reduction='sum') / X.size(0)
                acc_avg.update(acc_xi.data.item(), bs)
                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_Xi = torchvision.utils.make_grid(xi[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_Xi.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Xi)]}, commit=False)
                grid_X_Xi = torchvision.utils.make_grid((X[:batch_size] - xi[:batch_size]).data, nrow=8, padding=2,
                                                        normalize=True)
                wandb.log({"_Batch_{batch}_X-Xi.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X_Xi)]}, commit=False)
                wandb.log({'acc': acc_avg.avg}, commit=False)
    print("\nreconstruction complete!\t\tAcc: %.4f " % (acc_avg.avg))

def train(args, epoch, model, vae, optimizer, trainloader, attack):
    model.train()
    model.training = True
    vae.train()

    loss_avg = AverageMeter()
    loss_rec = AverageMeter()
    loss_ce = AverageMeter()
    loss_entropy = AverageMeter()
    loss_kl = AverageMeter()
    top1 = AverageMeter()
    adv_top1 = AverageMeter()

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda().view(-1, )
        inputs, labels = Variable(inputs), Variable(labels)
        bs = inputs.size(0)
        optimizer.zero_grad()
        adv_inputs = attack(inputs, labels)

        mu, logvar, xi = vae(inputs)
        adv_mu, adv_logvar, adv_xi = vae(adv_inputs)

        out = model(torch.cat((normalize(xi), normalize(inputs)-normalize(xi)), dim=0))
        out1 = out[0:inputs.size(0)]
        out2 = out[inputs.size(0):]

        adv_out = model(torch.cat((normalize(adv_xi), normalize(adv_inputs)-normalize(adv_xi)), dim=0))
        adv_out1 = adv_out[0:inputs.size(0)]
        adv_out2 = adv_out[inputs.size(0):]

        if epoch < 100:
            re = args.re[0]
        elif epoch < 200:
            re = args.re[1]
        else:
            re = args.re[2]

        l1 = F.mse_loss(normalize(xi), normalize(inputs)) \
            + F.mse_loss(normalize(adv_xi), normalize(adv_inputs)) \
            + F.mse_loss(normalize(adv_inputs) - normalize(inputs)+normalize(xi), normalize(adv_xi))
        entropy = (F.softmax(out1, dim=1) * F.log_softmax(out1, dim=1)).sum(dim=1).mean() \
                + (F.softmax(adv_out1, dim=1) * F.log_softmax(adv_out1, dim=1)).sum(dim=1).mean()
        cross_entropy = F.cross_entropy(out2, labels) + F.cross_entropy(adv_out2, labels)
        l2 = cross_entropy + entropy
        l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())\
            -0.5 * torch.sum(1 + adv_logvar - adv_mu.pow(2) - adv_logvar.exp())
        l3 /= bs * 3 * args.dim
        loss = re * l1 + args.ce * l2 + args.kl * l3
        loss.backward()
        optimizer.step()

        prec1, prec5, correct, pred = accuracy(out2.data, labels.data, topk=(1, 5))
        prec1_adv, prec5, correct, pred = accuracy(adv_out2.data, labels.data, topk=(1, 5))
        loss_avg.update(loss.data.item(), bs)
        loss_rec.update(l1.data.item(), bs)
        loss_ce.update(cross_entropy.data.item(), bs)
        loss_entropy.update(entropy.data.item(), bs)
        loss_kl.update(l3.data.item(), bs)
        top1.update(prec1.item(), bs)
        adv_top1.update(prec1_adv.item(), bs)

        n_iter = (epoch - 1) * len(trainloader) + batch_idx
        wandb.log({'loss': loss_avg.avg, \
                   'loss_rec': loss_rec.avg, \
                   'loss_ce': loss_ce.avg, \
                   'loss_entropy': loss_entropy.avg, \
                   'loss_kl': loss_kl.avg, \
                   'orig_acc': top1.avg,
                   'adv_acc': adv_top1.avg,
                   're_weight': re,
                   'lr':optimizer.param_groups[0]['lr']}, step=n_iter)
        if (batch_idx + 1) % 30 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Loss_rec: %.4f Loss_ce: %.4f Loss_entropy: %.4f Loss_kl: %.4f Orig_Acc@1: %.3f Adv_Acc@1: %.3f%%'
                % (epoch, args.epochs, batch_idx + 1,
                   len(trainloader), loss_avg.avg, loss_rec.avg, loss_ce.avg, loss_entropy.avg, loss_kl.avg, top1.avg, adv_top1.avg))

def main(args):
    learning_rate = 1.e-3
    learning_rate_min = 2.e-4
    CNN_embed_dim = args.dim
    feature_dim = args.fdim
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    start_epoch = 1
    batch_size = args.batch_size
    optim_type = args.optim

    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        num_classes = 10
    elif (args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Model
    print('\n[Phase 2] : Model setup')
    vae = CVAE_cifar(d=feature_dim, z=CNN_embed_dim)
    #model = resnet50(pretrained=False)
    model = Wide_ResNet(28, 10, 0.3, 10)
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        vae.cuda()
        vae = torch.nn.DataParallel(vae, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    attack = eval(args.attack)
    validation_attacks = [
        NoAttack(),
        DeltaAttack(model, vae, num_iterations=100, norm='linf', eps_max=8 / 255),
        DeltaAttack(model, vae, num_iterations=100, norm='l2', eps_max=1.0),
        XAttack(model, vae, num_iterations=100, norm='linf', eps_max=8 / 255),
        XAttack(model, vae, num_iterations=100, norm='l2', eps_max=1.0)
        ]

    optimizer = AdamW([
        {'params': model.parameters()},
        {'params': vae.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    if args.optim == 'consine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,
                                                        eta_min=learning_rate_min)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=0.1, last_epoch=-1)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        start_time = time.time()
        train(args, epoch, model, vae, optimizer, trainloader, attack)
        scheduler.step()
        if epoch % 10 == 1:
            print('\n=> Begin to Validation Epoch #%d' % (epoch))
            model.eval()
            vae.eval()
            evaluation_class_dependent.evaluate_against_attacks(model, vae, validation_attacks, testloader, wandb = wandb, num_batches = 100 )
            reconst_images(epoch=epoch, batch_size=64, batch_num=2, dataloader=testloader, model=vae)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    wandb.finish()
    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--attack', type=str, default='DeltaAttack(model, vae, num_iterations=10,eps_max=8 / 255)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5/', type=str, help='save_dir')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--optim', default='consine', type=str, help='optimizer')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--alpha', default=2.0, type=float, help='mix up')
    parser.add_argument('--epochs', default=300, type=int, help='training_epochs')
    parser.add_argument('--dim', default=2048, type=int, help='CNN_embed_dim')
    parser.add_argument('--fdim', default=32, type=int, help='featdim')
    parser.add_argument('--re', nargs='+', type=int)
    parser.add_argument('--step', nargs='+', type=int)
    parser.add_argument('--kl', default=1.0, type=float, help='kl weight')
    parser.add_argument('--ce', default=1.0, type=float, help='cross entropy weight')
    args = parser.parse_args()
    wandb.init(config=args, name=args.save_dir.replace("results/", ''))
    set_random_seed(args.seed)
    main(args)
