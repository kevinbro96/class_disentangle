from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import copy
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

        model.eval()
        vae.eval()
        adv_inputs = attack(inputs, labels)

        vae.train()
        optimizer.zero_grad()

        all_inputs = torch.cat((inputs, adv_inputs))
        bs = all_inputs.size(0)
        mu, logvar, xi = vae(all_inputs)

        out1 = model(normalize(xi))

        model.train()
        out2 = model(normalize(all_inputs) - normalize(xi))

        # model_state_dict = copy.deepcopy(model.state_dict())
        # # for key in model_state_dict:
        # #     if 'bn' in key:
        # #         old_tensor = model_state_dict[key]
        # #         print(f' {key} : {old_tensor}')
        if epoch < 100:
            re = args.re[0]
        elif epoch < 200:
            re = args.re[1]
        else:
            re = args.re[2]

        l1 = F.mse_loss(normalize(xi), normalize(all_inputs)) \
            + F.mse_loss(normalize(adv_inputs) - normalize(inputs)+normalize(xi[0:inputs.size(0)]), normalize(xi[inputs.size(0):]))
        entropy = (F.softmax(out1, dim=1) * F.log_softmax(out1, dim=1)).sum(dim=1).mean()
        cross_entropy = F.cross_entropy(out2, torch.cat((labels, labels)))
        l2 = cross_entropy + entropy
        l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l3 /= bs * 3 * args.dim
        loss = re * l1 + args.ce * l2 + args.kl * l3
        loss.backward()
        optimizer.step()

        prec1, prec5, correct, pred = accuracy(out2[0:inputs.size(0)].data, labels.data, topk=(1, 5))
        prec1_adv, prec5, correct, pred = accuracy(out2[inputs.size(0):].data, labels.data, topk=(1, 5))
        loss_avg.update(loss.data.item(), bs)
        loss_rec.update(l1.data.item(), bs)
        loss_ce.update(cross_entropy.data.item(), bs)
        loss_entropy.update(entropy.data.item(), bs)
        loss_kl.update(l3.data.item(), bs)
        top1.update(prec1.item(), bs/2)
        adv_top1.update(prec1_adv.item(), bs/2)

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
    model = Wide_ResNet(28, 10, 0.3, 10, args.mom)
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
            evaluation_class_dependent.evaluate_against_attacks(model, vae, validation_attacks, testloader, wandb = wandb, num_batches = 10 )
            reconst_images(epoch=epoch, batch_size=64, batch_num=2, dataloader=testloader, model=vae)
            checkpoint_fname = os.path.join(args.save_dir, f'{epoch:04d}.ckpt.pth')
            checkpoint_model = model
            if isinstance(checkpoint_model, nn.DataParallel):
                checkpoint_model = checkpoint_model.module
            checkpoint_vae = vae
            if isinstance(checkpoint_vae, nn.DataParallel):
                checkpoint_vae = checkpoint_vae.module
            state = {
                'model': checkpoint_model.state_dict(),
                'vae': checkpoint_vae.state_dict()
            }
            torch.save(state, checkpoint_fname)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    print('\n=> Begin to Validation Epoch #%d' % (epoch))
    model.eval()
    vae.eval()
    evaluation_class_dependent.evaluate_against_attacks(model, vae, validation_attacks, testloader, wandb=wandb,
                                                        num_batches=10)
    reconst_images(epoch=epoch, batch_size=64, batch_num=2, dataloader=testloader, model=vae)

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
    parser.add_argument('--mom', default=0.1, type=float, help='bn momentum')
    args = parser.parse_args()
    wandb.init(config=args, name=args.save_dir.replace("results/", ''))
    set_random_seed(args.seed)
    main(args)
