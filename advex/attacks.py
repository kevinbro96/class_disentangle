import torch
from torch import nn
from torch.nn import functional as F

from utils.normalize import *
normalize = CIFARNORMALIZE(32)
innormalize = CIFARINNORMALIZE(32)

class NoAttack(nn.Module):
    """
    Attack that does nothing.
    """

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels):
        return inputs

class XAttack(nn.Module):
    def __init__(self, model,  vae, eps_max=8/255, step_size=None,  num_iterations=7, norm='linf', rand_init=True, scale_each=False, loss='ce'):
        super().__init__()
        self.nb_its = num_iterations
        self.eps_max = eps_max
        if step_size is None:
            step_size = eps_max / (self.nb_its ** 0.5)
        self.step_size = step_size

        self.norm = norm
        self.rand_init = rand_init
        self.scale_each = scale_each
        self.loss = loss

        if self.loss == 'margin':
            self.criterion = MarginLoss(kappa=1000)
        else:
            self.criterion = nn.CrossEntropyLoss().cuda()
        self.model = model
        self.vae = vae

    def _init(self, shape, eps):
        if self.rand_init:
            if self.norm == 'linf':
                init = torch.rand(shape, dtype=torch.float32, device='cuda') * 2 - 1
            elif self.norm == 'l2':
                init = torch.randn(shape, dtype=torch.float32, device='cuda')
                init_norm = torch.norm(init.view(init.size()[0], -1), 2.0, dim=1)
                normalized_init = init / init_norm[:, None, None, None]
                dim = init.size()[1] * init.size()[2] * init.size()[3]
                rand_norms = torch.pow(torch.rand(init.size()[0], dtype=torch.float32, device='cuda'), 1/dim)
                init = normalized_init * rand_norms[:, None, None, None]
            else:
                raise NotImplementedError
            init = eps[:, None, None, None] * init
            init.requires_grad_()
            return init
        else:
            return torch.zeros(shape, requires_grad=True, device='cuda')

    def forward(self, img, labels):

        base_eps = self.eps_max * torch.ones(img.size()[0], device='cuda')
        step_size = self.step_size * torch.ones(img.size()[0], device='cuda')

        img = img.detach()
        img.requires_grad = True
        delta = self._init(img.size(), base_eps)

        s = self.model(normalize(img + delta))
        if self.norm == 'l2':
            l2_max = base_eps
        for it in range(self.nb_its):
            loss = self.criterion(s, labels)

            if self.loss == 'margin':
                loss.sum().backward()
            else:
                loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            grad = delta.grad.data

            if self.norm == 'linf':
                grad_sign = grad.sign()
                delta.data = delta.data + step_size[:, None, None, None] * grad_sign
                delta.data = torch.max(torch.min(delta.data, base_eps[:, None, None, None]), -base_eps[:, None, None, None])
                delta.data = torch.clamp(img.data + delta.data, 0., 1.) - img.data
            elif self.norm == 'l2':
                batch_size = delta.data.size()[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), 2.0, dim=1)
                normalized_grad = grad / grad_norm[:, None, None, None]
                delta.data = delta.data + step_size[:, None, None, None]   * normalized_grad
                l2_delta = torch.norm(delta.data.view(batch_size, -1), 2.0, dim=1)
                # Check for numerical instability
                proj_scale = torch.min(torch.ones_like(l2_delta, device='cuda'), l2_max / l2_delta)
                delta.data *= proj_scale[:, None, None, None]
                delta.data = torch.clamp(img.data + delta.data, 0., 1.) - img.data
            else:
                raise NotImplementedError

            if it != self.nb_its - 1:
                s = self.model(normalize(img + delta))
                delta.grad.data.zero_()

        delta.data[torch.isnan(delta.data)] = 0
        adv_sample = img + delta
        return torch.clamp(adv_sample.detach(), 0, 1)

class DeltaAttack(nn.Module):
    def __init__(self, model,  vae, eps_max=8/255, step_size=None,  num_iterations=7, norm='linf', rand_init=True, scale_each=False, loss='ce'):
        super().__init__()
        self.nb_its = num_iterations
        self.eps_max = eps_max
        if step_size is None:
            step_size = eps_max / (self.nb_its ** 0.5)
        self.step_size = step_size

        self.norm = norm
        self.rand_init = rand_init
        self.scale_each = scale_each
        self.loss = loss

        if self.loss == 'margin':
            self.criterion = MarginLoss(kappa=1000)
        else:
            self.criterion = nn.CrossEntropyLoss().cuda()
        self.model = model
        self.vae = vae

    def _init(self, shape, eps):
        if self.rand_init:
            if self.norm == 'linf':
                init = torch.rand(shape, dtype=torch.float32, device='cuda') * 2 - 1
            elif self.norm == 'l2':
                init = torch.randn(shape, dtype=torch.float32, device='cuda')
                init_norm = torch.norm(init.view(init.size()[0], -1), 2.0, dim=1)
                normalized_init = init / init_norm[:, None, None, None]
                dim = init.size()[1] * init.size()[2] * init.size()[3]
                rand_norms = torch.pow(torch.rand(init.size()[0], dtype=torch.float32, device='cuda'), 1/dim)
                init = normalized_init * rand_norms[:, None, None, None]
            else:
                raise NotImplementedError
            init = eps[:, None, None, None] * init
            init.requires_grad_()
            return init
        else:
            return torch.zeros(shape, requires_grad=True, device='cuda')

    def forward(self, img, labels):

        base_eps = self.eps_max * torch.ones(img.size()[0], device='cuda')
        step_size = self.step_size * torch.ones(img.size()[0], device='cuda')

        img = img.detach()
        img.requires_grad = True
        delta = self._init(img.size(), base_eps)

        _, _, delta_rec = self.vae(img+delta)
        s = self.model(normalize(img + delta) - normalize(delta_rec))
        if self.norm == 'l2':
            l2_max = base_eps
        for it in range(self.nb_its):
            loss = self.criterion(s, labels)

            if self.loss == 'margin':
                loss.sum().backward()
            else:
                loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            grad = delta.grad.data

            if self.norm == 'linf':
                grad_sign = grad.sign()
                delta.data = delta.data + step_size[:, None, None, None] * grad_sign
                delta.data = torch.max(torch.min(delta.data, base_eps[:, None, None, None]), -base_eps[:, None, None, None])
                delta.data = torch.clamp(img.data + delta.data, 0., 1.) - img.data
            elif self.norm == 'l2':
                batch_size = delta.data.size()[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), 2.0, dim=1)
                normalized_grad = grad / grad_norm[:, None, None, None]
                delta.data = delta.data + step_size[:, None, None, None]   * normalized_grad
                l2_delta = torch.norm(delta.data.view(batch_size, -1), 2.0, dim=1)
                # Check for numerical instability
                proj_scale = torch.min(torch.ones_like(l2_delta, device='cuda'), l2_max / l2_delta)
                delta.data *= proj_scale[:, None, None, None]
                delta.data = torch.clamp(img.data + delta.data, 0., 1.) - img.data
            else:
                raise NotImplementedError

            if it != self.nb_its - 1:
                _, _, delta_rec = self.vae(img + delta)
                s = self.model(normalize(img + delta) - normalize(delta_rec))
                delta.grad.data.zero_()

        delta.data[torch.isnan(delta.data)] = 0
        adv_sample = img + delta
        return torch.clamp(adv_sample.detach(), 0, 1 ) # , delta_in.mean().item(), delta_dn.mean().item()