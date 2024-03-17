import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys

class LinfFgsmAttack(object):
    def __init__(self, model, epsilon):
        self.model = model
        self.epsilon = epsilon

    def perturb(self, x_natural, y, attack_iter):

        self.model.eval()
  
        x = x_natural.detach()

        x.requires_grad_()
        with torch.enable_grad():
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x + self.epsilon * torch.sign(grad.detach())
        x = torch.clamp(x, 0, 1).detach()

        return x

class LinfPGDAttack(object):
    def __init__(self, model, epsilon, alpha):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

    def perturb(self, x_natural, y, k_step):

        self.model.eval()
  
        x = x_natural.detach()
        if k_step > 0 :
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(k_step):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1).detach()

        return x





class LinfPeerAttack(object):
    def __init__(self, model, epsilon, alpha):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

    def perturb(self, teacher_logits, x_natural, y, k_step):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
        self.model.eval()
        batch_size = len(x_natural)

        if k_step > 0 :
            x = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for i in range(k_step):
            x.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(self.model(x), dim=1),
                                       F.softmax(teacher_logits, dim=1))
                loss_kl = torch.sum(loss_kl)
            grad = torch.autograd.grad(loss_kl, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1).detach()

        return x

    
        
    

def evaluate_adversary(net, adversary, test_loader, attack_iter, args, device):

    total = 0
    adv_correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            if args.debug_mode:
                if batch_idx == 2 :
                    break

            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            adv = adversary.perturb(inputs, targets, attack_iter)

            adv_outputs = net(adv)

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

    robust_acc = 100. * adv_correct / total

    return robust_acc
