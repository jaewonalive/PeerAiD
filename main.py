import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import json
from models import *
from utils import *
from attack_function import *
from datasets import get_dataset

from robustbench import benchmark

import sys
import time
import numpy as np

from datetime import datetime


parser = argparse.ArgumentParser(description='PeerAiD adversarial distillation')

parser.add_argument('--p_type', help='the peer model type.')
parser.add_argument('--s_type', help='the student model type.')
parser.add_argument('--kd', action='store_true', help='whether to use adversarial knowledge distillation.')
parser.add_argument('--k_train', type = int, default=10, help='pgd k step in training')
parser.add_argument('--exp_id', type = int, default=0, help='experiment id.')
parser.add_argument('--temperature', type = float, default=6, help='temperature of the distillation term in the loss of the student model.')
parser.add_argument('--s_attack_type', type=str, default='pgd', help='adversarial training method.')


parser.add_argument('--config_path', type=str, help='json path which contains the hyperparameter config.')
parser.add_argument('--start_epoch', type = int, default=0, help='The epoch at which training epoch starts.')
parser.add_argument('--total_epoch', type = int, default=300, help='total training epoch.')
parser.add_argument('--lr_student', type = float, default=0.1, help='learning rate of the student model.')
parser.add_argument('--lr_peer', type = float, default=0.1, help='learning rate of the peer model.')
parser.add_argument('--batch_size', type = int, default=128, help='batch size.')
parser.add_argument('--weight_decay', type = float, default=0.0002, help='weight decay.')


parser.add_argument('--AA', action='store_true', help='whether to perform autoattack.')
parser.add_argument('--fgsm_eval', action='store_true', help='whether to test fgsm attack after finishing training.')
parser.add_argument('--pgd_eval', action='store_true', help='whether to test pgd attack after finishing training.')
parser.add_argument('--n_examples', type = int, default=10000, help='the number of samples you will test with AutoAttack.')

parser.add_argument('--dataset', type=str, default='cifar10', help='the name of the training dataset. CIFAR-10, CIFAR-100 and TinyImageNet are available.')
parser.add_argument('--data_path', type=str, default='./data', help='dataset path.')



parser.add_argument('--swa_s', action='store_true', help='whether to use stochastic weight averaging with the student model')
parser.add_argument('--swa_s_start', type = int, default=99, help='the epoch when the stochastic weight averaging of the student model starts')

parser.add_argument('--save_path', type=str, default='./checkpoint/', help='the path in which checkpoint is saved.')
parser.add_argument('--save_interval', type = int, default=30, help='the interval at which checkpoint is saved.')
parser.add_argument('--json_path', type=str, default='./json_logs/', help='json path which will be used to save the training result in the json format.')
parser.add_argument('--debug_mode', action='store_true', help='this mode only uses two batches.')
parser.add_argument('--resume', action='store_true', help='whether to resume your training from checkpoints.')
parser.add_argument('--resume_s_path', type=str, help='the path in which checkpoint of the student model exists.')
parser.add_argument('--resume_t_path', type=str, help='the path in which checkpoint of the peer model exists.')
parser.add_argument('--resume_s_swa_path', type=str, help='the path in which checkpoint of the SWA student network exists.')

parser.add_argument('--re_kd_temperature', type = float, default=1, help='the temperature parameter of the distillation term in the loss of the peer model.')
parser.add_argument('--lamb1', type = float, default=1, help='lambda1 hyperparameter in the loss of the student model.')
parser.add_argument('--lamb2', type = float, default=1, help='lambda2 hyperparameter in the loss of the student model.')
parser.add_argument('--lamb3', type = float, default=1, help='lambda3 hyperparameter in the loss of the student model.')
parser.add_argument('--gamma1', type = float, default=1, help='gamma1 hyperparameter in the loss of the peer model.')
parser.add_argument('--gamma2', type = float, default=1, help='gamma2 hyperparameter in the loss of the peer model.')


args=parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

print("Start time : " , datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if args.config_path is not None:
    with open(args.config_path, "r") as config_file:
        print("Config file opened!")
        config_json = json.load(config_file)
        print(config_json)

        args.lr_student = config_json['lr_student']
        args.lr_peer = config_json['lr_peer']
        args.total_epoch = config_json['epochs']
        
        lr_decay_epochs_peer = []
        lr_decay_epochs_student = []

        for i in config_json['lr_decay_epochs_peer']:
            lr_decay_epochs_peer.append(i)
            

        for i in config_json['lr_decay_epochs_student']:
            lr_decay_epochs_student.append(i)
            

        args.batch_size = config_json['batch_size']
        args.weight_decay = config_json['weight_decay']

    args.lr_decay_epochs_peer = lr_decay_epochs_peer
    args.lr_decay_epochs_student = lr_decay_epochs_student

    print("lr decay epochs of the peer model : " ,args.lr_decay_epochs_peer)
    print("lr decay epochs of the student model : " ,args.lr_decay_epochs_student)

    if args.swa_s:
        args.swa_s_start = args.lr_decay_epochs_student[0]

        if args.debug_mode:
            args.swa_s_start = 0

        print("SWA student starts at ", args.swa_s_start)
        

    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)
    print("New directory created ! : ", args.save_path)

if args.debug_mode:
    args.n_examples = 2
    args.total_epoch = 4

learning_rate_student = args.lr_student
learning_rate_peer = args.lr_peer
epsilon = 8/255 
k_train = args.k_train

alpha = 2/255 

file_name = 'PeerAiD'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device : ", device)


kwargs = {'pin_memory': True, 'num_workers': 8}
train_dataset, test_dataset, image_size, num_classes = get_dataset(dataset=args.dataset, data_path=args.data_path)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    saving_filename = args.save_path + filename
    torch.save(state, saving_filename)



if args.kd:

    peer_net = model_builder(args.p_type, num_classes=num_classes, dataset=args.dataset)
    peer_net = peer_net.to(device)

    student_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)
    student_net = student_net.to(device)


    swa_n_student = 0
    if args.swa_s:
        student_swa_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)
        student_swa_net = student_swa_net.to(device)

else:

    swa_n_student = 0
    student_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)

    student_net = student_net.to(device)


cudnn.benchmark = True

if args.kd:
    adversary_peer = LinfPGDAttack(peer_net, epsilon, alpha)
    adversary_student = LinfPGDAttack(student_net, epsilon, alpha)

    adversary_student_training = LinfPeerAttack(student_net, epsilon, alpha)
    
    if args.swa_s:
        adversary_swa_student = LinfPGDAttack(student_swa_net, epsilon, alpha)

else: 
    adversary_student = LinfPGDAttack(student_net, epsilon, alpha)
    adversary_student_training = LinfPGDAttack(student_net, epsilon, alpha)



def test_resume(net):
    net.eval()
    benign_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

    print('\nTotal natural test accuarcy of the resumed model:', 100. * benign_correct / total)


if args.kd:
    peer_criterion = nn.CrossEntropyLoss()
    student_criterion = nn.CrossEntropyLoss() 

    peer_optimizer = optim.SGD(peer_net.parameters(), lr=learning_rate_peer, momentum=0.9, weight_decay=args.weight_decay)
    student_optimizer = optim.SGD(student_net.parameters(), lr=learning_rate_student, momentum=0.9, weight_decay=args.weight_decay)


    if args.resume:
        if os.path.isfile(args.resume_s_path):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            checkpoint = torch.load(args.resume_s_path)
            student_net.load_state_dict(checkpoint['state_dict'])
            student_optimizer.load_state_dict(checkpoint['optimizer'])
            print("The checkpoint of student net successfully loaded.")
            test_resume(student_net)

            checkpoint = torch.load(args.resume_t_path)
            peer_net.load_state_dict(checkpoint['state_dict'])
            peer_optimizer.load_state_dict(checkpoint['optimizer'])
            print("The checkpoint of peer net successfully loaded.")
            test_resume(peer_net)

            if args.swa_s and (args.resume_s_swa_path is not None):
                checkpoint = torch.load(args.resume_s_swa_path)
                student_swa_net.load_state_dict(checkpoint['state_dict'])
                print("The checkpoint of SWA student net successfully loaded.")
                test_resume(student_swa_net)


else:
    student_criterion = nn.CrossEntropyLoss()
    student_optimizer = optim.SGD(student_net.parameters(), lr=learning_rate_student, momentum=0.9, weight_decay=args.weight_decay)


    if args.resume:
        if os.path.isfile(args.resume_s_path):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            checkpoint = torch.load(args.resume_s_path)
            student_net.load_state_dict(checkpoint['state_dict'])
            student_optimizer.load_state_dict(checkpoint['optimizer'])
            print("The checkpoint of student net successfully loaded.")
            test_resume(student_net)



def train(epoch):

    if args.swa_s:
        global swa_n_student


    print('\n[ Train epoch: %d ]' % epoch)
    student_net.train()
    student_train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if args.debug_mode:
            if batch_idx == 2 :
                break

        inputs, targets = inputs.to(device), targets.to(device)
        student_optimizer.zero_grad()

        if args.s_attack_type == 'natural_training':
            adv_outputs = student_net(inputs)
        else:
            adv = adversary_student_training.perturb(inputs, targets, k_train)
            student_net.train()
            adv_outputs = student_net(adv)


        loss = student_criterion(adv_outputs, targets)
        
        loss.backward()

        student_optimizer.step()
        student_train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx == 20  :
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item()*100 / targets.size(0)))
            print('Current adversarial train loss:', loss.item())
        
        elif batch_idx % 10 == 0 :
            print('\nCurrent batch:', str(batch_idx))


    print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    print('Total adversarial train loss:', student_train_loss)


def train_kd(epoch):
    if args.swa_s:
        global swa_n_student
    
    print('\n[ Train epoch: %d ]' % epoch)
    peer_net.train()
    student_net.train()


    T = args.temperature


    peer_train_loss = 0
    student_train_loss = 0
    correct = 0
    total = 0
    correct_student = 0
    total_student = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):


        if args.debug_mode:
            if batch_idx == 2 :
                break

        inputs, targets = inputs.to(device), targets.to(device)
        
        student_optimizer.zero_grad()
        peer_optimizer.zero_grad()


        peer_net.eval()
        with torch.no_grad():
            peer_logits = peer_net(inputs)

        peer_net.train()

        adv = adversary_student_training.perturb(peer_logits, inputs, targets, k_train)


        student_net.train()
        adv.detach_()


        adv_outputs = peer_net(adv)

        student_logit_target = student_net(adv)

        loss_peer = peer_criterion(adv_outputs, targets) * args.gamma1 +  nn.KLDivLoss()(F.log_softmax(adv_outputs/args.re_kd_temperature, dim=1), F.softmax(student_logit_target/args.re_kd_temperature, dim=1)) * args.gamma2 * args.re_kd_temperature * args.re_kd_temperature

        peer_train_loss += loss_peer.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
            
        peer_outputs = adv_outputs.clone()
        peer_outputs.detach_()

        adv.detach_()

        student_outputs = student_net(adv)

        _, predicted_student = student_outputs.max(1)
        total_student += targets.size(0)
        pred_result = predicted_student.eq(targets)
        correct_student +=  pred_result.sum().item()


        inputs.detach_()


        logit_nat = student_net(inputs)

        student_net.train()



        loss_student = F.cross_entropy(student_outputs, targets) * (args.lamb1) + \
                         nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1),
                                F.softmax(peer_outputs/T, dim=1)) * (args.lamb2 * T * T) + \
                        nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1),
                                F.softmax(logit_nat/T, dim=1)) * (args.lamb3 * T * T)


        loss_total = loss_peer + loss_student
        loss_total.backward()

        peer_optimizer.step()
        student_optimizer.step()


                    
        student_train_loss += loss_student.item()
        
        if batch_idx % 20 == 0:
            print('\nCurrent batch:', str(batch_idx))

        if batch_idx == 20:
            print('Current adversarial peer train accuracy:', str(predicted.eq(targets).sum().item()*100 / targets.size(0)))
            print('Current adversarial peer train loss:', loss_peer.item())
            print('Current adversarial student train accuracy:', str(predicted_student.eq(targets).sum().item()*100 / targets.size(0)))
            print('Current adversarial student train loss:', loss_student.item())          


    print('\nTotal adversarial peer train accuarcy:', 100. * correct / total)
    print('Total adversarial peer train loss:', peer_train_loss)
    print('Total adversarial student train loss:', student_train_loss)
    print('Total adversarial student train accuarcy:', 100. * correct_student / total_student)



    if args.swa_s and epoch >= args.swa_s_start:

        # SWA student
        moving_average(student_swa_net, student_net, 1.0/(swa_n_student + 1))
        swa_n_student += 1
        bn_update(train_loader, student_swa_net)    



def test(epoch):

    global best_student_natural_acc
    global best_student_robust_acc
    global best_epoch

    global best_student_swa_natural_acc
    global best_student_swa_robust_acc
    global best_student_swa_epoch

    global best_natural_epoch
    global at_natural_best_natural_acc
    global at_natural_best_robust_acc

    global best_swa_natural_epoch
    global at_natural_best_swa_natural_acc
    global at_natural_best_swa_robust_acc

    
    if args.swa_s:
        global swa_n_student



    print('\n[ Test epoch: %d ]' % epoch)

    student_net.eval()


    student_benign_loss = 0
    student_adv_loss = 0
    student_benign_correct = 0
    student_adv_correct = 0
    student_total = 0


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            if args.debug_mode:
                if batch_idx == 2 :
                    break


            inputs, targets = inputs.to(device), targets.to(device)
            student_total += targets.size(0)

            outputs = student_net(inputs)
            loss = student_criterion(outputs, targets)
            student_benign_loss += loss.item()

            _, predicted = outputs.max(1)
            student_benign_correct += predicted.eq(targets).sum().item()


            adv = adversary_student.perturb(inputs, targets, 10)
            adv_outputs = student_net(adv)
            loss = student_criterion(adv_outputs, targets)
            student_adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            student_adv_correct += predicted.eq(targets).sum().item()


    student_robust_acc = 100. * student_adv_correct / student_total
    student_natural_acc = 100. * student_benign_correct / student_total 


    print('\nTotal test clean accuarcy of the student model :', 100. * student_benign_correct / student_total)
    print('Total adversarial test Accuarcy of the student model against PGD-10 :', 100. * student_adv_correct / student_total)
    print('Total test clean loss of the student model:', student_benign_loss)
    print('Total adversarial test loss of the student model:', student_adv_loss)


    if (student_robust_acc > best_student_robust_acc) : 
        best_epoch = epoch

        best_student_natural_acc = student_natural_acc
        best_student_robust_acc = student_robust_acc

        print('Best robust acc achieved!')

        net_state_dict = student_net.state_dict()

        save_checkpoint({
            'epoch':epoch,
            'best_epoch' : best_epoch,
            'exp_id': args.exp_id,
            'state_dict': net_state_dict,
            'student_best_natural_acc' : best_student_natural_acc,
            'student_best_robust_acc': best_student_robust_acc,
            'optimizer' : student_optimizer.state_dict(),
            'best_student_swa_natural_acc' : best_student_swa_natural_acc,
            'best_student_swa_robust_acc' : best_student_swa_robust_acc,
            'best_student_swa_epoch' : best_student_swa_epoch,
            'swa_n_student':swa_n_student
        }, filename=str(args.exp_id)+'_baseline_' + file_name + '_student_best'+'.pth.tar')

        print('Model saved!')




    if (args.s_attack_type == 'natural_training') and (student_natural_acc > at_natural_best_natural_acc) :
        best_natural_epoch = epoch

        at_natural_best_natural_acc = student_natural_acc
        at_natural_best_robust_acc = student_robust_acc

        print("Best natural acc achieved!")


        net_state_dict = student_net.state_dict()


        save_checkpoint({
            'epoch':epoch,
            'best_natural_epoch' : best_natural_epoch,
            'at_natural_best_natural_acc' : at_natural_best_natural_acc,
            'at_natural_best_robust_acc' : at_natural_best_robust_acc,
            'best_epoch' : best_epoch,
            'exp_id': args.exp_id,
            'state_dict': net_state_dict,
            'student_best_natural_acc' : best_student_natural_acc,
            'student_best_robust_acc': best_student_robust_acc,
            'optimizer' : student_optimizer.state_dict(),
            'best_student_swa_natural_acc' : best_student_swa_natural_acc,
            'best_student_swa_robust_acc' : best_student_swa_robust_acc,
            'best_student_swa_epoch' : best_student_swa_epoch,
            'swa_n_student':swa_n_student
        }, filename=str(args.exp_id)+'_baseline_' + file_name + '_student_natural_best'+'.pth.tar')

        print("With best natural acc, the model saved!")



    if (epoch == (args.total_epoch - 1)) :

        net_state_dict = student_net.state_dict()

        save_checkpoint({
            'epoch':epoch,
            'best_epoch' : best_epoch,
            'exp_id': args.exp_id,
            'state_dict': net_state_dict,
            'student_best_natural_acc' : best_student_natural_acc,
            'student_best_robust_acc': best_student_robust_acc,
            'optimizer' : student_optimizer.state_dict(),
            'best_student_swa_natural_acc' : best_student_swa_natural_acc,
            'best_student_swa_robust_acc' : best_student_swa_robust_acc,
            'best_student_swa_epoch' : best_student_swa_epoch,
            'swa_n_student':swa_n_student
        }, filename=str(args.exp_id)+'_baseline_' + file_name + '_last_epoch'+'.pth.tar')

        print('At last epoch, Model Saved!')
        


def test_kd(epoch):

    global best_epoch 
    global best_student_natural_acc 
    global best_student_robust_acc
 
    global best_student_swa_natural_acc 
    global best_student_swa_robust_acc 
    global best_student_swa_epoch


    if args.swa_s:
        global swa_n_student


    print('\n[ Test epoch: %d ]' % epoch)
    student_net.eval()



    peer_net.eval()


    student_benign_loss = 0
    student_adv_loss = 0
    student_benign_correct = 0
    student_adv_correct = 0
    student_total = 0



    if args.swa_s and epoch >= args.swa_s_start:
        student_swa_net.eval()

        student_swa_benign_loss = 0
        student_swa_adv_loss = 0
        student_swa_benign_correct = 0
        student_swa_adv_correct = 0
        student_swa_total = 0


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            if args.debug_mode :
                if batch_idx == 2 :
                    break


            inputs, targets = inputs.to(device), targets.to(device)
            student_total += targets.size(0)

            outputs = student_net(inputs)


            loss = student_criterion(outputs, targets)
            student_benign_loss += loss.item()

            _, predicted = outputs.max(1)
            student_benign_correct += predicted.eq(targets).sum().item()


            adv = adversary_student.perturb(inputs, targets, 10)
            adv_outputs = student_net(adv)

            loss = student_criterion(adv_outputs, targets)
            student_adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            student_adv_correct += predicted.eq(targets).sum().item()

            


    student_robust_acc = 100. * student_adv_correct / student_total
    student_natural_acc = 100. * student_benign_correct / student_total

    print('\nTotal test clean accuarcy of the student model :', student_natural_acc)
    print('Total adversarial test Accuarcy of the student model against PGD-10 :', student_robust_acc)
    print('Total test clean loss of the student model:', student_benign_loss)
    print('Total adversarial test loss of the student model:', student_adv_loss)





    if (student_robust_acc > best_student_robust_acc) :
        
        best_epoch = epoch


        best_student_natural_acc = student_natural_acc
        best_student_robust_acc = student_robust_acc


        print('Best student robust acc achieved !')


        net_state_dict = student_net.state_dict()


        save_checkpoint({
            'epoch':epoch,
            'best_epoch' : best_epoch,
            'exp_id': args.exp_id,
            'state_dict': net_state_dict,
            'student_best_natural_acc' : best_student_natural_acc,
            'student_best_robust_acc': best_student_robust_acc,
            'optimizer' : student_optimizer.state_dict(),
            'best_student_swa_natural_acc' : best_student_swa_natural_acc,
            'best_student_swa_robust_acc' : best_student_swa_robust_acc,
            'best_student_swa_epoch' : best_student_swa_epoch,
            'swa_n_student':swa_n_student
        }, filename='student_net_' + file_name+ '_student_best_' + str(args.exp_id)+'.pth.tar')        

        print('Student Model Saved!')

    if (epoch == (args.total_epoch - 1)) :



        net_state_dict = student_net.state_dict()


        save_checkpoint({
            'epoch':epoch,
            'best_epoch' : best_epoch,
            'exp_id': args.exp_id,
            'state_dict': net_state_dict,
            'student_best_natural_acc' : best_student_natural_acc,
            'student_best_robust_acc': best_student_robust_acc,
            'optimizer' : student_optimizer.state_dict(),
            'best_student_swa_natural_acc' : best_student_swa_natural_acc,
            'best_student_swa_robust_acc' : best_student_swa_robust_acc,
            'best_student_swa_epoch' : best_student_swa_epoch,
            'swa_n_student':swa_n_student
        }, filename='student_net_' + file_name+ '_last_epoch_' + str(args.exp_id)+'.pth.tar')        

        print('At last epoch, Student Model Saved!')


    if args.swa_s and epoch >= args.swa_s_start:

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):

                if args.debug_mode :
                    if batch_idx == 2 :
                        break

                inputs, targets = inputs.to(device), targets.to(device)
                student_swa_total += targets.size(0)

                outputs = student_swa_net(inputs)

                loss = student_criterion(outputs, targets)
                student_swa_benign_loss += loss.item()

                _, predicted = outputs.max(1)
                student_swa_benign_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    print('\nCurrent batch:', str(batch_idx))

                adv = adversary_swa_student.perturb(inputs, targets, 10)


                adv_outputs = student_swa_net(adv)


                loss = student_criterion(adv_outputs, targets)
                student_swa_adv_loss += loss.item()

                _, predicted = adv_outputs.max(1)
                student_swa_adv_correct += predicted.eq(targets).sum().item()
                


    if args.swa_s and epoch >= args.swa_s_start:

        student_swa_robust_acc = 100. * student_swa_adv_correct / student_swa_total
        student_swa_natural_acc = 100. * student_swa_benign_correct / student_swa_total

        print('\nTotal test clean accuarcy of the SWA student model :', student_swa_natural_acc)
        print('Total adversarial test Accuarcy of the SWA student model against PGD-10 :', student_swa_robust_acc)
        print('Total test clean loss of the SWA student model:', student_swa_benign_loss)
        print('Total adversarial test loss of the SWA student model:', student_swa_adv_loss)





    if args.swa_s and epoch >= args.swa_s_start and (student_swa_robust_acc > best_student_swa_robust_acc) :

        best_student_swa_epoch = epoch


        best_student_swa_natural_acc = student_swa_natural_acc
        best_student_swa_robust_acc = student_swa_robust_acc


        print('Best SWA student robust acc achieved !')


        net_state_dict = student_swa_net.state_dict()


        save_checkpoint({
            'epoch':epoch,
            'best_epoch' : best_epoch,
            'exp_id': args.exp_id,
            'state_dict': net_state_dict,
            'student_best_natural_acc' : best_student_natural_acc,
            'student_best_robust_acc': best_student_robust_acc,
            'best_student_swa_natural_acc' : best_student_swa_natural_acc,
            'best_student_swa_robust_acc' : best_student_swa_robust_acc,
            'best_student_swa_epoch' : best_student_swa_epoch,
            'swa_n_student':swa_n_student
        }, filename='swa_student_net_' + file_name+ '_student_swa_best_' + str(args.exp_id)+'.pth.tar')   


        print('Best SWA Student Model Saved!')
    
    if args.swa_s and (epoch >= args.swa_s_start) and (epoch == args.total_epoch - 1) :

        net_state_dict = student_swa_net.state_dict()


        save_checkpoint({
            'epoch':epoch,
            'best_epoch' : best_epoch,
            'exp_id': args.exp_id,
            'state_dict': net_state_dict,
            'student_best_natural_acc' : best_student_natural_acc,
            'student_best_robust_acc': best_student_robust_acc,
            'best_student_swa_natural_acc' : best_student_swa_natural_acc,
            'best_student_swa_robust_acc' : best_student_swa_robust_acc,
            'best_student_swa_epoch' : best_student_swa_epoch,
            'swa_n_student':swa_n_student
        }, filename='swa_student_net_' + file_name+ '_swa_last_epoch_' + str(args.exp_id)+'.pth.tar')   


        print('At last epoch, SWA Student Model Saved!')



def adjust_learning_rate(optimizer, epoch, lr_decay_epochs, learning_rate):
    lr = learning_rate

    for i in lr_decay_epochs :
        if epoch < i :
            break
        else:
            lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr



best_epoch = -1
best_natural_epoch = -1
at_natural_best_natural_acc = -1
at_natural_best_robust_acc = -1
best_student_natural_acc = -1
best_student_robust_acc =-1


best_student_swa_natural_acc = -1
best_student_swa_robust_acc = -1
best_student_swa_epoch = -1


best_swa_natural_epoch = -1
at_natural_best_swa_natural_acc = -1
at_natural_best_swa_robust_acc = -1


if args.resume:
    args.start_epoch = checkpoint['epoch'] + 1
    best_epoch = checkpoint['best_epoch']
    best_student_natural_acc = checkpoint['student_best_natural_acc']
    best_student_robust_acc = checkpoint['student_best_robust_acc']


    if args.swa_s:
        best_student_swa_natural_acc = checkpoint['best_student_swa_natural_acc']
        best_student_swa_robust_acc = checkpoint['best_student_swa_robust_acc']
        best_student_swa_epoch = checkpoint['best_student_swa_epoch']
        swa_n_student = checkpoint['swa_n_student']

    print("Resuming from epoch ", args.start_epoch)

for epoch in range(args.start_epoch, args.total_epoch):

    if args.kd:
        print("Epoch : ", epoch)
        adjusted_lr = adjust_learning_rate(peer_optimizer, epoch, lr_decay_epochs_peer, learning_rate_peer)
        print("Teacher lr : ", adjusted_lr)
        adjusted_lr = adjust_learning_rate(student_optimizer, epoch, lr_decay_epochs_student, learning_rate_student)
        print("Student lr :", adjusted_lr)
    else:
        print("Epoch : ", epoch)
        adjusted_lr = adjust_learning_rate(student_optimizer, epoch, lr_decay_epochs_student, learning_rate_student)
        print("Current learning rate : ", adjusted_lr)

    if args.kd:

        train_kd(epoch)
        test_kd(epoch)
    else:

        train(epoch)
        test(epoch)


    if (epoch % args.save_interval == 0) :
        print("The number of epoch reached save interval. Checkpoints are being saved.")


        net_state_dict = student_net.state_dict()


        save_checkpoint({
            'epoch':epoch,
            'best_epoch' : best_epoch,
            'exp_id': args.exp_id,
            'state_dict': net_state_dict,
            'student_best_natural_acc' : best_student_natural_acc,
            'student_best_robust_acc': best_student_robust_acc,
            'optimizer' : student_optimizer.state_dict(),
            'best_student_swa_natural_acc' : best_student_swa_natural_acc,
            'best_student_swa_robust_acc' : best_student_swa_robust_acc,
            'best_student_swa_epoch' : best_student_swa_epoch,
            'swa_n_student':swa_n_student
        }, filename='student_net_' + file_name+ '_save_interval_' + str(args.exp_id)+'.pth.tar')        

        print('At epoch ', epoch, ', Student Model Saved!')


        if args.kd:

            net_state_dict = peer_net.state_dict()


            save_checkpoint({
                'epoch':epoch,
                'best_epoch' : best_epoch,
                'exp_id': args.exp_id,
                'state_dict': net_state_dict,
                'student_best_natural_acc' : best_student_natural_acc,
                'student_best_robust_acc': best_student_robust_acc,
                'optimizer' : peer_optimizer.state_dict(),
                'best_student_swa_natural_acc' : best_student_swa_natural_acc,
                'best_student_swa_robust_acc' : best_student_swa_robust_acc,
                'best_student_swa_epoch' : best_student_swa_epoch,
                'swa_n_student':swa_n_student
            }, filename='peer_net_' + file_name + '_save_interval_' + str(args.exp_id)+'.pth.tar')

            print('At epoch ', epoch, ', Teacher Model Saved!')

        if args.swa_s:

            net_state_dict = student_swa_net.state_dict()


            save_checkpoint({
                'epoch':epoch,
                'best_epoch' : best_epoch,
                'exp_id': args.exp_id,
                'state_dict': net_state_dict,
                'student_best_natural_acc' : best_student_natural_acc,
                'student_best_robust_acc': best_student_robust_acc,
                'best_student_swa_natural_acc' : best_student_swa_natural_acc,
                'best_student_swa_robust_acc' : best_student_swa_robust_acc,
                'best_student_swa_epoch' : best_student_swa_epoch,
                'swa_n_student':swa_n_student
            }, filename='swa_student_net_' + file_name+ '_save_interval_' + str(args.exp_id)+'.pth.tar')

            print('At epoch ', epoch, ', SWA student Model Saved!')



info_dict = dict()


if args.AA:

    threat_model = "Linf"  
    dataset = args.dataset  
    device = torch.device("cuda")
    if (args.dataset=='cifar10') or (args.dataset=='cifar100'):
        AA_data_path = args.data_path
    elif args.dataset=='tinyimagenet':
        AA_data_path= os.path.join(args.data_path, 'tiny-imagenet-200')


    # Student best model test.
    if args.kd:
        model_path = args.save_path +'student_net_' + file_name+ '_student_best_' + str(args.exp_id)+'.pth.tar'
    else:
        model_path = args.save_path + str(args.exp_id)+'_baseline_' + file_name + '_student_best'+'.pth.tar'



    model_name = model_path.split('/')[-1]

    target_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)
    target_net = target_net.to(device)
    checkpoint = torch.load(model_path)
    target_net.load_state_dict(checkpoint['state_dict'])

    target_net.eval()
    print("Student best model AA test starts.")
    clean_acc, robust_acc = benchmark(target_net, model_name=model_name, n_examples=args.n_examples, dataset=dataset,
                                    threat_model=threat_model, eps=8/255, device=device, data_dir=AA_data_path, 
                                    to_disk=True)

    info_dict['AA_student_clean_acc'] = clean_acc
    info_dict['AA_student_robust_acc'] = robust_acc
    print("Student best model AA test finished.")




    if args.swa_s:
        ## Student SWA net best test.
        if args.kd:
            model_path = args.save_path + 'swa_student_net_' + file_name+ '_student_swa_best_' + str(args.exp_id)+'.pth.tar'
        else:
            model_path = args.save_path + str(args.exp_id)+'_baseline_' + file_name + '_swa_student_best'+'.pth.tar'

        model_name = model_path.split('/')[-1]

        target_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)
        target_net = target_net.to(device)
        checkpoint = torch.load(model_path)
        target_net.load_state_dict(checkpoint['state_dict'])

        target_net.eval()
        print("SWA Student best model AA test starts.")
        clean_acc_swa, robust_acc_swa = benchmark(target_net, model_name=model_name, n_examples=args.n_examples, dataset=dataset,
                                        threat_model=threat_model, eps=8/255, device=device, data_dir=AA_data_path, 
                                        to_disk=True)


        info_dict['AA_swa_student_clean_acc'] = clean_acc_swa
        info_dict['AA_swa_student_robust_acc'] = robust_acc_swa
        print("SWA Student best model AA test finished.")



if args.pgd_eval:
    # Student best model test.
    device = torch.device("cuda")
    if args.kd:
        model_path = args.save_path +'student_net_' + file_name+ '_student_best_' + str(args.exp_id)+'.pth.tar'
    else:
        model_path = args.save_path + str(args.exp_id)+'_baseline_' + file_name + '_student_best'+'.pth.tar'

    model_name = model_path.split('/')[-1]

    target_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)
    target_net = target_net.to(device)
    checkpoint = torch.load(model_path)
    target_net.load_state_dict(checkpoint['state_dict'])

    target_net.eval()
    print("Student best model pgd evaluation starts.")

    adversary_student_best_pgd = LinfPGDAttack(target_net, epsilon, alpha)
    pgd_robust_acc = evaluate_adversary(target_net, adversary_student_best_pgd, test_loader, 20 , args, device)

    print("The PGD-20 robust accuracy of the Student best model:", pgd_robust_acc)
    info_dict['student_best_pgd_acc'] = pgd_robust_acc

    if args.swa_s:
        ## Student SWA net best test.
        if args.kd:
            model_path = args.save_path + 'swa_student_net_' + file_name+ '_student_swa_best_' + str(args.exp_id)+'.pth.tar'
        else:
            model_path = args.save_path + str(args.exp_id)+'_baseline_' + file_name + '_swa_student_best'+'.pth.tar'

        model_name = model_path.split('/')[-1]

        target_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)
        target_net = target_net.to(device)
        checkpoint = torch.load(model_path)
        target_net.load_state_dict(checkpoint['state_dict'])

        target_net.eval()
        print("SWA Student best model pgd evaluation starts.")
        
        adversary_swa_student_best_pgd = LinfPGDAttack(target_net, epsilon, alpha)
        pgd_robust_acc = evaluate_adversary(target_net, adversary_swa_student_best_pgd, test_loader, 20, args, device)

        print("The PGD-20 robust accuracy of the SWA Student best model: ", pgd_robust_acc)
        info_dict['swa_student_best_pgd_acc'] = pgd_robust_acc



if args.fgsm_eval:
    # Student best model test.
    device = torch.device("cuda")
    if args.kd:
        model_path = args.save_path +'student_net_' + file_name+ '_student_best_' + str(args.exp_id)+'.pth.tar'
    else:
        model_path = args.save_path + str(args.exp_id)+'_baseline_' + file_name + '_student_best'+'.pth.tar'

    model_name = model_path.split('/')[-1]

    target_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)
    target_net = target_net.to(device)
    checkpoint = torch.load(model_path)
    target_net.load_state_dict(checkpoint['state_dict'])

    target_net.eval()
    print("Student best model fgsm evaluation starts.")

    adversary_student_best_fgsm = LinfFgsmAttack(target_net, epsilon)
    fgsm_robust_acc = evaluate_adversary(target_net, adversary_student_best_fgsm, test_loader, 1, args, device)

    print("The FGSM robust accuracy of the Student best model : ", fgsm_robust_acc)
    info_dict['student_best_fgsm_acc'] = fgsm_robust_acc


    if args.swa_s:
        ## Student SWA net best test.
        if args.kd:
            model_path = args.save_path + 'swa_student_net_' + file_name+ '_student_swa_best_' + str(args.exp_id)+'.pth.tar'
        else:
            model_path = args.save_path + str(args.exp_id)+'_baseline_' + file_name + '_swa_student_best'+'.pth.tar'

        model_name = model_path.split('/')[-1]

        target_net = model_builder(args.s_type, num_classes=num_classes, dataset=args.dataset)
        target_net = target_net.to(device)
        checkpoint = torch.load(model_path)
        target_net.load_state_dict(checkpoint['state_dict'])

        target_net.eval()
        print("SWA Student best model fgsm evaluation starts.")
        
        adversary_swa_student_best_fgsm = LinfFgsmAttack(target_net, epsilon)
        fgsm_robust_acc = evaluate_adversary(target_net, adversary_swa_student_best_fgsm, test_loader, 1, args, device)

        print("The FGSM robust accuracy of the SWA Student best model : ", fgsm_robust_acc)
        info_dict['swa_student_best_fgsm_acc'] = fgsm_robust_acc



print("End time : " , datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



if args.AA:
    print("="*30)
    print("AA robust accuracy of the student model at the best epoch : " , robust_acc)

    if args.swa_s:
        print("AA robust accuracy of the SWA student model at the best epoch : " , robust_acc_swa)

print("="*30)



if not os.path.isdir(args.json_path):
    os.mkdir(args.json_path)
    print("New directory created ! : ", args.json_path)

json_file_name = args.json_path + str(args.exp_id)+ "_"+ args.s_type + "_"+ args.dataset + "_" +  str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

with open(json_file_name, "w") as outfile:
    json.dump(info_dict, outfile)
    print("json was succesfully dumped!")

