from .resnet import *
from .wide_resnet import *
from .mobilenetv2 import *
from pytorchcv.model_provider import get_model as ptcv_get_model

def model_builder(model_str, num_classes, dataset):

    if model_str == 'resnet18':
        target_net = ResNet18(num_classes=num_classes, dataset=dataset)

    elif model_str == 'resnet34':
        target_net = ResNet34(num_classes=num_classes, dataset=dataset)

    elif model_str == 'resnet50':
        target_net = ResNet50(num_classes=num_classes, dataset=dataset)

    elif model_str == 'resnet101':
        target_net = ResNet101(num_classes=num_classes, dataset=dataset)

    elif model_str == 'preactresnet18':
        target_net = PreActResNet18(num_classes=num_classes, dataset=dataset)

    elif model_str == 'wideresnet28x10':
        target_net = WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dataset=dataset)

    elif model_str == 'wideresnet34x10':
        target_net = WideResNet(depth=34, widen_factor=10, num_classes=num_classes, dataset=dataset)

    elif model_str == 'wideresnet70x16':
        target_net = WideResNet(depth=70, widen_factor=16, num_classes=num_classes, dataset=dataset)

    elif model_str == 'mobilenetv2':
        target_net = MobileNetV2(dataset=dataset, num_classes=num_classes)


    elif model_str == 'densenet40_k12_bc':
        if dataset == 'cifar10':
            target_net = ptcv_get_model("densenet40_k12_bc_cifar10", pretrained=False)

        elif dataset == 'cifar100':
            target_net = ptcv_get_model("densenet40_k12_bc_cifar100", pretrained=False)

        else:
            raise NotImplementedError

    elif model_str == 'densenet40_k12':
        if dataset == 'cifar10':
            target_net = ptcv_get_model("densenet40_k12_cifar10", pretrained=False)

        elif dataset == 'cifar100':
            target_net = ptcv_get_model("densenet40_k12_cifar100", pretrained=False)

        else:
            raise NotImplementedError

        
    else:
        raise NotImplementedError

    return target_net