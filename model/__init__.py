from model.deeplab import Deeplab
from model.deeplab_multi import DeeplabMulti
import torch.optim as optim

def CreateModel(args):
    if args.model == 'DeepLab':
        phase = 'test'
        if args.set == 'train' or args.set == 'trainval':
            phase = 'train'
        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from, phase=phase)
        if args.set == 'train' or args.set == 'trainval':
            optimizer = optim.SGD(model.optim_parameters(args),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            return model, optimizer
        else:
            return model
        
    if args.model == 'DeepLab_Multi':
        phase = 'test'
        if args.set == 'train' or args.set == 'trainval':
            phase = 'train'
        # train_bn = False in stage 1 
        # train_bn = True in stage 2
        model = DeeplabMulti(num_classes=args.num_classes, use_se = True, train_bn = args.train_bn, norm_style = 'gn', droprate = 0.1, init_weights = args.init_weights, restore_from=args.restore_from)
        if args.set == 'train' or args.set == 'trainval':
            optimizer = optim.SGD(model.optim_parameters(args),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            return model, optimizer
        else:
            return model
        

def CreateSSLModel(args):
    if args.model == 'DeepLab_Multi': 
        model = DeeplabMulti(num_classes=args.num_classes, use_se = True, train_bn = False, norm_style = 'gn', droprate = 0.1, init_weights = args.init_weights, restore_from=args.restore_from)
        
    elif args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from, phase=args.set)
    else:
        raise ValueError('The model mush be either deeplab-101 or vgg16-fcn')
    return model

