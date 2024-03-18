import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from options.train_options import TrainOptions
from utils.timer import Timer
import os
from data.polyp_dataset import PolypDataset
from model import CreateModel
#import tensorboardX
import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
from utils import FDA_source_to_target
import warnings
from utils.metrics import evaluate, Metrics
warnings.filterwarnings("ignore")

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1) )

def batch_dice_score(input, target):
    smooth = 1e-6
    bs = input.shape[0] 
    iflat = torch.sigmoid(input).view(bs,-1)
    iflat = (iflat > 0.5)
    tflat = target.view(bs,-1)
    intersection = (iflat * tflat).sum(dim=1)
    return (2.0 * intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth)

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

class MinEntLoss(nn.Module):
    """
    Loss for the minimization of the entropy map
    Source for version 1: https://github.com/valeoai/ADVENT
    Version 2 adds the variance of the entropy map in the computation of the loss
    """
    def __init__(self, ita=2., logits = True):
        super(MinEntLoss, self).__init__()
        self.ita = ita
        self.logits = logits

    def forward(self, x):
        if self.logits:
            P = torch.cat([torch.sigmoid(x), 1 - torch.sigmoid(x)], axis = 1) + 1e-10
        else:
            P = x + 1e-10
        logP = torch.log(P)
        PlogP = P * logP
        ent = -1.0 * PlogP.sum(dim=1)
        ent = ent / 0.69
        ent = ent ** 2.0 + 1e-8
        ent = ent ** self.ita
        return ent.mean() 
    

def main():
    opt = TrainOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time' : Timer()}

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    opt.print_options(args)
    
    # normal dataset setting
    sourceloader = torch.utils.data.DataLoader(
        PolypDataset(args.source + '/images/', args.source + '/masks/', args.data_size), batch_size = args.batch_size, shuffle = True, num_workers = 4)
    
    target_txt = args.target + '/UDA_train.txt'
    targetloader = torch.utils.data.DataLoader(
        PolypDataset(args.target + '/images/', args.target + '/masks/', args.data_size, txt = target_txt), batch_size = args.batch_size, shuffle = True, num_workers = 4)# worker_init_fn=init_fn
    
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)
    print('size of source dataloader:', len(sourceloader)*args.batch_size)
    print('size of target dataloader:', len(targetloader)*args.batch_size)

    args.train_bn = False
    model, optimizer = CreateModel(args)
    model.cuda()
    start_iter = 0
    cudnn.enabled = True
    cudnn.benchmark = True
    # losses to log
    loss_train = 0.0

    entloss = MinEntLoss()
    mean_img = torch.zeros(1, 1)

    best_test_dice = 0
    best_test_iter = 0
    best_train_dice = 0
    best_train_iter = 0
    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        model.train()
        model.adjust_learning_rate(args, optimizer, i)                               # adjust learning rate
        optimizer.zero_grad()                                                        # zero grad
        
        try:
            src_img, src_lbl = sourceloader_iter.next()                            # new batch source, img∈[0,255]
        except StopIteration:
            sourceloader_iter = iter(sourceloader)
            src_img, src_lbl = sourceloader_iter.next()  
        try:
            trg_img, _ = targetloader_iter.next()                            # new batch target
        except StopIteration:
            targetloader_iter = iter(targetloader)
            trg_img, _ = targetloader_iter.next() 
        
        if trg_img.size()[0] < src_img.size()[0]:
            src_img = src_img[:trg_img.size()[0], :, :, :]
            src_lbl = src_lbl[:trg_img.size()[0], :, :, :]

        if mean_img.shape[0] != src_img.size()[0]:
            B, C, H, W = src_img.shape
            mean_img = IMG_MEAN.repeat(B,1,H,W)

        #-------------------------------------------------------------------#

        # 1. source to target, target to target
        src_in_trg = FDA_source_to_target(src_img, trg_img, L=args.LB)   # src_lbl
        trg_in_trg = trg_img

        # 2. subtract mean(Normalization)
        src_img = src_in_trg.clone() - mean_img    # src, src_lbl
        trg_img = trg_in_trg.clone() - mean_img    # trg, trg_lbla

        #-------------------------------------------------------------------#

        # evaluate and update params 
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl).cuda() 
        pred_src = model(src_img)     
        loss_seg_src = structure_loss(pred_src, src_lbl) 

        # get target loss, only entropy for backpro
        trg_img = Variable(trg_img).cuda()
        pred_trg = model(trg_img)       
        loss_ent_trg = entloss(pred_trg)

        triger_ent = 0.0
        if i > args.switch2entropy:
            triger_ent = 1.0

        loss_all = (loss_seg_src) + triger_ent * args.entW * loss_ent_trg  # loss of seg on src, and ent on t

        loss_all.backward()
        optimizer.step()

        loss_train += loss_seg_src.item()
        
        del loss_seg_src, loss_ent_trg
                
        if (i+1) % args.print_freq == 0:

            temp_train_dice = run_target_train(model, args)
            if temp_train_dice > best_train_dice:
                best_train_dice = temp_train_dice
                best_train_iter = (i+1)
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'BEST_TRAIN.pth'))
                print('best train dice:', best_train_dice)

            temp_test_dice = run_target_test(model, args)
            if temp_test_dice > best_test_dice:
                best_test_dice = temp_test_dice
                best_test_iter = (i+1)
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'BEST_TEST.pth'))
                print('best test dice:', best_test_dice)
            
            _t['iter time'].toc(average=False)
            loss_train /= args.print_freq
            print('[it %d][src seg loss %.4f][train dice %.4f][test dice %.4f][lr %.8f][%.2fs]' % \
                    (i + 1, loss_train, temp_train_dice, temp_test_dice, optimizer.param_groups[0]['lr'], _t['iter time'].diff) )

            loss_train = 0.0

            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()
    print('best train iter:{}  best train dice:{}'.format(best_train_iter, best_train_dice))
    print('best test iter:{}  best test dice:{}'.format(best_test_iter, best_test_dice))
    
    
def run_target_test(model, args):
    mean_img = torch.zeros(1, 1)
    target_txt = args.target + '/UDA_test.txt'
    targetloader = torch.utils.data.DataLoader(
        PolypDataset(args.target + '/images/', args.target + '/masks/', args.data_size, txt = target_txt, mode = 'test'), batch_size = 1, shuffle = False, num_workers = 4)
    dices = []
    model.eval()
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])
    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            image, label, name = batch
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)
            image = image.clone() - mean_img
            image = Variable(image).cuda()
            label =Variable(label).cuda()

            output = model(image)  
            output = nn.functional.interpolate(output, label.squeeze().size(), mode='bilinear', align_corners=False)#.cpu().data[0].numpy()
            dice = batch_dice_score(output, label).item()
            dices.append(dice)

            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(torch.sigmoid(output), label)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, 
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean)
    dice_score = sum(dices)/len(dices)
    metrics_result = metrics.mean(len(targetloader))
    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
          'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
          % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
             metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
             metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))
    return dice_score  

def run_target_train(model, args):
    mean_img = torch.zeros(1, 1)
    target_txt = args.target + '/UDA_train.txt'
    targetloader = torch.utils.data.DataLoader(
        PolypDataset(args.target + '/images/', args.target + '/masks/', args.data_size, txt = target_txt, mode = 'test'), batch_size = 1, shuffle = False, num_workers = 4)
    dices = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            image, label, name = batch
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)
            image = image.clone() - mean_img
            image = Variable(image).cuda()
            label =Variable(label).cuda()

            output = model(image)  
            output = nn.functional.interpolate(output, label.squeeze().size(), mode='bilinear', align_corners=False)#.cpu().data[0].numpy()
            dice = batch_dice_score(output, label).item()
            dices.append(dice)
    dice_score = sum(dices)/len(dices)
    return dice_score

if __name__ == '__main__':
    main()

