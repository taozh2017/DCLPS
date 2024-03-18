import torch.nn.functional as F
import numpy as np
from options.train_options import TrainOptions
from utils.timer import Timer
import os
from model import CreateModel
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
from data.polyp_dataset import PolypDataset
from utils.loss import  MPCL, mpcl_loss_calc
import warnings
import random
from utils.metrics import evaluate, Metrics
import imageio
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

def generate_pseudo_label(cla_feas_trg, class_centers):
    '''
    class_centers: C*N_fea  2*2048
    cla_feas_trg: N*N_fea*H*W  bs*2048*h*w
    '''
    cla_feas_trg_de     = cla_feas_trg.detach()
    batch,N_fea,H,W     = cla_feas_trg_de.size()
    cla_feas_trg_de     = F.normalize(cla_feas_trg_de,p=2,dim=1)
    class_centers_norm  = F.normalize(class_centers,p=2,dim=1)
    cla_feas_trg_de     = cla_feas_trg_de.transpose(1,2).contiguous().transpose(2,3).contiguous() # N*H*W*N_fea
    cla_feas_trg_de     = torch.reshape(cla_feas_trg_de,[-1,N_fea]) # (bs*h*w)* 2048 
    class_centers_norm  = class_centers_norm.transpose(0,1)  # 2048*2
    batch_pixel_cosine  = torch.matmul(cla_feas_trg_de,class_centers_norm) # (bs*h*w)*2
    # threshold = 0
    # pixel_mask, weight  = pixel_selection(batch_pixel_cosine,threshold)
    # hard_pixel_label    = torch.argmax(batch_pixel_cosine,dim=1)  # (bs*h*w)
    # batch_pixel_cosine    = torch.reshape(batch_pixel_cosine, [batch,2,H,W])  # bs*2*h*w
    batch_pixel_cosine    = torch.reshape(batch_pixel_cosine, [batch,H,W,2])  
    batch_pixel_cosine   = batch_pixel_cosine.transpose(2,3).contiguous().transpose(1,2).contiguous() # bs*2*h*w
    # weight = torch.reshape(weight, [batch, H, W, -1])
    # weight = weight.transpose(2,3).contiguous().transpose(1,2).contiguous() # bs*2*h*w
    # return hard_pixel_label, pixel_mask
    return batch_pixel_cosine

def pixel_selection(batch_pixel_cosine,th):
    one_tag = torch.ones([1]).float().cuda()
    zero_tag = torch.zeros([1]).float().cuda()
    # sort默认升序
    batch_sort_cosine,_ = torch.sort(batch_pixel_cosine,dim=1)
    pixel_sub_cosine    = batch_sort_cosine[:,-1]-batch_sort_cosine[:,-2]  # cos_max - cos_submax
    weight = batch_sort_cosine
    
    pixel_mask          = torch.where(pixel_sub_cosine>th,one_tag,zero_tag)
    return pixel_mask, weight

def label_downsample(labels,fea_h,fea_w):
    '''
    labels: N*H*W
    '''
    labels = labels.float().cuda()
    labels = F.interpolate(labels, size=fea_w, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()
    labels = F.interpolate(labels, size=fea_h, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()  # n*fea_h*fea_w
    labels = labels.int()
    return labels

# get initial category prototype when iteration=0
def gen_class_center_feas(cla_src_feas, batch_src_labels):
    '''
    cla_src_feas: bs c h w
    batch_src_labels: bs h w
    '''
    batch_src_feas = cla_src_feas.detach()
    batch_src_labels = batch_src_labels
    n, c, fea_h, fea_w = batch_src_feas.size()
    batch_y_downsample = label_downsample(batch_src_labels, fea_h, fea_w)  # n*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(2): # n_class
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        batch_class_center_fea = class_fea_sum / class_num  # c
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0)  # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list, dim=0)  # n_class * c
    return batch_class_center_feas

def update_class_center_iter(cla_src_feas, batch_src_labels, class_center_feas, m):

    '''
    batch_src_feas  : n*c*h*w
    batch_src_labels: n*h*w
    '''
    batch_src_feas     = cla_src_feas.detach()
    batch_src_labels   = batch_src_labels
    n,c,fea_h,fea_w    = batch_src_feas.size()
    batch_y_downsample = label_downsample(batch_src_labels, fea_h, fea_w)  # n*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(2):
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c
    class_center_feas = m * class_center_feas + (1-m) * batch_class_center_feas

    return class_center_feas

# wbce+wiou
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def weight_bce_loss(pred, mask, weight=None):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, weight, reduction='mean')
    return wbce

def weight_bce_iou_loss(pred, mask, weight=None):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weight*wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weight).sum(dim=(2, 3))
    union = ((pred + mask)*weight).sum(dim=(2, 3))
    wiou = 1 - (2*inter)/(union)
    return (wbce + wiou).mean()

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return wiou.mean()

def dice_loss(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)
    size = pred.size(0)
    pred_flat = pred.view(size, -1)
    target_flat = target.view(size, -1)
    intersection = pred_flat * target_flat
    dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
    dice_loss = 1 - dice_score.sum()/size
    return dice_loss

def init_fn(worker_id):
    np.random.seed(int(worker_id))

def seed_it(seed):
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

def main():
    
    opt = TrainOptions()
    args = opt.initialize()
    seed_it(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time' : Timer()}
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    opt.print_options(args)

    # using pseudo labels 
    target_txt = args.target + '/UDA_train.txt'
    pseudotrgloader = torch.utils.data.DataLoader(
        PolypDataset(args.target + '/images/', args.target + args.pseudo_dir, args.data_size, txt = target_txt), batch_size = args.batch_size, shuffle = True, worker_init_fn=init_fn) # , worker_init_fn=init_fn
    pseudoloader_iter = iter(pseudotrgloader)      
    print('size of dataset:', len(pseudoloader_iter)*args.batch_size)
                  
    args.train_bn = True
    model, optimizer = CreateModel(args)
    model.cuda()
    

    start_iter = 0

    # losses to log
    loss_val = 0.0
    loss_contrastive = 0.0
    loss_consis = 0.0
    mpcl_loss = MPCL(num_class=2, temperature=2.0, base_temperature=1.0, m=0.2)
    bce_loss = nn.BCEWithLogitsLoss()
    l1_consistency = nn.L1Loss()
    l2_consistency = nn.MSELoss()
    mean_img = torch.zeros(1, 1)
    best_test_dice = 0
    best_test_iter = 0
    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        if i==0:
            test_dice = run_target_test(model, args, i)
            print('initial model test dice:', test_dice)
        model.train()
        model.adjust_learning_rate(args, optimizer, i)                               # adjust learning rate
        optimizer.zero_grad()                                                        # zero grad

        try:
            psu_img, psu_lbl = pseudoloader_iter.next()
        except StopIteration:
            pseudoloader_iter = iter(pseudotrgloader)
            psu_img, psu_lbl = pseudoloader_iter.next()
        
        if mean_img.shape[0] != psu_img.size()[0]:
            B, C, H, W = psu_img.shape
            mean_img = IMG_MEAN.repeat(B,1,H,W)

        psu_img = psu_img.clone()  - mean_img
        psu_img, psu_lbl = Variable(psu_img).cuda(), Variable(psu_lbl).cuda()
        cls_feature, pred_trg = model(psu_img, require_embedding_feature=True)

        # load prototype 
        if args.loss == 'wbce' or args.mpcl_weight != 0:
            if i == 0:
                if args.pro_plan == 'A':
                    # plan A: like proca, get prototype from one epoch and average
                    pass
                elif args.pro_plan == 'B':
                    # plan B: get prototype from one batch and update
                    cls_center_feas = gen_class_center_feas(cls_feature, psu_lbl.squeeze()).detach()  # 2(fore+back) * 2048
                    print('Generating initial prototypes!')
                elif args.pro_plan == 'C':
                    # plan C: load prototype from proca
                    cls_center_feas = torch.load(args.prototype_dir)
                    cls_center_feas = cls_center_feas.cuda().detach()
                    print('Loading prototypes from:', args.prototype_dir)
            else:       
                cls_center_feas = update_class_center_iter(cls_feature.detach(), psu_lbl.squeeze(), cls_center_feas, 0.2).detach()
        
        # pseudo rectify loss
        if args.loss == 'bce':
            # loss_seg = bce_loss(pred_trg, psu_lbl)
            loss_seg = weight_bce_loss(pred_trg, psu_lbl)
        if args.loss == 'wbce':
            # loss_seg = bce_loss(pred_trg, psu_lbl)
            batch_pixel_cosine = generate_pseudo_label(cls_feature.detach(), cls_center_feas) 
            batch_pixel_cosine = F.interpolate(batch_pixel_cosine, scale_factor=8, mode='bilinear') # bs 2 H W
            select_mask = torch.eq(torch.argmax(batch_pixel_cosine, dim=1).unsqueeze(1), psu_lbl).float()  # Eq.8 m
            # weight = batch_pixel_cosine/batch_pixel_cosine.sum(dim=1).unsqueeze(1)
            weight = torch.softmax(batch_pixel_cosine, dim=1)
            # weight = torch.where(psu_lbl.bool(), weight[:,1,:,:].unsqueeze(1), weight[:,0,:,:].unsqueeze(1))
            weight = torch.where(pred_trg.sigmoid().bool(), weight[:,1,:,:].unsqueeze(1), weight[:,0,:,:].unsqueeze(1)) * select_mask 

            loss_seg = weight_bce_loss(pred_trg, psu_lbl, weight.detach())
        elif args.loss == 'iou':
            loss_seg = iou_loss(pred_trg, psu_lbl)
        elif args.loss == 'bce+iou':
            loss_seg = bce_loss(pred_trg, psu_lbl) + iou_loss(pred_trg, psu_lbl)
        elif args.loss == 'bce+dice':
            loss_seg = bce_loss(pred_trg, psu_lbl) + dice_loss(pred_trg, psu_lbl)
        elif args.loss == 'wbce+wiou':
            loss_seg = structure_loss(pred_trg, psu_lbl)
 
        # mpcl
        if args.mpcl_weight != 0:
            loss_mpcl = mpcl_loss_calc(feas=cls_feature, labels=psu_lbl.squeeze(), class_center_feas=cls_center_feas.detach(), loss_func=mpcl_loss, tag='source')

            loss_all = loss_seg + loss_mpcl * args.mpcl_weight
            loss_contrastive += loss_mpcl.item()
        else:
            loss_all = loss_seg

        loss_all.backward()
        optimizer.step()

        loss_val += loss_seg.item()
        

        if (i+1) % args.print_freq == 0:  # print loss
            # temp_train_dice = run_target_train(model, args)
            temp_train_dice = 0 
            temp_test_dice = run_target_test(model, args, i)
            if temp_test_dice > best_test_dice:
                best_test_dice = temp_test_dice
                best_test_iter = (i+1)
                # torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'BEST_TEST_' + str(i+1) + '.pth'))
                print('best test dice:', best_test_dice)
            
            _t['iter time'].toc(average=False)
            loss_val  /= args.print_freq
            loss_contrastive   /= args.print_freq
            loss_consis /= args.print_freq
            print('[it %d][trg seg loss %.4f][trg mpcl loss %.4f][train dice %.4f][test dice %.4f][lr %.8f][%.2fs]' % \
                    (i + 1, loss_val, loss_contrastive, temp_train_dice, temp_test_dice, optimizer.param_groups[0]['lr'], _t['iter time'].diff))
            loss_val = 0.0
            loss_contrastive = 0.0
            loss_consis = 0.0

            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()

    print('best test iter:{}  best test dice:{}'.format(best_test_iter, best_test_dice))

def run_target_test(model, args, i):
    mean_img = torch.zeros(1, 1)
    target_txt = args.target + '/UDA_test.txt'
    targetloader = torch.utils.data.DataLoader(
        PolypDataset(args.target + '/images/', args.target + '/masks/', args.data_size, txt = target_txt, mode = 'test'), batch_size = 1, shuffle = False)
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
            label = Variable(label).cuda()

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

