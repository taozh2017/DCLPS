# train this to get compact embbedding feature distribution decreasing domain gap
# source: structure loss + contrastive loss
# target: contrastive loss
import torch.nn.functional as F
import numpy as np
from options.train_options import TrainOptions
from utils.timer import Timer
import os
import random
from model import CreateModel
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import FDA_source_to_target
from data.polyp_dataset import PolypDataset
from utils.loss import  MPCL, mpcl_loss_calc
from itertools import cycle
import warnings
from utils.metrics import evaluate, Metrics
from model.decoder import *
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
    class_centers: C*N_fea
    cla_feas_trg: N*N_fea*H*W
    '''
    cla_feas_trg_de     = cla_feas_trg.detach()
    batch,N_fea,H,W     = cla_feas_trg_de.size()
    cla_feas_trg_de     = F.normalize(cla_feas_trg_de,p=2,dim=1)
    class_centers_norm  = F.normalize(class_centers,p=2,dim=1)
    cla_feas_trg_de     = cla_feas_trg_de.transpose(1,2).contiguous().transpose(2,3).contiguous() # N*H*W*N_fea
    cla_feas_trg_de     = torch.reshape(cla_feas_trg_de,[-1,N_fea])
    class_centers_norm  = class_centers_norm.transpose(0,1)  # N_fea*C
    batch_pixel_cosine  = torch.matmul(cla_feas_trg_de,class_centers_norm) #N*N_class
    threshold = 0.25
    pixel_mask          = pixel_selection(batch_pixel_cosine,threshold)
    hard_pixel_label    = torch.argmax(batch_pixel_cosine,dim=1)

    return hard_pixel_label, pixel_mask

def pixel_selection(batch_pixel_cosine,th):
    one_tag = torch.ones([1]).float().cuda()
    zero_tag = torch.zeros([1]).float().cuda()

    batch_sort_cosine,_ = torch.sort(batch_pixel_cosine,dim=1)
    pixel_sub_cosine    = batch_sort_cosine[:,-1]-batch_sort_cosine[:,-2]
    pixel_mask          = torch.where(pixel_sub_cosine>th,one_tag,zero_tag)

    return pixel_mask
    
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
    cla_src_feas: bs 2048 h w
    batch_src_labels: bs h w
    '''
    batch_src_feas = cla_src_feas.detach()
    n, c, fea_h, fea_w = batch_src_feas.size()
    batch_y_downsample = label_downsample(batch_src_labels, fea_h, fea_w)  # bs*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # bs*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(2): # n_class
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #bs*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # bs*2048*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # 2048
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        batch_class_center_fea = class_fea_sum / class_num  # 2048
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0)  # 1 * 2048
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list, dim=0)  # n_class * c
    return batch_class_center_feas

# output-level prototypes
def gen_class_center_out(cla_src_feas, batch_src_labels):
    '''
    cla_src_feas: bs 2 h w
    batch_src_labels: bs 1 h w
    '''
    batch_src_feas = cla_src_feas.detach()
    batch_class_center_fea_list = []
    for i in range(2): # n_class
        fea_mask        = torch.eq(batch_src_labels.unsqueeze(1), i).float().cuda()  #bs*1*h*w
        class_feas      = batch_src_feas * fea_mask  # bs*2*h*w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # 2
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        batch_class_center_fea = class_fea_sum / class_num  # 2
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0)  # 1 * 2
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list, dim=0)  # n_class * 2
    return batch_class_center_feas

def update_class_center_iter_out(cla_src_feas, batch_src_labels, class_center_feas, m):

    '''
    batch_src_feas  : n*c*h*w
    batch_src_labels: n*h*w
    '''
    # source 
    batch_src_feas     = cla_src_feas.detach()
    batch_class_center_fea_list = []
    for i in range(2):
        fea_mask        = torch.eq(batch_src_labels.unsqueeze(1),i).float().cuda()  #n*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas_src = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c


    class_center_feas = m * class_center_feas + (1-m) * batch_class_center_feas_src

    return class_center_feas

def update_class_center_iter(cla_src_feas, batch_src_labels, cls_trg_feas, batch_trg_labels, class_center_feas, m):

    '''
    batch_src_feas  : n*c*h*w
    batch_src_labels: n*h*w
    '''
    # source 
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
    batch_class_center_feas_src = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c

    # target
    batch_trg_feas     = cls_trg_feas.detach()
    batch_trg_labels   = batch_trg_labels
    n,c,fea_h,fea_w    = batch_trg_feas.size()
    batch_y_downsample = label_downsample(batch_trg_labels, fea_h, fea_w)  # n*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(2):
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
        class_feas      = batch_trg_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas_trg = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c

    class_center_feas = m * class_center_feas + ((1-m)/2.0) * batch_class_center_feas_src + ((1-m)/2.0) * batch_class_center_feas_trg

    return class_center_feas

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

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
    seed_it(args.seed)
    opt = TrainOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time' : Timer()}
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    opt.print_options(args)
    
    # run a epoch to get initial prototype
    # sourceloader_ = torch.utils.data.DataLoader(
    #     PolypDataset(args.source + '/images/', args.source + '/masks/', args.data_size), batch_size = 1, shuffle = False, num_workers = 4)
    # pseudotrgloader_ = torch.utils.data.DataLoader(
    #     PolypDataset(args.target + '/images/', args.target + '/masks/', args.data_size, txt = target_txt), batch_size = 1, shuffle = False, num_workers = 4)
    
    # to train proca
    sourceloader = torch.utils.data.DataLoader(
        PolypDataset(args.source + '/images/', args.source + '/masks/', args.data_size), batch_size = args.batch_size, shuffle = True, num_workers = 4, worker_init_fn=init_fn)
    sourceloader_iter = iter(sourceloader)
    target_txt = args.target + '/UDA_train.txt'
    pseudotrgloader = torch.utils.data.DataLoader(
        PolypDataset(args.target + '/images/', args.target + '/masks/', args.data_size, txt = target_txt), batch_size = args.batch_size, shuffle = True, num_workers = 4, worker_init_fn=init_fn)# 
    pseudoloader_iter = iter(pseudotrgloader)

    print('size of dataset:', len(sourceloader_iter)*args.batch_size)
    print('size of dataset:', len(pseudoloader_iter)*args.batch_size)
    args.train_bn = False
    model, optimizer = CreateModel(args)
    model.cuda()

    # aux decoders 
    if args.aux:
        num_classes = 1
        vat_decoder = VATDecoder(num_classes, xi=1e-6, eps=2.0).cuda()
        drop_decoder = DropOutDecoder(num_classes, drop_rate=0.5, spatial_dropout=True).cuda()
        cut_decoder = CutOutDecoder(num_classes, erase=0.4).cuda()
        context_m_decoder = ContextMaskingDecoder(num_classes).cuda()
        object_masking = ObjectMaskingDecoder(num_classes).cuda()
        feature_drop = FeatureDropDecoder(num_classes).cuda()
        feature_noise = FeatureNoiseDecoder(num_classes, uniform_range=0.3).cuda()

        vat_decoder = load_weight(args.restore_from, vat_decoder)
        drop_decoder = load_weight(args.restore_from, drop_decoder)
        cut_decoder = load_weight(args.restore_from, cut_decoder)
        context_m_decoder = load_weight(args.restore_from, context_m_decoder)
        object_masking = load_weight(args.restore_from, object_masking)
        feature_drop = load_weight(args.restore_from, feature_drop)
        feature_noise = load_weight(args.restore_from, feature_noise)

        aux_decoders = nn.ModuleList([vat_decoder, drop_decoder, cut_decoder, context_m_decoder, object_masking, feature_drop, feature_noise])
        optimizer_aux = optim.SGD([{'params': aux_decoders.parameters(), 'lr': args.learning_rate*10}], momentum=args.momentum, weight_decay=args.weight_decay)
    start_iter = 0
    cudnn.enabled = True
    cudnn.benchmark = True
    
    mean_img_ = torch.zeros(1, 1)
    mean_img_ = IMG_MEAN.repeat(1, 1, 256, 256)
    # run a epoch get initial prototypes
    # if start_iter == 0:
    #     print('start generating initial prototypes!')
    #     cls_center_feas_src = torch.zeros((2, 2048)).cuda()  # prototype shape: num_class * 2048
    #     model.eval()
    #     with torch.no_grad():
    #         for i, pack in enumerate(zip(sourceloader_, cycle(pseudotrgloader_))):
    #             [(src_img_, src_lbl_), (psu_img_, _)] = pack

    #             src_in_trg_ = FDA_source_to_target(src_img_, psu_img_, L=args.LB)     
    #             src_img_ = src_in_trg_.clone() - mean_img_                               

    #             src_img_, src_lbl_ = Variable(src_img_).cuda(), Variable(src_lbl_).cuda()
    #             cls_feature_src_, _ = model(src_img_, require_embedding_feature=True)
    #             src_lbl_ = src_lbl_.squeeze(1)
    #             cls_center_feas_src_ = gen_class_center_feas(cls_feature_src_, src_lbl_)
    #             cls_center_feas_src += cls_center_feas_src_
    #             del cls_center_feas_src_, src_img_, src_lbl_, _

    # cls_center_feas_src /= len(sourceloader_)
    # print(i+1)
    # print('initial prototypes finished!')

    # losses to log
    loss_train = 0.0
    loss_mpcl_src = 0.0
    loss_mpcl_trg = 0.0   
    loss_mpcl_trg_out = 0.0   
    loss_mpcl_trg_out_aux = 0.0   
    loss_out_mse = 0.0
    mean_img = torch.zeros(1, 1)
    mpcl_loss_src = MPCL(num_class=2, temperature=2.0, base_temperature=1.0, m=0.2)
    mpcl_loss_trg = MPCL(num_class=2, temperature=1.0, base_temperature=1.0, m=0.2)
    mpcl_loss_trg_out = MPCL(num_class=2, temperature=1.0, base_temperature=1.0, m=0.2)
    best_test_dice = 0
    best_test_iter = 0
    best_train_dice = 0
    best_train_iter = 0

    mse = nn.MSELoss()

    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        if i == 0: 
            print('taking initial snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, '0.pth'))
            temp_train_dice = run_target_train(model, args)
            temp_test_dice = run_target_test(model, args)
            print('initial model train dice:', temp_train_dice)
            print('initial model test dice:', temp_test_dice)
        model.train()
        model.adjust_learning_rate(args, optimizer, i)                               # adjust learning rate
        optimizer.zero_grad()   
        if args.aux:
            optimizer_aux.param_groups[0]['lr'] = optimizer.param_groups[1]['lr']
            optimizer_aux.zero_grad()
        try:
            src_img, src_lbl = sourceloader_iter.next()                            # new batch source
        except StopIteration:
            sourceloader_iter = iter(sourceloader)
            src_img, src_lbl = sourceloader_iter.next()
        try:
            psu_img, _ = pseudoloader_iter.next()
        except StopIteration:
            pseudoloader_iter = iter(pseudotrgloader)
            psu_img, _ = pseudoloader_iter.next()

        if psu_img.size()[0] < src_img.size()[0]:
            src_img = src_img[:psu_img.size()[0], :, :, :]
            src_lbl = src_lbl[:psu_img.size()[0], :, :, :]

        if mean_img.shape[0] != src_img.size()[0]:
            B, C, H, W = src_img.shape
            mean_img = IMG_MEAN.repeat(B,1,H,W)

        src_in_trg = FDA_source_to_target(src_img, psu_img, L=args.LB)     
        src_img = src_in_trg.clone() - mean_img                                
        psu_img = psu_img.clone()    - mean_img

        # source 
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl).cuda()
        cls_feature_src, pred_src = model(src_img, require_embedding_feature=True)   
        loss_seg_src = structure_loss(pred_src, src_lbl)                                

        # target
        psu_img = Variable(psu_img).cuda()
        cls_feature_trg, pred_trg = model(psu_img, require_embedding_feature=True)
        # to-do: change threshold
        psu_lbl = (pred_trg.sigmoid()>0.5)
    
        # update prototypes
        src_lbl = src_lbl.squeeze()
        psu_lbl = psu_lbl.squeeze()
        assert not src_lbl.requires_grad
        assert not psu_lbl.requires_grad
        # to-do: change m
        if i==0:
            cls_center_feas_src = gen_class_center_feas(cls_feature_src, src_lbl)
        else:
            cls_center_feas_src = update_class_center_iter(cls_feature_src, src_lbl, cls_feature_trg, psu_lbl, cls_center_feas_src, 0.8)
        
        # mpcl loss
        mpcl_src = mpcl_loss_calc(feas=cls_feature_src, labels=src_lbl.detach(), class_center_feas=cls_center_feas_src, loss_func=mpcl_loss_src, tag='source')
        mpcl_trg = mpcl_loss_calc(feas=cls_feature_trg, labels=psu_lbl.detach(), class_center_feas=cls_center_feas_src, loss_func=mpcl_loss_trg, tag='source')

        # calculate out-level contrastive loss
        if args.output_level:
            
            if i==0:
                cls_center_out = gen_class_center_out(torch.cat([1-pred_trg, pred_trg], dim=1), psu_lbl)
            else:
                cls_center_out = update_class_center_iter_out(torch.cat([1-pred_trg, pred_trg], dim=1), psu_lbl, cls_center_out, 0.8)

            mpcl_trg_out = mpcl_loss_calc(feas=torch.cat([1-pred_trg, pred_trg], dim=1), labels=psu_lbl.detach(), class_center_feas=cls_center_out, loss_func=mpcl_loss_trg_out)
            loss_mpcl_trg_out += mpcl_trg_out.item()

            # calculate cross contrastive loss
            if args.aux:
                # #  get aux preds
                aux_outputs = [aux_decoder(cls_feature_trg, pred_trg.detach()) for aux_decoder in aux_decoders]
                
                # # if contrastive loss between main output and aux prototypes
                # aux_lbls = [(aux_output.sigmoid()>0.5).squeeze() for aux_output in aux_outputs]
                # aux_cls_centers = [gen_class_center_out() for i in range(len(aux_outputs))]
                
                # if contrastive loss between aux output and main prototypes
                aux_contrastive_loss_out = [mpcl_loss_calc(feas=torch.cat([1-aux_output, aux_output], dim=1), labels=psu_lbl.detach(), class_center_feas=cls_center_out.detach(), loss_func=mpcl_loss_trg_out) for aux_output in aux_outputs]
                mpcl_aux = sum(aux_contrastive_loss_out)/len(aux_contrastive_loss_out)
                loss_mpcl_trg_out_aux += mpcl_aux.item()

                # calculate cross consistency loss
                if args.aux_mse:
                    pred = pred_trg.sigmoid().detach()
                    loss_mse = sum([mse(aux_output.sigmoid(), pred) for aux_output in aux_outputs])/len(aux_outputs)
                    loss_out_mse += loss_mse.item()
                    loss_all = loss_seg_src + mpcl_src + mpcl_trg*args.mpcl_weight + mpcl_trg_out*0.1 + mpcl_aux + loss_mse
                else:
                    loss_all = loss_seg_src + mpcl_src + mpcl_trg*args.mpcl_weight + (mpcl_trg_out + mpcl_aux)*0.1 
            else:
                loss_all = loss_seg_src + mpcl_src + mpcl_trg*args.mpcl_weight + mpcl_trg_out * 0.1
        else:
            loss_all = loss_seg_src + (mpcl_src + mpcl_trg*args.mpcl_weight)


        loss_all.backward()
        optimizer.step()
        if args.aux:
            optimizer_aux.step()

        loss_train += loss_seg_src.item()
        loss_mpcl_src += mpcl_src.item()
        loss_mpcl_trg += mpcl_trg.item()


        del loss_seg_src, mpcl_src, mpcl_trg, psu_lbl, src_lbl, src_img, psu_img
            
        if (i+1) % args.print_freq == 0:  # print loss
            
            temp_train_dice = run_target_train(model, args)
            if temp_train_dice > best_train_dice:
                best_train_dice = temp_train_dice
                best_train_iter = (i+1)
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'BEST_TRAIN.pth'))
                torch.save(cls_center_feas_src.cpu(), os.path.join(args.snapshot_dir, 'BEST_TRAIN_PROTOTYPE.pth'))
                print('best train dice:', best_train_dice)

            temp_test_dice = run_target_test(model, args)
            if temp_test_dice > best_test_dice:
                best_test_dice = temp_test_dice
                best_test_iter = (i+1)
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'BEST_TEST.pth'))
                torch.save(cls_center_feas_src.cpu(), os.path.join(args.snapshot_dir, 'BEST_TEST_PROTOTYPE.pth'))
                print('best test dice:', best_test_dice)

            _t['iter time'].toc(average=False)
            loss_train /= args.print_freq
            loss_mpcl_src /= args.print_freq
            loss_mpcl_trg   /= args.print_freq
            loss_mpcl_trg_out   /= args.print_freq
            loss_mpcl_trg_out_aux   /= args.print_freq
            loss_out_mse /= args.print_freq
            print('[it %d][src seg loss %.4f][src mpcl loss %.4f][trg mpcl loss %.4f][trg mpcl out loss %.4f][trg mpcl aux loss %.4f][trg mse loss %.4f][train dice %.4f][test dice %.4f][lr %.8f][%.2fs]' % \
                    (i + 1, loss_train, loss_mpcl_src, loss_mpcl_trg, loss_mpcl_trg_out, loss_mpcl_trg_out_aux, loss_out_mse, temp_train_dice, temp_test_dice, optimizer.param_groups[0]['lr'], _t['iter time'].diff) )
            loss_train = 0.0
            loss_mpcl_src = 0.0
            loss_mpcl_trg = 0.0
            loss_mpcl_trg_out = 0.0
            loss_mpcl_trg_out_aux = 0.0
            loss_out_mse = 0.0

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

