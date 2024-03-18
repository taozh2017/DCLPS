from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from data.polyp_dataset import PolypDataset
from model import CreateSSLModel
import os
from options.test_options import TestOptions
import imageio
from utils.pseudo import PseudoLabel
# import pandas as pd

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (1,3,1,1))

def batch_dice_score(input, target):
    smooth = 1e-6
    
    bs = input.shape[0]
    iflat = torch.sigmoid(input).view(bs,-1)
    iflat = (iflat > 0.5)
    tflat = target.view(bs,-1)
    intersection = (iflat * tflat).sum(dim=1)
    
    return (2.0 * intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth)

def main():
    opt = TestOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    args.restore_from = args.restore_opt1
    model1 = CreateSSLModel(args)
    model1.eval()
    model1.cuda()

    # normal dataset setting
    # target_txt = args.target + '/UDA_train.txt'
    # targetloader = torch.utils.data.DataLoader(
    #     PolypDataset(args.target + '/images/', args.target + '/masks/', args.data_size, target_txt, mode = 'test'), batch_size = 1, shuffle = False)
    target_txt = args.target + '/UDA_train.txt'
    targetloader = torch.utils.data.DataLoader(
        PolypDataset(args.target + '/images/', args.target + '/masks/', args.data_size, target_txt, mode = 'test'), batch_size = 1, shuffle = False)

    mean_img = torch.zeros(1, 1)
    dices = []
    save_path = args.target + args.pseudo_dir 
    # save_path = './pred_rectify/pseudo/'   # 被覆盖了
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('size of test data:', len(targetloader))

    # pseudo_label = PseudoLabel()
    # # get threshold
    # with torch.no_grad():
    #     for index, batch in enumerate(targetloader):
    #         # print(index)
    #         image, label, name = batch
    #         if mean_img.shape[-1] < 2:
    #             B, C, H, W = image.shape
    #             mean_img = IMG_MEAN.repeat(B,1,H,W)
    #         image = image.clone() - mean_img
    #         image = Variable(image).cuda()
    #         label =Variable(label).cuda()

    #         output = model1(image)
    #         output = nn.functional.interpolate(output, label.squeeze().size(), mode='bilinear', align_corners=False)#.cpu().data[0].numpy()
    #         pseudo_label.update_pseudo_label(output)
    # args.thres = pseudo_label.get_threshold_const(thred=args.thres, percent=0.4)
    # print('thres:', args.thres)


    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            # print(index)
            image, label, name = batch
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)
            image = image.clone() - mean_img
            image = Variable(image).cuda()
            label =Variable(label).cuda()

            output = model1(image)

            output = nn.functional.interpolate(output, label.squeeze().size(), mode='bilinear', align_corners=False)#.cpu().data[0].numpy()
            dice = batch_dice_score(output, label)
            dices.append(dice)
            print(dice)
            # save
            res = output.sigmoid()
            # res = (res>=args.thres).float().data.cpu().numpy().squeeze()
            res = (res>0.5).float().data.cpu().numpy().squeeze()
            imageio.imsave((save_path + name[0]), res)
    print('avg dice:', sum(dices)/len(dices))
    
if __name__ == '__main__':
    main()
    
