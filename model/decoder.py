import math
import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np
import cv2
from torch.distributions.uniform import Uniform

# get decoder dict
def load_weight(init_weights, model):
    saved_state_dict = torch.load(init_weights, map_location=lambda storage, loc: storage)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if i_parts[1] == 'layer5': # decoder
            new_params['decoder.' + '.'.join(i_parts[2:])] = saved_state_dict[i]
    model.load_state_dict(new_params)
    return model

# layer5 in resnet
# decoder structure
class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out
    

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)


class MainDecoder(nn.Module):
    def __init__(self, num_classes):
        super(MainDecoder, self).__init__()
        self.decoder = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        
    def forward(self, x):
        x = self.decoder(x)
        return x

def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv

def guided_cutout(output, resize, erase=0.4, use_dropout=False):

    masks = (output > 0).float().squeeze()

    if use_dropout:
        p_drop = random.randint(3, 6)/10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try: # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except: # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w-min_w, max_h-min_h
            rnd_start_w = random.randint(0, int(bb_w*(1-erase)))
            rnd_start_h = random.randint(0, int(bb_h*(1-erase)))
            h_start, h_end = min_h+rnd_start_h, min_h+rnd_start_h+int(bb_h*erase)
            w_start, w_end = min_w+rnd_start_w, min_w+rnd_start_w+int(bb_w*erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


def guided_masking(x, output, resize, return_msk_context=True):

    masks_context = (output > 0).float()
    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class VATDecoder(nn.Module):
    def __init__(self, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        # self.decoder = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, x, _):
        r_adv = get_r_adv(x, self.decoder, self.it, self.xi, self.eps)
        # x = self.decoder(x + r_adv)
        # x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)

        return x
    
class DropOutDecoder(nn.Module):
    def __init__(self, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        # self.decoder = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, x, _):
        x = self.dropout(x)
        # x = self.decoder(x)
        # x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)

        return x

class CutOutDecoder(nn.Module):
    def __init__(self, num_classes, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        # self.decoder = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, x, pred=None):
        maskcut = guided_cutout(pred, erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        # x = self.decoder(x)
        # x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)

        return x

class ContextMaskingDecoder(nn.Module):
    def __init__(self, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        # self.decoder = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, x, pred=None):
        x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)), return_msk_context=True)
        # x_masked_context = self.decoder(x_masked_context)
        # x_masked_context = nn.functional.interpolate(x_masked_context, size=(256, 256), mode='bilinear', align_corners=True)

        return x_masked_context


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        # self.decoder = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, x, pred=None):
        x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)), return_msk_context=False)
        # x_masked_obj = self.decoder(x_masked_obj)
        # x_masked_obj = nn.functional.interpolate(x_masked_obj, size=(256, 256), mode='bilinear', align_corners=True)
        return x_masked_obj

class FeatureDropDecoder(nn.Module):
    def __init__(self, num_classes):
        super(FeatureDropDecoder, self).__init__()
        # self.decoder = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _):
        x = self.feature_dropout(x)
        # x = self.decoder(x)
        # x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        # self.decoder = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _):
        x = self.feature_based_noise(x)
        # x = self.decoder(x)
        # x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x














