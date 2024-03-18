import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy import io

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    root = ..
    """
    def __init__(self, image_root, gt_root, trainsize, txt=None, mode='train', scale=255):
        self.mode = mode
        self.trainsize = trainsize
        self.scale = scale
        if txt is not None:  # read txt 
            f=open(txt, encoding='gbk')
            self.images = [image_root + line.strip() for line in f]
            f=open(txt, encoding='gbk')
            self.gts = [gt_root + line.strip() for line in f]
        else:
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
                                ])
        if mode == 'train':
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
                ])  
        elif mode == 'test':
            self.gt_transform = transforms.Compose([
                transforms.ToTensor()
                ]) 

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if self.mode == 'test':
            name = self.images[index].split('/')[-1]
            return image*self.scale, gt, name
        elif self.mode == 'train':
            return image*self.scale, gt


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

if __name__ == '__main__':
    f=open('/opt/data/private/datasets/Polyp/EndoScene/UDA_train.txt', encoding='gbk')
    txt=[line.strip() for line in f]
    print(txt)