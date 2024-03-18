import numpy as np

class PseudoLabel:
    def __init__(self):
        h, w = 256,256
        self.prob_tar = np.zeros([1, h, w])
        self.thres = 0
        self.iter = 0


    def update_pseudo_label(self, input):
        input = input.sigmoid().detach()
        prob_np = input.cpu().numpy()
        if self.iter==0:
            self.prob_tar = prob_np
        else:
            self.prob_tar = np.append(self.prob_tar, prob_np, axis=0)
        self.iter += 1

    def get_threshold_const(self, thred, percent=0.6):

        x = self.prob_tar[self.prob_tar >= 0.5]
        if len(x) == 0:
            self.thres=thred
            return self.thres
        x = np.sort(x)
        index = np.int(np.round(len(x) * percent))
        print(x,index)
        self.thres = x[index]
        print(self.thres)
        if self.thres > thred:
            self.thres = thred

        return self.thres
