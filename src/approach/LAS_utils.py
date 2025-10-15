import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotation(inputs):
    batch = inputs.shape[0]
    target = torch.Tensor(np.random.permutation([0, 1, 2, 3] * (int(batch / 4) + 1)), device=inputs.device)[:batch]
    target = target.long()
    image = torch.zeros_like(inputs)
    image.copy_(inputs)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(inputs[i, :, :, :], target[i], [1, 2])

    return image, target

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion( 0,pred, y_a) + (1 - lam) * criterion(0,pred, y_b)

def H2T(x1, x2, rho = 0.3):
    if type(x1).__name__ == 'Tensor':
        fea_num = x1.shape[1]
        index = torch.randperm(fea_num).cuda()
        slt_num = int(rho*fea_num)
        index = index[:slt_num]
        x1[:,index,:,:] = x2[:,index,:,:]
                          #torch.rand(x2[:,index,:,:].shape).to(x2.device) #torch.zeros(x2[:,index,:,:].shape).to(x2.device)
                          #x2[:,index,:,:]
    else:
        for i in range(len(x1)):
            fea_num = x1[i].shape[1]
            index = torch.randperm(fea_num).cuda()
            slt_num = int(rho*fea_num)
            index = index[:slt_num]
            x1[i][:,index,:,:] = x2[i][:,index,:,:]
    return x1


'''-------------五、iFFM模块-----------------------------'''
class iFFM(torch.nn.Module):
    def __init__(self, channels=512, r=4):
        super().__init__()
        inter_channels = int(channels / r)
        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        x = input_1 + input_2
        xl1 = self.local_att(x)
        xg1 = self.global_att(x)
        xlg1 = xl1 + xg1
        wei1 = self.sigmoid(xlg1)
        xi = input_1 * wei1 + input_2 * (1 - wei1)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        x_end = x + x * wei2
        return x_end


class iFFM_avg(torch.nn.Module):
    def __init__(self, channels=512, r=4):
        super().__init__()
        # 保留通道数参数但不使用注意力相关层
        self.channels = channels

    def forward(self, input_1, input_2):
        # 第一次融合：直接平均代替注意力
        xi = (input_1 + input_2) / 2  # 用平均代替注意力权重

        # 第二次融合：直接残差连接（相当于固定权重为0.5）
        x_end = xi + xi * 0.5  # 固定权重融合
        return x_end


class LabelAwareSmoothing(nn.Module):
    def __init__(self, cls_num_list, smooth_head, smooth_tail, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, x, target):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()

class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
        # self.learned_norm = nn.Linear(num_classes,num_classes)

    def forward(self, x):
        # return F.normalize(self.learned_norm) * x
        return self.learned_norm * x
        # return self.learned_norm(x)
