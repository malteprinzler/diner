# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision
import torch as th


class Vgg19(th.nn.Module):
    ''' This Module is adapted from:
        "Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale
        image recognition. arXiv preprint arXiv:1409.1556 (2014)"
    '''

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = th.nn.Sequential()
        self.slice2 = th.nn.Sequential()
        self.slice3 = th.nn.Sequential()
        self.slice4 = th.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)

        out = [h_relu1, h_relu2, h_relu3, h_relu4]
        return out


class VGGLoss(th.nn.Module):
    ''' This Module is adapted from:
        "Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale
        image recognition. arXiv preprint arXiv:1409.1556 (2014)"
    '''

    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg_net = Vgg19().cuda()
        self.l1_loss = th.nn.L1Loss()
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x, y = self.normalize(x), self.normalize(y)
        x_vgg, y_vgg = self.vgg_net(x), self.vgg_net(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.l1_loss(x_vgg[i], y_vgg[i].detach())
        return loss
