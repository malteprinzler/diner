import torch


class AntibiasLoss(torch.nn.Module):
    def __init__(self, n_downsampling, metric=torch.nn.L1Loss()):
        super().__init__()
        self.downsampling = torch.nn.AvgPool2d(kernel_size=2 ** n_downsampling, stride=2 ** n_downsampling)
        self.metric = metric

    def forward(self, x, y):
        x = self.downsampling(x)
        y = self.downsampling(y)
        loss = self.metric(x, y)
        return loss
