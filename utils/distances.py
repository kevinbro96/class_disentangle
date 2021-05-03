from torch import nn

class L2Distance(nn.Module):
    def forward(self, img1, img2):
        return (img1 - img2).reshape(img1.shape[0], -1).norm(dim=1)


class LinfDistance(nn.Module):
    def forward(self, img1, img2):
        return (img1 - img2).reshape(img1.shape[0], -1).abs().max(dim=1)[0]