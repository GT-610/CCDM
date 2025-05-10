import torch.nn as nn

class SNPatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, img_size=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False)),  # 添加谱归一化
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),  # 添加谱归一化
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False))  # 添加谱归一化，移除 Sigmoid
        )

    def forward(self, x):
        return self.main(x).view(-1)  # 输出标量值