class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, img_size=64):
        super().__init__()
        # 示例结构（基于SNGAN架构）
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # 输出真假概率
        )

    def forward(self, x):
        return self.main(x).view(-1)