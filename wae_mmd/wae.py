import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, bias=False, padding=1),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, bias=False, padding=1),
            nn.BatchNorm2d(16, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, bias=False, padding=1),
            nn.BatchNorm2d(8, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, 3, bias=False, padding=1),
            nn.BatchNorm2d(4, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(4 * 8 * 8, self.n_z, bias=False)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x=x.view(-1,4*8*8)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h *8* 4 * 4),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h * 8, self.dim_h * 4, 3, padding=1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 2, 3, padding=1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h * 2, self.dim_h , 3, padding=1),
            nn.BatchNorm2d(self.dim_h ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h, self.dim_h//2, 3, padding=1),
            nn.BatchNorm2d(self.dim_h//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h//2, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 4, 4)
        x = self.main(x)
        return x