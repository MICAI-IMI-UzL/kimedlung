from torch import nn
import torch


class CNN(nn.Module):

    def __init__(self, config: dict, in_channels: int, n_feat: int):
        super().__init__()
        self.config = config
        p = 0.2
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=2 * n_feat, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(2 * n_feat),
            nn.Dropout3d(0.25 * p, True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=2 * n_feat, out_channels=4 * n_feat, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(4 * n_feat),
            nn.Dropout3d(0.5 * p, True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=4 * n_feat, out_channels=8 * n_feat, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(8 * n_feat),
            nn.Dropout3d(p, True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=8 * n_feat, out_channels=16 * n_feat, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(16 * n_feat),
            nn.Dropout3d(p, True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=16 * n_feat, out_channels=32 * n_feat, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(32 * n_feat),
            nn.Dropout3d(p, True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=32 * n_feat, out_channels=64 * n_feat, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(64 * n_feat),
            nn.Dropout3d(p, True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ReLU(True),
        )
        num_feat = n_feat * 64
        if self.config['encode_lnstation']:
            num_feat += 1
        if self.config['encode_pt_suv']:
            num_feat += 1
        self.fc = nn.Sequential(
            nn.Linear(num_feat, self.config['num_classes'])
        )
        self.enc_ln = nn.Linear(1, 1)
        self.enc_pt_suv = nn.Linear(1, 1)
        self.flatten = nn.Flatten()

    def forward(self, x, ln=None, pt_suv=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        if self.config['encode_lnstation']:
            x = torch.concat([self.flatten(x), self.enc_ln(ln)], dim=1)
        if self.config['encode_pt_suv']:
            x = torch.concat([self.flatten(x), self.enc_pt_suv(pt_suv)], dim=1)
        x = self.flatten(x)
        out = self.fc(x)
        return out