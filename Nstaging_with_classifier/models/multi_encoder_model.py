import torch
from torch import nn
from Nstaging_with_classifier.models.CNN_for_size_64 import CNN as CNN_64
from Nstaging_with_classifier.models.CNN_for_size_32 import CNN as CNN_32
from monai.data import MetaTensor


class MultiEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config['spatial_size'] == (32, 32, 32):
            CNN = CNN_32
        else:
            CNN = CNN_64
        num_channels = 1
        if self.config['mask_as_prior']:
            num_channels += 1
        if self.config['modality'] == 'PET':
            num_channels += 1
        if self.config['share_weights']:
            self.enc = CNN(self.config, in_channels=num_channels, n_feat=4)
        else:
            enc = CNN(self.config, in_channels=num_channels, n_feat=4)
            self.enc_list = nn.ModuleList([enc for _ in range(len(self.config['lymphnode_names']))])
        num_features = len(self.config['lymphnode_names']) + 1
        self.linear = nn.Sequential(nn.Linear(num_features, 32), nn.Linear(32, 4))
        self.pt_loc_encoder = nn.Linear(1, 1)
        # Initialize weights of pt_encoder
        self.pt_loc_encoder.weight.data.fill_(1)
        self.pt_loc_encoder.bias.data.zero_()

    def forward(self, x, pt_loc, pt_suv=None):
        pred_list = []
        for channel in range(len(self.config['lymphnode_names'])):
            # Attention: There is no way to process 3 channels on series level.
            if self.config['modality'] == 'PET' or self.config['mask_as_prior']:
                input = torch.concat([x[:, channel].unsqueeze(1), x[:, channel+len(self.config['lymphnode_names'])].unsqueeze(1)], dim=1)
            elif self.config['modality'] == 'CT':
                input = x[:, channel].unsqueeze(1)
            else:
                raise Exception('Unknown modality.')
            if self.config['encode_lnstation']:
                if self.config['share_weights']:
                    pred = self.enc(input, ln=MetaTensor(torch.full([x.shape[0], 1], channel).type(x.dtype).to(x.device)))
                else:
                    pred = self.enc_list[channel](input, ln=MetaTensor(torch.full([x.shape[0], 1], channel).type(x.dtype)).to(x.device))
            elif self.config['encode_lnstation'] and self.config['encode_pt_suv']:
                if self.config['share_weights']:
                    pred = self.enc(input, ln=MetaTensor(torch.full([x.shape[0], 1], channel).type(x.dtype)).to(x.device), pt_suv= pt_suv)
                else:
                    pred = self.enc_list[channel](input, ln=MetaTensor(torch.full([x.shape[0], 1], channel).type(x.dtype)).to(x.device), pt_suv=pt_suv)
            elif not self.config['encode_lnstation'] and self.config['encode_pt_suv']:
                if self.config['share_weights']:
                    pred = self.enc(input, pt_suv= pt_suv)
                else:
                    pred = self.enc_list[channel](input, pt_suv=pt_suv)
            else:
                if self.config['share_weights']:
                    pred = self.enc(input)
                else:
                    pred = self.enc_list[channel](input)
            pred_list.append(pred)
        pred_lnlevel = torch.concat(pred_list, dim=1)
        feat_pt_loc = self.pt_loc_encoder(pt_loc)
        pred_nstage = self.linear(torch.concat((pred_lnlevel, feat_pt_loc), dim=1))
        return pred_lnlevel.view(-1), pred_nstage