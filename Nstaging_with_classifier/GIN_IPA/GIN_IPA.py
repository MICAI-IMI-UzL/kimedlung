import torch
from Nstaging_with_classifier.GIN_IPA.rand_conv3d import GINGroupConv3D
from monai.transforms import Randomizable, MapTransform, Transform
from typing import Any, Optional
import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return np.exp(-5.0 * phase * phase).astype(float)


def get_current_weight(epoch, saturation_epoch, max_weight=1):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return max_weight * sigmoid_rampup(epoch, saturation_epoch)

def rescale_intensity(data, new_min=0, new_max=1, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c, d, h, w = data.size(0), data.size(1), data.size(2), data.size(3), data.size(4)
    data = data.view(bs*c, -1)
    # pytorch 1.3
    old_max = torch.max(data, dim=1, keepdim=True)[0]
    old_min = torch.min(data, dim=1, keepdim=True)[0]

    new_data = (data - old_min+eps) / (old_max - old_min + eps)*(new_max-new_min)+new_min
    new_data = new_data.view(bs, c, d, h, w)
    return new_data


class GIN_IPA(Randomizable, Transform):
    def __init__(self, prob: float = 0.5, alpha_ramp: bool = False, num_epochs: int = 1000) -> None:
        self.prob = np.clip(prob, 0.0, 1.0)
        self.alpha_ramp = alpha_ramp
        self.num_epochs = num_epochs

    def apply_GIN_IPA(self, img, current_epoch):
        if self.alpha_ramp:
            alpha_max = get_current_weight(current_epoch, self.num_epochs)
        else:
            alpha_max = 1.0
        augmenter = GINGroupConv3D(in_channel=img.shape[1], out_channel=1, interm_channel=4, n_layer=4, alpha_max=alpha_max)#.to(self.device, non_blocking=True)
        img_aug = rescale_intensity(augmenter(img))
        # import SimpleITK as sitk
        # sitk.WriteImage(sitk.GetImageFromArray(img[0].squeeze().cpu().numpy()), '/workspace/img.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(img_aug[0].squeeze().cpu().numpy()), '/workspace/img_aug.nii.gz')
        return img_aug

    def __call__(self, img: torch.Tensor, current_epoch: int = None) -> torch.Tensor:
        if self.R.random() < self.prob:
            return self.apply_GIN_IPA(img, current_epoch)
        else:
            return img
