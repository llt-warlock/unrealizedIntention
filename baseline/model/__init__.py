from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from pytorch_lightning.utilities.seed import isolate_rng

from .accel import (
    SegmentationHead as AccelSegmentationhead,
    ResNetBodyNoChannelPool as AccelModelBodyNoPool)


class SegmentationFusionModel(torch.nn.Module):
    def __init__(self,
                 modalities,
                 mask_len=45):
        """
        """
        super().__init__()

        self.modalities = modalities

        if 'accel' in modalities:
            self.accel_feature_extractor = AccelModelBodyNoPool(c_in=3)
            self.accel_head = AccelSegmentationhead(c_out=1, output_len=mask_len)

    def forward(self, batch: dict):
        """
        """
        masks = []
        if 'accel' in batch:

            print("batch ", batch['accel'])

            print(" in train 0: ", batch['accel'].shape)
            f = self.accel_feature_extractor(batch['accel'])
            print(" in train 1: ", f.shape)
            u = self.accel_head(f)
            print(" in train 2: ", u.shape)
            masks.append(u)

        masks = torch.stack(masks, dim=0)
        masks = masks.mean(dim=0)
        #print("mask shape : ", masks.shape)
        # average over the new mask dim
        return masks
