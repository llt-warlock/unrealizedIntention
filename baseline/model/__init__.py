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

    def forward(self, batch:dict):
        """
        """
        masks = []
        if 'accel' in batch:
            f = self.accel_feature_extractor(batch['accel'])
            masks.append(self.accel_head(f))

        masks = torch.stack(masks, dim=2)
        masks = masks.mean(dim=2)

        # average over the new mask dim
        return masks