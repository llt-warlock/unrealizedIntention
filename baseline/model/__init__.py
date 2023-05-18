
import torch


from .accel import (
    SegmentationHead as AccelSegmentationhead,
    ResNetBodyNoChannelPool as AccelModelBodyNoPool)


class SegmentationFusionModel(torch.nn.Module):
    def __init__(self,
                 modalities,
                 mask_len):
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
            #print("label : ", batch)
            #print("input ", batch['accel'].shape, "  ", batch['accel'])
            f = self.accel_feature_extractor(batch['accel'])
            #print(" in train 1: ", f)
            u = self.accel_head(f)
            #print(" in train 2: ", u.shape)
            #print("in train 2 : ", u)

            if u.size(dim=0) == 100:
                print(u.size())
                print("reshape ?")
                u = u[None, :]
                print(u.size())


            masks.append(u)

            print("u:", u.shape)




        masks = torch.stack(masks, dim=2)

        masks = masks.mean(dim=2)


        #print("mask shape : ", masks.shape)
        # average over the new mask dim
        return masks
