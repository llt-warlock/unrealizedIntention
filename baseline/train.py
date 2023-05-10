import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, \
    precision_recall_curve, accuracy_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler, WeightedRandomSampler

from model import SegmentationFusionModel

import matplotlib.pyplot as plt

class System(pl.LightningModule):
    def __init__(self, modalities, task='classification'):
        super().__init__()
        self.save_hyperparameters()
       
        # self.model = SegmentationFusionModel(modalities, mask_len=60)
        self.model = SegmentationFusionModel(modalities, mask_len=100)
        self.loss_fn = {
            'classification':F.binary_cross_entropy_with_logits,
            'regression': F.mse_loss,
        }[task]

        self.performance_metric = {
            'classification': lambda input, target: roc_auc_score(target.flatten(), input.flatten()),
            'regression': F.mse_loss,
        }[task]
        self.training_loss = []
        self.training_metric = []
        self.val_loss_list = []
        self.val_metric_list = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        output = self.model(batch).squeeze()
        #output = self.model(batch)

        print("in training step : ", "  output : ", output.shape, " target : ", batch['label'].shape)
        print(output, "   ", batch['label'])


        # if output.size(dim=0) == 200:
        #     loss = self.loss_fn(output, batch['label'].float().reshape(-1,))
        # else:
        loss = self.loss_fn(output, batch['label'].float())

        # loss = self.loss_fn(output, batch['label'].float())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        # optimizer = torch.optim.SGD(self.parameters(), lr=.001, momentum=0.9)
        return optimizer

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        # for k, y in batch.items():
        #     print("key : ", k, " value type : " , type(y), " y_shape : ", y.shape)
        #print("output shape : ", t.shape, "  batch shape : ", batch['label'].shape, " output : ", t, " label : ", batch['label'])
        #val_loss = self.loss_fn(t, batch['label'].float())
        # if output.size(dim=0) == 400:
        #     val_loss = self.loss_fn(output, batch['label'].float().reshape(-1,))
        # else:
        val_loss = self.loss_fn(output, batch['label'].float())

        #val_loss = self.loss_fn(t, batch['label'].float().reshape(-1,))

        # val_loss = self.loss_fn(t, batch['label'].float())
        self.log('val_loss', val_loss)
        print("in validation_step : ", val_loss)

        return (output, batch['label'])

    def validation_epoch_end(self, validation_step_outputs):
        # for o in validation_step_outputs:
        #     if o[0].size(dim=0) == 200:
        #         all_outputs = torch.cat([o[0].reshape(-1,) for o in validation_step_outputs]).cpu()
        #     else:
        #         all_outputs = torch.cat([o[0] for o in validation_step_outputs]).cpu()

        print("validaion : " )

        all_outputs = torch.cat([o[0] for o in validation_step_outputs]).cpu()
        all_labels = torch.cat([o[1] for o in validation_step_outputs]).cpu()


        print("all_unsuccessful _ output : ", all_outputs)
        print("all_labels : ", all_outputs)

        # 1-6 valid code
        val_metric = self.performance_metric(all_outputs, all_labels)
        self.log('val_metric', val_metric)


        self.val_metric_list.append(val_metric)
        print("all_unsuccessful metric in val : ", val_metric)



    def test_step(self, batch, batch_idx):
        output = self.model(batch).squeeze()

        return (output, batch['index'], batch['label'])

    def test_epoch_end(self, test_step_outputs):

        # print("In test_epoch_end : ")
        # for i in range(0,len(test_step_outputs)):
        #     print(type(test_step_outputs[i]), " : ", test_step_outputs[i][0] , "  # ", test_step_outputs[i][1])

            #print(i ,"  ####################### ", temp[2].shape)

        all_outputs = torch.cat([o[0] for o in test_step_outputs]).cpu()
        all_indices = torch.cat([o[1] for o in test_step_outputs]).cpu()
        all_labels = torch.cat([o[2] for o in test_step_outputs]).cpu()

        print("all_unsuccessful output : ", len(all_outputs), " all_unsuccessful labels : ", len(all_labels))
        # modify here
        test_metric = self.performance_metric(all_outputs, all_labels)

       #pre = average_precision_score(all_labels.flatten(), all_outputs.flatten())
        precision, recall, t = precision_recall_curve(all_labels.flatten(), all_outputs.flatten())
        self.test_results = {
            'metric': test_metric,
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'index': all_indices,
            'proba': all_outputs
        }
        self.log('test_metric', test_metric)

def _collate_fn(batch):
    batch = batch[0]
    #print(batch)
    return {k: torch.tensor(v) for k,v in batch.items()}


def train(i, train_ds, val_ds, modalities,
        trainer_params={}, prefix=None, task='classification', 
        deterministic=False, eval_every_epoch=False, weights_path=None):

    num_epochs = {
        ('audio',): 10,
        ('accel',): 10,
        ('video',): 15,
        ('audio', 'video', 'accel'): 15
    }

    print("type : ", train_ds)


    # data loaders
    batch_size = 32

    g = torch.Generator()
    g.manual_seed(729387+i)


    # create WeightedRandomSampler to solve the unblanced data

    data_loader_train = DataLoader(
        dataset=train_ds,
        # This line below!
        sampler=BatchSampler(
            RandomSampler(train_ds, generator=g), batch_size=batch_size, drop_last=False
        ),
        num_workers=8,
        generator=g,
        collate_fn=_collate_fn
    )

    # data_loader_train = DataLoader(
    #     dataset=train_ds,
    #     # This line below!
    #     sampler=WeightedRandomSampler(train_sample_weight, len(train_sample_weight), replacement=True),
    #     batch_size=32,
    #     num_workers=8,
    #     generator=g,
    #     collate_fn=_collate_fn
    # )


    # g = torch.Generator()
    # g.manual_seed(897689769+i)

    data_loader_val = DataLoader(
        dataset=val_ds,
        # This line below!
        sampler=BatchSampler(
            SequentialSampler(val_ds), batch_size=batch_size, drop_last=False
        ),
        #sampler=WeightedRandomSampler(val_sample_weight, len(val_sample_weight), replacement=True),
        #batch_size=32,
        num_workers=8,
        generator=g,
        collate_fn=_collate_fn
    )

    system = System(modalities, task=task)
    trainer_fn = partial(pl.Trainer, **trainer_params)
    trainer = trainer_fn(
        accelerator='gpu',
        check_val_every_n_epoch=1 if eval_every_epoch else 10000,
        max_epochs=num_epochs[modalities],
        logger= pl.loggers.TensorBoardLogger(
            save_dir='logs/', name='', 
            version=prefix),
        deterministic=deterministic,
        enable_checkpointing=False)
        
    trainer.fit(system, data_loader_train, data_loader_val)

    if weights_path is not None:
        trainer.save_checkpoint(weights_path)


    # trainer.model.test_results

    return trainer, trainer.model.training_loss, trainer.model.training_metric, \
           trainer.model.val_loss_list, trainer.model.val_metric_list #system.test_results

def test(i, model, test_ds, prefix=None):

    batch_size=32
    # data loaders
    g = torch.Generator()
    g.manual_seed(897689769+i)
    test_dl = DataLoader(
        dataset=test_ds,
        # This line below!
        sampler=BatchSampler(
            SequentialSampler(test_ds), batch_size=batch_size, drop_last=False
        ),
        num_workers=0,
        generator=g,
        collate_fn=_collate_fn
    )

    print("model version is : ", prefix)

    trainer = pl.Trainer(
                logger= pl.loggers.TensorBoardLogger(
                    save_dir='logs/', name='', 
                    version=prefix))

    trainer.test(
        model=model, 
        dataloaders=test_dl)
    
    return trainer.model.test_results