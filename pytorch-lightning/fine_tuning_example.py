import os
from typing import Callable
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import Trainer
from albumentations.pytorch import ToTensor

import numpy as np
import PIL
import pytorch_lightning as pl
from fastai.vision.learner import create_cnn_model
from fastai.vision.models import resnet34
from fastai.callbacks.hooks import num_features_model
from fastai.metrics import error_rate
from fastai.torch_core import requires_grad, bn_types, split_model, apply_init
from fastai.vision.learner import create_cnn_model, cnn_config, create_body, create_head
from fastai.layers import FlattenedLoss
from fastai.datasets import URLs, untar_data
import albumentations as A

IMG_NAME = r"(.*?)_\d+.jpg$"

class CNNPretrainedModel(nn.Module):
    """
    Customized from fastai learner
    """
    def __init__(self, base_arch, no_classes, dropout=0.5, init=nn.init.kaiming_normal_):
        super(CNNPretrainedModel, self).__init__()

        self.model = create_cnn_model(base_arch, no_classes, ps=dropout)
        self.meta = cnn_config(base_arch)
        self.split(self.meta['split'])
        self.freeze()

        apply_init(self.model[1], init)

    def split(self, split_on):
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on(self.model)
        self.layer_groups = split_model(self.model, split_on)
        return self

    def freeze_to(self, n):
        "Freeze layers up to layer group `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not isinstance(l, bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)

    def freeze(self):
        "Freeze up to last layer group."
        assert(len(self.layer_groups) > 1)
        self.freeze_to(-1)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def forward(self, x):
        return self.model.forward(x)


class ImageListDataset(Dataset):

    def __init__(self, base_folder, transforms, labels):
        self.base_folder = base_folder
        self.transforms = transforms
        self.image_list = [img for img in self.base_folder.ls() if img.name.endswith('.jpg')]
        self.labels = labels

    def __len__(self):
        return len(self.image_list)

    def open_image(self, img_path):
        img = np.array(PIL.Image.open(img_path).convert('RGB')).astype('float32')
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, axis=2)
        img = img/255.
        return img

    def __getitem__(self, idx):
        file_name = self.image_list[idx].name
        img = self.open_image(self.image_list[idx])
        img = self.transforms(image=img)['image']

        lbl = re.findall(IMG_NAME, file_name.lower())[0]
        return {
            'image': img,
            'label': np.array([self.labels[lbl]])
        }


class IITPetClassification(pl.LightningModule):

    def __init__(self, path, transforms, params):
        super(IITPetClassification, self).__init__()

        self.params = params
        self.path = path
        self.transforms = transforms
        self.loss_func = FlattenedLoss(nn.CrossEntropyLoss)

        img_names = set([re.findall(IMG_NAME, x.name.lower())[0] for x in self.path.ls() if x.name.lower().endswith('.jpg')])
        self.labels = {lbl: i for i, lbl in enumerate(img_names)}

        self.model = CNNPretrainedModel(resnet34, len(self.labels))

    def create_lr_scheduler(self, each_step, optimizer):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        each_step['lr'],
                                                        steps_per_epoch=self.steps_per_epoch,
                                                        epochs=each_step['epochs'])
        return scheduler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.trainer.batch_idx == 0:
            if self.trainer.current_epoch == self.params['stages'][0]['epochs']:
                self.model.freeze_to(-2)
                self.trainer.lr_schedulers[0]['scheduler'] = self.create_lr_scheduler(self.params['stages'][1], self.trainer.optimizers[0])
            elif self.trainer.current_epoch == self.params['stages'][0]['epochs'] + self.params['stages'][1]['epochs']:
                self.model.freeze_to(0)
                self.trainer.lr_schedulers[0]['scheduler'] = self.create_lr_scheduler(self.params['stages'][2], self.trainer.optimizers[0])

        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)

        tensorboard_logs = {
            'train_loss': loss,
            'error_rate': error_rate(F.softmax(y_hat, dim=-1), y)
        }
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        train_dl = self.train_dataloader()
        self.steps_per_epoch = len(train_dl)

        schedulers = []
        current_epochs = 0
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        schedulers.append({
            "scheduler": self.create_lr_scheduler(self.params['stages'][0], optimizer),
            "interval" : "step"
            })
        return [optimizer], schedulers

    def train_dataloader(self):
        return DataLoader(
            ImageListDataset(self.path, self.transforms, self.labels),
            batch_size=self.params['batch_size'],
            shuffle=True,
            num_workers=4)


if __name__ == "__main__":

    dataset_path = untar_data(URLs.PETS)
    tfms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightness(),
            ], p=0.3),
        A.ShiftScaleRotate(),
        A.Normalize(max_pixel_value=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(),
        ])

    params = {
        "batch_size": 64,
        "stages": [
            {
                "epochs": 4,
                "lr": 0.001,
                "freeze_to": -1
            },
            {
                "epochs": 4,
                "lr": 0.0001,
                "freeze_to": -2
            },
            {
                "epochs": 4,
                "lr": 0.00001,
                "freeze_to": 0
            }
        ]
    }
    system = IITPetClassification(dataset_path/'images', tfms, params)
    trainer = Trainer(max_epochs=12, gpus=1)
    trainer.fit(system)
