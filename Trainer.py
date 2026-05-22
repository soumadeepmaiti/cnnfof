#Trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from Tools import HaloSimuData, Trainer, get_total_datasets
from Model import He2019ClassificationNet, BasicBlockHaloClassification
import logging

#-- Settings

lbox = 93.75
Ngrid = 128

train_size = 350
val_size = 50
test_size = 100

train_start_idx = 0
val_start_idx = 350
test_start_idx = 400

total_datasets = get_total_datasets(lbox, Ngrid)

if (train_start_idx + train_size > total_datasets) or (val_start_idx + val_size > total_datasets) or (test_start_idx + test_size > total_datasets):
    raise ValueError(f"Dataset sizes exceed available datasets ({total_datasets}).")

batch_size = 4
num_workers = 16
num_epochs = 150

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-- Data set     

train_runs = np.arange(train_start_idx, train_start_idx + train_size)
val_runs = np.arange(val_start_idx, val_start_idx + val_size)
test_runs = np.arange(test_start_idx, test_start_idx + test_size)

trainSet = HaloSimuData(train_runs, Ngrid, lbox)
valSet = HaloSimuData(val_runs, Ngrid, lbox)
testSet = HaloSimuData(test_runs, Ngrid, lbox)

trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
valLoader = DataLoader(valSet, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

#-- Model

model = He2019ClassificationNet(BasicBlockHaloClassification, in_channels=6, out_channels=1)
model = nn.DataParallel(model)
model = model.to(device)

#-- Optimization

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = np.inf  
optimal_epoch = -1  

#-- Training

t = Trainer(model, criterion, optimizer, device=device)
#t.load('L200-N128_model_checkpoint')

t.train(num_epochs, trainLoader=trainLoader, validationLoader=valLoader)
t.save("L93.75-N128_new")

#t.save("L93.75-N128_halo_classification")

#-- Testing (optional)

# t.test(testLoader)
print(f'Optimal epoch based on validation loss: {t.optimal_epoch}')