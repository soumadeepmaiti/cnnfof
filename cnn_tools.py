#Tools.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
import copy

#--
def get_total_datasets(lbox, Ngrid):
    dataset_dir = f"/u/masou/My_Workspace/Git/haloes/halo_finder/R200b/L{lbox}-N{Ngrid}/dataset/bin/"
    #dataset_dir = f"/u/masou/My_Workspace/Git/Aletheia_halo_finder/data/L{lbox}-N{Ngrid}/dataset/bin/"
    #dataset_dir = f"/ptmp/masou/My_Workspace/Git/L{lbox}-N{Ngrid}/dataset/bin/"
    
    #all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.ascii')]
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.bin')]
    total_datasets = len(all_files)
    
    if total_datasets == 0:
        raise ValueError(f"No datasets found in the directory {dataset_dir}.")
    
    return total_datasets

#--
def fnameBuilder_nn(run, lbox, Ngrid, zeros=4):
    base_path = f"/u/masou/My_Workspace/Git/haloes/halo_finder/R200b/L{lbox}-N{Ngrid}/dataset/bin/"
    #base_path = f"/u/masou/My_Workspace/Git/Aletheia_halo_finder/data/L{lbox}-N{Ngrid}/dataset/bin/"
    #base_path = f"/ptmp/masou/My_Workspace/Git/L{lbox}-N{Ngrid}/dataset/bin/"

    if isinstance(run, (list, np.ndarray)):
        run = run[0]

    #filename = f"{str(run).zfill(zeros)}.ascii"
    filename = f"{str(run).zfill(zeros)}.bin"
    full_path = base_path + filename
    return full_path

#--
def numpy_reader(filename, field='pos_vel_host', verbose=False):
    logging.debug(f"Reading file: {filename}")
    # data = np.genfromtxt(filename, dtype=None, encoding='utf-8', names=True)
    data = pd.read_pickle(filename)
    result = {'ids': data['Particle_ID'].to_numpy()}
    
    if 'pos' in field:
        pos = np.vstack((data['Px'].to_numpy(), data['Py'].to_numpy(), data['Pz'].to_numpy()))
        result['pos'] = pos
    
    if 'vel' in field:
        vel = np.vstack((data['PVx'].to_numpy(), data['PVy'].to_numpy(), data['PVz'].to_numpy()))
        result['vel'] = vel
    
    if 'host' in field:
        host = data['Host'].to_numpy()
        result['host'] = host
    
    idx = np.argsort(result['ids'])
    result['ids'] = result['ids'][idx]
    
    if 'pos' in result:
        result['pos'] = result['pos'][:, idx]
    
    if 'vel' in result:
        result['vel'] = result['vel'][:, idx]
    
    if 'host' in result:
        result['host'] = result['host'][idx]

    return result

#--
def psField_hl(run, lbox=200, Ngrid=128, disp_factor=1.0, vel_factor=1.0):
    file_path = fnameBuilder_nn(run, lbox, Ngrid)
    nbody = numpy_reader(file_path, field='pos_vel_host', verbose=False)

    pos = nbody['pos'] * disp_factor  
    vel = nbody['vel'] * vel_factor

    pos = pos.reshape(3, Ngrid, Ngrid, Ngrid)
    vel = vel.reshape(3, Ngrid, Ngrid, Ngrid)

    input_tensor = torch.tensor(np.concatenate([pos, vel]), dtype=torch.float32)
    halo_labels = nbody['host']
    output_tensor = torch.tensor(halo_labels.reshape(1, Ngrid, Ngrid, Ngrid), dtype=torch.float32)

    particle_ids = nbody['ids']
    return input_tensor, output_tensor, particle_ids

#--
class HaloSimuData(Dataset):
    def __init__(self, sim_range, Ngrid, lbox, **kwargs):
        super(HaloSimuData, self).__init__()
        self.range = sim_range
        self.ngrid = Ngrid
        self.lbox = lbox

        Omh = 0.31864557807975047
        h = 0.67
        f = Omh ** 0.55

        self.disp_factor = 1 
        if 'disp_factor' in kwargs:
            self.disp_factor = kwargs.pop('disp_factor')

        self.vel_factor = 1 / (100*f) # IMPORTANT CHANGE
        if 'vel_factor' in kwargs:
            self.vel_factor = kwargs.pop('vel_factor')

    def __len__(self):
        return len(self.range)

    def __getitem__(self, index):
        run = self.range[index]
        qv_gadget, hl_gadget, particle_ids = psField_hl(run, Ngrid=self.ngrid, lbox=self.lbox,
                                                        disp_factor=self.disp_factor, vel_factor=self.vel_factor)
        return qv_gadget, hl_gadget, particle_ids.reshape(self.ngrid, self.ngrid, self.ngrid)

#--
class Trainer:

    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model.to(device)
        self.criterion = criterion  
        self.optimizer = optimizer  
        self.device = device
        self.loss_values = None
        self.state_dict = None
        self.epoch = 0
        self.loss_validation = None
        self.validation_test = None
        self.last_val_epoch = None
        self.best_val_loss = float('inf')
        self.optimal_epoch = 0

    def _train_epoch_(self, dataLoader):
        self.model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        start_time = time.time()

        #ti = time.time()
        for batch, data in enumerate(tqdm(dataLoader, desc="Training", leave=False)):
            inputs, targets = data[0], data[1]
            #tf = time.time()
            #print('Time for loading data: ', tf-ti)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            
            predicted_labels = (outputs >= 0.5).float()
            correct_predictions += (predicted_labels == targets).sum().item()
            total_predictions += targets.numel()

            #ti = time.time()

        epoch_loss /= len(dataLoader)
        accuracy = correct_predictions / total_predictions * 100
        epoch_time = time.time() - start_time

        return epoch_loss, accuracy, epoch_time

    def _eval_validation_(self, validationLoader):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for data in tqdm(validationLoader, desc="Validating", leave=False):
                inputs, targets = data[0], data[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                predicted_labels = (outputs >= 0.5).float()
                correct_predictions += (predicted_labels == targets).sum().item()
                total_predictions += targets.numel()

        val_loss /= len(validationLoader)
        accuracy = correct_predictions / total_predictions * 100

        return val_loss, accuracy

    def train(self, num_epochs, trainLoader, validationLoader=None, optimal_model=True):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        epoch_times = []
        
        total_start_time = time.time()

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            train_loss, train_accuracy, epoch_time = self._train_epoch_(trainLoader)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            epoch_times.append(epoch_time)

            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Time: {epoch_time:.2f}s')
            
            if validationLoader:
                val_loss, val_accuracy = self._eval_validation_(validationLoader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.state_dict = copy.deepcopy(self.model.state_dict())
                    self.optimal_epoch = epoch + 1  # Save the 1-indexed epoch
                    print(f'Best validation loss so far: {val_loss:.4f} at epoch {self.optimal_epoch}')
            else:
                val_losses.append(None)
                val_accuracies.append(None)
        
        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time

        hours, remainder = divmod(total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'Total training time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds')

        if optimal_model and self.state_dict:
            self.model.load_state_dict(self.state_dict)

        return train_losses, val_losses, train_accuracies, val_accuracies, epoch_times

    def test(self, testLoader):
        self.model.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for data in tqdm(testLoader, desc="Testing", leave=True):
                inputs, targets = data[0], data[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                predicted_labels = (outputs >= 0.5).float()
                correct_predictions += (predicted_labels == targets).sum().item()
                total_predictions += targets.numel()

        test_loss /= len(testLoader)
        test_accuracy = (correct_predictions / total_predictions) * 100
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        return test_loss, test_accuracy

    def save(self, fn):
        fn_model = fn + '_model.pth'
        fn_loss = fn + '_loss.txt'
        
        torch.save(self.state_dict, fn_model)
        print(f'Model saved to: {fn_model}')

    def load(self, fn):
        fn_model = fn + '_model.pth'
        self.model.load_state_dict(torch.load(fn_model, map_location=self.device))
        print(f'Model loaded from: {fn_model}')