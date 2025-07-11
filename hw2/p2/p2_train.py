# ============================================================================
# File: p2_train.py
# Date: 2025-03-11
# Author: TA
# Description: Training a model and save the best model.
# ============================================================================

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

import config as cfg
from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log


def plot_learning_curve(
        logfile_dir: str,
        result_lists: list
    ):
    '''
    Plot and save the learning curves under logfile_dir.
    - Args:
         - logfile_dir: str, the directory to save the learning curves.
         - result_lists: dict, the dictionary contains the training and
                         validation results with keys
                         'train_acc', 'train_loss', 'val_acc', 'val_loss'.
     - Returns:
         - None
    '''
    ################################################################
    # TODO:                                                        #
    # Plot and save the learning curves under logfile_dir, you can #
    # use plt.plot() and plt.savefig().                            #
    #                                                              #
    # NOTE:                                                        #
    # You have to attach four plots of your best model in your     #
    # report, which are:                                           #
    #   1. training accuracy                                       #
    #   2. training loss                                           #
    #   3. validation accuracy                                     #
    #   4. validation loss                                         #
    #                                                              #
    # NOTE:                                                        #
    # This function is called at end of each epoch to avoid the    #
    # plot being unsaved if early stop, so the result_lists's size #
    # is not fixed.                                                #
    ################################################################
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(result_lists['train_acc'], label='train')
    plt.plot(result_lists['val_acc'], label='val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(result_lists['train_loss'], label='train')
    plt.plot(result_lists['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(logfile_dir, 'learning_curve.png'))
    plt.close()


def train(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        unlabeled_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        logfile_dir: str,
        model_save_dir: str,
        criterion: nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim,
        device: torch.device
    ):
    '''
    Training and validation process.
    - Args:
        - model: nn.Module, the model to be trained.
        - train_loader: DataLoader, the dataloader of training set.
        - val_loader: DataLoader, the dataloader of validation set.
        - logfile_dir: str, the directory to save the log files.
        - model_save_dir: str, the directory to save the best model.
        - criterion: nn.Module, the loss function.
        - optimizer: torch.optim, the optimizer.
        - scheduler: torch.optim.lr_scheduler, the learning rate scheduler.
        - device: torch.device, the device to run the model.
    - Returns:
        - None
    '''

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0
    patience = 10  # Number of epochs to wait for improvement
    counter = 0  # To count the number of epochs with no improvement
    
    best_val_loss = np.inf  # Best validation loss observed
    for epoch in range(cfg.epochs):
        ##### TRAINING #####
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0
        model.train()
        for batch, data in enumerate(train_loader):
            sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] Train batch: {batch + 1} / {len(train_loader)}')
            sys.stdout.flush()
            # Data loading. (batch_size, 3, 32, 32), (batch_size)
            images, labels = data['images'].to(device), data['labels'].to(device)
            # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
            pred = model(images)
            # Calculate loss.
            loss = criterion(pred, labels)
            # Backprop. (update model parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate.
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()
        # Print training result
        train_time = time.time() - train_start_time
        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc.cpu().numpy())
        train_loss_list.append(train_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}')

        ##### VALIDATION #####
        model.eval()
        
        confident_samples = [] 
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0
            
            #############################################################
            # TODO:                                                     #
            # Finish forward part in validation, you can refer to the   #
            # training part.                                            #
            #                                                           #
            # NOTE:                                                     #
            # You don't have to update parameters, just record the      #
            # accuracy and loss.                                        #
            #############################################################
            for batch, data in enumerate(val_loader):
                sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] Val batch: {batch + 1} / {len(val_loader)}')
                sys.stdout.flush()
                images, labels = data['images'].to(device), data['labels'].to(device)
                pred = model(images)
                loss = criterion(pred, labels)
                val_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                val_loss += loss.item()
            #semi supervice
            threshold = 0.91 - 0.005 * epoch
            if epoch in [20, 21, 22, 23]:
                for batch, data in enumerate(unlabeled_loader):
                    sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] unlabel batch: {batch + 1} / {len(unlabeled_loader)}')
                    sys.stdout.flush()
                    images = data['images'].to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    max_probs, preds = torch.max(probs, 1) 

                    confident_samples.append((images[max_probs > threshold], preds[max_probs > threshold]))
            ######################### TODO End ##########################

        # Print validation result
        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc.cpu().numpy())
        val_loss_list.append(val_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
        
        # Scheduler step
        scheduler.step() 

        model.train()
        
        for images, pseudo_labels in []:
            outputs = model(images)
            loss = criterion(outputs, pseudo_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(logfile_dir, 'result_log.txt'),
                         epoch, epoch_time,
                         train_acc, val_acc,
                         train_loss, val_loss,
                         is_better)

        ##### SAVE THE BEST MODEL #####
        if is_better:
            print(f'[{epoch + 1}/{cfg.epochs}] Save best model to {model_save_dir} ...')
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, 'model_best.pth'))
            best_acc = val_acc

         # Early stopping check
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset counter if validation loss improves
        else:
            counter += 1  # Increment counter if no improvement

        # If counter exceeds patience, stop training
        if counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break  # Exit the loop and stop training

        ##### PLOT LEARNING CURVE #####
        ##### TODO: check plot_learning_curve() in this file #####
        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }
        plot_learning_curve(logfile_dir, current_result_lists)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', 
                        help='dataset directory', 
                        type=str, 
                        default='../hw2_data/p2_data/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Experiment name
    exp_name = cfg.model_type \
        + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S') \
        + '_' + cfg.exp_name

    # Write log file for config
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    write_config_log(os.path.join(logfile_dir, 'config_log.txt'))

    # Fix a random seed for reproducibility
    set_seed(2025)

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ##### MODEL #####
    ##### TODO: check model.py #####
    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    if cfg.model_type == 'mynet':
        model = MyNet()
    elif cfg.model_type == 'resnet18':
        model = ResNet18()
    else:
        raise NameError('Unknown model type')

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    ##### DATALOADER #####
    ##### TODO: check dataset.py #####
    train_loader = get_dataloader(os.path.join(dataset_dir, 'train'),
                                  batch_size=cfg.batch_size, split='train')
    val_loader   = get_dataloader(os.path.join(dataset_dir, 'val'),
                                  batch_size=cfg.batch_size, split='val')
    unlabeled_loader   = get_dataloader(os.path.join(dataset_dir, 'unlabel'),
                                  batch_size=cfg.batch_size, split='unlabel')

    ##### LOSS & OPTIMIZER #####
    criterion = nn.CrossEntropyLoss()
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                    momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=0.1)
    
    ##### TRAINING & VALIDATION #####
    ##### TODO: check train() in this file #####
    train(model=model,
          train_loader=train_loader,
          unlabeled_loader=unlabeled_loader,
          val_loader=val_loader,
          logfile_dir=logfile_dir,
          model_save_dir=model_save_dir,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device)
    
if __name__ == '__main__':
    main()
