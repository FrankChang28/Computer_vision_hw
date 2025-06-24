# ============================================================================
# File: config.py
# Date: 2025-03-11
# Author: TA
# Description: Experiment configurations.
# ============================================================================

################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = 'default'  # name of experiment

# Model Options
model_type = 'resnet18'  # 'mynet' or 'resnet18'

# Learning Options
epochs = 80                # train how many epochs
batch_size = 32            # batch size for dataloader 
use_adam = True           # Adam or SGD optimizer
lr = 1e-4                  # learning rate
milestones = [16, 32, 45, 68, 80]  # reduce learning rate at 'milestones' epochs