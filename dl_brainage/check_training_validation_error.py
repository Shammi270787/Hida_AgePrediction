#!/home/smore/.venvs/py3smore/bin/python3
import torch
import random
import argparse
from dp_model.model_files.sfcn import SFCN
import torchvision.transforms as transforms
from load_data import Brainage_Dataset
from train_validation import *
from pathlib import Path
import matplotlib

# parser.add_argument("--trained_model_weights", type=str, help="Trained model parameters",
#                     default='./results/T1/models/sgd_0.0001_model_1.pt')  # model weights for pre-trained model (run_20190719_00_epoch_best_mae for T1)
#
# parser.add_argument("--trained_checkpoint", type=str, help="Trained model parameters",
#                     default='./results/T1/models/sgd_0.0001_checkpoint_1.pt')  # model weights for pre-trained model (run_20190719_00_epoch_best_mae for T1)

# Read all the arguments
# args = parser.parse_args()
# trained_weights = args.trained_model_weights
# checkpoint = args.trained_checkpoint
# print('Saved model weights: ', trained_weights)
# print('Last saved checkpoint: ', checkpoint)

model_name = 'sgd_0.001_all'
trained_weights = './results/T1/models/' + model_name + '_model_1.pt'
checkpoint = './results/T1/models/' + model_name + '_checkpoint_1.pt'
print('Saved model weights: ', trained_weights)
print('Last saved checkpoint: ', checkpoint)


# Check if cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

checkpoint_dict = torch.load(checkpoint, map_location=torch.device(device))

for key, value in checkpoint_dict.items():
    print(key)

print('Epoch', checkpoint_dict['epoch'])
print('validation_loss', len(checkpoint_dict['validation_loss']))
print('training_loss', len(checkpoint_dict['training_loss']))

plt.figure()
plt.scatter(range(0, len(checkpoint_dict['validation_loss'])), checkpoint_dict['validation_loss'], label ='validation loss')
plt.scatter(range(0, len(checkpoint_dict['training_loss'])), checkpoint_dict['training_loss'], label ='training loss')
plt.gca().set(xlabel='Epoch', ylabel='Loss', title = model_name)
plt.legend()
plt.savefig('./train_validation_loss/' + model_name + '_LossVsEpoch.png')

plt.figure()
plt.scatter(range(0, len(checkpoint_dict['validation_MAE'])), checkpoint_dict['validation_MAE'], label ='validation MAE')
plt.scatter(range(0, len(checkpoint_dict['training_MAE'])), checkpoint_dict['training_MAE'], label ='training MAE')
plt.gca().set(xlabel='Epoch', ylabel='MAE', title = model_name)
plt.legend()
plt.savefig('./train_validation_loss/' + model_name + '_MAEVsEpoch.png')



