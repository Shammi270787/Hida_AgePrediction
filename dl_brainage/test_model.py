#!/home/smore/.venvs/py3smore/bin/python3
import torch
import argparse
from train_validation import SFCN_mod
from dp_model.model_files.sfcn import SFCN
import torchvision.transforms as transforms
from load_data import Brainage_Dataset
from train_validation import *
from pathlib import Path
import nibabel as nib
import matplotlib
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input file path")  # Path to csv file with subject info
    parser.add_argument("--save_path", type=str, help="Output file path")  # Output filename name
    parser.add_argument("--save_filename", type=str, help="Output file name")  # Output filename name
    parser.add_argument("--age_range", type=str, help="Age range",
                        default='42,82')  # age range for the dataset
    parser.add_argument("--trained_weights", type=str, help="Trained model parameters",
                        default='./brain_age/run_20190719_00_epoch_best_mae.p')  # model weights file

    args = parser.parse_args()
    input_file = args.input_file  # contains site, participant_id, age, sex and file_path
    save_path = Path(args.save_path)
    save_filename = args.save_filename
    age_range = [int(x.strip()) for x in args.age_range.split(',')]
    trained_weights = Path(args.trained_weights)

    print(input_file, save_path, save_filename, age_range, trained_weights)

    print('\nInput file used: ', input_file)
    print('Path to results: ', save_path)
    print('File extension to save results: ', save_filename)
    print('Age range ', age_range)
    print('Pre-trained weights file used: ', trained_weights)
    # input_file, save_path, save_filename, age_range, trained_weights = './ixi_subject_list_test.csv', './results', 'xx', [42, 82], './results/train_samples_42_82_model.pt'

    # Check if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)

    # Create test dataset (train_status = 'test')
    test_dataset = Brainage_Dataset(csv_file=input_file, train_status='test', transform=None, age_range=age_range) #or transform=data_transform
    print('\nLength of test sets:', len(test_dataset))

    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False)

    # Load the model architecture
    cnn_model = SFCN_mod(trained_weights='', device=device, age_range=age_range)

    print(trained_weights)
    print(trained_weights.name)

    # Load the trained weights in the model
    if trained_weights.name == 'run_20190719_00_epoch_best_mae.p' or trained_weights.name == 'run_20190914_10_epoch_best_mae.p':
        cnn_model.model.load_state_dict(torch.load(trained_weights, map_location=torch.device(device)))
    else:
        cnn_model.load_state_dict(torch.load(trained_weights, map_location=torch.device(device)))

    print('\nModel architecture \n', cnn_model)
    cnn_model.to(device)

    # test the model
    mae, mse, corr = test_model(test_dataset, cnn_model, age_range, device, save_path, save_filename)
    print('\nMAE:', mae, 'MSE:', mae, 'Correlation:', corr)


