#!/home/smore/.venvs/py3smore/bin/python3
import torch
import time
import random
import argparse
from dp_model.model_files.sfcn import SFCN
import torchvision.transforms as transforms
from load_data import Brainage_Dataset
from train_validation import *
from pathlib import Path
import matplotlib


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.from_numpy(sample)


class Normalize(object):
    """Normalize the image."""
    def __init__(self, mean, std):
        print('in normalization')
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return sample-self.mean/self.std


def random_seed(seed_value, use_cuda):
    """Set the random seed"""
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input file path (csv) ")  # Path to csv file with subject info
    parser.add_argument("--save_path", type=str, help="Output file path")  # Output filename path
    parser.add_argument("--save_filename", type=str, help="Output file name")  # Output filename name
    parser.add_argument("--learning_rate", type=float, help="Learning rate",
                        default=0.0001)  # Learning rate for optimizer
    parser.add_argument("--num_epochs", type=int, help="Number of epochs",
                        default=240)  # Number of epochs for model training
    parser.add_argument("--batch_size", type=int, help="Batch size",
                        default=3)  # Number of samples to be used in each batch
    parser.add_argument("--train_all_flag", type=int, help="Train all the layers or only outermost",
                        default=0)  # True: Train all the layers, False: train only outermost layer
    parser.add_argument("--optimizer_name", type=str, help="Optimizer name",
                        default='adam')  # the optimizer to be use ('adam' or 'sgd')
    parser.add_argument("--age_range", type=str, help="Age range of the dataset",
                        default='42,82')  # age range for the dataset used: [min, max]
    parser.add_argument("--trained_weights", type=str, help="Trained model parameters",
                        default='./brain_age/run_20190914_10_epoch_best_mae.p')  # model weights for pre-trained model (run_20190719_00_epoch_best_mae for T1)

    # Read all the arguments
    args = parser.parse_args()
    input_file = args.input_file
    save_path = Path(args.save_path)
    save_filename = args.save_filename
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    train_all_flag = args.train_all_flag
    batch_size = args.batch_size
    optimizer_name = args.optimizer_name
    age_range = [int(x.strip()) for x in args.age_range.split(',')]
    trained_weights = Path(args.trained_weights)

    print('Input file used: ', input_file)
    print('Path to results: ', save_path)
    print('File extension to save results: ', save_filename)
    print('Number of epochs: ', num_epochs)
    print('Learning rate: ', learning_rate)
    print('Train all layers flag: ', train_all_flag)
    print('Batch size: ', batch_size)
    print('Optimizer: ', optimizer_name)
    print('Age range ', age_range)
    print('Pre-trained weights file: ', trained_weights)

    # Check if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    start_time = time.time()

    # # define random seeds for reproducibility (if needed)
    # use_cuda = 'False'
    # if device == "cuda:0":
    #     use_cuda = 'True'
    # random_seed(200, use_cuda)

    # calculate mean and standard deviation in batch wise manner and get normalized batches for train and validation (if needed)
    # mean, std = mean_std_batchwise(input_file, batch_size)
    # print('mean', mean, 'std', std)
    # data_transform = transforms.Compose([ToTensor(), Normalize(mean, std)])

    # Create train and validation dataset (train_status = 'train' to get train data, 'validate' to get validation data
    train_dataset = Brainage_Dataset(csv_file=input_file, train_status='train', transform=None, age_range=age_range) #or transform=data_transform
    validation_dataset = Brainage_Dataset(csv_file=input_file, train_status='validate', transform=None, age_range=age_range) #or transform=data_transform
    print('\nLength of train and validation sets', len(train_dataset), len(validation_dataset))

    # Create train and validation dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # check the data
    print("Image dimensions:")
    data = next(iter(train_loader))
    print('mean and std of train', data[0].mean(), data[0].std())
    data = next(iter(validation_loader))
    print('mean and std of val', data[0].mean(), data[0].std())

    # save one image to check
    img = data[0][1,:,:,:,:].numpy()
    b, c, d, h, w = data[0].shape
    img = img.reshape(d, h, w)
    matplotlib.image.imsave('img_90.png', img[:,:,90])

    # Load the model architecture and initial it with weights from initial_weights file
    model = SFCN_mod(trained_weights=trained_weights, device=device, age_range=age_range)
    print('\nModel architecture: \n', model)

    # If False trains only the outermost layer (freeze model weights for all the layers by setting requires_grad=False and set it
    # true only for the outermost layers)
    if train_all_flag == 0: # if 1: trains all the layers, if 0: trains the outermost layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.model.module.classifier.conv_6.parameters():
            param.requires_grad = True

    # Optimizer initialization
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9,
                                weight_decay=0.001)

    # send the model on the device (CPU or GPU)
    model.to(device)

    # print model state dictionary and optimizer state dictionary
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print("\nOptimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name], '\n')

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # Train the model
    model = train_model(model, num_epochs, train_loader, validation_loader, optimizer, save_path, save_filename,
                        age_range, device, 1)

    time_sec = (time.time() - start_time)
    time_hr = time_sec/(60*60)

    print(f'Training time is {time_hr} hours or {time_sec} seconds')




