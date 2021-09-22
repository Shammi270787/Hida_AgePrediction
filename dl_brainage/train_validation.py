import numpy as np
import torch
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from torch.autograd import Variable
import matplotlib.pyplot as plt
from dp_model.model_files.sfcn import SFCN
import torch.nn as nn
import time
import torch.nn.init
import os
import pandas as pd


def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class SFCN_mod(nn.Module):

    """Taken from original SFCN and adapted it for any age range"""

    def __init__(self, trained_weights, device, age_range):
        super(SFCN_mod, self).__init__()
        self.model = SFCN()  # call the model
        self.model = torch.nn.DataParallel(self.model)

        if trained_weights:
            self.model.load_state_dict(torch.load(trained_weights, map_location=torch.device(device))) # load the weights
            print('** Weights from pre-trained model loaded **')
        else:
            print('** No weights loaded from pre-trained model **')

        n_out = age_range[1] - age_range[0]

        if age_range != [42, 82]: # change the output layer if age range doesn't match
            self.model.module.classifier.conv_6 = nn.Conv3d(64, n_out, padding=0, kernel_size=1)
            torch.nn.init.kaiming_normal_(self.model.module.classifier.conv_6.weight)

    def forward(self, x):
        out = self.model(x)
        return out



def train_model(model, num_epochs, train_loader, validation_loader, optimizer, out_dir, out_filenm, age_range, device, fold):
    """
    :param model: Initial defined model
    :param num_epochs: Number of epochs for model training and validation
    :param train_loader: Dataloader for training dataset
    :param validation_loader: Dataloader for validation dataset
    :param optimizer: Optimizer algorithm to be used
    :param out_dir: Output directory to save trained model and checkpoint
    :param out_filenm: Output filename
    :param age_range: Age range of the dataset
    :param device: CPU or GPU
    :return: The trained model
    """

    if not os.path.exists(out_dir):  # creates the output directory if it doesn't exists
        os.makedirs(out_dir)
        print(os.path.exists(out_dir))

    path_model = os.path.join(out_dir, out_filenm + f'_model_{fold}.pt')
    path_checkpoint = os.path.join(out_dir, out_filenm + f'_checkpoint_{fold}.pt')
    print(f'Model path is {path_model}')
    print(f'Model checkpoint is {path_checkpoint}')

    n_epochs_stop = 20
    min_val_loss = np.Inf
    epochs_no_improve = 0

    bin_step = 1
    sigma = 1
    validation_loss, validation_MSE, validation_MAE = [], [], []
    training_loss, training_MSE, training_MAE = [], [], []

    for epoch in range(num_epochs):

        start_time = time.time()
        print('\n----Epoch number: %d------' %(epoch))
        train_loss = 0
        train_mse, train_mae = [], []

        for i, (data, label) in enumerate(train_loader):
            print('\nTrain loader-%d' %(i))

            # Transforming the age to soft label (probability distribution)
            label = label.numpy().reshape(-1)
            y, bc = dpu.num2vect(label, age_range, bin_step, sigma) # probabilities, bin centers
            y = torch.tensor(y, dtype=torch.float32, device=device)

            # Preprocessing done while loading the data
            # data = data/data.mean(); data = dpu.crop_center(data, (160, 192, 160))
            b, c, d, h, w = data.shape
            input_data = torch.as_tensor(data, dtype=torch.float32, device=device)
            # print(f'Input data shape: {input_data.shape}', f'dtype: {input_data.dtype}')

            # Training
            model.train()
            optimizer.zero_grad()

            output = model(input_data)
            out = output[0].reshape([b, -1])
            loss = dpl.my_KLDivLoss(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Prediction, Visualisation and Summary
            out = out.detach().cpu().numpy()
            y = y.cpu().numpy()
            prob = np.exp(out)
            pred = prob@bc

            train_mse.append(mean_squared_error(label, pred))
            train_mae.append(mean_absolute_error(label, pred))

        train_loss = train_loss/len(train_loader)
        training_loss.append(train_loss)
        training_MSE.append(np.mean(train_mse))
        training_MAE.append(np.mean(train_mae))

        val_loss = 0
        val_mse, val_mae = [], []
        true_labels, predictions = [], []

        for i, (data, label) in enumerate(validation_loader):
            print('\nValidation loader-%d' % (i))

            label = label.numpy().reshape(-1)
            y, bc = dpu.num2vect(label, age_range, bin_step, sigma)
            y = torch.tensor(y, dtype=torch.float32, device=device)

            # Pre-processing done while loading the data
            # data = data / data.mean() data = dpu.crop_center(data, (160, 192, 160))
            b, c, d, h, w = data.shape
            input_data = torch.as_tensor(data, dtype=torch.float32, device=device)
            # print(f'Input data shape: {input_data.shape}', f'dtype: {input_data.dtype}')

            # Evaluation
            model.eval()  # Don't forget this. BatchNorm will be affected if not in eval mode.
            with torch.no_grad():
                output = model(input_data)

            # Output, loss, visualisation
            out = output[0].reshape([b, -1]) # bring it back to cpu if you want out = output[0].cpu().reshape([b, -1])
            loss = dpl.my_KLDivLoss(out, y)
            val_loss += loss.item()

            # Prediction, Visualisation and Summary
            out = out.cpu().numpy()
            y = y.cpu().numpy()
            prob = np.exp(out)
            pred = prob@bc

            true_labels.append(label)
            predictions.append(pred)
            val_mse.append(mean_squared_error(label, pred))
            val_mae.append(mean_absolute_error(label, pred))

        true_labels = np.concatenate(true_labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        val_loss = val_loss/len(validation_loader)
        validation_loss.append(val_loss)
        validation_MSE.append(np.mean(val_mse))
        validation_MAE.append(np.mean(val_mae))

        print('\ntraining_loss', training_loss, '\ntraining_MSE', training_MSE, '\ntraining_MAE', training_MAE)
        print('\nvalidation_loss', validation_loss, '\nvalidation_MSE', validation_MSE, '\nvalidation_MAE',
              validation_MAE)
        print('\ntrue_labels', true_labels, '\npredictions', predictions)

        print("--- %s seconds ---" % (time.time() - start_time))

        if val_loss < min_val_loss:
            print(f'Saving model, current loss: {val_loss}, previous minimum loss: {min_val_loss}')
            torch.save(model.state_dict(), path_model)  # sav the model state dictionary
            epochs_no_improve = 0
            min_val_loss = val_loss
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'validation_loss': validation_loss, 'training_loss': training_loss, 'validation_MSE': validation_MSE,
                 'training_MSE': training_MSE, 'validation_MAE': validation_MAE, 'training_MAE': training_MAE,
                 'true_labels': true_labels, 'predictions': predictions, 'age_range': age_range}, path_checkpoint) # save checkpoint

        else:
            print('Error not reducing')
            epochs_no_improve += 1
            print('Epochs_no_improve', epochs_no_improve)
            if epochs_no_improve == n_epochs_stop: # Check early stopping condition
                epochs_no_improve = 0
                print('Early stopping!') # stop training here
    # torch.save(model.state_dict(), path_model)
    print('Training process has finished')

    return model


def test_model(test_data, model, age_range, device, out_dir, out_filenm):
    """
    :param test_data: Test data, single data at a time
    :param model: final trained model
    :param age_range: age range of data
    :param device: cpu or gpu
    :return: MAE, MSE, CORR between true and predicted
            saves the predictions in a csv file and scatter plot between true and predicted
    """
    df = pd.DataFrame()
    bin_step = 1
    sigma = 1
    true_labels, predictions  = [], []

    test_loss = 0
    print('out_dir', out_dir)

    if not os.path.exists(out_dir):  # creates the output directory if it doesn't exists
        os.makedirs(out_dir)
        print(os.path.exists(out_dir))

    for i, (data, label) in enumerate(test_data):
        y, bc = dpu.num2vect(label, age_range, bin_step, sigma)
        y = torch.tensor(y, dtype=torch.float32, device=device)

        # Preprocessing done while loading the data (don't need anymore)
        # data = data / data.mean(), data = dpu.crop_center(data, (160, 192, 160))

        c, d, h, w = data.shape
        print(c, d, h, w)
        b = 1 # batch size would be one here
        data = data.reshape(1, 1, d, h, w)
        input_data = torch.as_tensor(data, dtype=torch.float32, device=device)

        # Evaluation
        model.eval()  # Don't forget this. BatchNorm will be affected if not in eval mode.
        with torch.no_grad():
            output = model(input_data)

        # Output, loss, visualisation
        out = output[0].cpu().reshape([b, -1])  # bring it back to cpu
        y = y.cpu()
        loss = dpl.my_KLDivLoss(out, y)
        test_loss += loss.item()

        # Prediction, Visualisation and Summary
        out = out.numpy()
        y = y.numpy()
        prob = np.exp(out)
        pred = prob@bc
        print('---i---', 'True:', label,  '   pred:', pred)

        # stores the true labels and predictions in a dataframe
        temp = pd.DataFrame(
            {'i': [i], 'label': int(label * 100) / 100, 'prediction': int(pred*100)/100})
        df = pd.concat([df, temp])

        true_labels.append(label)
        predictions.append(pred)

    # saves the df with true ans predicted age in a csv
    csv_path = os.path.join(out_dir,  out_filenm + '_predictions.csv')
    print('csv_path', csv_path)
    df.to_csv(csv_path, index=False)

    # calculate MSE, MAE and corr between true and predicted
    predictions = np.concatenate(predictions, axis=0)
    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    corr2, p = pearsonr(true_labels, predictions)

    mse = int(mse * 100) / 100
    mae = int(mae * 100) / 100
    corr2 = int(corr2 * 100) / 100

    print('Results',  mse, mae, corr2, p)
    print('\ntrue_labels', true_labels, '\npredictions', predictions)

    # Scatter plot for true vs predicted
    plt.figure()
    plt.plot(true_labels, true_labels, linestyle='-')
    plt.scatter(true_labels, predictions, label=out_filenm)
    plt.title('MSE:' + str(mse) + '   MAE:' + str(mae) + '   Corr:' + str(corr2))
    plt.xlabel('True age')
    plt.ylabel('Predicted age')
    plt.legend()

    file_nm_out = os.path.join(out_dir, out_filenm + '_true_vs_pred.png')
    print('file_nm_out', file_nm_out)
    plt.savefig(file_nm_out)
    plt.close()

    return mae, mse, corr2

