import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torch.nn.functional as F
from dp_model import dp_utils as dpu


class Brainage_Dataset(Dataset):

    """Brainage dataset."""

    def __init__(self, csv_file, transform=None, train_status='train', age_range=[42, 82]):
        """
        Args:
            csv_file (string): Path to the csv file with paths and other phenotypes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            train_status: train or validate or test
            age_range: age range for the used dataset

        """

        # Load the csv file as a dataframe
        phenotype = pd.read_csv(csv_file, sep=',')

        # check if age and file_path are present as columns in phenotype dataframe
        assert (pd.Series(['file_path', 'age']).isin(phenotype.columns).all()), "File must contain age and file_path"

        # select samples for required age range
        phenotype = phenotype[(phenotype['age'] >= age_range[0]) & (phenotype['age'] <= age_range[1])]
        phenotype.sort_values(by='age', inplace=True, ignore_index=True) # sort data based on age

        self.paths = phenotype['file_path'].values  # takes the data path from csv
        self.age = phenotype['age'].values # takes the age from csv
        self.transform = transform
        train_status = train_status

        # create indices for train and test (every 4 sample is used for validation/test)
        # One can use CV from sklearn to create test and train splits
        all_idx = np.arange(0, len(phenotype), 1).tolist()
        test_idx = np.arange(0, len(phenotype), 4).tolist()
        train_idx = list(set(all_idx) - set(test_idx))

        if train_status == 'train':
            self.paths = self.paths[train_idx]
            self.age = self.age[train_idx]
        elif train_status == 'validate':
            self.paths = self.paths[test_idx]
            self.age = self.age[test_idx]
        else:
            self.paths = self.paths[all_idx]
            self.age = self.age[all_idx]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        sub_age = self.age[idx]
        sub_img = nib.load(path)
        sub_data = sub_img.get_fdata()

        sub_data = sub_data / sub_data.mean()
        sub_data = dpu.crop_center(sub_data, (160, 192, 160))  # pre-process the image as required by the CNN model

        if self.transform:
            sub_data = self.transform(sub_data)

        sub_data = np.expand_dims(sub_data, axis=0)

        # print('Image dimension: ', sub_data.shape)

        return sub_data, sub_age


