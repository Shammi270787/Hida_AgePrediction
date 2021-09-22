import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_train_test_csv(csv_file):

    phenotype = pd.read_csv(csv_file, sep=',')
    # phenotype.drop(columns='age', inplace=True)

    if 'age' in phenotype.columns:
        qc = pd.cut(phenotype['age'].tolist(), bins=5, precision=1)  # create bins for age
        X_train, X_test = train_test_split(phenotype, test_size=0.2, random_state=42, stratify=qc.codes)

        # check the ages included in train and test set
        # X_train.sort_values(by='age', inplace=True, ignore_index=True)
        # X_test.sort_values(by='age', inplace=True, ignore_index=True)
        # print(X_train['age'].astype(int).unique())
        # print(X_test['age'].astype(int).unique())

    else:
        raise ValueError("The file doesn't not have a 'age' column")

    return X_train, X_test


if __name__ == '__main__':
    csv_file = './ixi_subject_list.csv'

    X_train, X_test = create_train_test_csv(csv_file)
    X_train.to_csv('ixi_subject_list_train.csv', index=False)
    X_test.to_csv('ixi_subject_list_test.csv', index=False)






