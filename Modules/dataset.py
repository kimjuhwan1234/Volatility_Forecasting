from torch.utils.data import Dataset

import torch


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.window_size = 5 * 4

    def __len__(self):
        return len(self.data) - self.window_size - 1

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.window_size

        if end_index + 1 > len(self.data):
            raise IndexError("Index out of bounds. Reached the end of the dataset.")

        X_train = self.data.iloc[start_index:end_index, :]
        y_train = self.data.iloc[end_index:end_index + 1, 0]

        X_train_tensor = torch.Tensor(X_train.values)
        y_train_tensor = torch.Tensor(y_train.values)

        return X_train_tensor, y_train_tensor
