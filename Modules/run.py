from utils.Metrics import *
from Modules.train import *
from torch.optim import AdamW
from utils.seed import seed_everything
from torch.utils.data import DataLoader
from Modules.dataset import CustomDataset, TestDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Run:
    def __init__(self, file_path, config):
        self.config = config
        self.file_path = file_path

        self.lr = self.config['train'].lr
        self.epochs = self.config['train'].epochs
        self.batch_size = self.config['train'].batch_size

        self.seed = self.config['train'].seed
        self.device = self.config['train'].device
        self.num_workers = self.config['train'].num_workers
        self.model_saving_strategy = self.config['train'].model_saving_strategy

        # Seed everything
        seed_everything(self.seed)

    def load_data(self):
        print('Loading data...')
        train = pd.read_csv(self.file_path, index_col=0)
        if self.config['model'].Transfer:
            train_data = train.loc[:'2021-01-01']
            val_data = train.loc['2021-01-01':'2023-01-01']
            test_data = train.loc['2023-01-01':]
            test_dataset = TestDataset(test_data)
            self.test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
            self.test_index = test_data.index[len(test_data) - len(self.test_dl):]

        if not self.config['model'].Transfer:
            train_data = train.loc[:'2012-01-01']
            val_data = train.loc['2012-01-01':]

        train_dataset = CustomDataset(train_data)
        val_dataset = CustomDataset(val_data)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False),
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
        }
        print('Finished loading data!')
        self.dataloaders = dataloaders

    def set_path(self):
        if not self.config['model'].Transfer:
            if self.config['model'].backbone1:
                self.weight_path = f'Weight/Backbone/BiLSTM_{self.file_path[-10:-8]}.pth'

            if self.config['model'].backbone2:
                self.weight_path = f'Weight/Backbone/DLinear_{self.file_path[-10:-8]}.pth'

            if self.config['model'].backbone3:
                self.weight_path = f'Weight/Backbone/MLP_{self.file_path[-10:-8]}.pth'

            if self.config['model'].backbone4:
                self.weight_path = f'Weight/Backbone/NBEATSx_{self.file_path[-10:-8]}.pth'

            if self.config['model'].backbone5:
                self.weight_path = f'Weight/Backbone/Prophet_{self.file_path[-10:-8]}.pth'

        if self.config['model'].Transfer:
            if self.config['model'].backbone1:
                if self.config['model'].additional:
                    self.weight_path = f'Weight/BiLSTM/additional_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/BiLSTM/additional_{self.file_path[-10:-8]}.csv'

                if not self.config['model'].additional:
                    self.weight_path = f'Weight/BiLSTM/additionalX_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/BiLSTM/additionalX_{self.file_path[-10:-8]}.csv'

            if self.config['model'].backbone2:
                if self.config['model'].additional:
                    self.weight_path = f'Weight/DLinear/additional_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/DLinear/additional_{self.file_path[-10:-8]}.csv'

                if not self.config['model'].additional:
                    self.weight_path = f'Weight/DLinear/additionalX_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/DLinear/additionalX_{self.file_path[-10:-8]}.csv'

            if self.config['model'].backbone3:
                if self.config['model'].additional:
                    self.weight_path = f'Weight/MLP/additional_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/MLP/additional_{self.file_path[-10:-8]}.csv'

                if not self.config['model'].additional:
                    self.weight_path = f'Weight/MLP/additionalX_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/MLP/additionalX_{self.file_path[-10:-8]}.csv'

            if self.config['model'].backbone4:
                if self.config['model'].additional:
                    self.weight_path = f'Weight/NBEATSx/additional_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/NBEATSx/additional_{self.file_path[-10:-8]}.csv'

                if not self.config['model'].additional:
                    self.weight_path = f'Weight/NBEATSx/additionalX_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/NBEATSx/additionalX_{self.file_path[-10:-8]}.csv'

            if self.config['model'].backbone5:
                if self.config['model'].additional:
                    self.weight_path = f'Weight/Prophet/additional_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/Prophet/additional_{self.file_path[-10:-8]}.csv'

                if not self.config['model'].additional:
                    self.weight_path = f'Weight/Prophet/additionalX_{self.file_path[-10:-8]}.pth'
                    self.saving_path = f'Files/Prophet/additionalX_{self.file_path[-10:-8]}.csv'

    def run_model(self):
        TM = Train_Module(self.device)
        self.load_data()
        self.set_path()
        print(' ')
        print('Training model...')
        print(' ')

        model = self.config['structure']
        model.to(self.device)

        opt = AdamW(model.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=self.config['train'].patience)

        parameters = {
            'num_epochs': self.epochs,
            'weight_path': self.weight_path,

            'train_dl': self.dataloaders['train'],
            'val_dl': self.dataloaders['val'],

            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
        }

        model, loss_hist, metric_hist = TM.train_and_eval(model, parameters)
        print('Finished training model!')

        self.model = model
        self.loss_hist = loss_hist
        self.metric_hist = metric_hist

    def check_validation(self):
        print(' ')
        print('Check loss and RMSE...')

        loss_hist_numpy = self.loss_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
        metric_hist_numpy = self.metric_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)

        # plot loss progress
        plt.title("Train-Val Loss")
        plt.plot(range(1, self.epochs + 1), loss_hist_numpy.iloc[:, 0], label="train")
        plt.plot(range(1, self.epochs + 1), loss_hist_numpy.iloc[:, 1], label="val")
        plt.ylabel("Loss")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        # plot accuracy progress
        plt.title("Train-Val RMSE")
        plt.plot(range(1, self.epochs + 1), metric_hist_numpy.iloc[:, 0], label="train")
        plt.plot(range(1, self.epochs + 1), metric_hist_numpy.iloc[:, 1], label="val")
        plt.ylabel("RMSE")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        print('Finished checking loss and RMSE!')

    def evaluate_testset(self):
        print(' ')
        print('Evaluation in progress for testset...')
        TM = Train_Module(self.device)
        all_predictions = []
        all_gt = []
        pred = pd.DataFrame(columns=['Predictions', 'Ground Truths'])

        # Needed to load weights after model training and test them later.
        self.model = self.config['structure']
        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.to(self.device)

        self.model.eval()
        with ((torch.no_grad())):
            i = 0
            for X_train, gt in self.test_dl:
                i += 1
                TM.plot_bar('Test', i, len(self.test_dl))
                X_train = X_train.to(self.device)
                output = self.model(X_train)
                output = output.cpu().detach().numpy().tolist()
                gt = gt.squeeze().detach().numpy().tolist()

                all_predictions.append(output)
                all_gt.append(gt)

        pred['Predictions'] = all_predictions
        pred['Ground Truths'] = all_gt
        pred.index = self.test_index

        nd = calculate_nd(all_gt, all_predictions)
        mae = calculate_mae(all_gt, all_predictions)
        rmse = calculate_rmse(all_gt, all_predictions)
        ad_r2 = calculate_adjusted_r2_score(all_gt, all_predictions, 20, 2)

        print(' ')
        print(f'Model: {self.weight_path}')
        print(f'ND: {nd:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'adjusted-R^2: {ad_r2:.4f}')
        print("Finished evaluation!")

        if self.config['model'].Transfer:
            pred.to_csv(self.saving_path)
