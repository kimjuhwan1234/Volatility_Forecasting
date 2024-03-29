from utils.Metrics import *
from Modules.train import *
from Modules.model import *
from torch.optim import AdamW
from utils.seed import seed_everything
from torch.utils.data import DataLoader
from Modules.dataset import CustomDataset
from sklearn.model_selection import train_test_split
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
        self.saving_path = self.config['train'].saving_path
        self.model_saving_strategy = self.config['train'].model_saving_strategy

        # Seed everything
        seed_everything(self.seed)

    def load_data(self):
        print('Loading data...')
        train = pd.read_csv(self.file_path, index_col=0)
        self.test = train.iloc[-52:]
        train = train.iloc[:-52]

        val_ratio = 0.1

        train_data, val_data = train_test_split(train, test_size=val_ratio, shuffle=False)

        train_dataset = CustomDataset(train_data)
        val_dataset = CustomDataset(val_data)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False),
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True),
        }
        print('Finished loading data!')
        self.dataloaders = dataloaders

    def run_model(self):
        TL = Transfer_Learning(self.device)
        self.load_data()
        print(' ')
        print('Training model...')
        print(' ')

        # model = RegressionModel(input_size=self.config['model'].input_size,
        #                         hidden_size=self.config['model'].hidden_size,
        #                         num_layers=self.config['model'].num_layers,
        #                         output_size=self.config['model'].output_size,
        #                         additional=self.config['model'].additional,
        #                         bidirectional=self.config['model'].bidirectional
        #                         )

        # model = single_biLSTM(input_size=self.config['model'].input_size,
        #                       hidden_size=self.config['model'].hidden_size,
        #                       num_layers=self.config['model'].num_layers,
        #                       output_size=self.config['model'].output_size,
        #                       additional=self.config['model'].additional,
        #                       )

        model = MLP(input_size=self.config['model'].input_size,
                    hidden_size=self.config['model'].hidden_size,
                    output_size=self.config['model'].output_size,
                    )

        model.to(self.device)

        if self.config['model'].additional:
            self.weight_path = f'Weight/bi_LSTM_GRU_{self.file_path[-11:-8]}.pth'

        if not self.config['model'].additional:
            self.weight_path = f'Weight/bi_LSTM_{self.file_path[-11:-8]}.pth'

        # model.load_state_dict(torch.load(self.weight_path))

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

        model, loss_hist, metric_hist = TL.train_and_eval(model, parameters)
        print('Finished training model!')

        self.model = model
        self.loss_hist = loss_hist
        self.metric_hist = metric_hist

    def check_validation(self):
        print(' ')
        print('Check loss and adjusted_R_square...')

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
        plt.title("Train-Val adjusted_R_square")
        plt.plot(range(1, self.epochs + 1), metric_hist_numpy.iloc[:, 0], label="train")
        plt.plot(range(1, self.epochs + 1), metric_hist_numpy.iloc[:, 1], label="val")
        plt.ylabel("adjusted_R_square")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        print('Finished checking loss and adjusted R_square!')

    def evaluate_testset(self, saving_path):
        print(' ')
        print('Evaluation in progress for testset...')
        TL = Transfer_Learning(self.device)
        all_predictions = []
        all_gt = []
        pred = pd.DataFrame(columns=['Predictions', 'Ground Truths'])

        self.model.eval()
        with ((torch.no_grad())):
            for i in range(len(self.test) - 21):
                TL.plot_bar('Test', i, len(self.test) - 21)
                gt = self.test.iloc[i + 21, 0]
                data = self.test.iloc[i:i + 20]
                data_tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)
                data_tensor = data_tensor.to(self.device)
                output = self.model(data_tensor)
                output = output.cpu().detach().numpy().tolist()

                all_predictions.append(output)
                all_gt.append(gt)

        pred['Predictions'] = all_predictions
        pred['Ground Truths'] = all_gt
        all_nd = []
        all_rmse = []
        all_rou50 = []
        all_rou90 = []

        for i in range(len(all_predictions)):
            predictions = all_predictions[i]
            ground_truth = all_gt[i]

            nd = calculate_nd(ground_truth, predictions)
            rmse = calculate_rmse(ground_truth, predictions)
            rou50 = calculate_rou50(ground_truth, predictions)
            rou90 = calculate_rou90(ground_truth, predictions)
            all_nd.append(nd)
            all_rmse.append(rmse)
            all_rou50.append(rou50)
            all_rou90.append(rou90)

        print(' ')
        print(f'Model: {self.weight_path}')
        print(f'ND: {np.mean(all_nd):.4f}')
        print(f'RMSE: {np.mean(all_rmse):.4f}')
        print(f'rou50: {np.mean(all_rou50):.4f}')
        print(f'rou90: {np.mean(all_rou90):.4f}')
        print("Finished evaluation!")

        pred.to_csv(f'{saving_path}/{self.weight_path[7:14]}_{self.file_path[-11:-8]}.csv')
