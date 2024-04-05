from utils.Metrics import *
from Modules.train import *
from torch.optim import Adam
from utils.seed import seed_everything
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Modules.dataset import CustomDataset, TestDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import pandas as pd
import matplotlib.pyplot as plt


class Run:
    def __init__(self, file_path, config):
        seed_everything(self.seed)
        self.seed = self.config['train'].seed

        self.config = config
        self.file_path = file_path
        self.lr = self.config['train'].lr

        self.epochs = self.config['train'].epochs
        self.device = self.config['train'].device
        self.batch_size = self.config['train'].batch_size
        self.train = pd.read_csv(self.file_path, index_col=0)

        # Transfer learning and Backbone Model 여부에 따라 weight 저장 위치를 달리함.
        if not self.config['model'].Transfer:
            if self.config['model'].backbone1:
                self.weight_path = f'Weight/Backbone/BiLSTM_{self.file_path[-10:-8]}.pth'
                self.saving_path = f'Files/Backbone_{self.file_path[-10:-8]}.csv'

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

    def load_data(self, retraining):
        print('Loading data...')

        # 처음 훈련할 때 load_data 설정.
        if not retraining:
            # Transfer Learning을 할때는 코로나를 반영하기 위해 2015년 부터 2020년 까지를 훈련데이터로 씀.
            if self.config['model'].Transfer:
                # 전처리에서 애초에 2015년부터 시작하게 해둠.
                train_data = self.train.loc[:'2021-01-01']
                # valindation은 2년을 진행하는데, 2023-01-01이 아니라 2022-11-01인 이유는 if retraining에서 설명하겠음.
                # 참고로 test set은 2023-01-01부터 시작함.
                val_data = self.train.loc['2021-01-01':'2022-11-01']
                train_dataset = CustomDataset(train_data)
                val_dataset = CustomDataset(val_data)
                dataloaders = {
                    'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False),
                    'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
                }
            # Backbone 훈련을 위한 설정 1986년 부터 2011년 까지를 훈련데이터로 씀
            if not self.config['model'].Transfer:
                train_data = self.train.loc[:'2012-01-01']
                # 2012년부터 2014년 까지가 validation. 전처리에서 애초에 2014년 12월에 끝나게 해둠.
                val_data = self.train.loc['2012-01-01':]
                train_dataset = CustomDataset(train_data)
                val_dataset = CustomDataset(val_data)
                dataloaders = {
                    'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False),
                    'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
                }

        if retraining:
            '''위에서 전이학습 validation을 2023-01-01이 아니라 2022-01-01로 한 이유는 다음과 같음. 만약 2023-01-01까지
            validation을 진행하면 첫 60일 예측 이후 retraining에 이용할 data size는 무조건 2023-01-01부터 시작해야 함.(미래참조이슈)
            그럼 data size가 60이 되고 60*0.2=12인데 이것은 retraining에 이용할 val_dataset window_size보다 작음.그래서 
            2022-11-01까지를 validation으로 이용하고 retrain에 이용할 data size가 103*0.2>20 이 되게 하여 오류가 안나게 함.
            
            train val split을 0.2로 안하고 더 늘리거나, retrain 주기를 처음부터 100이상으로 하는 방법도 있음. 전자의 경우는 train
            set의 크기가 줄어 예측성능이 엉망이 되는 것을 확인함. 현재 retrain 주기는 60인데 이때의 예측성능도 retrain을 하지 않았을 때
            보다 안좋아서 retrain 주기를 100이상으로 늘릴 것을 고민 중임. 그러면 validation을 2023-01-01까지 해도 됨.
            
            self.test_index는 밑에서 저장되는데, retrain 주기만큼의 예측이 끊났을 때 마지막 날짜-window_size의 날짜를 저장한 것임.
            이유는 밑에서 더 설명함.'''
            data = self.train.loc['2022-11-01':self.test_index]
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
            train_dataset = CustomDataset(train_data)
            val_dataset = CustomDataset(val_data)
            dataloaders = {
                'train': DataLoader(train_dataset, batch_size=4, shuffle=False),
                'val': DataLoader(val_dataset, batch_size=4, shuffle=False),
            }

            '''retrain 주기만큼 예측이 끊나면 주기의 마지막 날짜 -20을 self.test_index에 저장하게 됨. 그 이유는 retrain 예측이 
            끊났을 때 마지막 날짜를 저장하게 되면 초반 20일은 항상 사용하지 않기 때문에 마지막 예측 날짜의 20일을 뺀 값을 사용해야
            예측이 연속적으로 이뤄지기 때문. 때문에 self.test_data는 마지막 예측 날짜의 -20일 부터 시작하는 것이 됨. '''
            self.test_data = self.train.loc[self.test_index:]
            self.test_dataset = TestDataset(self.test_data)
            self.test_dl = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        self.dataloaders = dataloaders

        print('Finished loading data!')

    def run_model(self, retraining):
        print(' ')
        print('Training model...')
        print(' ')

        self.load_data(retraining)

        # retraining이 True이면 self.model이 이미 선언되었을 것임.
        if retraining:
            self.model.to(self.device)

        # retraining이 False이면 self.model이 없기 때문에 선언해줘야 함.
        if not retraining:
            self.model = self.config['structure']
            self.model.to(self.device)

        opt = Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=self.config['train'].patience)
        parameters = {
            'num_epochs': self.epochs,
            'weight_path': self.weight_path,

            'train_dl': self.dataloaders['train'],
            'val_dl': self.dataloaders['val'],

            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
        }

        TM = Train_Module(self.device)
        self.model, self.loss_hist, self.metric_hist = TM.train_and_eval(self.model, parameters)

        print('Finished training model!')

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
        print('Saving evaluations and predictions for a test set...')

        # 처음 while문 들어갈때는 test_dl이 없으므로 만들어 줘야 함. 또한 retrain을 하지 않을 때를 위해서 필요함.
        self.test_data = self.train.loc['2023-01-01':]
        self.test_dataset = TestDataset(self.test_data)
        self.test_dl = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        # retrain을 하기 때문에 pred dataframe을 while문 밖에서 저장할 필요가 있음.
        self.pred = pd.DataFrame(columns=['Predictions', 'Ground Truths'])
        pred_index = self.test_data.index[len(self.test_data) - len(self.test_dl):]

        '''2023-01-01부터 시작하는 첫 test_data의 index가 저장될 필요가 있음. self.test_data는 계속 변하기 때문에 
        self_data.index를 사용하면 안됨. 이유는 밑에서 더 설명.'''
        retrain_index = self.test_data.index

        # Needed to load weights after model training and test them later.
        self.model = self.config['structure']
        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.to(self.device)

        j = 0
        # 294는 총 예측 날짜임. self.pred길이가 294가 되면 종료.
        while len(self.pred) < 294:
            # 처음은 retrain을 스킵해야 하기 때문에
            if j > 0:
                self.run_model(True)

            self.model.eval()
            with ((torch.no_grad())):
                for X_train, gt in self.test_dl:
                    X_train = X_train.to(self.device)
                    output = self.model(X_train)
                    output = output.cpu().detach().numpy().tolist()
                    gt = gt.squeeze().detach().numpy().tolist()
                    self.pred.loc[len(self.pred)] = [output, gt]

                    if len(self.pred) % 60 == 0:
                        '''이 부분에서 self.test_index를 저장하는데 retrain_index를 사용해야 2023-01-01부터 
                        총 예측한 날짜 이후 날짜가 저장됨. <=> 주기 마지막 날짜 - 20일과 동치'''
                        self.test_index = retrain_index[len(self.pred)]
                        break
            j += 1

        self.pred.index = pred_index
        nd = calculate_nd(self.pred['Ground Truths'].values, self.pred['Predictions'].values)
        mae = calculate_mae(self.pred['Ground Truths'].values, self.pred['Predictions'].values)
        rmse = calculate_rmse(self.pred['Ground Truths'].values, self.pred['Predictions'].values)
        ad_r2 = calculate_adjusted_r2_score(self.pred['Ground Truths'].values, self.pred['Predictions'].values, 20, 2)
        self.pred.loc[len(self.pred)] = [nd, ad_r2]
        self.pred.loc[len(self.pred)] = [mae, rmse]
        self.pred.to_csv(self.saving_path)

        print(' ')
        print(f'Model: {self.weight_path}')
        print(f'ND: {nd:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'adjusted-R^2: {ad_r2:.4f}')
        print(f"Saved Result in {self.saving_path}!")
