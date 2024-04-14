from torch.nn.functional import mse_loss
import torch
import torch.nn as nn


class stack_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional, additional):
        super(stack_BiLSTM, self).__init__()

        self.additional = additional

        # 출력을 위한 선형 레이어
        self.additional_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2),
        )

        if bidirectional:
            hidden_size2 = hidden_size * 2

        if not bidirectional:
            hidden_size2 = hidden_size

        # 첫 번째 LSTM 층
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=0.2, batch_first=True)

        # 두 번째 LSTM 층
        self.lstm2 = nn.LSTM(hidden_size2, hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=0.2, batch_first=True)

        # 세 번째 LSTM 층
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=0.1, batch_first=True)

        # 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, train, gt=None):
        # 첫 번째 LSTM 층
        out, _ = self.lstm1(train)

        # 두 번째 LSTM 층
        out, _ = self.lstm2(out)

        # 세 번째 LSTM 층
        out, _ = self.lstm3(out)

        if self.additional:
            out = self.additional_layer(out)

        # 출력을 위한 선형 레이어
        output = self.fc(out)

        output = output[:, -1, :]

        output = output.squeeze()

        if gt != None:
            gt = gt.squeeze()
            loss = torch.sqrt(mse_loss(output, gt))
            return output, loss

        return output


class single_biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, additional):
        super(single_biLSTM, self).__init__()

        self.additional = additional

        # 출력을 위한 선형 레이어
        self.additional_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2),
        )

        self.backbone = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                bidirectional=True, batch_first=True)

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, train, gt=None):
        out, _ = self.backbone(train)

        if self.additional:
            out = self.additional_layer(out)

        output = self.fc(out)

        output = output[:, -1, :]

        output = output.squeeze()

        if gt != None:
            gt = gt.squeeze()
            loss = torch.sqrt(mse_loss(output, gt))
            return output, loss

        return output


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.fc = nn.Linear(hidden_size, output_size),

    def forward(self, train, gt=None):
        output = self.MLP(train)
        output = output[:, -1, :]
        output = output.squeeze()

        if gt != None:
            gt = gt.squeeze()
            loss = torch.sqrt(mse_loss(output, gt))
            return output, loss

        return output
