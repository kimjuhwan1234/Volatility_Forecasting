import torch.nn as nn
from torch.nn.functional import mse_loss


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super(StackedLSTM, self).__init__()

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

    def forward(self, x):
        # 첫 번째 LSTM 층
        out, _ = self.lstm1(x)

        # 두 번째 LSTM 층
        out, _ = self.lstm2(out)

        # 세 번째 LSTM 층
        out, _ = self.lstm3(out)

        # 출력을 위한 선형 레이어
        out = self.fc(out)

        out = out[:, :5, :]

        return out


class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super(StackedGRU, self).__init__()

        if bidirectional:
            hidden_size2 = hidden_size * 2

        if not bidirectional:
            hidden_size2 = hidden_size

        # 첫 번째 GRU 층
        self.GRU1 = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=0.4, batch_first=True)

        # 두 번째 GRU 층
        self.GRU2 = nn.GRU(hidden_size2, hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=0.3, batch_first=True)

        # 세 번째 GRU 층
        self.GRU3 = nn.GRU(hidden_size2, hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=0.1, batch_first=True)

        # 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # 첫 번째 GRU 층
        out, _ = self.GRU1(x)

        # 두 번째 GRU 층
        out, _ = self.GRU2(out)

        # 세 번째 GRU 층
        out, _ = self.GRU3(out)

        # 출력을 위한 선형 레이어
        out = self.fc(out)

        return out


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, additional, bidirectional):
        super(RegressionModel, self).__init__()

        self.additional = additional

        self.backbone = StackedLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    output_size=output_size, bidirectional=bidirectional)

        if additional:
            self.additional_layer = StackedGRU(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers,
                                               output_size=output_size, bidirectional=bidirectional)

    def forward(self, train, gt=None):
        output = self.backbone(train)

        if self.additional:
            output = self.additional_layer(output)

        output = output.squeeze()

        if gt != None:
            loss = mse_loss(output, gt)
            return output, loss

        return output
