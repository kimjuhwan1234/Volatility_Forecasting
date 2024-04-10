from torch.nn.functional import mse_loss
import torch
import torch.nn as nn


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

        out = out[:, -1, :]

        return out


class single_biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, additional):
        super(single_biLSTM, self).__init__()

        self.additional = additional

        self.backbone = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                bidirectional=True, batch_first=True)

        # 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.additional_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, train, gt=None):
        out, _ = self.backbone(train)

        if not self.additional:
            output = self.fc(out)

        # Todo: additional도 구현했는데 이유는 전체 weight 수정과 마지막 레이어 수정의 절충안이기 때문에 시도해보고자 함.
        if self.additional:
            output = self.additional_layer(out)

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

        self.fc= nn.Linear(hidden_size, output_size),

    def forward(self, train, gt=None):
        output = self.MLP(train)
        output = output[:, -1, :]
        output = output.squeeze()

        if gt != None:
            gt = gt.squeeze()
            loss = torch.sqrt(mse_loss(output, gt))
            return output, loss

        return output


# class Transfer_Learning(nn.Module):
#     def __init__(self, backbone, output_size, hidden_size, additional):
#         super(Transfer_Learning, self).__init__()
#
#         self.backbone = backbone
#
#         self.additional = additional
#
#         self.additional_layer = nn.Sequential(
#             nn.Linear(output_size, hidden_size),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, output_size),
#         )
#
#     def forward(self, train, gt=None):
#         output = self.backbone(train)
#
#         if self.additional:
#             if output.size() == torch.Size([]):
#                 output = output.unsqueeze(0)
#             output = output.unsqueeze(dim=1)
#             output = self.additional_layer(output)
#
#         output = output.squeeze()
#
#         if gt != None:
#             gt = gt.squeeze()
#             loss = torch.sqrt(mse_loss(output, gt))
#             return output, loss
#
#         return output
