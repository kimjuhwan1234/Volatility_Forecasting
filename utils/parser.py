import argparse
from Modules.model import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_size", type=int, default=1, help="input_size")
parser.add_argument("--hidden_size", type=int, default=64, help="hidden_size")
parser.add_argument("--num_layers", type=int, default=2, help="num_layers")
parser.add_argument("--output_size", type=int, default=1, help="output_size")
parser.add_argument("--bidirectional", type=bool, default=True, help="bidirectional")

parser.add_argument("--Transfer", type=bool, default=False, help="Transfer")
parser.add_argument("--additional", type=bool, default=True, help="additional")

parser.add_argument("--backbone1", type=bool, default=True, help="biLSTM")
parser.add_argument("--backbone2", type=bool, default=False, help="DLinear")
parser.add_argument("--backbone3", type=bool, default=False, help="MLP")
parser.add_argument("--backbone4", type=bool, default=False, help="NBEATSx")
parser.add_argument("--backbone5", type=bool, default=False, help="Prophet")

opt_model = parser.parse_args()
print(opt_model)
# ---------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--model_saving_strategy", type=str, default='better', help="model_saving_strategy")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--device", type=str, default='cuda', help="device")
parser.add_argument("--epochs", type=int, default=1000, help="epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
parser.add_argument("--patience", type=int, default=3, help="patience")
parser.add_argument("--use_accelerator", type=bool, default=False, help="use_accelerator")
parser.add_argument("--use_wandb", type=bool, default=False, help="use_wandb")

opt_train = parser.parse_args()
print(opt_train)
# ---------------------------------------------------------------------------------------------------------------------#
backbone1 = single_biLSTM(input_size=opt_model.input_size, hidden_size=opt_model.hidden_size,
                         num_layers=opt_model.num_layers, output_size=opt_model.output_size)
backbone3 = MLP(input_size=opt_model.input_size, hidden_size=opt_model.hidden_size, output_size=opt_model.output_size)
# ---------------------------------------------------------------------------------------------------------------------#
config = dict()
config['model'] = opt_model
config['train'] = opt_train

if not opt_model.Transfer:
    if opt_model.backbone1:
        config['structure'] = backbone1

    if opt_model.backbone2:
        config['structure'] = backbone1

    if opt_model.backbone3:
        config['structure'] = backbone3

    if opt_model.backbone4:
        config['structure'] = backbone1

    if opt_model.backbone5:
        config['structure'] = backbone1

if opt_model.Transfer:
    if opt_model.backbone1:
        backbone_weight_path = 'Weight/Backbone/BiLSTM_SP.pth'
        backbone1.load_state_dict(torch.load(backbone_weight_path))
        config['structure'] = Transfer_Learning(backbone1, opt_model.output_size,
                                                opt_model.hidden_size, opt_model.additional,
                                                )

    if opt_model.backbone2:
        backbone_weight_path = 'Weight/Backbone/DLinear_SP.pth'
        backbone1.load_state_dict(torch.load(backbone_weight_path))
        config['structure'] = Transfer_Learning(backbone1, opt_model.output_size,
                                                opt_model.hidden_size, opt_model.additional,
                                                )

    if opt_model.backbone3:
        backbone_weight_path = 'Weight/Backbone/MLP_SP.pth'
        backbone3.load_state_dict(torch.load(backbone_weight_path))
        config['structure'] = Transfer_Learning(backbone3, opt_model.output_size,
                                                opt_model.hidden_size, opt_model.additional,
                                                )

    if opt_model.backbone4:
        backbone_weight_path = 'Weight/Backbone/NBEATSx_SP.pth'
        backbone1.load_state_dict(torch.load(backbone_weight_path))
        config['structure'] = Transfer_Learning(backbone1, opt_model.output_size,
                                                opt_model.hidden_size, opt_model.additional,
                                                )

    if opt_model.backbone5:
        backbone_weight_path = 'Weight/Backbone/Prophet_SP.pth'
        backbone1.load_state_dict(torch.load(backbone_weight_path))
        config['structure'] = Transfer_Learning(backbone1, opt_model.output_size,
                                                opt_model.hidden_size, opt_model.additional,
                                                )
