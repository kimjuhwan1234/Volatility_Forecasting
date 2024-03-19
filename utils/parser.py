import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_size", type=int, default=1, help="input_size")
parser.add_argument("--hidden_size", type=int, default=64, help="hidden_size")
parser.add_argument("--num_layers", type=int, default=2, help="num_layers")
parser.add_argument("--output_size", type=int, default=1, help="output_size")
parser.add_argument("--additional", type=bool, default=False, help="additional")
parser.add_argument("--bidirectional", type=bool, default=True, help="bidirectional")

opt_model = parser.parse_args()
print(opt_model)

parser = argparse.ArgumentParser()
parser.add_argument("--model_saving_strategy", type=str, default='better', help="model_saving_strategy")
parser.add_argument("--saving_path", type=str, default='Database', help="saving_path")
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

config = dict()
config['model'] = opt_model
config['train'] = opt_train
