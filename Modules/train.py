from utils.Metrics import calculate_R2

import sys
import time
import torch
import pandas as pd


class Train_Module:
    def __init__(self, device, patience):
        self.device = device
        self.patience = patience

    def plot_bar(self, mode, i, len_data):
        progress = i / len_data
        bar_length = 30
        block = int(round(bar_length * progress))
        progress_bar = f'{mode}: [{"-" * block}{"." * (bar_length - block)}] {progress * 100:.2f}%'
        sys.stdout.write('\r' + progress_bar)

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def eval_fn(self, model, dataset_dl):
        total_loss = 0.0
        total_accuracy = 0.0
        len_data = len(dataset_dl)

        model.eval()

        with torch.no_grad():
            i = 0
            for data, gt in dataset_dl:
                i += 1
                self.plot_bar('Val', i, len_data)

                data = data.to(self.device)
                gt = gt.to(self.device)

                output, loss = model(data, gt)
                total_loss += loss

                # accuracy = self.calculate_adjusted_r2_score(output, gt, data.shape[1], data.shape[2])
                accuracy = calculate_R2(gt, output)
                total_accuracy += accuracy

            total_loss = total_loss / len_data
            total_accuracy = total_accuracy / len_data

        return total_loss, total_accuracy

    def train_fn(self, model, dataset_dl, opt):
        total_loss = 0.0
        total_accuracy = 0.0
        len_data = len(dataset_dl)

        model.train()
        i = 0
        for data, gt in dataset_dl:
            i += 1
            self.plot_bar('Train', i, len_data)

            data = data.to(self.device)
            gt = gt.to(self.device)

            opt.zero_grad()
            output, loss = model(data, gt)
            loss.backward()
            opt.step()

            total_loss += loss
            # accuracy = self.calculate_adjusted_r2_score(output, gt, data.shape[1], data.shape[2])
            accuracy = calculate_R2(gt, output)
            total_accuracy += accuracy

        total_loss = total_loss / len_data
        total_accuracy = total_accuracy / len_data

        return total_loss, total_accuracy

    def train_and_eval(self, model, params):
        num_epochs = params['num_epochs']
        weight_path = params["weight_path"]

        train_dl = params["train_dl"]
        val_dl = params["val_dl"]

        opt = params["optimizer"]
        lr_scheduler = params["lr_scheduler"]

        loss_history = pd.DataFrame(columns=['train', 'val'])
        accuracy_history = pd.DataFrame(columns=['train', 'val'])

        val_loss, val_accuracy = self.eval_fn(model, val_dl)
        best_loss = val_loss
        start_time = time.time()

        for epoch in range(num_epochs):
            current_lr = self.get_lr(opt)
            print(f'Epoch {epoch + 1}/{num_epochs}, current lr={current_lr}')

            train_loss, train_accuracy = self.train_fn(model, train_dl, opt)
            loss_history.loc[epoch, 'train'] = train_loss
            accuracy_history.loc[epoch, 'train'] = train_accuracy

            val_loss, val_accuracy = self.eval_fn(model, val_dl)
            loss_history.loc[epoch, 'val'] = val_loss
            accuracy_history.loc[epoch, 'val'] = val_accuracy

            lr_scheduler.step(val_loss)
            print(' ')

            if val_loss < best_loss:
                counter = 0
                best_loss = val_loss
                torch.save(model.state_dict(), weight_path)
                print('Saved model Weight!')
            else:
                counter += 1
                if counter >= self.patience:
                    print("Early stopping")
                    break

            print(f'train loss: {train_loss:.10f}, val loss: {val_loss:.10f}')
            print(f'R2: {val_accuracy:.4f}, time: {(time.time() - start_time) / 60:.2f}')
            print(' ')

        return model, loss_history, accuracy_history
