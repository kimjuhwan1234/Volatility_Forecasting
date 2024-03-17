import os
import torch


def save_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


def auto_save_model(
        model,
        accelerator,
        saving_path: str = None,
        model_saving_strategy: str = None,
        training_finished: bool = True,
        saving_name: str = None,
):
    state_dict = accelerator.get_state_dict(model)

    if saving_path is not None and model_saving_strategy is not None:
        name = __name__ if saving_name is None else saving_name
        if not training_finished and model_saving_strategy == "better":
            accelerator.save(state_dict, os.path.join(saving_path, name))

        elif training_finished and model_saving_strategy == "best":
            accelerator.save(state_dict, os.path.join(saving_path, name))
    else:
        return
