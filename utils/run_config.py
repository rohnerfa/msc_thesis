import utils
from utils.HeightNet import HeightNet
from utils.plotting import plot_model_2d

def run_config(data_dict, model_hyperparams, train_hyperparams, log_path, return_model=False):
    names = ['mlp', 'nest1', 'nest2']
    name = names[model_hyperparams['n_height']-1]
    model_hyperparams['name'] = name
    model = HeightNet(path=log_path, **train_hyperparams, **model_hyperparams, **data_dict)

    model.train()
    model.test()

    if return_model:
        return model.test_error, model.training_time, model
    else:
        return model.test_error, model.training_time