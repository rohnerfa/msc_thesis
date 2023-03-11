import utils
from utils.func_yarotsky import get_func

import torch
from torch.utils.data import TensorDataset


def get_data_dict(n_peaks, n_train, n_test, n_val=None):

    func = get_func(n_peaks)
    eng = torch.quasirandom.SobolEngine(dimension=2)
    eng.reset()
    X_train = eng.draw(n_train**2)
    y_train = func(X_train.unsqueeze(-1))

    X_test1 = torch.linspace(0,1,n_test)
    X_test2 = torch.linspace(0,1,n_test)

    X_test = torch.cartesian_prod(X_test1,X_test2)
    y_test = func(X_test.unsqueeze(-1))

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    data_dict = {
        'train_set': train_set,
        'test_set': test_set
    }  

    if n_val is not None:
        X_val1 = torch.linspace(0,1,n_val)
        X_val2 = torch.linspace(0,1,n_val)
        X_val = torch.cartesian_prod(X_val1,X_val2)
        y_val = func(X_val.unsqueeze(-1))
        val_set = TensorDataset(X_val, y_val)

        data_dict['val_set'] = val_set
    
    return data_dict