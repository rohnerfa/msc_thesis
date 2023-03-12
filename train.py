import os
import sys
import time
from pathlib import Path
import utils
from utils.data import get_data_dict
from utils.config_gen import product_dict
from utils.run_config import run_config
import numpy as np
import json

mode = sys.argv[1]
skip_ensemble = (mode == 'retrain')
skip_retrain = (mode == 'ensemble')

cwd = os.getcwd()


if not skip_ensemble:
    print('ensemble training')
    path = cwd + '/logs/ensemble/'
    Path(path).mkdir(parents=True, exist_ok=True)

    data_hyperparams = {
            'n_peaks': 5,
            'n_train': 100,
            'n_test': 500,
            'n_val': 50
        }

    train_hyperparams = {
        'epochs': 80,
        'batch_size_train': 32,
        'learning_rate':3e-3,
        'seed': 1
    }

    model_hyperparams = {
        'n_width': [20,30,35],
        'n_layers': [4,6],
        'n_height': [1], 
        'input_dim':[2],
        'output_dim':[1]
    }

    data_dict = get_data_dict(**data_hyperparams)

    train_list_1 = []
    train_list_2 = []
    train_list_3 = []

    for config in product_dict(**model_hyperparams):
        print(f"running config: width {config['n_width']}, depth {config['n_layers']}, height {config['n_height']}")
        test_error, training_time = run_config(data_dict, config, train_hyperparams, path)
        if config['n_height'] == 1:
            train_list_1.append([test_error, training_time, config])
        elif config['n_height'] == 2:
            train_list_2.append([test_error, training_time, config])
        else:
            train_list_3.append([test_error, training_time, config])

    if len(train_list_1)>0:
        amin_1 = np.argmin(np.array([model[0] for model in train_list_1]))
        _, _, best_config_1 = train_list_1[amin_1]
        with open(path + "best_config_1.json", "w") as outfile:
            json.dump(best_config_1, outfile)
        with open(path + "ensemble_training_1.json", "w") as outfile:
            json.dump(train_list_1, outfile)

    if len(train_list_2)>0:
        amin_2 = np.argmin(np.array([model[0] for model in train_list_2]))
        _, _, best_config_2 = train_list_2[amin_2]
        with open(path + "best_config_2.json", "w") as outfile:
            json.dump(best_config_2, outfile) 
        with open(path + "ensemble_training_2.json", "w") as outfile:
            json.dump(train_list_2, outfile) 

    if len(train_list_3)>0:
        amin_3 = np.argmin(np.array([model[0] for model in train_list_3]))
        _, _, best_config_3 = train_list_3[amin_3]
        with open(path + "best_config_3.json", "w") as outfile:
            json.dump(best_config_3, outfile)  
        with open(path + "ensemble_training_3.json", "w") as outfile:
            json.dump(train_list_3, outfile)

    
if not skip_retrain:
    print('Retraining best models')
    path = cwd + '/logs/retrain/'
    Path(path).mkdir(parents=True, exist_ok=True)
    
    n_retrain = 5

    data_hyperparams = {
            'n_peaks': 5,
            'n_train': 100,
            'n_test': 500,
            'n_val': 50
        }

    train_hyperparams = {
        'epochs': 120,
        'batch_size_train': 32,
        'learning_rate': 3e-3
    }

    with open(cwd + "/logs/ensemble/best_config_1.json", "r") as infile:
        model_hyperparams_1 = json.load(infile)
    
    with open(cwd + "/logs/ensemble/best_config_2.json", "r") as infile:
        model_hyperparams_2 = json.load(infile)
    
    with open(cwd + "/logs/ensemble/best_config_3.json", "r") as infile:
        model_hyperparams_3 = json.load(infile)

    data_dict = get_data_dict(**data_hyperparams)

    train_dict_1 = {}
    train_dict_2 = {}
    train_dict_3 = {}

    for k in range(n_retrain):
        train_hyperparams['seed'] = k
        test_error, training_time = run_config(data_dict, model_hyperparams_1, train_hyperparams, path)
        train_dict_1['run_'+str(k)] = {'test_error': test_error,
                                       'training_time': training_time,
                                       }
        
        test_error, training_time = run_config(data_dict, model_hyperparams_2, train_hyperparams, path)
        train_dict_2['run_'+str(k)] = {'test_error': test_error,
                                       'training_time': training_time,
                                       }
        
        test_error, training_time = run_config(data_dict, model_hyperparams_3, train_hyperparams, path)
        train_dict_3['run_'+str(k)] = {'test_error': test_error,
                                       'training_time': training_time,
                                       }

    with open(path + "retrain_1.json", "w") as outfile:
            json.dump(train_dict_1, outfile)
    
    with open(path + "retrain_2.json", "w") as outfile:
            json.dump(train_dict_2, outfile)

    with open(path + "retrain_3.json", "w") as outfile:
            json.dump(train_dict_3, outfile)