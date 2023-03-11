import json
import numpy as np
import torch
import pathlib

from utils.HeightNet import MLP
from utils.common import NN
from utils.nest1 import NestNet
from utils.nest2 import NestNet2
from utils.parse_log import parse_tensorboard_log

def get_stats(path_):
    with open(path_, "r") as fp:
        retrain_dict = json.load(fp)
    test_errors = np.array([run['test_error'] for run in retrain_dict.values()])
    training_times = np.array([run['training_time'] for run in retrain_dict.values()])
    return {'mean_mse': test_errors.mean(),
           'min_mse': test_errors.min(),
           'max_mse': test_errors.max(),
           'training_time': training_times.mean(),
           'argmin': test_errors.argmin()}

def get_training_dict(log_path):
    training_dict = {}
    training_dict['height_1'] = get_stats(log_path + "retrain_1.json")
    training_dict['height_2'] = get_stats(log_path + "retrain_2.json")
    training_dict['height_3'] = get_stats(log_path + "retrain_3.json")
    return training_dict

def get_model(height, cwd, training_dict):
    path_ = cwd + '/logs/ensemble/best_config_' + str(height) + '.json'
    with open(path_,"r") as fp:
        model_hyperparams = json.load(fp)
    model_hyperparams['seed'] = 0
    
    if height == 3:
        network = NestNet2(**model_hyperparams)
    elif height == 2:
        network = NestNet(**model_hyperparams)
    else:
        network = MLP(**model_hyperparams)
        
    model = NN(network, learning_rate=0)
    ckp_root = cwd + '/logs/retrain/' + model_hyperparams['name'] + '/version_' + str(training_dict['height_' + str(height)]['argmin'])
    ckp_path = str(list(pathlib.Path(ckp_root).rglob('*.ckpt'))[0])
    ckp = torch.load(ckp_path)
    model.load_state_dict(ckp['state_dict'])
    return model

def parse_log(height, cwd, training_dict):
    path_ = cwd + '/logs/ensemble/best_config_' + str(height) + '.json'
    with open(path_,"r") as fp:
        model_hyperparams = json.load(fp)
    log_root = cwd + '/logs/retrain/' + model_hyperparams['name'] + '/version_' + str(training_dict['height_' + str(height)]['argmin'])
    log_path = str(list(pathlib.Path(log_root).rglob('*.local'))[0])
    return parse_tensorboard_log(log_path)