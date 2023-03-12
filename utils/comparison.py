import json
import numpy as np
import pandas as pd
import torch
import pathlib
import matplotlib.pyplot as plt

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

def parse_log_all(height, cwd):
    path_ = cwd + '/logs/ensemble/best_config_' + str(height) + '.json'
    with open(path_,"r") as fp:
        model_hyperparams = json.load(fp)
    log_root = cwd + '/logs/retrain/' + model_hyperparams['name']
    df = pd.DataFrame()
    for path in list(pathlib.Path(log_root).rglob('*.local')):
        df = pd.concat((df,parse_tensorboard_log(str(path))))
    by_row_index = df.groupby(df.index)
    df_means = by_row_index.mean()

    return df_means

def plot_training_comparison(cwd):
    df_height1 = parse_log_all(height=1, cwd=cwd)
    df_height2 = parse_log_all(height=2, cwd=cwd)
    df_height3 = parse_log_all(height=3, cwd=cwd)

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    axs[0].semilogy(df_height1['val_loss'])
    axs[0].semilogy(df_height2['val_loss'])
    axs[0].semilogy(df_height3['val_loss'])
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('validation loss')

    axs[1].semilogy(df_height1['wall_time'], df_height1['val_loss'], label='Standard')
    axs[1].semilogy(df_height2['wall_time'], df_height2['val_loss'], label='Nest 2')
    axs[1].semilogy(df_height3['wall_time'], df_height3['val_loss'], label='Nest 3')
    axs[1].set_xlabel('training time (s)')
    axs[1].set_ylabel('validation loss')
    axs[1].legend()

    plt.savefig(cwd + '/plot.png')

def get_training_summary(log_path):
    training_dict = get_training_dict(log_path)
    names = list(training_dict.keys())
    print(f"{'model' : <15}{'training time' : >15}{'mean mse' : >20}{'min mse' : >20}{'max mse' : >20}")
    print(f'{"="*90}')
    for i, key in enumerate(training_dict.keys()):
        print(f"{key: <15}{training_dict[key]['training_time']:>15.2f}{training_dict[key]['mean_mse']:>20.4e}{training_dict[key]['min_mse']:>20.4e}{training_dict[key]['max_mse']:>20.4e}")
