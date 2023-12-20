
from TrainUtils import training, get_environment, get_architectures, get_train_params
import torch
import numpy as np
import wandb
from quadrotor import Quadrotor

device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_env(env_name = "quadrotor", sigma=1, device=device_default,
              activation='tanh', load_model=False, num_train=10, num_warmup=10,
              num_episodes_hnet=1024, num_episodes_adj=2048, rate=1.5,
              num_hnet_train_max=10000, num_adj_train_max=1000,
              batch_size_hnet=32, batch_size_hnet_sample=256, batch_size_adj=64,
              update_interval_custom=-1, log_interval_custom=-1, stop_train_condition=0.01,
              model_dir='models/', lr_change=False, lr_hnet_custom=1e-3, lr_adj_custom=1e-3):
    # Initialize networks with specific architectures detailed in arch_file
    _, adj_net, hnet, hnet_target = get_architectures()
    print("get_architectures")
    # Initialize hyperparameters detailed in param_file
    T_hnet, T_adj, n_timesteps, control_coef, lr_hnet, lr_adj, update_interval, log_interval_hnet = get_train_params()
    # Get environment (with specific (quadratic) control coefficient)
    env = get_environment(env_name)
    # Set custom hyperparameter if allowed
    if update_interval_custom != -1:
        update_interval = update_interval_custom
    if log_interval_custom != -1:
        log_interval_hnet = log_interval_custom
    lr_hnet = lr_hnet_custom if lr_change else lr_hnet
    lr_adj = lr_adj_custom if lr_change else lr_adj

    # Print hyperparameters info
    print(f'\nDevice: {device}')
    print('\nHamiltonian training:')
    print(f'total number of episodes: {num_episodes_hnet}, max number of iterations: {num_hnet_train_max}.')
    print(f'update interval: {update_interval}, rate to train while sampling: {rate}.')
    print(f'sample_batch_size: {batch_size_hnet_sample}, batch size: {batch_size_hnet}, learning_rate: {lr_hnet}.')
    print(f'\nAdjoint net training:')
    print(f'total number of episodes: {num_episodes_adj}, max number of iterations: {num_adj_train_max}')
    print(f'batch size: {batch_size_adj}, learning rate: {lr_adj}')
    print(f'\nThe sigma constant for the diffusion is {sigma}.')

    # Training step
    print(f'\nTraining environment {env_name}')

    log_interval_adj = 100
    training(sigma, device, env, env_name,
             adj_net, hnet, hnet_target,
             num_train, num_warmup, load_model,
             T_hnet, T_adj, n_timesteps, control_coef,
             num_episodes_hnet, num_episodes_adj,
             update_interval, rate,
             batch_size_hnet_sample, batch_size_hnet, batch_size_adj,
             lr_hnet, lr_adj, log_interval_hnet, log_interval_adj,
             num_hnet_train_max, num_adj_train_max, stop_train_condition)

if __name__ == "__main__":
    train_env()