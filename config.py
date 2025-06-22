import torch
import os
import socket


def create_dir_if_necessary(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created '{dir_path}' directory.")


curr_dir = os.path.dirname(os.path.abspath(__file__))


root_dir = ''  # Please write down your data directory, e.g., '/home/data/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device={device.__str__().upper()}")


ckp_dir = os.path.join(curr_dir, 'checkpoints')
create_dir_if_necessary(ckp_dir)

fig_dir = os.path.join(curr_dir, 'figs')
create_dir_if_necessary(fig_dir)
