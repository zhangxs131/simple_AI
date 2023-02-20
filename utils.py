import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_label_list(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read().splitlines()

    return content