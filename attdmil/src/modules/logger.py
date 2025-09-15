import os
import shutil
from abc import ABC, abstractmethod
import wandb
import yaml
import numpy as np

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def get_run_name(log_path: str, run_name: str = None, cleanup: bool = False) -> str:
    if run_name is None:
        prefix = "run_"
        for i in range(1, 100):
            path_cand = os.path.join(log_path, f"{prefix}{i:02d}")
            if not os.path.exists(path_cand):
                os.mkdir(path_cand)
                break
        return f"{prefix}{i:02d}"
    else:
        path = os.path.join(log_path, run_name)
        if not os.path.exists(path):
            os.mkdir(path)
        elif cleanup:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            os.mkdir(path)
        return run_name

def save_config(log_path: str, run_name: str, config: dict):
    yaml.Dumper.ignore_aliases = lambda *args: True
    config_path = os.path.join(log_path, run_name, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(file_path: str):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def prepare_folder(log_dir, run_name):
    ckpt_save_path = os.path.join(log_dir, run_name, "checkpoints")
    misc_save_path = os.path.join(log_dir, run_name, "misc")
    if not (os.path.isdir(ckpt_save_path)):
        os.makedirs(ckpt_save_path, exist_ok=True)

    if not (os.path.isdir(misc_save_path)):
        os.makedirs(misc_save_path, exist_ok=True)

    return ckpt_save_path, misc_save_path

class AbstractLogger(ABC):
    @abstractmethod
    def log_scalar(self, tag, scalar_value, global_step):
        raise NotImplementedError
    
    @abstractmethod
    def log_scalar_test(self, tag, scalar_value):
        raise NotImplementedError

    @abstractmethod
    def log_AUC(self, misc_save_path, value, name):
        raise NotImplementedError

    @abstractmethod
    def log_names(self, misc_save_path, name_lst, name):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        pass

class WandbLogger(AbstractLogger):
    def __init__(self, *, run_name=None, log_dir=None):
        self.wandb = wandb
        self.dir = os.path.join(log_dir, run_name)

    def log_scalar(self, tag, scalar_value, global_step):
        self.wandb.log({tag: scalar_value}, step=global_step)
    
    def log_scalar_test(self, tag, scalar_value):
        self.wandb.log({tag: scalar_value})
    
    def log_AUC(self, misc_save_path, value, name):
        file_path = os.path.join(misc_save_path, f'{name}_5runs.txt')

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                values = [float(line.strip()) for line in lines if line.strip().replace('.', '', 1).isdigit()]
        else:
            values = []

        values.append(value)

        if len(values) == 5:
            mean_value = np.mean(values)
            std_value = np.std(values)

            with open(file_path, 'a') as file:
                file.write(f'{value}\n')
                file.write(f'Mean: {mean_value:.3f}\n')
                file.write(f'Std: {std_value:.3f}\n\n')

            values.clear()
        else:
            with open(file_path, 'a') as file:
                file.write(f'{value}\n')
    
    def log_names(self, misc_save_path, name_lst, name):
        file_path = os.path.join(misc_save_path, f'{name}.txt')
        with open(file_path, 'w') as file:
            for item in name_lst:
                file.write(f"{item}\n")
        
        print(f"Names have been written to {file_path}")

    def finish(self):
        self.wandb.finish()