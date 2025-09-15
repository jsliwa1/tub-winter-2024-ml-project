import wandb
import os
import time
import torch as th
import torch.utils.data as data_utils
from torchinfo import summary

from src.modules.config import MILModelConfig, HistoBagsConfig, MILPoolingConfig
from src.dataset.HistoDataset import HistoDataset
from src.model.histo_mil_wrapper import HistoMILWrapper
from src.model.mil import MILModel
from src.modules.logger import bcolors
from src.modules.trainer import Trainer
from src.modules.logger import load_config

def test():
    pml_cluster = False
    ckpt_save_path = "/home/pml06/dev/attdmil/logs/histo/1/embedding_poolattention_lr_0.0001_wd_0.001/checkpoints/last.pt"
    base_log_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_save_path)))
    misc_save_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_save_path)), 'misc')
    run_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_save_path)))

    print(f"{bcolors.OKBLUE}Start test with run_name: {bcolors.BOLD}{run_name}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Log dir: {bcolors.BOLD}{base_log_dir}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Checkpoint save path: {bcolors.BOLD}{ckpt_save_path}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Misc save path: {bcolors.BOLD}{misc_save_path}{bcolors.ENDC}")

    loaded_config = load_config(os.path.join(base_log_dir, run_name, 'config.yaml'))

    test_config = MILModelConfig(**loaded_config)
    test_config.ckpt_path = ckpt_save_path
    test_config.misc_save_path = misc_save_path
    test_config.ckpt_save_path = None
    test_config.val_every = None
    test_config.save_max = None
    test_config.patience = None
    test_config.train_dataset_config = None
    test_config.val_dataset_config = None
    test_config.test_dataset_config = HistoBagsConfig(
        seed=1,
        prop_num_bags=1,
        h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
        color_normalize=False,
        datatype="features_for_vis",
        mode="test",
        val_mode=False,
        split=0.8,
        pml_cluster=pml_cluster,
    )
    test_loader = data_utils.DataLoader(
        HistoDataset(**test_config.test_dataset_config.__dict__),
        batch_size=test_config.batch_size,
        shuffle=False
    )
    model = MILModel(test_config).to(test_config.device)
    wrapper = HistoMILWrapper(model=model, config=test_config, epochs=test_config.epochs)
    summary(model, input_data=th.rand(1000, 768).to(test_config.device))

    trainer = Trainer(
        device=test_config.device,
        wrapper=wrapper,
        misc_save_path=test_config.misc_save_path,
    )
    trainer.test_visualize(test_loader)

if __name__ == '__main__':
    test()
    print(f"{bcolors.OKGREEN}Test finished{bcolors.ENDC}")

