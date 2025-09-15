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
from src.modules.logger import prepare_folder, get_run_name, bcolors, WandbLogger, save_config
from src.modules.trainer import Trainer


def train(config=None):

    base_log_dir = '/home/pml06/dev/attdmil/logs/m3histo'
    pml_cluster = False

    with wandb.init(
            dir=base_log_dir,
            config=config,
        ):
        config = wandb.config
        base_log_dir = base_log_dir + f"/{config.num_bags}"
        if not os.path.exists(base_log_dir):
            os.makedirs(base_log_dir)
        run_name = get_run_name(base_log_dir, f"{config.mode}_pool{config.pooling_type}")
        wandb.run.name = run_name  
        wandb.run.save()
        ckpt_save_path, misc_save_path = prepare_folder(base_log_dir, run_name)
        print(f"{bcolors.OKBLUE}Start training with run_name: {bcolors.BOLD}{run_name}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Log dir: {bcolors.BOLD}{base_log_dir}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Checkpoint save path: {bcolors.BOLD}{ckpt_save_path}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Misc save path: {bcolors.BOLD}{misc_save_path}{bcolors.ENDC}")
    
        # Configure the model with parameters from the sweep
        train_config = MILModelConfig(
            device=th.device("cuda" if th.cuda.is_available() else "cpu"),
            mode=config.mode,
            epochs=20,
            batch_size=1,
            train_dataset_config=HistoBagsConfig(
                seed=1,
                prop_num_bags=config.num_bags,
                h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
                color_normalize=False,
                datatype="features",
                mode="train",
                val_mode=False,
                split=0.8,
                pml_cluster=pml_cluster
            ),
            val_dataset_config=HistoBagsConfig(
                seed=1,
                prop_num_bags=config.num_bags,
                h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
                color_normalize=False,
                datatype="features",
                mode="train",
                val_mode=True,
                split=0.8,
                pml_cluster=pml_cluster
            ),
            val_dataset_vis_config=HistoBagsConfig(
                seed=1,
                prop_num_bags=config.num_bags,
                h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
                color_normalize=False,
                datatype="features_for_vis",
                mode="test",
                val_mode=False,
                split=0.8,
                pml_cluster=pml_cluster
            ),
            test_dataset_config=HistoBagsConfig(
                seed=1,
                prop_num_bags=1,
                h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
                color_normalize=False,
                datatype="features",
                mode="test",
                val_mode=False,
                split=0.8,
                pml_cluster=pml_cluster
            ),
            mil_pooling_config=MILPoolingConfig(
                pooling_type=config.pooling_type,
                feature_dim=768,
                attspace_dim=config.attspace_dim,
                attbranches=1
            ),
            just_features=True,
            ckpt_path=None,
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
            T_0=50,
            T_mult=2,
            eta_min=1e-6,
            step_size=10000000,
            gamma=0.1,
            ckpt_save_path=ckpt_save_path,
            misc_save_path=misc_save_path,
            val_every=1,
            save_max=2,
            patience=2,
        )
        save_config(base_log_dir, run_name, train_config.__dict__)

        train_loader = data_utils.DataLoader(
            HistoDataset(**train_config.train_dataset_config.__dict__),
            batch_size=train_config.batch_size,
            shuffle=True
        )
        val_loader = data_utils.DataLoader(
            HistoDataset(**train_config.val_dataset_config.__dict__),
            batch_size=train_config.batch_size,
            shuffle=False
        )
        val_vis_loader = data_utils.DataLoader(
            HistoDataset(**train_config.val_dataset_vis_config.__dict__),
            batch_size=train_config.batch_size,
            shuffle=False
        )
        test_loader = data_utils.DataLoader(
            HistoDataset(**train_config.test_dataset_config.__dict__),
            batch_size=train_config.batch_size,
            shuffle=False
        )

        model = MILModel(mil_model_config=train_config).to(train_config.device)
        wrapper = HistoMILWrapper(model=model, config=train_config, epochs=train_config.epochs)

        summary(model, input_data=th.rand(1000, 768).to(train_config.device))

        trainer = Trainer(
            device=train_config.device,
            wrapper=wrapper,
            misc_save_path=train_config.misc_save_path,
        )

        trainer.train(
            epochs=train_config.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            val_vis_loader=val_vis_loader,
            logger=WandbLogger(log_dir=base_log_dir, run_name=run_name),
            ckpt_save_path=train_config.ckpt_save_path,
            ckpt_save_max=train_config.save_max,
            val_every=train_config.val_every,
            patience=train_config.patience,
        )

        trainer.test(
            test_loader=test_loader,
        )


def main_sweep():
    """
    Define the sweep configuration
    """
    
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'val/auc',
            'goal': 'maximize' 
            },
        'parameters': {
            'lr': {
                'values': [0.005]     # [0.0005, 0.0001, 0.00005]
            },
            'weight_decay': {
                'values': [1e-3]     # [1e-4, 1e-5], 1e-3 keep
            },
            'num_bags': {
                'values': [0.5]     # proportion 1 for all bags float for less     [50, 100, 150, 200, 300, 400, 500]
            },
            'mode': {
                'values': ['embedding']     # ['embedding', 'instance']
            },
            'pooling_type': {
                'values': ['attention']       # ['max', 'mean', 'attention', 'gated_attention']
            },
            'attspace_dim': {
                'values': [256]     # [128, 256, 512]
            },
        }
    }
    return sweep_config

# run 5 experiments per sweep configuration
if __name__ == "__main__":

    for i in range(1):
        project_name = 'AttDMIL-PML-HISTO-XAI'
        # Initialize a sweep
        sweep_config = main_sweep()
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        wandb.agent(sweep_id, function=train, count=1)
        print(f"{bcolors.OKGREEN}Sweep {i} completed!{bcolors.ENDC}")
        time.sleep(4)
    print("All sweeps completed successfully!")
