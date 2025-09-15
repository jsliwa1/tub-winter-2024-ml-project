import wandb
import time
import torch as th
import torch.utils.data as data_utils
from torchinfo import summary

from src.modules.config import MILModelConfig, MNISTBagsConfig, MILPoolingConfig
from src.dataset.dataset import MNISTBags
from model.mnist_mil_wrapper import AttDMILWrapper
from src.model.mil import MILModel
from src.modules.logger import prepare_folder, get_run_name, bcolors, WandbLogger
from src.modules.trainer import Trainer


def train(config=None):
    """
    Trains the MIL model using the provided configuration

    Args:
        config (dict): Dictionary containing the configuration parameters
    """

    base_log_dir = '/home/pml06/dev/attdmil/logs/local_gpu'

    with wandb.init(
            dir=base_log_dir,
            config=config,
        ):
        config = wandb.config
        base_log_dir = base_log_dir + f"/new_mu{config.mean_bag_size}"
        run_name = get_run_name(base_log_dir, f"{config.mode}_pool{config.pooling_type}_mu{config.mean_bag_size}_var{config.var_bag_size}_num{config.num_bags}")
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
            epochs=200,
            batch_size=1,
            img_size=(1, 28, 28),
            train_dataset_config=MNISTBagsConfig(
                seed=1,
                positive_num=9,
                mean_bag_size=config.mean_bag_size,
                var_bag_size=config.var_bag_size,
                num_bags=config.num_bags,
                train=True,
                test_attention=False
            ),
            val_dataset_config=MNISTBagsConfig(
                seed=1,
                positive_num=9,
                mean_bag_size=config.mean_bag_size,
                var_bag_size=config.var_bag_size,
                num_bags=1000,
                train=False,
                test_attention=False
            ),
            mil_pooling_config=MILPoolingConfig(
                pooling_type=config.pooling_type,
                feature_dim=500,
                attspace_dim=128,
                attbranches=1
            ),
            ckpt_path=None,
            lr=0.0005,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
            step_size=1000000,
            gamma=0.1,
            ckpt_save_path=ckpt_save_path,
            misc_save_path=misc_save_path,
            val_every=10,
            save_max=2,
            patience=3,
        )

        train_loader = data_utils.DataLoader(
            MNISTBags(**train_config.train_dataset_config.__dict__),
            batch_size=train_config.batch_size,
            shuffle=True
        )
        val_loader = data_utils.DataLoader(
            MNISTBags(**train_config.val_dataset_config.__dict__),
            batch_size=train_config.batch_size,
            shuffle=False
        )

        # Initialize model and wrapper
        model = MILModel(mil_model_config=train_config).to(train_config.device)
        wrapper = AttDMILWrapper(model=model, config=train_config, epochs=train_config.epochs)

        summary(model, input_data=th.rand(train_config.batch_size, *train_config.img_size).to(train_config.device))

        trainer = Trainer(
            device=train_config.device,
            wrapper=wrapper,
            misc_save_path=train_config.misc_save_path,
        )

        trainer.train(
            epochs=train_config.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=WandbLogger(log_dir=base_log_dir, run_name=run_name),
            ckpt_save_path=train_config.ckpt_save_path,
            ckpt_save_max=train_config.save_max,
            val_every=train_config.val_every,
            patience=train_config.patience,
        )


def main_sweep():
    """
    Define the sweep configuration
    """
    
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'val/loss',
            'goal': 'minimize' 
            },
        'parameters': {
            'mean_bag_size': {
                'value': 10             # [10, 50, 100] fixed
            },
            'var_bag_size': {
                'value': 2             # [2, 10, 20] fixed   
            },
            'num_bags': {
                'values': [50, 100, 150]     # [50, 100, 150, 200, 300, 400, 500]
            },
            'mode': {
                'values': ['embedding', 'instance']     # ['embedding', 'instance']
            },
            'pooling_type': {
                'values': ['max', 'mean', 'attention', 'gated_attention']       # ['max', 'mean', 'attention', 'gated_attention']
            },
        }
    }
    return sweep_config

# run 5 experiments per sweep configuration
if __name__ == "__main__":

    for i in range(5):
        project_name = 'AttDMIL-PML-MNIST'
        # Initialize a sweep
        sweep_config = main_sweep()
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        wandb.agent(sweep_id, function=train, count=56)
        print(f"{bcolors.OKGREEN}Sweep {i} completed!{bcolors.ENDC}")
        time.sleep(4)
    print("All sweeps completed successfully!")
