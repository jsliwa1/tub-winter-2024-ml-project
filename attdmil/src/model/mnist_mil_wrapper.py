import torch as th
import torch.utils.data as data_utils
from torchinfo import summary

from src.modules.ModelWrapperABC import ModelWrapper
from src.model.mil import MILModel
from src.dataset.dataset import MNISTBags
from src.modules.config import MILModelConfig, MNISTBagsConfig, MILPoolingConfig
from src.modules.metrics import LossErrorAccuracyPrecisionRecallF1Metric
from src.modules.plotting import visualize_gtbags, visualize_attMechanism


class AttDMILWrapper(ModelWrapper):
    """
    Wrapper class for attention-based deep MIL model.

    Attributes:
        model (MILModel): The MIL model to be trained.
        config (dict): The configuration parameters for the model.
        epochs (int): The number of training
    """
    def __init__(
            self,
            *,
            model: MILModel,
            config: dict,
            epochs: int,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.epochs = epochs

        if config.ckpt_path is not None:
            self.load_model_checkpoint(config.ckpt_path)
    
    def init_val_metrics(self):
        """
        Initializes the validation metrics.
        """
        self.val_metrics = LossErrorAccuracyPrecisionRecallF1Metric(model=self.model, device="cuda")

    def configure_optimizers(self):
        """
        Configures the optimizer and the learning rate scheduler.

        Returns:
            tuple: The optimizer and the learning rate scheduler.
        """
        print(f"Using base learning rate: {self.config.lr}")
        optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
        # optional: add lr_scheduler
        # lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.config.T_0,
        #     T_mult=self.config.T_mult,
        #     eta_min=self.config.eta_min
        # )

        # use lr_scheduler but constant lr
        lr_scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma
        )
        return optimizer, lr_scheduler

    def shared_step(
            self,
            model: MILModel,
            batch: tuple,
    ):
        """
        Shared step ideally for training and validation.

        Args:
            model (MILModel): The MIL model.
            batch (tuple): The bag and label.
        """
        bag, label = batch
        loss = self._loss(model, bag, label)
        error, acc = self._error(model, bag, label)
        
        return {"loss": loss, "error": th.tensor(error), "acc": th.tensor(acc)}

    def training_step(
            self,
            model: MILModel,
            batch: tuple,
    ):
        """
        Training step for the model.

        Args:
            model (MILModel): The MIL model.
            batch (tuple): The bag and label.

        Returns:
            loss_dict: The computed loss, error, and accuracy.
        """
        batch[0] = batch[0].squeeze(0)
        loss_dict = self.shared_step(model, batch)
        loss_dict["loss"].backward()
        
        return loss_dict

    def validation_step(
            self,
            batch: tuple,
    ):
        """
        Validation step for the model, update the validation metrics.

        Args:
            batch (tuple): The bag and label.
        """
        batch[0] = batch[0].squeeze(0)
        self.val_metrics.update(batch)

    def visualize_step(
            self,
            model: MILModel,
            batch: tuple,
            misc_save_path: str,
            global_step: int,
    ):
        """
        Visualizes the model predictions.

        Args:
            model (MILModel): The MIL model.
            batch (tuple): The bag and label.
            misc_save_path (str): The path to save the visualization files.
        """
        #visualize_gtbags(batch[0], batch[1], global_step, self.config.train_dataset_config.positive_num, False, misc_save_path)
        visualize_attMechanism(model, batch, self.config.train_dataset_config.positive_num, global_step, False, misc_save_path)

    def _error(
            self,
            model: MILModel,
            bag: th.Tensor,
            label: th.Tensor,
    ):
        """
        Computes the error and accuracy for the model.

        Args:
            model (MILModel): The MIL model.
            bag (th.Tensor): The input bag.
            label (th.Tensor): The input label.

        Returns:
            tuple: The error and accuracy values.
        """
        y_bag_true = label[0].float()
        y_bag_pred, y_instance_pred = model(bag)
        y_bag_pred_binary = th.where(y_bag_pred > 0.5, 1, 0)
        acc = th.mean((y_bag_pred_binary == y_bag_true).float()).item()
        error = 1.0 - acc
        return error, acc
    
    def _loss(
            self,
            model: MILModel,
            bag: th.Tensor,
            label: th.Tensor,
    ):
        """
        Computes the loss for the model.

        Args:
            model (MILModel): The MIL model.
            bag (th.Tensor): The input bag.
            label (th.Tensor): The input label.

        Returns:
            th.Tensor: The computed loss.
        """
        y_bag_true = label[0].float()
        y_bag_pred, y_instance_pred = model(bag)
        y_bag_pred = th.clamp(y_bag_pred, min=1e-4, max=1. - 1e-4)
        loss = th.nn.BCELoss()(y_bag_pred, y_bag_true)
        return loss

    def load_model_checkpoint(self, ckpt_path):
        """
        Loads the model from a checkpoint file.

        Args:
            ckpt_path (str): The path to the checkpoint file.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        try:
            model_state = th.load(ckpt_path) 
            self.model.load_state_dict(model_state["model"])
            print(f"Model loaded from {ckpt_path}")
        except Exception as e:
            print(f"Error loading model from {ckpt_path}: {e}")


# test functionality of the model wrapper
if __name__ == "__main__":

    train_config = MILModelConfig(
        mode='embedding',
        epochs=5,
        batch_size=1,
        img_size=(1, 28, 28),
        dataset_config=MNISTBagsConfig(
            seed=1,
            positive_num=2,
            mean_bag_size=10,
            var_bag_size=2,
            num_bags=10,
            train=True,
            test_attention=False
        ),
        mil_pooling_config=MILPoolingConfig(
            pooling_type='attention',
            feature_dim=500,
            attspace_dim=128,
            attbranches=1
        ),
        ckpt_path=None,
        lr=0.0005,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        step_size=1000000,
        gamma=0.1,
    )

    model = MILModel(mil_model_config=train_config)
    wrapper = AttDMILWrapper(model=model, config=train_config, epochs=train_config.epochs)

    summary(model, input_data=th.rand(train_config.batch_size, *train_config.img_size))

    data_loader = data_utils.DataLoader(
        MNISTBags(**train_config.dataset_config.__dict__),
        batch_size=train_config.batch_size,
        shuffle=True
    )

    optimizer, lr_scheduler = wrapper.configure_optimizers()

    for i, batch in enumerate(data_loader):
        result = wrapper.training_step(
            model=model,
            batch=batch,
        )
        print(f"Training Step {i + 1} - Loss: {result['loss']}, Error: {result['error']}, Accuracy: {result['acc']}")
        if i >= 2:
            break