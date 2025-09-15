import torch as th
import torch.utils.data as data_utils
from torchinfo import summary

from src.modules.ModelWrapperABC import ModelWrapper
from src.model.mil import MILModel
from src.dataset.HistoDataset import HistoDataset
from src.modules.config import MILModelConfig, HistoBagsConfig, MILPoolingConfig
from src.modules.metrics import LossErrorAccuracyPrecisionRecallF1Metric
from src.modules.plotting import visualize_histo_att, visualize_histo_gt, visualize_histo_patches, visualize_histo_smoothgrad, visualize_histo_shap, visualize_histo_lrp

class HistoMILWrapper(ModelWrapper):
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
        self.val_metrics = LossErrorAccuracyPrecisionRecallF1Metric(model=self.model, just_features=self.config.just_features, mode="val", device="cuda", misc_save_path=self.config.misc_save_path)
        self.test_metrics = LossErrorAccuracyPrecisionRecallF1Metric(model=self.model, just_features=self.config.just_features, mode="test", device="cuda", misc_save_path=self.config.misc_save_path)
    
    
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

        # use lr_scheduler but constant lr
        lr_scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma
        )
        return optimizer, lr_scheduler
    
    def training_step(
            self,
            model: MILModel,
            batch: tuple,
    ):
        features, label, cls, dict = batch
        features = features.squeeze(0)

        loss, attention_entropy_loss, cls_loss = self._loss(model, features, label)
        error, acc = self._error(model, features, label)

        loss.backward()

        return {"loss": loss, "error": th.tensor(error), "acc": th.tensor(acc), "attn_entropy_loss": attention_entropy_loss, "cls_loss": cls_loss}
    
    def validation_step(
            self,
            batch: tuple,
    ):
        batch[0] = batch[0].squeeze(0)
        self.val_metrics.update(batch)
    
    def test_step(
            self,
            batch: tuple,
    ):
        batch[0] = batch[0].squeeze(0)
        self.test_metrics.update(batch) 

    def visualize_step(
            self,
            model: MILModel,
            batch: tuple,
            misc_save_path: str,
            global_step: int,
            mode: str,
    ):
        if mode == "train":
            print("Visualizing train step")
            # visualize_histo_lrp(model, batch, misc_save_path, global_step, mode, "raw")
            # visualize_histo_lrp(model, batch, misc_save_path, global_step, mode, "percentile")
            visualize_histo_lrp(model, batch, misc_save_path, global_step, mode, "log")
            visualize_histo_smoothgrad(model, batch, misc_save_path, global_step, mode, "raw")
            # visualize_histo_smoothgrad(model, batch, misc_save_path, global_step, mode, "percentile")
            # visualize_histo_smoothgrad(model, batch, misc_save_path, global_step, mode, "log")
            visualize_histo_att(model, batch, misc_save_path, global_step, mode, "raw")
            # visualize_histo_gt(model, batch, misc_save_path)
            # visualize_histo_patches(model, batch, misc_save_path)
        # elif mode == "test":
        else:
            print("Visualizing test step")
            # visualize_histo_lrp(model, batch, misc_save_path, global_step, mode, "raw")
            # visualize_histo_lrp(model, batch, misc_save_path, global_step, mode, "percentile")
            visualize_histo_lrp(model, batch, misc_save_path, global_step, mode, "log")
            visualize_histo_smoothgrad(model, batch, misc_save_path, global_step, mode, "raw")
            # visualize_histo_smoothgrad(model, batch, misc_save_path, global_step, mode, "percentile")
            # visualize_histo_smoothgrad(model, batch, misc_save_path, global_step, mode, "log")
            visualize_histo_att(model, batch, misc_save_path, global_step, mode, "raw")
            # visualize_histo_shap(model, batch, misc_save_path, global_step, mode, "raw")
            # visualize_histo_att(model, batch, misc_save_path, global_step, mode, "raw")
            # visualize_histo_att(model, batch, misc_save_path, global_step, mode, "log")
            # visualize_histo_att(model, batch, misc_save_path, global_step, mode, "percentile")

    def _error(
            self,
            model: MILModel,
            features: th.Tensor,
            label: th.Tensor,
    ):
        y_bag_true = label
        y_bag_pred, y_instance_pred = model(features)
        y_bag_pred_binary = th.where(y_bag_pred > 0.5, 1, 0)
        acc = th.mean((y_bag_pred_binary == y_bag_true).float()).item()
        error = 1.0 - acc
        return error, acc
    
    def _loss(
            self,
            model: MILModel,
            features: th.Tensor,
            label: th.Tensor,
    ):
        att_weight = 0.0001

        y_bag_true = label
        y_bag_pred, y_instance_pred = model(features)
        y_bag_pred = th.clamp(y_bag_pred, min=1e-4, max=1. - 1e-4)
        loss = th.nn.BCELoss()(y_bag_pred, y_bag_true)
        attention_entropy_loss = th.tensor(0)
        cls_loss = th.tensor(0)
        return loss, att_weight*attention_entropy_loss, cls_loss
    
    def load_model_checkpoint(self, ckpt_path):
        try:
            model_state = th.load(ckpt_path) 
            self.model.load_state_dict(model_state["model"])
            print(f"Model loaded from {ckpt_path}")
        except Exception as e:
            print(f"Error loading model from {ckpt_path}: {e}")


if __name__ == "__main__":

    pml_cluster = False

    train_config = MILModelConfig(
        mode='embedding',
        epochs=5,
        batch_size=1,
        img_size=(1, 28, 28),
        dataset_config=HistoBagsConfig(
            seed=1,
            prop_num_bags=5,
            h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
            color_normalize=False,
            datatype="features",
            mode="train",
            val_mode=False,
            split=0.8,
            pml_cluster=pml_cluster
        ),
        mil_pooling_config=MILPoolingConfig(
            pooling_type='attention',
            feature_dim=768,
            attspace_dim=128,
            attbranches=1
        ),
        just_features=True,
        ckpt_path=None,
        lr=0.0005,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        step_size=1000000,
        gamma=0.1,
    )

    train_data_loader = data_utils.DataLoader(
        HistoDataset(**train_config.dataset_config.__dict__),
        batch_size=train_config.batch_size,
        shuffle=True
    )

    model = MILModel(mil_model_config=train_config)
    wrapper = HistoMILWrapper(model=model, config=train_config, epochs=train_config.epochs)

    summary(model, input_data=th.rand(1000, 768))


    optimizer, lr_scheduler = wrapper.configure_optimizers()
    wrapper.init_val_metrics()

    for batch_idx, batch in enumerate(train_data_loader):
        result = wrapper.training_step(
            model=model,
            batch=batch,
        )
        wrapper.validation_step(
            batch=batch,
        )
        print(f"Training Step {batch_idx} - Loss: {result['loss']}, Error: {result['error']}, Accuracy: {result['acc']}")
        if batch_idx >= 2:
            break