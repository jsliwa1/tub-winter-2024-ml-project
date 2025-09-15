from types import SimpleNamespace
from typing import Union

class MNISTBagsConfig(SimpleNamespace):
    seed: int
    positive_num: int
    mean_bag_size: int
    var_bag_size: float
    num_bags: int
    train: bool
    test_attention: bool


class HistoBagsConfig(SimpleNamespace):
    seed: int
    num_bags: int
    h5_path: str
    color_normalize: bool
    datatype: str
    mode: str
    split: float
    pml_cluster: bool


class MILPoolingConfig(SimpleNamespace):
    pooling_type: str
    feature_dim: int  #M
    attspace_dim: int  #L
    attbranches: int


class MILModelConfig(SimpleNamespace):
    device: str
    mode: str
    epochs: int
    batch_size: int
    img_size: tuple[int, int, int]
    dataset_config: Union[MNISTBagsConfig, HistoBagsConfig]
    mil_pooling_config: MILPoolingConfig
    ckpt_path: str
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    T_0: int
    T_mult: int
    eta_min: float
    step_size: int
    gamma: float
    ckpt_save_path: str
    misc_save_path: str
    val_every: int
    save_max: int
    patience: int

    # MS2 addition
    just_features: bool

