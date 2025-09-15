import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchinfo import summary

from src.dataset.dataset import MNISTBags
from src.dataset.HistoDataset import HistoDataset
from src.modules.config import MNISTBagsConfig, MILPoolingConfig, MILModelConfig, HistoBagsConfig
from src.modules.plotting import visualize_histo_att, visualize_histo_patches, visualize_histo_gt

class MILPooling(nn.Module):
    def __init__(
            self,
            mil_pooling_config: MILPoolingConfig
    ):
        super(MILPooling, self).__init__()
        assert mil_pooling_config.pooling_type in ['max', 'mean', 'attention', 'gated_attention'], \
            "pooling_type should be either 'max', 'mean', 'attention', or 'gated_attention'"
        
        self.pooling_type = mil_pooling_config.pooling_type
        self.feature_dim = mil_pooling_config.feature_dim
        self.attspace_dim = mil_pooling_config.attspace_dim
        self.attbranches = mil_pooling_config.attbranches

        if self.pooling_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.attspace_dim),
                nn.Tanh(),
                nn.Linear(self.attspace_dim, self.attbranches)
            )
        elif self.pooling_type == 'gated_attention':
            self.attention_V = nn.Sequential(
                nn.Linear(self.feature_dim, self.attspace_dim),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(self.feature_dim, self.attspace_dim),
                nn.Sigmoid()
            )
            self.attention_w = nn.Linear(self.attspace_dim, self.attbranches)

    def forward(self, x):

        if self.pooling_type == 'max':
            y_bag_pred = torch.max(x, dim=0).values
            return y_bag_pred, None                     # return (1, 1), None
        
        elif self.pooling_type == 'mean':
            y_bag_pred = torch.mean(x, dim=0)
            return y_bag_pred, None                     # return (1, 1), None
        
        elif self.pooling_type == 'attention':
            A = self.attention(x).squeeze(1)            # (num_instances, feature_dim) -> (num_instances, ATTENTION_BRANCHES)
            A = torch.softmax(A, dim=0)                 # (num_instances, ATTENTION_BRANCHES) -> (num_instances, ATTENTION_BRANCHES)
            x = A.T @ x                                 # (ATTENTION_BRANCHES, num_instances) @ (num_instances, feature_dim) -> (ATTENTION_BRANCHES, feature_dim)
            return x, A                                 # return (ATTENTION_BRANCHES, feature_dim), (num_instances, ATTENTION_BRANCHES)
        
        elif self.pooling_type == 'gated_attention':
            V = self.attention_V(x)                     # (num_instances, feature_dim) -> (num_instances, attspace_dim)
            U = self.attention_U(x)                     # (num_instances, feature_dim) -> (num_instances, attspace_dim)   
            A = self.attention_w(V * U).squeeze(1)      # (num_instances, attspace_dim) -> (num_instances, ATTENTION_BRANCHES)
            A = torch.softmax(A, dim=0)                 # (num_instances, ATTENTION_BRANCHES) -> (num_instances, ATTENTION_BRANCHES)
            x = A.T @ x                                 # (ATTENTION_BRANCHES, num_instances) @ (num_instances, feature_dim) -> (ATTENTION_BRANCHES, feature_dim)
            return x, A                                 # return (ATTENTION_BRANCHES, feature_dim), (num_instances, ATTENTION_BRANCHES)

class ConvFeatureExtractor(nn.Module):
    def __init__(
            self,
            mil_pooling_config: MILPoolingConfig
    ):
        self.feature_dim = mil_pooling_config.feature_dim
        
        super(ConvFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.fc = nn.Linear(50 * 4 * 4, self.feature_dim)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))    # (num_instances, c, h, w) -> (num_instances, 20, 24, 24)
        x = self.pool(x)                # (num_instances, 20, 24, 24) -> (num_instances, 20, 12, 12)
        x = self.relu(self.conv2(x))    # (num_instances, 20, 12, 12) -> (num_instances, 50, 8, 8)
        x = self.pool(x)                # (num_instances, 50, 8, 8) -> (num_instances, 50, 4, 4)
        x = x.view(x.size(0), -1)       # (num_instances, 50, 4, 4) -> (num_instances, 50 * 4 * 4)
        x = self.relu(self.fc(x))       # (num_instances, 50 * 4 * 4) -> (num_instances, feature_dim)
        return x

class MILModel(nn.Module):
    def __init__(
            self, 
            mil_model_config: MILModelConfig,
    ):
        super(MILModel, self).__init__()
        self.mode = mil_model_config.mode
        self.mil_model_config = mil_model_config
        self.mil_pooling_config = mil_model_config.mil_pooling_config

        if self.mode == 'instance':
            assert self.mil_pooling_config.pooling_type in ['mean', 'max'], \
                "For 'instance' mode, pooling_type must be 'mean' or 'max'"
        elif self.mode == 'embedding':
            assert self.mil_pooling_config.pooling_type in ['mean', 'max', 'attention', 'gated_attention'], \
                "For 'embedding' mode, pooling_type can be 'mean', 'max', 'attention', or 'gated_attention'"
        else:
            raise ValueError("mode should be either 'instance' or 'embedding'")

        if self.mil_model_config.just_features == False:
            self.feature_extractor = ConvFeatureExtractor(mil_pooling_config=self.mil_pooling_config)
        self.pooling = MILPooling(mil_pooling_config=self.mil_pooling_config)
        feature_dim = self.mil_pooling_config.feature_dim * self.mil_pooling_config.attbranches
        
        # Define the classifier
        if self.mode == 'instance':
            self.classifier = nn.Linear(feature_dim, 1)
        elif self.mode == 'embedding':
            self.bag_classifier = nn.Linear(feature_dim, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.mil_model_config.just_features == False:
            instance_features = self.feature_extractor(x)                           # (num_instances, c, h, w) -> (num_instances, feature_dim)
        else:
            instance_features = x 
                                        
        if self.mode == 'instance':
            y_instance_pred = self.sigmoid(self.classifier(instance_features))      # (num_instances, feature_dim) -> (num_instances, 1)
            y_bag_pred, _ = self.pooling(y_instance_pred)                           # (num_instances, 1) -> (1,)
            return y_bag_pred, y_instance_pred                                      
         
        elif self.mode == 'embedding':
            bag_representation, y_instance_pred = self.pooling(instance_features)   # (num_instances, feature_dim) -> (1, feature_dim), (num_instances, 1)/None
            y_bag_pred = self.sigmoid(self.bag_classifier(bag_representation))      # (1, feature_dim) -> (1, 1)
            return y_bag_pred, y_instance_pred


# test functionality of the different model approaches
if __name__ == "__main__":
    """
    test just one model approach
    """
    pml_cluster = False

    train_config = MILModelConfig(
        mode='embedding',
        batch_size=1,
        img_size=(1, 28, 28),
        dataset_config=HistoBagsConfig(
            seed=1,
            prop_num_bags=1,
            h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
            color_normalize=False,
            datatype="features",
            mode="test",
            val_mode=False,
            split=0.8,
            pml_cluster=pml_cluster,
        ),
        mil_pooling_config=MILPoolingConfig(
            pooling_type='attention',
            feature_dim=768,
            attspace_dim=128,
            attbranches=1
        ),
        just_features=True
    )
    train_data_loader = data_utils.DataLoader(
        HistoDataset(**train_config.dataset_config.__dict__),
        batch_size=train_config.batch_size,
        shuffle=False
    )
    model = MILModel(mil_model_config=train_config)
    summary(model,
           verbose=1,
           input_data={"x": torch.rand(1000, 768)},
    )
    for batch_idx, (features, label, cls, dict) in enumerate(train_data_loader):
            print(f"Batch {batch_idx}:")
            #print(f"Features shape: {features.shape}")  
            #print(f"Labels: {label}")                 
            #print(f"Classes: {cls}")
            #print(f"Dict: {dict}")

            # test histo vis
            #visualize_histo_att(model, (features, label, cls, dict), "/home/pml06/dev/attdmil/logs/histo/misc")
            #visualize_histo_patches(model, (features, label, cls, dict), "/home/pml06/dev/attdmil/logs/histo/misc")
            #visualize_histo_gt(model, (features, label, cls, dict), "/home/pml06/dev/attdmil/logs/histo/misc")
            
            features = features.squeeze(0)
            label = label            

            y_bag_pred, y_instance_pred = model(features)


            print(f"Bag {batch_idx} - True Label: {label}, Predicted Bag Score: {y_bag_pred.item()}, Predicted Instance Score: {y_instance_pred}")
            
            if batch_idx == 3:
                break

    # train_config = MILModelConfig(
    #     mode='embedding',
    #     batch_size=1,
    #     img_size=(1, 28, 28),
    #     dataset_config=MNISTBagsConfig(
    #         seed=1,
    #         positive_num=2,
    #         mean_bag_size=10,
    #         var_bag_size=2,
    #         num_bags=5,
    #         train=True,
    #         test_attention=False
    #     ),
    #     mil_pooling_config=MILPoolingConfig(
    #         pooling_type='attention',
    #         feature_dim=500,
    #         attspace_dim=128,
    #         attbranches=1
    #     ),
    #     just_features=False,
    # )
    # data_loader = data_utils.DataLoader(
    #     MNISTBags(**train_config.dataset_config.__dict__),
    #     batch_size=train_config.batch_size,
    #     shuffle=True
    # )
    # model = MILModel(mil_model_config=train_config)
    # summary(model,
    #         verbose=1,
    #         input_data={"x": torch.rand(train_config.batch_size, *train_config.img_size),},
    # )
    # for i, (bag, label) in enumerate(data_loader):
    #         bag = bag.squeeze(0)  # Remove batch dimension
    #         label = label[0].item()  # Convert label to a scalar

    #         # Forward pass through the model
    #         y_bag_pred, y_instance_pred = model(bag)

    #         print(f"Bag {i + 1} - True Label: {label}, Predicted Bag Score: {y_bag_pred.item()}, Predicted Instance Score: {y_instance_pred}")
    #         break 

    """
    test all model approach
    """
    # modes = ['instance', 'embedding']
    # pooling_types = ['max', 'mean', 'attention', 'gated_attention']

    # for mode in modes:
    #     for pooling_type in pooling_types:
    #         if mode == 'instance' and pooling_type in ['attention', 'gated_attention']:
    #             continue
    #         print(f"\nTesting {mode.capitalize()} Mode with {pooling_type.capitalize()} Pooling")

    #         train_config = MILModelConfig(
    #             mode=mode,
    #             batch_size=1,
    #             img_size=(1, 28, 28),
    #             dataset_config=MNISTBagsConfig(
    #                 seed=1,
    #                 positive_num=2,
    #                 mean_bag_size=10,
    #                 var_bag_size=2,
    #                 num_bags=5,
    #                 train=True,
    #                 test_attention=False
    #             ),
    #             mil_pooling_config=MILPoolingConfig(
    #                 pooling_type=pooling_type,
    #                 feature_dim=500,
    #                 attspace_dim=128,
    #                 attbranches=1
    #             )
    #         )
    #         data_loader = data_utils.DataLoader(
    #             MNISTBags(**train_config.dataset_config.__dict__),
    #             batch_size=train_config.batch_size,
    #             shuffle=True
    #         )
    #         model = MILModel(mil_model_config=train_config)
    #         summary(model,
    #                 verbose=1,
    #                 input_data={"x": torch.rand(train_config.batch_size, *train_config.img_size),}
    #         )
    #         for i, (bag, label) in enumerate(data_loader):
    #             bag = bag.squeeze(0)
    #             label = label[0].item()
    #             y_bag_pred, y_instance_pred = model(bag)

    #             print(f"Bag {i + 1} - True Label: {label}, Predicted Bag Score: {y_bag_pred.item()}, Predicted Instance Score: {y_instance_pred}")
    #             break
    
    # print("Model test passed!")