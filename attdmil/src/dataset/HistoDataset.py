import torch as th
import ast
import os
import h5py
import numpy as np
import pandas as pd
import random
from PIL import Image
import torch.utils.data as data_utils
from torchvision import datasets, transforms


from src.modules.config import HistoBagsConfig
from src.modules.utils import label_to_logits, cls_to_logits, create_metadata


class HistoDataset(data_utils.Dataset):
    def __init__(
        self, 
        *,
        seed: int,
        prop_num_bags: float,
        h5_path: str,
        color_normalize: bool,
        datatype: str,
        mode: str,
        val_mode: bool,
        split: float,
        pml_cluster: bool,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.color_normalize = color_normalize
        self.mode = mode
        self.datatype = datatype

        metadata_path_cluster = "/home/space/datasets/camelyon16/metadata/v001/slide_metadata.csv"
        case_metadata_path_cluster = "/home/space/datasets/camelyon16/metadata/v001/case_metadata.csv"

        self.modif_metadata_path_cluster = "/home/pml06/dev/attdmil/HistoData/metadata/split_metadata.csv"

        self.feature_path_cluster = "/mnt/datasets/camelyon16/features/20x/ctranspath_pt/"  if pml_cluster else "/home/space/datasets/camelyon16/features/20x/ctranspath_pt/"
        self.patch_metadata_path_cluster = "/mnt/datasets/camelyon16/patches/20x/"  if pml_cluster else "/home/space/datasets/camelyon16/patches/20x/"
        self.annotations_path = "/mnt/datasets/camelyon16/annotations" if pml_cluster else "/home/space/datasets/camelyon16/annotations"

        # main function to load the data from our .csv file
        if os.path.isfile(self.modif_metadata_path_cluster):
            print(f"Loading modified metadata from {self.modif_metadata_path_cluster}")
            slide_metadata = pd.read_csv(self.modif_metadata_path_cluster)

            train_lst = slide_metadata[slide_metadata['split'] == 'train']['slide_id'].tolist()
            val_lst = slide_metadata[slide_metadata['split'] == 'val']['slide_id'].tolist()
            test_lst = slide_metadata[slide_metadata['split'] == 'test']['slide_id'].tolist()

            print(f"Number of training cases: {len(train_lst)}")
            print(f"Number of validation cases: {len(val_lst)}")
            print(f"Number of test cases: {len(test_lst)}")

            random.seed(seed)
            random.shuffle(train_lst)
            random.shuffle(val_lst)

            if val_mode == False and mode == 'train':
                self.name_lst = train_lst if prop_num_bags == 1 else train_lst[:int(len(train_lst)*prop_num_bags)]
            elif val_mode == True and mode == 'train':
                self.name_lst = val_lst if prop_num_bags == 1 else val_lst[:int(len(val_lst)*prop_num_bags)]
            elif mode == 'test':
                self.name_lst = test_lst if prop_num_bags == 1 else test_lst[:int(len(test_lst)*prop_num_bags)]

        # old function with .h5 file was too slow
        else:

            with h5py.File(self.h5_path, "r+") as h5:
                if self.mode == 'train':
                    self.name_lst = list(h5["train"].keys())
                    print(f"Number of training cases: {len(self.name_lst)}")
                    print(f"Apply train-val split with split ratio: {split}")

                    slide_metadata = pd.read_csv(metadata_path_cluster)
                    case_metadata = pd.read_csv(case_metadata_path_cluster)

                    # Check for cases/patients with multiple slides
                    grouped_cases = slide_metadata.groupby('case_id')['slide_id'].apply(list).to_dict()
                    case_to_slides = {case: slides for case, slides in grouped_cases.items() if len(slides) > 1}

                    print(f"Number of cases with multiple slides: {len(case_to_slides)}")

                    def split_group(entries, split_ratio):
                        train_split = []
                        val_split = []

                        for case in entries:
                            case_slides = grouped_cases.get(case, [case])
                            if len(train_split) < int(len(entries) * split_ratio):
                                train_split.extend(case_slides)
                            else:
                                val_split.extend(case_slides)

                        return train_split, val_split

                    normal_entries = [name for name in self.name_lst if name.startswith("normal")]
                    tumor_entries = [name for name in self.name_lst if name.startswith("tumor")]
                    print(f"Number of normal cases: {len(normal_entries)} in proportion: {len(normal_entries)/len(self.name_lst)}")
                    print(f"Number of tumor cases: {len(tumor_entries)} in proportion: {len(tumor_entries)/len(self.name_lst)}")
                    micro_tumors = []
                    macro_tumors = []

                    for case in tumor_entries:
                        if h5[self.mode][case].attrs["class"] == "micro":
                            micro_tumors.append(case)
                        elif h5[self.mode][case].attrs["class"] == "macro":
                            macro_tumors.append(case)
                    print(f"Number of micro tumors: {len(micro_tumors)} in proportion: {len(micro_tumors)/len(tumor_entries)}")
                    print(f"Number of macro tumors: {len(macro_tumors)} in proportion: {len(macro_tumors)/len(tumor_entries)}")

                    train_lst = []
                    val_lst = []

                    normal_train, normal_val = split_group(normal_entries, split)
                    micro_train, micro_val = split_group(micro_tumors, split)
                    macro_train, macro_val = split_group(macro_tumors, split)

                    train_lst = normal_train + micro_train + macro_train
                    val_lst = normal_val + micro_val + macro_val

                    test_lst = list(h5["test"].keys())

                    # create a .csv with the fixed split for train and val
                    if not os.path.isfile(self.modif_metadata_path_cluster):
                        create_metadata(normal_train ,micro_train,macro_train, normal_val,micro_val,macro_val, test_lst, slide_metadata, case_metadata, modif_metadata_path_cluster)

                    random.seed(seed)
                    random.shuffle(train_lst)
                    random.shuffle(val_lst)

                    if val_mode == False:
                        self.name_lst = train_lst if prop_num_bags == 1 else train_lst[:int(len(train_lst)*prop_num_bags)]
                    elif val_mode == True:
                        self.name_lst = val_lst if prop_num_bags == 1 else val_lst[:int(len(val_lst)*prop_num_bags)]

                    if "splitting" not in h5:
                        print("Creating splitting table...")
                        all_cases = train_lst + val_lst
                        split_info = []
                        for case in all_cases:
                            split = "train" if case in train_lst else "val"

                            if case.startswith("normal"):
                                cls = "normal"
                                label = "normal"
                            elif case in micro_tumors:
                                cls = "micro"
                                label = "tumor"
                            elif case in macro_tumors:
                                cls = "macro"
                                label = "tumor"
                            else:
                                raise ValueError(f"Unknown case type for: {case}")

                            split_info.append({"case_name": case, "split": split, "class": cls, "label": label})

                        split_df = pd.DataFrame(split_info).sort_values(by="case_name")

                        csv_data = split_df.to_csv(index=False).encode("utf-8")
                        h5.create_dataset("splitting", data=csv_data)
                        print("Splitting table saved successfully.")
                    else:
                        print("Splitting table already exists in the HDF5 file.")
                    h5.close()

                elif self.mode == 'test':
                    self.name_lst = list(h5["test"].keys())
             
    def __len__(self):
        return len(self.name_lst)

    def __getitem__(self, idx):
        if self.datatype == "features":
            return self._get_features(idx)
        elif self.datatype == "features_for_vis":
            return self._get_features_for_vis(idx)
        elif self.datatype == "patches":
            return self._get_patches(idx)
        elif self.datatype == "coordinates":
            return self._get_patch_coordinates(idx)
    
    def _get_patch_coordinates(self, idx):
        
        coordinates = {}
        case_name = self.name_lst[idx]
        self.h5 = h5py.File(self.h5_path, "r")
        self.patch_lst = list(self.h5[self.mode][case_name].keys())
        self.patch_lst.remove("features")
        self.patch_lst.remove("metadata")

        print(f"Case: {case_name}")
        print(f"Number of patches: {len(self.patch_lst)}")
        for patch_name in self.patch_lst:
            coordinates[patch_name] = ast.literal_eval(self.h5[self.mode][case_name][patch_name].attrs["position_abs"])

        return coordinates

    def _get_patches(self, idx):
        pass
    
    def _get_features(self, idx):
        
        case_name = self.name_lst[idx]
        feature_path = os.path.join(self.feature_path_cluster, f"{case_name}.pt")
        features = th.load(feature_path, weights_only=True)


        slide_metadata = pd.read_csv(self.modif_metadata_path_cluster)
        label = slide_metadata[slide_metadata['slide_id'] == case_name].iloc[0]['label']
        label = label_to_logits(label)

        cls = slide_metadata[slide_metadata['slide_id'] == case_name].iloc[0]['class']
        cls = cls_to_logits(cls)

        patch_ordering_dict = {'case_name': case_name}

        return features.squeeze(1), label, cls, patch_ordering_dict
    
    def _get_features_for_vis(self, idx):
        
        case_name = self.name_lst[idx]
        feature_path = os.path.join(self.feature_path_cluster, f"{case_name}.pt")
        features = th.load(feature_path, weights_only=True)


        slide_metadata = pd.read_csv(self.modif_metadata_path_cluster)
        label = slide_metadata[slide_metadata['slide_id'] == case_name].iloc[0]['label']
        label = label_to_logits(label)

        cls = slide_metadata[slide_metadata['slide_id'] == case_name].iloc[0]['class']
        cls = cls_to_logits(cls)


        patch_meta_path = os.path.join(self.patch_metadata_path_cluster, case_name, "metadata", "df.csv")
        patch_meta_df = pd.read_csv(patch_meta_path)

        patch_id_lst = list(patch_meta_df["patch_id"])
        position_abs_lst = list(patch_meta_df["position_abs"])
        position_abs_lst = [ast.literal_eval(pos) for pos in position_abs_lst]
        patch_size_abs_lst = list(patch_meta_df["patch_size_abs"])

        annotations_path = os.path.join(self.annotations_path, f"{case_name}.png")
        if os.path.exists(annotations_path):
            annotations = Image.open(annotations_path)
            width, height = annotations.size
            original_shape = (width, height)

        patch_ordering_dict = self.patch_ordering(patch_id_lst, position_abs_lst, patch_size_abs_lst, original_shape, case_name)

        return features.squeeze(1), label, cls, patch_ordering_dict
    
    
        
    def patch_ordering(
            self,
            patch_id_lst: list,
            position_abs_lst: list,
            patch_size_abs_lst: list,
            original_shape: tuple,
            case_name: str,
    ):
        dict = {i: [patch_id_lst[i], position_abs_lst[i], patch_size_abs_lst[i]] for i in range(len(patch_id_lst))}
        dict['original_shape'] = original_shape
        dict['case_name'] = case_name
        return dict
    

if __name__ == "__main__":
    train_dataset = HistoDataset(
        seed=1,
        prop_num_bags=0.3, # 1.0 for all bags float for proportion of bags
        h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
        color_normalize=False,
        datatype="features",
        mode="train",
        val_mode=False,
        split=0.8,
        pml_cluster=False,
    )
    val_dataset = HistoDataset(
        seed=1,
        prop_num_bags=0.3, # 1.0 for all bags float for proportion of bags
        h5_path="/home/pml06/dev/attdmil/HistoData/camelyon16.h5",
        color_normalize=False,
        datatype="features",
        mode="train",
        val_mode=True,
        split=0.8,
        pml_cluster=False,
    )

    train_data = train_dataset[0]
    val_data = val_dataset[0]

    train_data_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=1,      
        shuffle=True,     
        num_workers=0,    
        pin_memory=True   
    )

    val_data_loader = data_utils.DataLoader(
        val_dataset, 
        batch_size=1,     
        shuffle=True,     
        num_workers=0,    
        pin_memory=True   
    )

    for batch_idx, (features, labels, classes, dict) in enumerate(train_data_loader):
        print(f"Batch {batch_idx}:")
        print(f"Features shape: {features.shape}")  
        print(f"Labels: {labels}")                      
        print(f"Classes: {classes}")                   
        #print(f"Dict: {dict}")                         
    
        if batch_idx == 10:
            break
    
    for batch_idx, (features, labels, classes, dict) in enumerate(val_data_loader):
        print(f"Batch {batch_idx}:")
        print(f"Features shape: {features.shape}")
        print(f"Labels: {labels}")
        print(f"Classes: {classes}")
        #print(f"Dict: {dict}")

        if batch_idx == 10:
            break